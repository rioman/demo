#!/usr/local/anaconda/bin/python
#command line arguments
#-r <int>: shift statistics calculations for denoted number of days ago
#-s <country>: start with selected country and go on in alphabetically order
#-t <id_test> calculate statistics for selected test only
#-c <country_list>: calculate statistics for selected countries.
#List should be without spaces and separated by comma

import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from psutil import virtual_memory
from clickhouse_driver import Client as CHClient
from pymongo import MongoClient

def min1(series):
    return min(series)


def transform(value):
    return np.arctan((value-1)*10)/10+1


def bootstrapStatistics(values, samples, percent=10):
    try:
        samples = np.mean(np.random.choice(values, [samples, len(values)]), axis=1)
        borders = np.percentile(samples, [percent/2, 100-percent/2])
        return np.mean(values), borders[0], borders[1]
    except ValueError:
        return np.NaN, np.NaN, np.NaN


def next_filename(template):
    if not os.path.isfile(template.format('')):
        return template.format('')
    template = template.format('({})')
    tr = 1
    while os.path.isfile(template.format(tr)):
        tr += 1
    return template.format(tr)


def full_revenue(trustI):
    try:
        nfanswer = pd.DataFrame.from_records(CH.execute(f"""
            SELECT search_date_, search_test_group, search_id,
                   sum(click_revenue*
                          if(isNaN(click_first_uniq), 0, click_first_uniq))
            FROM {CC}.search ALL INNER JOIN {CC}.impression USING (search_id)
            WHERE search_test_id = {test_id}
            AND search_date_ BETWEEN '{start_date}' AND '{end_date}'
            GROUP BY search_date_, search_test_group, search_id;"""),
            columns=['search_date_', 'search_test_group',
                     'search_id', 'obtained_revenue_sum'])
        nfanswer.loc[nfanswer['obtained_revenue_sum'].isnull(),
                     'obtained_revenue_sum'] = 0

        for i, row in trustI[trustI['level_2'] == 'revenue'].iterrows():
            trustI.iloc[i, -3:] = bootstrapStatistics(
                nfanswer.loc[(nfanswer['search_date_'] <= row['level_0'].date()) &
                (nfanswer['search_test_group'] == row['level_1']) &
                np.logical_not(np.isnan(nfanswer['obtained_revenue_sum'])),
                             'obtained_revenue_sum'], 1000)
    except Exception as e:
        for i, row in trustI[trustI['level_2'] == 'revenue'].iterrows():
            trustI.iloc[i, -3:] = np.NaN, np.NaN, np.NaN
        return np.NaN
    return len(set(nfanswer['search_id']))


parser = argparse.ArgumentParser(description='Get stats')

parser.add_argument('-r', action="store", dest="date_shift", default=0, type=int)
parser.add_argument('-s', action="store", dest="start_country", default='aa')
parser.add_argument('-t', action="store", dest="id_test", default=1, type=int)
parser.add_argument('-c', action="store", dest="countries")

args = parser.parse_args()

date_param = datetime.now()-timedelta(days=args.date_shift)
start_date = '{:%Y-%m-%d}'.format(date_param-timedelta(days=11))
end_date = '{:%Y-%m-%d}'.format(date_param)

base_folder = '/mnt/storage4/pyservices2/AB/static'
save_folder = '{}/plots/{:%Y-%m-%d}'.format(base_folder, date_param)
report_file = '{}/reports/{:%Y-%m-%d}.tsv'.format(base_folder, date_param)
log_file = '/mnt/storage/common/logs/automated_stats.txt'
mem_log_file = '/mnt/storage/common/logs/automated_stats_memory.txt'
model_repo = '...rescorer_data/logreg_models/'
graph_template = 'st{test_id}-{country}-{metrics}-top{top}-ntop{nontop}-' \
               'filter{filter}-abs{abs}-{date:%Y-%m-%d}{dup}.png'
graph_composite_template = 'st{test_id}-{country}-allmetrics-top{top}-' \
                'ntop{nontop}-{date:%Y-%m-%d}{dup}.png'
metrics = {'ctr':['click_first_uniq_sum', 'CTR'],
           'fraction':['click_first_uniq_max', '%'],
           'skipadd':['last_click_min', 'skip'],
           'skip':['last_click_min1', 'skip'],
           'jdp':['jdp_respond_uniq_sum', 'jdp'],
           'revenue':['obtained_revenue_sum', '$']}
CH_HOST = 'lacerta.portal.com'
CH_PORT_TCP = 9000
CH = CHClient(host=CH_HOST, port=CH_PORT_TCP)
MG = MongoClient('mongodb://equuleus.portal.com:27017')
colors = ["#56B4E9", "#009E73", "#FF3957", "#004ED3", "#E69F00", "#333333"]

logging.basicConfig(filename=log_file, level=logging.INFO)
mem_log = open(mem_log_file, 'a')

columns = [
    'search_id',
    'search_kw_hash',
    'search_test_id',
    'search_test_group',
    'search_date',
    'search_date_',
    'impr_position',
    'click_first_uniq',
    'click_destination',
    'jdp_respond_uniq',
    'click_revenue']

cclist = [] if args.countries is None else args.countries.split(',')

countries = [db[0] for db in CH.execute('show databases;')
             if ((len(db[0]) == 2) & (db[0] >= args.start_country) &
                ((len(cclist) == 0) | (db[0] in cclist)))]

testlist = []

for country in countries:
    testlist.extend([(country, test_id, cnt) for (test_id, cnt) in CH.execute("""
        SELECT search_test_id, count(distinct search_id)
        FROM {}.search
        WHERE
        search_date_ BETWEEN '{:%Y-%m-%d}' AND '{:%Y-%m-%d}'
        group by search_test_id
        ;""".format(country, date_param-timedelta(days=2), date_param))
                     if test_id is not None
                     and test_id > 9
                     and (args.id_test < 3
                   or test_id == args.id_test)])

if len(testlist) == 0:
    logging.info('{:%Y-%m-%d}: No data for today'.format(date_param))
else:
    try:
        os.makedirs(save_folder, exist_ok=True)
        os.chmod(save_folder, 0o777)
    except Exception as e:
        pass
    os.chdir(save_folder)
    report = open(report_file, 'a')
    for test_data in testlist:
        report.write('\t'.join(str(x) for x in test_data) + '\n')
        logging.info('{:%Y-%m-%d}: Test {} for {} will be proceeded'.format(
            date_param, test_data[1], test_data[0]))
    report.close()
    try:
        os.chmod(report_file, 0o777)
    except Exception as e:
        pass

for CC, test_id, _cnt in testlist:
    try:
        hash_by_groups = [set(rec['hashes']) for rec in
                          list(MG.searchTest.searchTestData.find(
                              {'country':CC, 'testId':test_id})
                               .sort('dateUpdated', -1))[0]['groups']
                          if len(rec['hashes']) > 0]
        hashes_top = hash_by_groups[0]
    except Exception as e:
        hashes_top = set()

    condition = f"""
    AND session_ip_cc = '{CC}'
    AND search_kw_hash != 0
    AND impr_max_position < 80
    AND impr_position <= 20
    AND (impr_on_screen > 0 OR jdp_respond_uniq >0)
    """
    query = f"""
    SELECT {', '.join(columns)}
    FROM {CC}.search ALL INNER JOIN {CC}.impression USING (search_id)
    WHERE search_test_id = {test_id}
    AND search_date_ BETWEEN '{start_date}' AND '{end_date}'
    """

    try:
        answer = pd.DataFrame.from_records(
            CH.execute(query + condition + ';'),
            columns=columns)
    except Exception as e:
        logging.error('{:%Y-%m-%d}: '.format(date_param) + str(e))
        answer = pd.DataFrame()
    #answer = answer.drop(['search_test_id', 'search_date'], axis=1).groupby(['search_date_', 'search_test_group', 'search_kw_hash', 'search_id', 'impr_position']).max().reset_index()

    if answer.shape[0] == 0:
        logging.info('{:%Y-%m-%d}: '.format(date_param) +
                     'No data for {}'.format(CC))
        continue

    logging.info('{:%Y-%m-%d}: '.format(date_param) +
                 '{} records obtainned for {}'.format(answer.shape[0], CC))

    marginBaseGroup = '0'# if len(set(answer['search_test_group'])) < 5 else '1'

    answer['last_click'] = (
        answer['impr_position']*answer['click_first_uniq'] +
        100*(1-answer['click_first_uniq']))
    answer['obtained_revenue'] = (
        answer['click_revenue']*answer['click_first_uniq'])
    logging.info('{:%Y-%m-%d}: '.format(date_param) + 'point 10')

    a1 = answer.groupby(['search_date_', 'search_test_group',
                         'search_kw_hash', 'search_id']).agg(
                             {'click_first_uniq':['sum', 'max'],
                              'jdp_respond_uniq':'sum',
                              'last_click':['min', min1],
                              'obtained_revenue':'sum'})
    a1.columns = ['_'.join(col) for col in a1.columns.values]
    a1.reset_index(inplace=True)
    a1.loc[a1['last_click_min'] == 100, 'last_click_min'] = 30
    a1.loc[a1['last_click_min1'] == 100, 'last_click_min1'] = np.nan
    a1['top_hash'] = a1['search_kw_hash'].isin(hashes_top).astype(int)

    logging.info('{:%Y-%m-%d}: '.format(date_param) + 'point 20')

    for idx, (top, nontop) in enumerate([(1, 1), (1, 0), (0, 1)]):
        trustI = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [sorted(set(answer['search_date_'])),
                 sorted(set(answer['search_test_group'])),
                 metrics.keys()]),
            columns=['cnt', 'mean', 'q05', 'q95'],
            dtype=np.float64).reset_index()

        for i, row in trustI.iterrows():
            search_mask = ((a1['search_date_'] <= row['level_0'].date()) &
                   (a1['search_test_group'] == row['level_1']) &
                   ((a1['top_hash'] == top) |
                   (a1['top_hash'] == 1 - nontop)) &
                    np.logical_not(np.isnan(a1[metrics[row['level_2']][0]])))
            trustI.iloc[i, -3:] = bootstrapStatistics(
                a1.loc[search_mask, metrics[row['level_2']][0]],
                1000)
            trustI.iloc[i, -4] = np.sum(search_mask)
        logging.info('{:%Y-%m-%d}: '.format(date_param) +
                     f'Trust intervals are calculated for {CC}, top = {top}')

        if idx == 1:
            nf_cnt = full_revenue(trustI)
            logging.info('{:%Y-%m-%d}: '.format(date_param) +
                         f'Full revenue data are obtained for {CC}')
        elif idx == 2:
            pass
        trustI['level_1'] = trustI['level_1'].astype(str)

        if np.sum(trustI['cnt'] > 300) > 0:
            trustI = trustI[trustI['cnt'] > 300]

        tttAll = trustI[trustI['level_0'] == max(trustI['level_0'])]
        tttAll = (pd.merge(tttAll,
                           tttAll.groupby(['level_2'])
                           .agg({'mean':['mean']})
                           .reset_index(),
                           on='level_2'))

        tttAll['mean'] = transform(tttAll['mean']/tttAll[('mean', 'mean')])
        tttAll['q95'] = transform(tttAll['q95']/tttAll[('mean', 'mean')])
        tttAll['q05'] = transform(tttAll['q05']/tttAll[('mean', 'mean')])
        tttAll.loc[(tttAll['level_2'] == 'skip') |
                   (tttAll['level_2'] == 'skipadd'),
                   ['mean', 'q05', 'q95']] = (2-
                            tttAll.loc[(tttAll['level_2'] == 'skip') |
                            (tttAll['level_2'] == 'skipadd'),
                                       ['mean', 'q05', 'q95']])

        logging.info('{:%Y-%m-%d}: '.format(date_param) +
                     f'Baseline is calculated for {CC}, top = {top}')

        plt.figure(figsize=(8, 6), dpi=120)
        plt.style.use('ggplot')
        plt.subplot(111, projection='polar')

        try:
            for i, group in enumerate(sorted(set(trustI['level_1']))):
                tttI = tttAll[(tttAll['level_1'] == group)]
                plt.errorbar(x=np.array(range(len(tttI['level_2'])))+i/30,
                             y=tttI['mean'],
                             yerr=[tttI['mean']-tttI['q05'],
                                   tttI['q95']-tttI['mean']],
                             color=colors[i], elinewidth=0.6, linewidth=0.8)
                plt.fill(np.array(range(len(tttI['level_2'])))+i/30, tttI['mean'],
                         edgecolor=colors[i], linewidth=0.8,
                         fill=False, label='_nolegend_')
            
            plt.xticks(range(len(tttI['level_2'])), tttI['level_2'])
            plt.ylim(ymin=min(tttAll['q05']), ymax=max(tttAll['q95']))
            yticks = plt.yticks()
            plt.yticks(yticks[0], ['']*len(yticks[0]))
            plt.title('{country}\t\t\ttest {test_id}\t\t({cnt} searches)'
                       .format(country=CC,
                               test_id=test_id,
                               cnt=sum((a1['top_hash'] == top) |
                                  (a1['top_hash'] == 1 - nontop)))
                       .expandtabs())
            L = plt.legend()
            for i in range(len(set(trustI['level_1']))):
                L.get_texts()[i].set_text(i)
            plt.savefig(next_filename(graph_composite_template.format(
                test_id=test_id, country=CC, top=top,
                nontop=nontop, date=date_param, dup='{}')))
    
            logging.info('{:%Y-%m-%d}: '.format(date_param) +
                         f'Stargrams are built for {CC}, top = {top}')
        except ValueError as e:
            logging.info('{:%Y-%m-%d}: '.format(date_param) +
                         f'Stargrams error built for {CC}: {e}')

        basedate = min(trustI['level_0'])
        numdays = (max(trustI['level_0']) - basedate).days + 1

        for abs in [1, 0]:
            for j, metric in enumerate(metrics.items()):
                plt.figure(figsize=(8, 6), dpi=120)
                plt.style.use('ggplot')
                for i, group in enumerate(sorted(set(trustI['level_1']))):
                    tttI = trustI[(trustI['level_2'] == metric[0]) &
                                  (trustI['level_1'] == group)]
                    try:
                        plt.errorbar(x=(tttI['level_0']-
                                        basedate).dt.days+i/30,
                                     y=tttI['mean'],
                                     yerr=[tttI['mean']-tttI['q05'],
                                           tttI['q95']-tttI['mean']],
                                     color=colors[i], elinewidth=0.8)
                    except Exception as e:
                        pass

                plt.xticks(range(numdays),
                           [(basedate + timedelta(days=dt)).strftime('%m-%d')
                                                    for dt in range(numdays)]
                           )
                plt.xlabel('Dates')
                plt.ylabel(metric[1][1])
                try:
                    L = plt.legend()
                    for i in range(len(set(trustI['level_1']))):
                        try:
                            L.get_texts()[i].set_text(i)
                        except Exception as e:
                            pass
                except IndexError as e:
                    pass
                if metric[0][:4] == 'skip':
                    plt.gca().invert_yaxis()

                if (metric[0] == 'ctr') & (abs == 0):
                    ylim = plt.gca().get_ylim()
                    plt.ylim(ymin=min(ylim[0], -0.1), ymax=max(ylim[1], 0.1))

                if (metric[0] == 'revenue') & (idx>0):
                    if idx == 1:
                        plt.title((CC+'\t\t\ttest {}'.format(test_id)+
                                   '\t\t({} searches)'
                                   .format(nf_cnt)).expandtabs())
                        plt.savefig(next_filename(
                            graph_template.format(test_id=test_id, country=CC,
                                                  metrics=metric[0], top=1,
                                                  nontop=1,
                                                  filter=0, abs=abs,
                                                  date=date_param, dup='{}')))
                    elif idx == 2:
                        pass
                else:
                    plt.title('{country}\t\t\ttest {test_id}\t\t({cnt} searches)'
                              .format(country=CC,
                               test_id=test_id,
                               cnt=sum((a1['top_hash'] == top) |
                                       (a1['top_hash'] == 1 - nontop)))
                              .expandtabs())

                    plt.savefig(next_filename(
                        graph_template.format(test_id=test_id, country=CC,
                                              metrics=metric[0], top=top,
                                              nontop=nontop, filter=1,
                                              abs=abs, date=date_param, dup='{}')))
                plt.clf()
            trustI = trustI.merge(
                trustI.loc[trustI['level_1'] <= marginBaseGroup,
                           ['level_0', 'level_2', 'mean']]
                .groupby(['level_0', 'level_2'])
                .agg('mean').reset_index()
                .rename({'mean':'base'}, axis='columns'),
                on=['level_0', 'level_2'])
            try:
                trustI['mean'] -= trustI['base']
                trustI['q05'] -= trustI['base']
                trustI['q95'] -= trustI['base']
            except Exception as e:
                pass

            logging.info('{:%Y-%m-%d}: '.format(date_param) +
                         f'Graphs are built for {CC}, top = {top}, abs = {abs}')

        if len(hashes_top) == 0:
            break

    mem_log.write(('\t'.join(['{:%Y-%m-%d}'.format(datetime.now()), CC,
                              str(virtual_memory().percent)])
                   + '\n'))
    mem_log.flush()

mem_log.close()
