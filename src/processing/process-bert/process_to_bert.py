import pandas as pd
import numpy as np

np.random.seed(42)
import random

random.seed(42)

from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt

from transformers import AutoTokenizer

from sklearn.preprocessing import LabelEncoder

from pdb import set_trace

BUILD_LSPC = True
BUILD_DEEPMATCHER = True
BUILD_DI2KG = True

def process_to_bert(dataset, attributes, tokenizer, comb_func, cutting_func=None, multi_encoder=None):
    dataset = dataset.fillna('')

    if multi_encoder is None:
        try:
            cluster_id_set_left = set()
            cluster_id_set_left.update(dataset['cluster_id_left'].tolist())
            cluster_id_set_right = set()
            cluster_id_set_right.update(dataset['cluster_id_right'].tolist())
            cluster_id_set_left.update(cluster_id_set_right)
            dataset = dataset.rename(columns={'cluster_id_left': 'label_multi1', 'cluster_id_right': 'label_multi2'})
            label_enc = LabelEncoder()
            label_enc.fit(list(cluster_id_set_left))
            dataset['label_multi1'] = label_enc.transform(dataset['label_multi1'])
            dataset['label_multi2'] = label_enc.transform(dataset['label_multi2'])

        except KeyError:
            pass
    else:
        dataset = dataset.rename(columns={'cluster_id_left': 'label_multi1', 'cluster_id_right': 'label_multi2'})
        try:
            dataset['label_multi1'] = multi_encoder.transform(dataset['label_multi1'])
            dataset['label_multi2'] = multi_encoder.transform(dataset['label_multi2'])
        except ValueError:
            dataset['label_multi1'] = 0
            dataset['label_multi2'] = 0

    print(f'Before cutting:')
    _print_attribute_stats(dataset, attributes)
    if cutting_func:
        tqdm.pandas(desc='Cutting attributes')
        dataset = dataset.progress_apply(cutting_func, axis=1)
        print(f'After cutting:')
        _print_attribute_stats(dataset, attributes)

    dataset['sequence_left'], dataset['sequence_left_titleonly'], dataset['sequence_right'], dataset[
        'sequence_right_titleonly'] = comb_func(dataset)

    dataset['sequence_left'] = dataset['sequence_left'].str.split()
    dataset['sequence_left'] = dataset['sequence_left'].str.join(' ')
    dataset['sequence_right'] = dataset['sequence_right'].str.split()
    dataset['sequence_right'] = dataset['sequence_right'].str.join(' ')

    dataset['sequence_left_titleonly'] = dataset['sequence_left_titleonly'].str.split()
    dataset['sequence_left_titleonly'] = dataset['sequence_left_titleonly'].str.join(' ')
    dataset['sequence_right_titleonly'] = dataset['sequence_right_titleonly'].str.split()
    dataset['sequence_right_titleonly'] = dataset['sequence_right_titleonly'].str.join(' ')

    tqdm.pandas(desc='Tokenizing left sequence for inspection')
    dataset['sequence_left_inspect'] = dataset['sequence_left'].progress_apply(lambda x: tokenizer.tokenize(x))
    dataset['sequence_left_titleonly_inspect'] = dataset['sequence_left_titleonly'].progress_apply(
        lambda x: tokenizer.tokenize(x))
    tqdm.pandas(desc='Tokenizing right sequence for inspection')
    dataset['sequence_right_inspect'] = dataset['sequence_right'].progress_apply(lambda x: tokenizer.tokenize(x))
    dataset['sequence_right_titleonly_inspect'] = dataset['sequence_right_titleonly'].progress_apply(
        lambda x: tokenizer.tokenize(x))

    dataset_combined_length = dataset.apply(
        lambda x: len(x['sequence_left_inspect']) + len(x['sequence_right_inspect']), axis=1)
    dataset_combined_length_binned = pd.cut(dataset_combined_length, [-1, 32, 64, 128, 256, 512, 50000],
                                            labels=['32', '64', '128', '256', '512', '50000'])
    print('Full sequence:')
    plt.hist(dataset_combined_length_binned)
    plt.show()

    dataset_combined_length = dataset.apply(
        lambda x: len(x['sequence_left_titleonly_inspect']) + len(x['sequence_right_titleonly_inspect']), axis=1)
    dataset_combined_length_binned = pd.cut(dataset_combined_length, [-1, 32, 64, 128, 256, 512, 50000],
                                            labels=['32', '64', '128', '256', '512', '50000'])
    print('Title only sequence:')
    plt.hist(dataset_combined_length_binned)
    plt.show()

    try:
        dataset_reduced = dataset[
            ['label', 'label_multi1', 'label_multi2', 'pair_id', 'sequence_left', 'sequence_right']]
        dataset_reduced_titleonly = dataset[
            ['label', 'label_multi1', 'label_multi2', 'pair_id', 'sequence_left_titleonly',
             'sequence_right_titleonly']].copy()
    except KeyError:
        dataset_reduced = dataset[['label', 'pair_id', 'sequence_left', 'sequence_right']]
        dataset_reduced_titleonly = dataset[
            ['label', 'pair_id', 'sequence_left_titleonly', 'sequence_right_titleonly']].copy()

    dataset_reduced_titleonly = dataset_reduced_titleonly.rename(columns={'sequence_left_titleonly': 'sequence_left',
                                                                          'sequence_right_titleonly': 'sequence_right'})

    dataset_inspect = dataset[
        ['sequence_left', 'sequence_left_inspect', 'sequence_left_titleonly', 'sequence_left_titleonly_inspect',
         'sequence_right', 'sequence_right_inspect', 'sequence_right_titleonly', 'sequence_right_titleonly_inspect',
         'pair_id']]

    return dataset_reduced, dataset_reduced_titleonly, dataset_inspect


def _att_to_seq_lspc(dataset):
    seq_left = dataset['brand_left'] + ' ' + dataset['title_left'] + ' ' + dataset['description_left'] + ' ' + dataset[
        'specTableContent_left']
    seq_left_titleonly = dataset['brand_left'] + ' ' + dataset['title_left']
    seq_right = dataset['brand_right'] + ' ' + dataset['title_right'] + ' ' + dataset['description_right'] + ' ' + \
                dataset['specTableContent_right']
    seq_right_titleonly = dataset['brand_right'] + ' ' + dataset['title_right']
    return seq_left, seq_left_titleonly, seq_right, seq_right_titleonly


def _att_to_seq_abtbuy(dataset):
    seq_left = dataset['name_left'] + ' ' + dataset['price_left'].astype(str) + ' ' + dataset['description_left']
    seq_left_titleonly = dataset['name_left']
    seq_right = dataset['name_right'] + ' ' + dataset['price_left'].astype(str) + ' ' + dataset['description_right']
    seq_right_titleonly = dataset['name_right']
    return seq_left, seq_left_titleonly, seq_right, seq_right_titleonly


def _att_to_seq_amazongoogle(dataset):
    seq_left = dataset['manufacturer_left'] + ' ' + dataset['title_left'] + ' ' + dataset['description_left'] + ' ' + \
               dataset['price_left'].astype(str)
    seq_left_titleonly = dataset['manufacturer_left'] + ' ' + dataset['title_left']
    seq_right = dataset['manufacturer_right'] + ' ' + dataset['title_right'] + ' ' + dataset[
        'description_right'] + ' ' + \
                dataset['price_right'].astype(str)
    seq_right_titleonly = dataset['manufacturer_right'] + ' ' + dataset['title_right']
    return seq_left, seq_left_titleonly, seq_right, seq_right_titleonly


def _att_to_seq_dblpscholar(dataset):
    seq_left = dataset['title_left'] + ' ' + dataset['authors_left'] + ' ' + dataset['venue_left'] + ' ' + dataset[
        'year_left'].astype(str)
    seq_left_titleonly = dataset['title_left']
    seq_right = dataset['title_right'] + ' ' + dataset['authors_right'] + ' ' + dataset['venue_right'] + ' ' + dataset[
        'year_right'].astype(str)
    seq_right_titleonly = dataset['title_right']
    return seq_left, seq_left_titleonly, seq_right, seq_right_titleonly


def _att_to_seq_company(dataset):
    seq_left = dataset['content_left']
    seq_left_titleonly = dataset['content_left']
    seq_right = dataset['content_right']
    seq_right_titleonly = dataset['content_right']
    return seq_left, seq_left_titleonly, seq_right, seq_right_titleonly


def _att_to_seq_di2kg(dataset):
    seq_left = dataset['title_left'] + ' ' + dataset['specs_left']
    # seq_left = dataset['title_left']
    seq_left_titleonly = dataset['title_left']
    seq_right = dataset['title_right'] + ' ' + dataset['specs_right']
    # seq_right = dataset['title_right']
    seq_right_titleonly = dataset['title_right']
    return seq_left, seq_left_titleonly, seq_right, seq_right_titleonly


def _print_attribute_stats(dataset, attributes):
    for attr in attributes:
        attribute = list(dataset[f'{attr}_left'].values)
        attribute.extend(list(dataset[f'{attr}_right'].values))
        attribute_clean = [x for x in attribute if x != '']
        attribute_tokens = [x.split(' ') for x in attribute_clean]
        att_len = [len(x) for x in attribute_tokens]
        att_len_max = max(att_len)
        att_len_avg = np.mean(att_len)
        att_len_median = np.median(att_len)
        print(f'{attr}: Max length: {att_len_max}, mean length: {att_len_avg}, median length: {att_len_median}')


def _cut_lspc(row):
    row[f'title_left'] = ' '.join(row[f'title_left'].split(' ')[:50])
    row[f'title_right'] = ' '.join(row[f'title_right'].split(' ')[:50])
    row[f'brand_left'] = ' '.join(row[f'brand_left'].split(' ')[:5])
    row[f'brand_right'] = ' '.join(row[f'brand_right'].split(' ')[:5])
    row[f'description_left'] = ' '.join(row[f'description_left'].split(' ')[:100])
    row[f'description_right'] = ' '.join(row[f'description_right'].split(' ')[:100])
    row[f'specTableContent_left'] = ' '.join(row[f'specTableContent_left'].split(' ')[:200])
    row[f'specTableContent_right'] = ' '.join(row[f'specTableContent_right'].split(' ')[:200])
    return row


def _cut_amazongoogle(row):
    row[f'description_left'] = ' '.join(row[f'description_left'].split(' ')[:100])
    row[f'description_right'] = ' '.join(row[f'description_right'].split(' ')[:100])
    return row


def get_encoder(name):
    name_dict = {
        'computers': '../../../data/interim/wdc-lspc/gold-standards/preprocessed_computers_gs.pkl.gz',
        'cameras': '../../../data/interim/wdc-lspc/gold-standards/preprocessed_cameras_gs.pkl.gz',
        'watches': '../../../data/interim/wdc-lspc/gold-standards/preprocessed_watches_gs.pkl.gz',
        'shoes': '../../../data/interim/wdc-lspc/gold-standards/preprocessed_shoes_gs.pkl.gz'
    }
    gs = pd.read_pickle(name_dict[name])
    enc = LabelEncoder()
    cluster_id_set_left = set()
    cluster_id_set_left.update(gs['cluster_id_left'].tolist())
    cluster_id_set_right = set()
    cluster_id_set_right.update(gs['cluster_id_right'].tolist())
    cluster_id_set_left.update(cluster_id_set_right)
    enc.fit(list(cluster_id_set_left))
    return enc


def get_encoder_deepmatcher(handle):
    train = pd.read_json(f'../../../data/interim/{handle}/{handle}-train.json.gz', lines=True)
    gs = pd.read_json(f'../../../data/interim/{handle}/{handle}-gs.json.gz', lines=True)
    full = train.append(gs)
    enc = LabelEncoder()
    cluster_id_set_left = set()
    cluster_id_set_left.update(full['cluster_id_left'].tolist())
    cluster_id_set_right = set()
    cluster_id_set_right.update(full['cluster_id_right'].tolist())
    cluster_id_set_left.update(cluster_id_set_right)
    enc.fit(list(cluster_id_set_left))
    return enc


def get_encoder_di2kg(handle):
    train = pd.read_json(f'../../../data/interim/di2kg/{handle}-train.json.gz', lines=True)
    #     gs = pd.read_json(f'../../../data/interim/di2kg/{handle}-gs.json.gz', lines=True)
    #     full = train.append(gs)
    enc = LabelEncoder()
    cluster_id_set_left = set()
    cluster_id_set_left.update(train['cluster_id_left'].tolist())
    cluster_id_set_right = set()
    cluster_id_set_right.update(train['cluster_id_right'].tolist())
    cluster_id_set_left.update(cluster_id_set_right)
    enc.fit(list(cluster_id_set_left))
    return enc


if __name__ == '__main__':
    if BUILD_LSPC:

        encoders = {
            'computers': get_encoder('computers'),
            'cameras': get_encoder('cameras'),
            'watches': get_encoder('watches'),
            'shoes': get_encoder('shoes'),
        }

        Path('../../../data/processed/wdc-lspc/bert/inspection/').mkdir(parents=True, exist_ok=True)
        path = '../../../data/processed/wdc-lspc/bert/'

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        attributes = ['title', 'description', 'brand', 'specTableContent']

        datasets_lspc_train = ['preprocessed_computers_train_small', 'preprocessed_computers_train_medium',
                               'preprocessed_computers_train_large', 'preprocessed_computers_train_xlarge',
                               'preprocessed_cameras_train_small', 'preprocessed_cameras_train_medium',
                               'preprocessed_cameras_train_large', 'preprocessed_cameras_train_xlarge',
                               'preprocessed_watches_train_small', 'preprocessed_watches_train_medium',
                               'preprocessed_watches_train_large', 'preprocessed_watches_train_xlarge',
                               'preprocessed_shoes_train_small', 'preprocessed_shoes_train_medium',
                               'preprocessed_shoes_train_large', 'preprocessed_shoes_train_xlarge'
                               ]

        datasets_lspc_gs = ['preprocessed_computers_gs', 'preprocessed_cameras_gs',
                            'preprocessed_watches_gs', 'preprocessed_shoes_gs', 'preprocessed_computers_new_testset_1500'
                            ]

        for ds in datasets_lspc_gs:
            enc = None
            for key in encoders.keys():
                if key in ds:
                    assert enc is None
                    enc = encoders[key]
            df = pd.read_pickle(f'../../../data/interim/wdc-lspc/gold-standards/{ds}.pkl.gz')
            df_gs, df_gs_titleonly, df_inspect = process_to_bert(df, attributes, tokenizer,
                                                                 _att_to_seq_lspc, _cut_lspc, multi_encoder=enc)

            df_gs.to_pickle(f'{path}{ds}_bert_cutBTDS.pkl.gz', compression='gzip')
            df_gs_titleonly.to_pickle(f'{path}{ds}_bert_cutBT_titleonly.pkl.gz',
                                      compression='gzip')
            df_inspect.to_csv(f'{path}inspection/{ds}_bert_cutBTDS.csv.gz', index=False)

        for ds in datasets_lspc_train:
            enc = None
            for key in encoders.keys():
                if key in ds:
                    assert enc is None
                    enc = encoders[key]
            df = pd.read_pickle(f'../../../data/interim/wdc-lspc/training-sets/{ds}.pkl.gz')
            df_train, df_train_titleonly, df_inspect = process_to_bert(df, attributes, tokenizer, _att_to_seq_lspc,
                                                                       _cut_lspc, multi_encoder=enc)

            df_train.to_pickle(f'{path}{ds}_bert_cutBTDS.pkl.gz', compression='gzip')
            df_train_titleonly.to_pickle(f'{path}{ds}_bert_cutBT_titleonly.pkl.gz',
                                         compression='gzip')
            df_inspect.to_csv(f'{path}inspection/{ds}_bert_cutBTDS.csv.gz', index=False)

    ###############################################################

    if BUILD_DEEPMATCHER:
        datasets = [
            ('abt-buy', ['name', 'description'], _att_to_seq_abtbuy, get_encoder_deepmatcher('abt-buy')),
            ('company', ['content'], _att_to_seq_company, get_encoder_deepmatcher('company')),
            ('dblp-scholar', ['title', 'authors', 'venue'], _att_to_seq_dblpscholar, get_encoder_deepmatcher('dblp-scholar'))
        ]

        for dataset in datasets:

            handle, attributes, sequencing_function_handle, enc = dataset

            print(f'{handle}')
            print(f'Multi-class encoder has {len(enc.classes_)} classes')

            Path(f'../../../data/processed/{handle}/bert/inspection/').mkdir(parents=True, exist_ok=True)
            path = f'../../../data/processed/{handle}/bert/'

            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

            datasets_train = [f'{handle}-train']

            datasets_gs = [f'{handle}-gs']

            for ds in datasets_train:
                df = pd.read_json(f'../../../data/interim/{handle}/{ds}.json.gz', lines=True)
                df_train, df_train_titleonly, df_inspect = process_to_bert(df, attributes, tokenizer,
                                                                           sequencing_function_handle,
                                                                           multi_encoder=enc)

                df_train.to_pickle(f'{path}{ds}-bert.pkl.gz', compression='gzip')
                df_train_titleonly.to_pickle(f'{path}{ds}-bert-titleonly.pkl.gz', compression='gzip')
                df_inspect.to_csv(f'{path}inspection/{ds}-bert.csv.gz', index=False)

            for ds in datasets_gs:
                df = pd.read_json(f'../../../data/interim/{handle}/{ds}.json.gz', lines=True)
                df_gs, df_gs_titleonly, df_inspect = process_to_bert(df, attributes, tokenizer,
                                                                     sequencing_function_handle, multi_encoder=enc)

                df_gs.to_pickle(f'{path}{ds}-bert.pkl.gz', compression='gzip')
                df_gs_titleonly.to_pickle(f'{path}{ds}-bert-titleonly.pkl.gz', compression='gzip')
                df_inspect.to_csv(f'{path}inspection/{ds}-bert.csv.gz', index=False)

    ######################################################

    if BUILD_DI2KG:
        datasets = [
            ('monitor', ['title', 'specs'], _att_to_seq_di2kg, get_encoder_di2kg('monitor'))
        ]

        for dataset in datasets:

            handle, attributes, sequencing_function_handle, enc = dataset

            print(f'{handle}')
            print(f'Multi-class encoder has {len(enc.classes_)} classes')

            Path(f'../../../data/processed/di2kg/bert/inspection/').mkdir(parents=True, exist_ok=True)
            path = f'../../../data/processed/di2kg/bert/'

            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

            datasets_train = [f'{handle}-train']

            datasets_gs = [f'{handle}-gs']

            for ds in datasets_train:
                df = pd.read_json(f'../../../data/interim/di2kg/{ds}.json.gz', lines=True)
                df_train, df_train_titleonly, df_inspect = process_to_bert(df, attributes, tokenizer,
                                                                           sequencing_function_handle,
                                                                           multi_encoder=enc)

                df_train.to_pickle(f'{path}{ds}-bert.pkl.gz', compression='gzip')
                df_train_titleonly.to_pickle(f'{path}{ds}-bert-titleonly.pkl.gz', compression='gzip')
                df_inspect.to_csv(f'{path}inspection/{ds}-bert.csv.gz', index=False)

            for ds in datasets_gs:
                df = pd.read_json(f'../../../data/interim/di2kg/{ds}.json.gz', lines=True)
                df_gs, df_gs_titleonly, df_inspect = process_to_bert(df, attributes, tokenizer,
                                                                     sequencing_function_handle, multi_encoder=enc)

                df_gs.to_pickle(f'{path}{ds}-bert.pkl.gz', compression='gzip')
                df_gs_titleonly.to_pickle(f'{path}{ds}-bert-titleonly.pkl.gz', compression='gzip')
                df_inspect.to_csv(f'{path}inspection/{ds}-bert.csv.gz', index=False)

    ######################################################