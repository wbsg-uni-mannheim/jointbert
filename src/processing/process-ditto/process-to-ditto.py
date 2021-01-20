import pandas as pd
import numpy as np

np.random.seed(42)
import random

random.seed(42)

from tqdm import tqdm

from pathlib import Path
import jsonlines

from pdb import set_trace

BUILD_LSPC = True
BUILD_DEEPMATCHER = True
BUILD_DI2KG = True


def write_datasets_to_file(dataset, handle, attributes, suffix):
    if handle != 'monitor':
        processed = dataset
        with open(f'../../../data/processed/{handle}/ditto/{handle}-{suffix}.txt', 'w') as file_object:
            for i, row in processed.iterrows():
                file_object.write(f'{row["ditto_left"]}\t{row["ditto_right"]}\t{row["label"]}\n')

        with open(f'../../../data/processed/{handle}/ditto/{handle}-{suffix}-titleonly.txt', 'w') as file_object:
            for i, row in processed.iterrows():
                file_object.write(f'{row["ditto_left_titleonly"]}\t{row["ditto_right_titleonly"]}\t{row["label"]}\n')

        with jsonlines.open(f'../../../data/processed/{handle}/ditto/jsonlines/{handle}-{suffix}.jsonl',
                            mode='w') as writer:
            for i, row in processed.iterrows():
                row_left = row[list(map(lambda x: x + '_left', attributes))]
                columns = row_left.index
                columns = [x.replace('_left', '') for x in columns]
                row_left.index = columns

                row_right = row[list(map(lambda x: x + '_right', attributes))]
                columns = row_right.index
                columns = [x.replace('_right', '') for x in columns]
                row_right.index = columns

                writer.write([row_left.to_dict(), row_right.to_dict()])

        with jsonlines.open(f'../../../data/processed/{handle}/ditto/jsonlines/{handle}-{suffix}-titleonly.jsonl',
                            mode='w') as writer:
            for i, row in processed.iterrows():
                row_left = row[[attributes[0] + '_left']]
                columns = row_left.index
                columns = [x.replace('_left', '') for x in columns]
                row_left.index = columns

                row_right = row[[attributes[0] + '_right']]
                columns = row_right.index
                columns = [x.replace('_right', '') for x in columns]
                row_right.index = columns

                writer.write([row_left.to_dict(), row_right.to_dict()])
    else:
        processed = dataset
        with open(f'../../../data/processed/di2kg/ditto/{handle}-{suffix}.txt', 'w') as file_object:
            for i, row in processed.iterrows():
                file_object.write(f'{row["ditto_left"]}\t{row["ditto_right"]}\t{row["label"]}\n')

        with open(f'../../../data/processed/di2kg/ditto/{handle}-{suffix}-titleonly.txt', 'w') as file_object:
            for i, row in processed.iterrows():
                file_object.write(f'{row["ditto_left_titleonly"]}\t{row["ditto_right_titleonly"]}\t{row["label"]}\n')

        with jsonlines.open(f'../../../data/processed/di2kg/ditto/jsonlines/{handle}-{suffix}.jsonl',
                            mode='w') as writer:
            for i, row in processed.iterrows():
                row_left = row[list(map(lambda x: x + '_left', attributes))]
                columns = row_left.index
                columns = [x.replace('_left', '') for x in columns]
                row_left.index = columns

                row_right = row[list(map(lambda x: x + '_right', attributes))]
                columns = row_right.index
                columns = [x.replace('_right', '') for x in columns]
                row_right.index = columns

                writer.write([row_left.to_dict(), row_right.to_dict()])

        with jsonlines.open(f'../../../data/processed/di2kg/ditto/jsonlines/{handle}-{suffix}-titleonly.jsonl',
                            mode='w') as writer:
            for i, row in processed.iterrows():
                row_left = row[[attributes[0] + '_left']]
                columns = row_left.index
                columns = [x.replace('_left', '') for x in columns]
                row_left.index = columns

                row_right = row[[attributes[0] + '_right']]
                columns = row_right.index
                columns = [x.replace('_right', '') for x in columns]
                row_right.index = columns

                writer.write([row_left.to_dict(), row_right.to_dict()])


def extract_ditto_sequence_lspc(row):
    row[
        'ditto_left'] = f'COL brand VAL {row["brand_left"]} COL title VAL {row["title_left"]} COL description VAL {row["description_left"]} COL specTable VAL {row["specTableContent_left"]}'
    row[
        'ditto_right'] = f'COL brand VAL {row["brand_right"]} COL title VAL {row["title_right"]} COL description VAL {row["description_right"]} COL specTable VAL {row["specTableContent_right"]}'
    row['ditto_left_titleonly'] = f'COL brand VAL {row["brand_left"]} COL title VAL {row["title_left"]}'
    row['ditto_right_titleonly'] = f'COL brand VAL {row["brand_right"]} COL title VAL {row["title_right"]}'
    return row


def extract_ditto_sequence_abtbuy(row):
    row[
        'ditto_left'] = f'COL name VAL {row["name_left"]} COL price VAL {row["price_left"]} COL description VAL {row["description_left"]}'
    row[
        'ditto_right'] = f'COL name VAL {row["name_right"]} COL price VAL {row["price_right"]} COL description VAL {row["description_right"]}'
    row['ditto_left_titleonly'] = f'COL name VAL {row["name_left"]}'
    row['ditto_right_titleonly'] = f'COL name VAL {row["name_right"]}'
    return row


def extract_ditto_sequence_dblpscholar(row):
    row[
        'ditto_left'] = f'COL title VAL {row["title_left"]} COL authors VAL {row["authors_left"]} COL venue VAL {row["venue_left"]} COL year VAL {row["year_left"]}'
    row[
        'ditto_right'] = f'COL title VAL {row["title_right"]} COL authors VAL {row["authors_right"]} COL venue VAL {row["venue_right"]} COL year VAL {row["year_right"]}'
    row['ditto_left_titleonly'] = f'COL title VAL {row["title_left"]}'
    row['ditto_right_titleonly'] = f'COL title VAL {row["title_right"]}'
    return row


def extract_ditto_sequence_company(row):
    row['ditto_left'] = f'COL content VAL {row["content_left"]}'
    row['ditto_right'] = f'COL content VAL {row["content_right"]}'
    row['ditto_left_titleonly'] = f'COL content VAL {row["content_left"]}'
    row['ditto_right_titleonly'] = f'COL content VAL {row["content_right"]}'
    return row


def extract_ditto_sequence_monitor(row):
    row['ditto_left'] = ' '.join(f'COL title VAL {row["title_left"]} COL specs VAL {row["specs_left"]}'.split())
    row['ditto_right'] = ' '.join(f'COL title VAL {row["title_right"]} COL specs VAL {row["specs_right"]}'.split())
    row['ditto_left_titleonly'] = ' '.join(f'COL title VAL {row["title_left"]}'.split())
    row['ditto_right_titleonly'] = ' '.join(f'COL title VAL {row["title_right"]}'.split())
    return row


if BUILD_DEEPMATCHER:
    datasets_deepmatcher = [
        ('abt-buy', ['name', 'price', 'description'], extract_ditto_sequence_abtbuy),
        ('company', ['content'], extract_ditto_sequence_company),
        ('dblp-scholar', ['title', 'authors', 'venue', 'year'], extract_ditto_sequence_dblpscholar),
    ]

    for ds in datasets_deepmatcher:
        handle, attributes, extraction_function = ds

        Path(f'../../../data/processed/{handle}/ditto/jsonlines/').mkdir(parents=True, exist_ok=True)

        dataset = pd.read_json(f'../../../data/interim/{handle}/{handle}-gs.json.gz', lines=True)
        dataset = dataset.fillna('')

        tqdm.pandas(desc="Extracting Ditto Sequences")
        processed = dataset.progress_apply(extraction_function, axis=1)
        write_datasets_to_file(processed, handle, attributes, 'gs')

        dataset = pd.read_json(f'../../../data/interim/{handle}/{handle}-train.json.gz', lines=True)
        dataset = dataset.fillna('')
        dataset = dataset.set_index('pair_id', drop=False)

        processed = dataset.progress_apply(extraction_function, axis=1)

        valid_ids = pd.read_csv(f'../../../data/interim/{handle}/{handle}-valid.csv')
        id_list = valid_ids['pair_id'].to_list()

        valid_set = processed.loc[id_list].copy()
        train_set = processed.drop(id_list)

        write_datasets_to_file(valid_set, handle, attributes, 'valid')
        write_datasets_to_file(train_set, handle, attributes, 'train')

if BUILD_DI2KG:
    datasets_di2kg = [
        ('monitor', ['title', 'specs'], extract_ditto_sequence_monitor)
    ]

    for ds in datasets_di2kg:
        handle, attributes, extraction_function = ds

        Path(f'../../../data/processed/di2kg/ditto/jsonlines/').mkdir(parents=True, exist_ok=True)

        dataset = pd.read_json(f'../../../data/interim/di2kg/{handle}-gs.json.gz', lines=True)
        dataset = dataset.fillna('')

        tqdm.pandas(desc="Extracting Ditto Sequences")
        processed = dataset.progress_apply(extraction_function, axis=1)
        write_datasets_to_file(processed, handle, attributes, 'gs')

        dataset = pd.read_json(f'../../../data/interim/di2kg/{handle}-train.json.gz', lines=True)
        dataset = dataset.fillna('')
        dataset = dataset.set_index('pair_id', drop=False)

        tqdm.pandas(desc="Extracting Ditto Sequences")
        processed = dataset.progress_apply(extraction_function, axis=1)

        valid_ids = pd.read_csv(f'../../../data/interim/di2kg/{handle}-valid.csv')
        id_list = valid_ids['pair_id'].to_list()

        valid_set = processed.loc[id_list].copy()
        train_set = processed.drop(id_list)

        write_datasets_to_file(valid_set, handle, attributes, 'valid')
        write_datasets_to_file(train_set, handle, attributes, 'train')

if BUILD_LSPC:
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

    Path('../../../data/processed/wdc-lspc/ditto/jsonlines/').mkdir(parents=True, exist_ok=True)

    for ds in datasets_lspc_gs:
        dataset = pd.read_pickle(f'../../../data/interim/wdc-lspc/gold-standards/{ds}.pkl.gz')
        dataset = dataset.fillna('')

        tqdm.pandas(desc="Extracting Ditto Sequences")
        processed = dataset.progress_apply(extract_ditto_sequence_lspc, axis=1)

        with jsonlines.open(f'../../../data/processed/wdc-lspc/ditto/jsonlines/{ds}.jsonl', mode='w') as writer:
            for i, row in processed.iterrows():
                row_left = row[['brand_left', 'title_left', 'description_left', 'specTableContent_left']]
                columns = row_left.index
                columns = [x.replace('_left', '') for x in columns]
                row_left.index = columns

                row_right = row[['brand_right', 'title_right', 'description_right', 'specTableContent_right']]
                columns = row_right.index
                columns = [x.replace('_right', '') for x in columns]
                row_right.index = columns

                writer.write([row_left.to_dict(), row_right.to_dict()])

        with jsonlines.open(f'../../../data/processed/wdc-lspc/ditto/jsonlines/{ds}_titleonly.jsonl',
                            mode='w') as writer:
            for i, row in processed.iterrows():
                row_left = row[['brand_left', 'title_left']]
                columns = row_left.index
                columns = [x.replace('_left', '') for x in columns]
                row_left.index = columns

                row_right = row[['brand_right', 'title_right']]
                columns = row_right.index
                columns = [x.replace('_right', '') for x in columns]
                row_right.index = columns

                writer.write([row_left.to_dict(), row_right.to_dict()])

        with open(f'../../../data/processed/wdc-lspc/ditto/{ds}.txt', 'w') as file_object:
            for i, row in processed.iterrows():
                file_object.write(f'{row["ditto_left"]}\t{row["ditto_right"]}\t{row["label"]}\n')

        with open(f'../../../data/processed/wdc-lspc/ditto/{ds}_titleonly.txt', 'w') as file_object:
            for i, row in processed.iterrows():
                file_object.write(f'{row["ditto_left_titleonly"]}\t{row["ditto_right_titleonly"]}\t{row["label"]}\n')

    for ds in datasets_lspc_train:
        dataset = pd.read_pickle(f'../../../data/interim/wdc-lspc/training-sets/{ds}.pkl.gz')
        dataset = dataset.fillna('')
        dataset = dataset.set_index('pair_id', drop=False)

        processed = dataset.progress_apply(extract_ditto_sequence_lspc, axis=1)

        filename_split = ds.split('_')
        valid_name = f'{filename_split[1]}_valid_{filename_split[3]}'
        valid_ids = pd.read_csv(f'../../../data/raw/wdc-lspc/validation-sets/{valid_name}.csv')
        id_list = valid_ids['pair_id'].to_list()

        valid_set = processed.loc[id_list].copy()
        train_set = processed.drop(id_list)

        with jsonlines.open(f'../../../data/processed/wdc-lspc/ditto/jsonlines/{ds}.jsonl', mode='w') as writer:
            for i, row in train_set.iterrows():
                row_left = row[['brand_left', 'title_left', 'description_left', 'specTableContent_left']]
                columns = row_left.index
                columns = [x.replace('_left', '') for x in columns]
                row_left.index = columns

                row_right = row[['brand_right', 'title_right', 'description_right', 'specTableContent_right']]
                columns = row_right.index
                columns = [x.replace('_right', '') for x in columns]
                row_right.index = columns

                writer.write([row_left.to_dict(), row_right.to_dict()])

        with jsonlines.open(f'../../../data/processed/wdc-lspc/ditto/jsonlines/{ds}_titleonly.jsonl',
                            mode='w') as writer:
            for i, row in train_set.iterrows():
                row_left = row[['brand_left', 'title_left']]
                columns = row_left.index
                columns = [x.replace('_left', '') for x in columns]
                row_left.index = columns

                row_right = row[['brand_right', 'title_right']]
                columns = row_right.index
                columns = [x.replace('_right', '') for x in columns]
                row_right.index = columns

                writer.write([row_left.to_dict(), row_right.to_dict()])

        with jsonlines.open(f'../../../data/processed/wdc-lspc/ditto/jsonlines/{valid_name}.jsonl', mode='w') as writer:
            for i, row in valid_set.iterrows():
                row_left = row[['brand_left', 'title_left', 'description_left', 'specTableContent_left']]
                columns = row_left.index
                columns = [x.replace('_left', '') for x in columns]
                row_left.index = columns

                row_right = row[['brand_right', 'title_right', 'description_right', 'specTableContent_right']]
                columns = row_right.index
                columns = [x.replace('_right', '') for x in columns]
                row_right.index = columns

                writer.write([row_left.to_dict(), row_right.to_dict()])

        with jsonlines.open(f'../../../data/processed/wdc-lspc/ditto/jsonlines/{valid_name}_titleonly.jsonl',
                            mode='w') as writer:
            for i, row in valid_set.iterrows():
                row_left = row[['brand_left', 'title_left']]
                columns = row_left.index
                columns = [x.replace('_left', '') for x in columns]
                row_left.index = columns

                row_right = row[['brand_right', 'title_right']]
                columns = row_right.index
                columns = [x.replace('_right', '') for x in columns]
                row_right.index = columns

                writer.write([row_left.to_dict(), row_right.to_dict()])

        with open(f'../../../data/processed/wdc-lspc/ditto/{ds}.txt', 'w') as file_object:
            for i, row in train_set.iterrows():
                file_object.write(f'{row["ditto_left"]}\t{row["ditto_right"]}\t{row["label"]}\n')

        with open(f'../../../data/processed/wdc-lspc/ditto/{ds}_titleonly.txt', 'w') as file_object:
            for i, row in train_set.iterrows():
                file_object.write(f'{row["ditto_left_titleonly"]}\t{row["ditto_right_titleonly"]}\t{row["label"]}\n')

        with open(f'../../../data/processed/wdc-lspc/ditto/{valid_name}.txt', 'w') as file_object:
            for i, row in valid_set.iterrows():
                file_object.write(f'{row["ditto_left"]}\t{row["ditto_right"]}\t{row["label"]}\n')

        with open(f'../../../data/processed/wdc-lspc/ditto/{valid_name}_titleonly.txt', 'w') as file_object:
            for i, row in valid_set.iterrows():
                file_object.write(f'{row["ditto_left_titleonly"]}\t{row["ditto_right_titleonly"]}\t{row["label"]}\n')

        with open(f'../../../data/processed/wdc-lspc/ditto/{ds}_full.txt', 'w') as file_object:
            for i, row in processed.iterrows():
                file_object.write(f'{row["ditto_left"]}\t{row["ditto_right"]}\t{row["label"]}\n')

        with open(f'../../../data/processed/wdc-lspc/ditto/{ds}_titleonly_full.txt', 'w') as file_object:
            for i, row in processed.iterrows():
                file_object.write(f'{row["ditto_left_titleonly"]}\t{row["ditto_right_titleonly"]}\t{row["label"]}\n')