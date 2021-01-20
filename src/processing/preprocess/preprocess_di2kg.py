import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

from tqdm import tqdm
from pathlib import Path
import shutil
import os

import json
from copy import deepcopy

from sklearn.preprocessing import LabelEncoder

import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

crappy_excel_df = pd.read_excel('../../../data/raw/di2kg/monitor_entity_resolution_labelled_data_v2_05_26_2020.xlsx')
left = []
right = []
label = []

for i, text in crappy_excel_df.itertuples():
    split = text.split(',')
    sample = random.sample(split[:1], 1)[0]
    split.remove(sample)
    left.append(sample)
    right.append(split[0])
    label.append(int(split[1]))

pairs_df = pd.DataFrame({
    'left':left,
    'right':right,
    'label':label
})

data_to_extract = set(pairs_df['left'])
data_to_extract.update(pairs_df['right'])

cluster_groundtruth = pd.read_excel('../../../data/raw/di2kg/monitor_entity_resolution_gt.xlsx')

entity_dict = {}

for i, text in cluster_groundtruth.itertuples():
    split = text.split(',')
    if split[1] in entity_dict.keys():
        continue
    else:
        entity_dict[split[1]] = split[0]

final_df = pd.DataFrame()
rest_df = pd.DataFrame()

all_files = []

for dirpath, dirnames, filenames in os.walk(f'../../../data/raw/di2kg/2013_monitor_specs/'):
    for f in filenames:
         all_files.append(os.path.join(dirpath, f))
            
for path in data_to_extract:
       
    split = path.split('//')
    folder = split[0]
    file = split[1]
    full_path = f'../../../data/raw/di2kg/2013_monitor_specs/{folder}/{file}.json'

    all_files.remove(full_path)
    
    with open(full_path, 'r') as f:
        data = json.load(f)
        
    data['specs'] = ''
        
    for key in deepcopy(data).keys():
        
        if key == '<page title>' or key == 'specs':
            continue
            
        else:
            current = data[key]
            specs = ''
            if type(current) != list and pd.isnull(current):
                del data[key]
                assert key not in data.keys()
                continue
            else:
                if type(current) == list:
                    values = set()
                    for value in current:
                        values.add(value)
                    specs += f'{key} '
                    for i, value in enumerate(values):
                        specs += f'{value} '
                        if i == len(values)-1:
                            specs += ', '
                else:
                    specs += f'{key} {current} , '
            if specs != '':
                specs = specs.rstrip(' , ')
                data['specs'] += f'{specs} '
            del data[key]
            assert key not in data.keys()
              
    data['specs'] = data['specs'].strip()
    
    data['id'] = path
    final_df = final_df.append(data, ignore_index=True)

final_df = final_df.set_index('id', drop=False)

for path in tqdm(all_files):
    
    sub_path = '//'.join(path.split('/')[-2:]).replace('.json', '')
    
    if sub_path not in entity_dict.keys():
        continue
    
    split = sub_path.split('//')
    folder = split[0]
    file = split[1]
    full_path = f'../../../data/raw/di2kg/2013_monitor_specs/{folder}/{file}.json'
    
    with open(full_path, 'r') as f:
        data = json.load(f)
        
    data['specs'] = ''
        
    for key in deepcopy(data).keys():
        
        if key == '<page title>' or key == 'specs':
            continue
            
        else:
            current = data[key]
            specs = ''
            if type(current) != list and pd.isnull(current):
                del data[key]
                assert key not in data.keys()
                continue
            else:
                if type(current) == list:
                    values = set()
                    for value in current:
                        values.add(value)
                    specs += f'{key} '
                    for i, value in enumerate(values):
                        specs += f'{value} '
                        if i == len(values)-1:
                            specs += ', '
                else:
                    specs += f'{key} {current} , '
            if specs != '':
                specs = specs.rstrip(' , ')
                data['specs'] += f'{specs} '
            del data[key]
            assert key not in data.keys()
              
    data['specs'] = data['specs'].strip()
    data['id'] = sub_path
    rest_df = rest_df.append(data, ignore_index=True)

rest_df = rest_df.set_index('id', drop=False)

print(f'Train offers: {len(final_df)}, Test offers: {len(rest_df)}')

final_df = final_df[['id','<page title>','specs']]
final_df = final_df.rename(columns={'<page title>':'title'})

rest_df = rest_df[['id','<page title>','specs']]
rest_df = rest_df.rename(columns={'<page title>':'title'})

final_df['cluster_id'] = final_df['id'].apply(lambda x: entity_dict[x])
rest_df['cluster_id'] = rest_df['id'].apply(lambda x: entity_dict[x])

train_ids = set(final_df['cluster_id'])
rest_ids = set(rest_df['cluster_id'])

def get_pair(combination):
    combination_list = list(combination)
    picked = random.sample(combination_list, 1)[0]
    combination_list.remove(picked)
    left = picked
    right = combination_list[0]
    label = 1 if entity_dict[left] == entity_dict[right] else 0
    pair_dict = {
        'left':left,
        'right':right,
        'label':label
    }
    
    return pair_dict

from itertools import combinations

test_combinations = combinations(rest_df['id'], 2)
test_combinations = list(test_combinations)

results = []

with tqdm(total=len(test_combinations)) as pbar:
    with ProcessPoolExecutor(max_workers=60) as executor:
        futures = []
        for combination in test_combinations:
            futures.append(executor.submit(get_pair, combination))

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            pbar.update(1)

test_pairs = pd.DataFrame.from_records(results)

left = final_df.loc[pairs_df['left']]
left = left.reset_index(drop=True)
right = final_df.loc[pairs_df['right']]
right = right.reset_index(drop=True)

merged = left.join(right, lsuffix='_left', rsuffix='_right')
merged['pair_id'] = merged['id_left'] + '#' + merged['id_right']
merged['label'] = pairs_df['label']

from sklearn.preprocessing import LabelEncoder

all_cluster_ids = set(merged['cluster_id_left'])
all_cluster_ids.update(merged['cluster_id_right'])
enc = LabelEncoder()
enc.fit(list(all_cluster_ids))

merged['cluster_id_left'] = enc.transform(merged['cluster_id_left'])
merged['cluster_id_right'] = enc.transform(merged['cluster_id_right'])

left = rest_df.loc[test_pairs['left']]
left = left.reset_index(drop=True)
right = rest_df.loc[test_pairs['right']]
right = right.reset_index(drop=True)

merged_test = left.join(right, lsuffix='_left', rsuffix='_right')
merged_test['pair_id'] = merged_test['id_left'] + '#' + merged_test['id_right']
merged_test['label'] = test_pairs['label']

unique_offer_dict = {}

for i, row in merged.iterrows():
    
    cid_left = row['cluster_id_left']
    cid_right = row['cluster_id_right']
    
    id_left = row['id_left']
    id_right = row['id_right']
    
    label = row['label']
    
    if cid_left not in unique_offer_dict:
        unique_offer_dict[cid_left] = {}
        
    if cid_right not in unique_offer_dict:
        unique_offer_dict[cid_right] = {}
        
    if id_left not in unique_offer_dict[cid_left]:
        unique_offer_dict[cid_left][id_left] = [(i, id_left, id_right, label)]
    else:
        unique_offer_dict[cid_left][id_left].append((i, id_left, id_right, label))
        
    if id_right not in unique_offer_dict[cid_right]:
        unique_offer_dict[cid_right][id_right] = [(i, id_left, id_right, label)]
    else:
        unique_offer_dict[cid_right][id_right].append((i, id_left, id_right, label))

rep_in_valid = set()
rep_in_train = set()
amounts = []

for key, representation_dict in unique_offer_dict.items():

    amount = len(representation_dict)
    amounts.append(amount)
    rep_set = set(representation_dict.keys())
    perc_20 = int(0.33 * amount)
    if perc_20 == 0:
        perc_20 = 1
    
    valid_reps = random.sample(rep_set, perc_20)
    
    for rep in valid_reps:
        rep_in_valid.add(rep)
        rep_set.remove(rep)
        
    for rep in deepcopy(rep_set):
        rep_in_train.add(rep)
        rep_set.remove(rep)
        
    assert len(rep_set) == 0

train_ids = []
valid_ids = []

for i, pair in merged['pair_id'].iteritems():
    
    split = pair.split('#')
    left = split[0]
    right = split[1]
    
    if left in rep_in_valid and right in rep_in_valid:
        valid_ids.append(i)
    elif left in rep_in_train and right in rep_in_train:
        train_ids.append(i)

merged['cluster_id_left'] = enc.inverse_transform(merged['cluster_id_left'])
merged['cluster_id_right'] = enc.inverse_transform(merged['cluster_id_right'])

train = merged.loc[train_ids]
valid = merged.loc[valid_ids]

train = train.sample(frac=1.0, random_state=42)
valid = valid.sample(frac=1.0, random_state=42)
test = merged_test
test = test.sample(frac=1.0, random_state=42)

Path('../../../data/interim/di2kg/').mkdir(parents=True, exist_ok=True)

valid['pair_id'].to_csv('../../../data/interim/di2kg/monitor-valid.csv', header=True, index=False)

train_full = train.append(valid)
train_full = train_full.reset_index(drop=True)

test = test.reset_index(drop=True)

cluster_ids_train = set()
cluster_ids_train.update(train_full['cluster_id_left'])
cluster_ids_train.update(train_full['cluster_id_right'])

test_combined = test[((test['cluster_id_left'].isin(cluster_ids_train) & test['cluster_id_right'].isin(cluster_ids_train))) | ((~test['cluster_id_left'].isin(cluster_ids_train) & ~test['cluster_id_right'].isin(cluster_ids_train)))]

train_full.to_json('../../../data/interim/di2kg/monitor-train.json.gz', compression='gzip', lines=True, orient='records')
test_combined.to_json('../../../data/interim/di2kg/monitor-gs.json.gz', compression='gzip', lines=True, orient='records')