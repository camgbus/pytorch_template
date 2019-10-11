################################################################################
# Functions to save and restore different data types
################################################################################

import os

# PICKLE
import pickle
def pkl_dump(obj, name, path = 'obj'):
    if '.p' not in name:
        name = name + '.pkl'
    path = os.path.join(path, name)
    pickle.dump(obj, open(path, 'wb'))

def pkl_load(name, path = 'obj'):
    if '.p' not in name:
        name = name + '.pkl'
    path = os.path.join(path, name)
    try:
        obj = pickle.load(open(path, 'rb'))
    except FileNotFoundError:
        obj = None
    return obj

# PANDAS
import pandas as pd
def save_df(df, name, path):
    full_path = os.path.join(path, name)
    df.to_pickle(full_path)

def load_df(name, path):
    full_path = os.path.join(path, name)
    return pd.read_pickle(full_path)

# JSON
import json
def save_json(dict_obj, name, path):
    if 'txt' not in name:
        name += '.txt'
    with open(os.path.join(path, name), 'w') as json_file:
        json.dump(dict_obj, json_file)

def load_json(name, path):
    if 'txt' not in name:
        name += '.txt'
    with open(os.path.join(path, name), 'r') as json_file:
        return json.load(json_file)

# PYTORCH
import torch
def save_model_state(model, name, path):
    full_path = os.path.join(path, name)
    torch.save(model.state_dict(), full_path)

def load_model_state(model, name, path):
    full_path = os.path.join(path, name)
    if os.path.isfile(full_path):
        model.load_state_dict(torch.load(full_path))
        return True
    return False

def save_optimizer_state(optimizer, name, path):
    full_path = os.path.join(path, name)
    torch.save(optimizer.state_dict(), full_path)

def load_optimizer_state(optimizer, name, path):
    full_path = os.path.join(path, name)
    optimizer.load_state_dict(torch.load(full_path))

# OTHERS
import functools
def join_path(list):
    """From a list of chained directories, forms a path"""
    return functools.reduce(os.path.join, list)
