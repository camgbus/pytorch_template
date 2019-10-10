import os
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

import json
def save_json(dict_obj, path, file_name):
    if 'txt' not in file_name:
        file_name += '.txt'
    with open(os.path.join(path, file_name), 'w') as json_file:
        json.dump(dict_obj, json_file)

def load_json(path, file_name):
    if 'txt' not in file_name:
        file_name += '.txt'
    with open(os.path.join(path, file_name), 'r') as json_file:
        return json.load(json_file)

import functools
def join_path(list):
    ''' From a list of chained directories, forms a path '''
    return functools.reduce(os.path.join, list)

import torch
def save_model_state(model, name, path, save_on_D = False):
    if save_on_D:
        root = 'D:\Benchmark_models'
        full_path = os.path.join(root, path)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        full_path = os.path.join(full_path, name)
    else:
        full_path = os.path.join(path, name)
    torch.save(model.state_dict(), full_path)

def load_model_state(model, name, path = 'obj'):
    full_path = os.path.join(path, name)
    if os.path.isfile(full_path):
        model.load_state_dict(torch.load(full_path))
        return True
    return False

def get_results(experiment_time_str):
    path = join_path(['obj', experiment_time_str, 'results'])
    return pkl_load(name='results', path=path)

def save_optimizer_state(optimizer, name, path = 'obj/optimizer_states'):
    full_path = os.path.join(path, name)
    torch.save(optimizer.state_dict(), full_path)

def load_optimizer_state(optimizer, name, path = 'obj/optimizer_states'):
    full_path = os.path.join(path, name)
    optimizer.load_state_dict(torch.load(full_path))