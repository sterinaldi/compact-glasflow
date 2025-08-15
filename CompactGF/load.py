import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from CompactGF.utils import ExtensionError
from CompactGF.flow import CompactFlow

def save_flow(flow, name = 'trained_flow', folder = '.'):
    out_folder = Path(folder)
    out_folder.mkdir(exist_ok = True, parents = True)
    with open(Path(out_folder, f'{name}.json'), 'w') as f:
        json.dump(json.dumps(flow.config_dict), f)
    torch.save(flow.state_dict(), Path(out_folder, f'{name}.pt'))

def load_flow(file):
    ext = Path(file).suffix
    if ext != '.json':
        if ext == '.pt':
            raise ExtensionError("Please pass a .json file instead of a .pt file.")
        else:
            raise ExtensionError("Please provide a .json file.")
    config_dict = json.load(file)
    for key in config_dict.keys():
        if key == 'probit':
            config_dict[key] = bool(config_dict[key])
        elif key == 'bounds':
            if config_dict[key] == 'None':
                config_dict[key] = None
            else:
                config_dict[key] = np.atleast_2d(config_dict[key])
        elif key != 'pt_file':
            config_dict[key] = int(config_dict[key])
    flow = CompactFlow(**config_dict)
    if not Path(flow.pt_file).exists():
        raise FileNotFoundError(f"Please ensure that {flow.pt_file} exists.")
    flow.load_state_dict(torch.load(flow.pt_file, weights_only=True))
    flow.eval()
    return flow

def _load_sevn_file(file, pars, hyperpars):
    """
    Ad-hoc SEVN data product reader
    """
    column_names = ['ID', 'mass_1','mass_2', 'cmu1','cmu2','z', 'z_form','t','Z']
    df =  pd.read_csv(Path(file), delimiter = ' ', header = 1, names = column_names)
    df['q'] = df['mass_2']/df['mass_1']
    return df[pars], df[hyperpars]

def load_sevn_file(file, pars, hyperpars, n_samples = None):
    df_pars, df_hyperpars = _load_sevn_file(file, pars, hyperpars)
    samples_pars      = df_pars.to_numpy()
    samples_hyperpars = df_hyperpars.to_numpy()
    if n_samples is not None:
        n_pts             = np.min([len(samples_pars), n_samples])
        idx               = np.random.choice(len(samples_pars), size = n_pts, replace = False)
        samples_pars      = samples_pars[idx]
        samples_hyperpars = samples_hyperpars[idx]
    return samples_pars, samples_hyperpars

def load_sevn(path, pars, hyperpars, pdf = False, n_samples = None, ext = '.dat'):
    """
    Load sevn files from a folder. If pdf is true, each file is downsampled to the same number of samples and ignored if below
    """
    data = [_load_sevn_file(file, pars, hyperpars) for file in Path(path).glob('*'+ext)]
    if pdf:
        if n_samples_per_file is None:
            raise ValueError("Please specify the desired number of samples")
        n_samples = int(n_samples)
        indexes   = [np.random.choice(len(df[0]), size = n_samples, replace = False) if len(df[0]) >= n_samples else None for df in data]
        data = [[df[0][idx], df[1][idx]] if len(df[0]) >= n_samples else [None, None] for df, idx in zip(data, indexes)]
    samples_pars      = pd.concatenate([df[0] for df in data]).to_numpy()
    samples_hyperpars = pd.concatenate([df[1] for df in data]).to_numpy()
    return samples_pars, samples_hyperpars
