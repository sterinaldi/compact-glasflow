import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from CompactGF.utils import ExtensionError
from CompactGF.flow import CompactFlow
from CompactGF._loaders import loaders, available_codes_list

def available_codes():
    print(available_codes_list)

def save_flow(flow, name = 'trained_flow', folder = '.'):
    out_folder = Path(folder)
    out_folder.mkdir(exist_ok = True, parents = True)
    flow.config_dict['pt_file'] = str(Path(out_folder, f'{name}.pt'))
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
    with open(file, 'r') as f:
        config_dict = json.loads(json.load(f))
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

def load_file(file, pars, hyperpars, code, pdf = False, n_samples = None):
    if not code in loaders.keys():
        raise NotImplementedError(f"Code not implemented (yet). Available codes: {available_codes_list}")
    df_pars, df_hyperpars = _filter_df(*loaders[code](file, pars, hyperpars), pdf, n_samples)
    return df_pars.to_numpy(), df_hyperpars.to_numpy()

def load_data(path, pars, hyperpars, code, pdf = False, n_samples = None, ext = '.dat'):
    """
    Load sevn files from a folder. If pdf is true, each file is downsampled to the same number of samples and ignored if below
    """
    if not code in loaders.keys():
        raise NotImplementedError(f"Code not implemented (yet). Available codes: {available_codes_list}")
    data                  = [loaders[code](file, pars, hyperpars) for file in Path(path).glob('*'+ext)]
    df_pars               = pd.concat([df[0] for df in data])
    df_hyperpars          = pd.concat([df[1] for df in data])
    df_pars, df_hyperpars = _filter_df(df_pars, df_hyperpars, pdf, n_samples)
    return df_pars.to_numpy(), samples_hyperpars.to_numpy()

def _filter_df(df_pars, df_hyperpars, pdf = False, n_samples = None):
    if pdf:
        if n_samples is None:
            raise ValueError("Please specify the desired number of samples")
        n_samples = int(n_samples)
        red_dfs_hyperpars = []
        red_dfs_pars      = []
        for v in df_hyperpars.drop_duplicates().values:
            idx = np.array(np.prod(df_hyperpars == v, axis = 1).tolist(), dtype = bool)
            n_avail_samples = np.sum(idx)
            if n_avail_samples < n_samples:
                print(f"Hyperparameter(s) {v} excluded due to limited number of samples ({n_avail_samples} available, {n_samples} required).")
            else:
                selected = np.random.choice(n_avail_samples, size = n_samples, replace = False)
                red_dfs_pars.append(df_pars[idx].iloc[selected])
                red_dfs_hyperpars.append(df_hyperpars[idx].iloc[selected])
        df_pars = pd.concat(red_dfs_pars)
        df_hyperpars = pd.concat(red_dfs_hyperpars)
    else:
        if n_samples is not None:
            n_pts             = np.min([len(samples_pars), int(n_samples)])
            idx               = np.random.choice(len(samples_pars), size = n_pts, replace = False)
            samples_pars      = samples_pars[idx]
            samples_hyperpars = samples_hyperpars[idx]
    return df_pars, df_hyperpars
