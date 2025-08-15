import numpy as np
import pandas as pd
from pathlib import Path

def _SEVN_loader(file, pars, hyperpars):
    """
    Ad-hoc SEVN data product reader
    """
    column_names = ['ID', 'mass_1','mass_2', 'cmu1','cmu2','z', 'z_form','t','Z']
    df =  pd.read_csv(Path(file), delimiter = ' ', header = 1, names = column_names)
    df['q'] = df['mass_2']/df['mass_1']
    return df[pars], df[hyperpars]

loaders = {'SEVN': _SEVN_loader}
available_codes_list = ', '.join(list(loaders.keys()))
