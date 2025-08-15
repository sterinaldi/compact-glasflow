import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from corner import corner

class ExtensionError(Exception):
    pass

def compare_with_data(flow, samples_pars, samples_hyperpars, labels = None, name = 'comparison', out_folder = '.', samples_label = r'\mathrm{Samples}'):
    """
    Cornerplot comparing trained flow with data
    """
    if labels is not None:
        if type(labels) == str:
            labels = [f'${labels}$']
        else:
            labels = [f'${l}$' for l in labels]
    samples_label = f'${samples_label}$'
    # Corner
    fig = corner(samples_pars, labels = labels, show_titles = False, quiet = True, color = 'firebrick',  hist_kwargs = {'density': True, 'label': samples_label}, plot_density = False, no_fill_contours = True, plot_datapoints = False, hist_bin_factor = int(np.sqrt(len(samples_pars)))/20)
    fig = corner(flow.rvs(samples_hyperpars, len(samples_pars)), labels = labels, show_titles = False, quiet = True, color = 'steelblue',  hist_kwargs = {'density': True, 'label': r'$\mathrm{CompactGF}$'}, plot_density = False, no_fill_contours = True, plot_datapoints = False, hist_bin_factor = int(np.sqrt(len(samples_pars)))/20, fig = fig)
    fig.axes[samples_pars.shape[-1]-1].legend(loc = 5,  *fig.axes[0].get_legend_handles_labels())
    fig.savefig(Path(out_folder, f'{name}.pdf'), bbox_inches = 'tight')
