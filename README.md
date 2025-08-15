# compact-glasflow
[glasflow](https://github.com/uofgravity/glasflow) wrapper targeted at synthetic catalogues of compact binaries. The `CompactFlow` class includes the basic `scipy.stats` methods `pdf`, `logpdf` and `rvs` 

## Usage
An example of how to use `CompactGF` to train a normalising flow:

```python
from CompactGF.flow import CompactFlow
from CompactGF.load import load_file, load_flow, save_flow
from CompactGF.utils import compare_with_data

postprocess = False
pdf         = True
probit      = True
code        = 'SEVN'
file        = './my_synthetic_catalogue.dat'
name        = 'trained_flow'
bounds      = [[0,50],[0,1]]
pars        = ['mass_1','q']
hyperpars   = ['Z']
labels      = [r'\mathrm{M}_1',r'q']
n_samples   = 1000
n_epochs    = 500

samples_pars, samples_hyperpars = load_file(file,
                                            code      = code,
                                            pars      = pars,
                                            hyperpars = hyperpars,
                                            n_samples = n_samples,
                                            pdf       = pdf,
                                            )

if not postprocess:
    flow = CompactFlow(n_dimensions      = len(pars),
                       n_hyperparameters = len(hyperpars),
                       probit            = probit,
                       bounds            = bounds,
                       )

    flow.train_flow(samples_pars, samples_hyperpars, n_epochs = n_epochs)
    save_flow(flow, name = name)

else:
    flow = load_flow(name+'.json')

# Comparison plot
compare_with_data(flow, samples_pars, samples_hyperpars)
```

## Acknowledgements

Training parameters are taken from [Colloms et al. (2025)](). Please cite it if you make use of this code in your work:
```
@ARTICLE{2025ApJ...988..189C,
       author = {{Colloms}, Storm and {Berry}, Christopher P.~L. and {Veitch}, John and {Zevin}, Michael},
        title = "{Exploring the Evolution of Gravitational-wave Emitters with Efficient Emulation: Constraining the Origins of Binary Black Holes Using Normalizing Flows}",
      journal = {\apj},
         year = 2025,
        month = aug,
       volume = {988},
       number = {2},
          eid = {189},
        pages = {189},
          doi = {10.3847/1538-4357/ade546},
archivePrefix = {arXiv},
       eprint = {2503.03819},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025ApJ...988..189C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
