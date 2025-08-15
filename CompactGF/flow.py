import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from glasflow import CouplingNSF
from torch import from_numpy, no_grad
from torch.utils.data import TensorDataset, DataLoader

from CompactGF._probit import transform_to_probit, probit, from_probit, probit_logJ

class CompactFlow(CouplingNSF):
    """
    Wrapper for the CouplingNSF class from glasflow to add straightforward scipy-like pdf, logpdf, rvs methods.
    It also includes a user-transparent probit transformation, if activated.
    Heavily inspired (if not straightforwardly following) by glasflow's user guide.
    Default values from Colloms et al. (2025)
    """
    def __init__(n_dimensions,
                 n_hyperparameters,
                 n_transforms         = 6,
                 n_neurons            = 128,
                 n_bins               = 5,
                 bounds               = None,
                 probit               = False,
                 batch_size           = 1000,
                 learning_rate        = 0.001,
                 *args,
                 **kwargs
                 ):
        # Store config parameters
        self.n_dimensions      = int(n_dimensions)
        self.n_hyperparameters = int(n_hyperparameters)
        self.n_transforms      = int(n_transforms)
        self.n_neurons         = int(n_neurons)
        self.n_bins            = int(n_bins)
        self.batch_size        = int(batch_size)
        self.learning_rate     = learning_rate
        self.bounds            = bounds
        self.probit            = probit
        self.device            = 'cpu' # Hardcoded for now
        # Consistency checks
        if self.bounds is not None:
            if not self.n_dimensions == len(self.bounds):
                raise ValueError("Bounds are inconsistent with the provided number of dimensions.")
        
        # config_dict (for saving purposes)
        self.config_dict = {}
        self.config_dict['n_dimensions']      = self.n_dimensions
        self.config_dict['n_hyperparameters'] = self.n_hyperparameters
        self.config_dict['n_transforms']      = self.n_transforms
        self.config_dict['n_neurons']         = self.n_neurons
        self.config_dict['n_bins']            = self.n_bins
        self.config_dict['probit']            = self.probit
        if self.bounds is None:
            self.config_dict['bounds'] = 'None'
        else:
            self.config_dict['bounds'] = self.bounds.tolist()
        # Instantiate actual NF
        super().__init__(n_inputs             = self.n_dimensions,
                         n_conditional_inputs = self.n_hyperparameters,
                         n_transforms         = self.n_transforms,
                         n_neurons            = self.n_neurons,
                         num_bins             = n_bins
                         *args,
                         **kwargs)
        
    def _prepare_training_validation_datasets(self, data, hyperpars):
        """
        Prepares training and validation datasets using SciKit Learn routine
        """
        # Split data
        x_train, x_val, y_train, y_val = train_test_split(data, hyperpars)
        # Training
        x_train_tensor    = from_numpy(x_train.astype(np.float32))
        y_train_tensor    = from_numpy(y_train.astype(np.float32))
        train_dataset     = TensorDataset(x_train_tensor, y_train_tensor)
        self.train_loader = DataLoader(train_dataset,
                                         batch_size = selfbatch_size,
                                         shuffle    = True,
                                         )
        # Validation
        x_val_tensor    = from_numpy(x_val.astype(np.float32))
        y_val_tensor    = from_numpy(y_val.astype(np.float32))
        val_dataset     = torch.utils.data.TensorDataset(x_val_tensor, y_val_tensor)
        self.val_loader = DataLoader(val_dataset,
                                     batch_size = self.batch_size,
                                     shuffle    = False
                                     )

    def _train_flow(self, n_epochs = 1000, verbose = True):
        """
        Internal optimisation routine.
        """
        # Preparation
        self.loss = dict(train=[], val=[])
        optimiser = torch.optim.Adam(self.parameters(),
                                     lr           = self.learning_rate,
                                     weight_decay = 0,
                                     )
        # Training
        for _ in tqdm(range(n_epochs), desc = 'Training', disable = not(verbose)):
            self.train()
            train_loss = 0.
            for batch in self.train_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                optimiser.zero_grad()
                _loss = -self.log_prob(x, conditional=y).mean()
                _loss.backward()
                optimiser.step()
                train_loss += _loss.item()
            self.loss["train"].append(train_loss / len(train_loader))

            flow.eval()
            val_loss = 0.
            for batch in self.val_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                with no_grad():
                    _loss = -self.log_prob(x, conditional=y).mean().item()
                val_loss += _loss
            self.loss["val"].append(val_loss / len(val_loader))
        self.eval()
    
    def plot_loss(self, out_folder = '.'):
        """
        Plots the loss function after training for diagnostic purposes.
        """
        # Check that the flow has actually been trained
        if not hasattr(self, 'loss'):
            raise AttributeError("Loss not available. Please train the flow first.")
        # Plot
        fig, ax = plt.subplots()
        ax.plot(self.loss['train'], color = 'steelblue', label = r'$\mathrm{Train}$')
        ax.plot(self.loss['val'], color = 'firebrick', linestyle = '--', label = r'$\mathrm{Validation}$')
        ax.set_xlabel(r'$\mathrm{Epoch}$')
        ax.set_ylabel(r'$\mathrm{Loss}$')
        ax.legend(loc = 0)
        fig.savefig(Path(out_folder, 'loss_function.pdf'), bbox_inches = 'tight')
    
    def train(self, data, hyperpars, n_epochs = 1000, verbose = True):
        """
        Training routine.
        """
        # Consistency checks
        if not data.shape[-1] == self.n_dimensions:
            raise ValueError("Data dimensionality is inconsistent with the expected number of dimensions.")
        if not hyperpars.shape[-1] == self.n_hyperparameters:
            raise ValueError("Hyperparameter dimensionality is inconsistent with the expected number of hyperparameters.")
        # Prepare dataset
        if self.probit:
            data = transform_to_probit(data, self.bounds)
        self._prepare_training_validation_datasets(data, hyperpars)
        # Actual optimisation
        self._train_flow(n_epochs = n_epochs, verbose = verbose)
        # Plot loss function
        if verbose:
            self.plot_loss()
    
    def pdf(self, x, *hyperpars):
        """
        pdf evaluation function
        """
        hyperpars = np.atleast_2d(hyperpars)
        x         = np.atleast_2d(x)
        # Sanity check
        if not x.shape[-1] == self.n_dimensions:
            raise ValueError("Data dimensionality is inconsistent with the expected number of dimensions.")
        if not hyperpars.shape[-1] == self.n_hyperparameters:
            raise ValueError("Hyperparameter dimensionality is inconsistent with the expected number of hyperparameters.")
        return np.exp(self._logpdf(x, hyperpars))
    
    def logpdf(self, x, *hyperpars):
        """
        logpdf evaluation function
        """
        hyperpars = np.atleast_2d(hyperpars)
        x         = np.atleast_2d(x)
        # Sanity check
        if not x.shape[-1] == self.n_dimensions:
            raise ValueError("Data dimensionality is inconsistent with the expected number of dimensions.")
        if not hyperpars.shape[-1] == self.n_hyperparameters:
            raise ValueError("Hyperparameter dimensionality is inconsistent with the expected number of hyperparameters.")
        return self._logpdf(x, hyperpars)
        
    @probit
    def _logpdf(self, x, hyperpars):
        """
        Internal logpdf evaluation. Points are already transformed in probit space by the decorator.
        """
        # Repeat value for each point
        if hyperpars.shape[0] == 1:
            hyperpars = np.repeat(hyperpars, x.shape[0], axis = 0)
        # Prepare arrays and tensors
        old_shape    = x.shape
        x_reshaped   = x.reshape(x, (-1, self.n_dimensions))
        x_tensor     = from_numpy(x_reshaped).to(self.device)
        hyperpars    = from_numpy(hyperpars.astype(np.float32)).to(self.device)
        # Evaluate
        with no_grad():
            log_pdf      = self.log_prob(x_tensor, hyperpars).cpu().numpy()
        log_jacobian = -probit_logJ(x_reshaped, self.bounds, flag = self.probit)
        return np.reshape(log_pdf + log_jacobian, old_shape)
    
    def rvs(self, hyperpars, n = 1):
        """
        Random sampling
        """
        hyperpars = np.atleast_2d(hyperpars)
        n         = int(n)
        # Sanity check
        if not hyperpars.shape[-1] == self.n_hyperparameters:
            raise ValueError("Hyperparameter dimensionality is inconsistent with the expected number of hyperparameters.")
        return self._rvs(hyperpars, n)
    
    @from_probit
    def _rvs(self, hyperpars, n = 1):
        """
        Internal rvs routine
        """
        # Repeat value for each point
        if hyperpars.shape[0] == 1:
            hyperpars = np.repeat(hyperpars, n, axis = 0)
        # Prepare tensor
        hyperpars = from_numpy(hyperpars.astype(np.float32)).to(self.device)
        # Sampling
        with no_grad():
            samples = self.sample(n, hyperpars).cpu().numpy()
        return samples
