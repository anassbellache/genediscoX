import torch
from botorch.models import SingleTaskVariationalGP
from gpytorch.mlls import VariationalELBO
from botorch.fit import fit_gpytorch_model
from slingpy.models.abstract_base_model import AbstractBaseModel

class GeneDiscoSTVGPModel(AbstractBaseModel):
    """
    A Single-Task Variational GP model that can be trained on CPU or GPU,
    conforming to GeneDisco's AbstractBaseModel interface (example).
    """
    def __init__(self, 
                 num_inducing: int = 30, 
                 lr: float = 0.01, 
                 training_iters: int = 200,
                 device: str = None):
        """
        Args:
            num_inducing: # of inducing points M for the variational GP
            lr: learning rate
            training_iters: # of training iterations
            device: 'cuda' or 'cpu' or None. If None, auto-select based on availability.
        """
        super().__init__()  # ensure we call the base class init

        # If device not specified, pick automatically
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.num_inducing = num_inducing
        self.lr = lr
        self.training_iters = training_iters
        self._device = torch.device(device)

        self.model = None
        self.likelihood = None
        self.mll = None

    @property
    def device(self):
        """Expose the device for external usage (e.g. by DiscoBAX)."""
        return self._device

    def train_model(self, X_train, Y_train):
        """
        Train the STVGP model on the given data. 
        X_train, Y_train can be NumPy arrays or torch Tensors. 
        We'll convert them to torch Tensors on self.device.
        """
        if not torch.is_tensor(X_train):
            X_train = torch.from_numpy(X_train).float()
        if not torch.is_tensor(Y_train):
            Y_train = torch.from_numpy(Y_train).float()

        X_train = X_train.to(self._device)
        Y_train = Y_train.view(-1, 1).to(self._device)  # shape [N,1]

        # 1) Choose M random inducing points from training data
        n = X_train.size(0)
        if n <= self.num_inducing:
            inducing_points = X_train.clone()
        else:
            rand_idx = torch.randperm(n)[: self.num_inducing]
            inducing_points = X_train[rand_idx].clone()

        # 2) Create the BoTorch STVGP model
        self.model = SingleTaskVariationalGP(
            train_X=X_train, 
            train_Y=Y_train,
            inducing_points=inducing_points,
            learn_inducing_points=True,
        ).to(self._device)

        # 3) Prepare MLL with the model's likelihood
        self.likelihood = self.model.likelihood
        self.mll = VariationalELBO(
            self.likelihood, self.model, num_data=n
        ).to(self._device)

        # 4) Fit hyperparameters + variational params
        fit_gpytorch_model(
            self.mll, 
            options={"maxiter": self.training_iters, "lr": self.lr}
        )

    def predict(self, X_test, return_std=True):
        """
        Return predictive mean (and optional std) for X_test.

        Args:
            X_test: shape [M, d], can be NumPy or torch Tensor
            return_std: if True, return standard deviation

        Returns:
            mean: shape [M]
            std: shape [M], or None if return_std=False
        """
        if not torch.is_tensor(X_test):
            X_test = torch.from_numpy(X_test).float()
        X_test = X_test.to(self._device)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), torch.autograd.profiler.record_function("predict"):
            posterior = self.model.posterior(X_test)
            pred_mean = posterior.mean.squeeze(-1)   # [M]
            pred_var = posterior.variance.squeeze(-1)  # [M]
            if return_std:
                return (
                    pred_mean.cpu().numpy(), 
                    torch.sqrt(pred_var).cpu().numpy()
                )
            else:
                return pred_mean.cpu().numpy(), None

    def posterior(self, X_test):
        """
        Return a BoTorch Posterior object for X_test, useful for sampling.
        """
        if not torch.is_tensor(X_test):
            X_test = torch.from_numpy(X_test).float()
        X_test = X_test.to(self._device)

        return self.model.posterior(X_test)

    def condition_on_observations(self, X_new, Y_new):
        """
        Return a fantasy model conditioned on new observations (X_new, Y_new).
        For DiscoBAX, we might need to re-fit or use GPyTorch's 
        condition_on_observations.
        """
        if not torch.is_tensor(X_new):
            X_new = torch.from_numpy(X_new).float()
        if not torch.is_tensor(Y_new):
            Y_new = torch.from_numpy(Y_new).float()
        X_new = X_new.to(self._device)
        Y_new = Y_new.view(-1,1).to(self._device)

        # GPyTorch has .condition_on_observations for approximate GPs as well:
        # This returns a new model (a Posterior object) rather than an entire GP instance 
        # in current GPyTorch versions. The approach below is conceptual 
        # and may vary with GPyTorch version.
        conditioned_model = self.model.condition_on_observations(X_new, Y_new)
        return conditioned_model
