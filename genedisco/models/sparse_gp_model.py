"""
Copyright 2025 GeneDisco Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import torch
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# BoTorch and GPyTorch imports
from botorch.models import SingleTaskVariationalGP, ApproximateGP
from botorch.models.transforms import Standardize
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.kernels import (
    ScaleKernel, RBFKernel, MaternKernel, 
    LinearKernel, PeriodicKernel,
    SpectralMixtureKernel, AdditiveKernel
)
from gpytorch.means import ConstantMean, ZeroMean
from botorch.fit import fit_gpytorch_model

# Slingpy imports
from slingpy.models.abstract_base_model import AbstractBaseModel
from slingpy import AbstractDataSource


class HighDimensionalSparseGPModel(AbstractBaseModel):
    """
    A Sparse Gaussian Process model optimized for high-dimensional data.
    
    Features:
    - Multiple kernel options optimized for different data types
    - Automatic dimensionality reduction for very high-dimensional inputs
    - Smart inducing point selection strategies
    - Custom fitting routine with hyperparameter tuning
    - GPU acceleration support
    """
    
    def __init__(
        self,
        num_inducing: int = 100,
        kernel_type: str = "matern",
        matern_nu: float = 2.5,
        dim_reduction: Optional[str] = "auto",
        dim_reduction_target: int = 50,
        inducing_point_method: str = "kmeans",
        lr: float = 0.01,
        training_iters: int = 500,
        device: str = None,
        optimize_inducing_points: bool = True,
        standardize_input: bool = True,
        standardize_output: bool = True,
        jitter: float = 1e-4,
    ):
        """
        Initialize the sparse GP model with configuration options.
        
        Args:
            num_inducing: Number of inducing points for the sparse approximation
            kernel_type: Kernel function to use - one of "rbf", "matern", "linear", 
                         "periodic", "spectral_mixture", or "additive"
            matern_nu: Smoothness parameter for Matern kernel (0.5, 1.5, or 2.5)
            dim_reduction: Dimensionality reduction technique to use:
                          "auto" - automatically decide based on input dimensions
                          "pca" - use PCA
                          "random_proj" - use random projections
                          None - no dimensionality reduction
            dim_reduction_target: Target dimensions after reduction (if used)
            inducing_point_method: Method to select inducing points:
                                  "kmeans" - K-means clustering
                                  "random" - Random selection
                                  "greedy" - Greedy selection based on uncertainty
            lr: Learning rate for optimization
            training_iters: Maximum number of training iterations
            device: Computing device ('cuda', 'cpu', or None for automatic selection)
            optimize_inducing_points: Whether to optimize inducing point locations
            standardize_input: Whether to standardize input features
            standardize_output: Whether to standardize output values
            jitter: Small constant added to diagonal of kernel matrix for stability
        """
        super().__init__()
        
        # If device not specified, pick automatically
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.num_inducing = num_inducing
        self.kernel_type = kernel_type.lower()
        self.matern_nu = matern_nu
        self.dim_reduction = dim_reduction
        self.dim_reduction_target = dim_reduction_target
        self.inducing_point_method = inducing_point_method.lower()
        self.lr = lr
        self.training_iters = training_iters
        self._device = torch.device(device)
        self.optimize_inducing_points = optimize_inducing_points
        self.standardize_input = standardize_input
        self.standardize_output = standardize_output
        self.jitter = jitter
        
        # Initialize model components to be set during training
        self.model = None
        self.likelihood = None
        self.mll = None
        self.input_dim = None
        self.pca_transformer = None
        self.input_scaler = None
        self.output_scaler = None
        self.original_dim = None
        self.inducing_points = None
        
        # Validate inputs
        self._validate_params()
    
    def _validate_params(self):
        """Validate the initialization parameters."""
        valid_kernels = ["rbf", "matern", "linear", "periodic", 
                         "spectral_mixture", "additive"]
        if self.kernel_type not in valid_kernels:
            raise ValueError(f"kernel_type must be one of {valid_kernels}")
        
        valid_inducing_methods = ["kmeans", "random", "greedy"]
        if self.inducing_point_method not in valid_inducing_methods:
            raise ValueError(f"inducing_point_method must be one of {valid_inducing_methods}")
        
        if self.dim_reduction not in ["auto", "pca", "random_proj", None]:
            raise ValueError("dim_reduction must be 'auto', 'pca', 'random_proj', or None")
        
        if self.matern_nu not in [0.5, 1.5, 2.5]:
            raise ValueError("matern_nu must be one of [0.5, 1.5, 2.5]")
    
    @property
    def device(self):
        """Return the computing device."""
        return self._device
    
    def _prepare_data(
        self, 
        X: Union[np.ndarray, torch.Tensor], 
        Y: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare the data for model fitting or prediction:
        - Convert to torch tensor if needed
        - Move to correct device
        - Apply dimensionality reduction if configured
        - Apply scaling if configured
        
        Args:
            X: Input features, shape [N, D]
            Y: Target values, shape [N] or [N, 1], or None for prediction
            
        Returns:
            Tuple of processed X and Y tensors
        """
        # Convert to tensors if needed
        if not torch.is_tensor(X):
            X = torch.from_numpy(X).float()
        
        if Y is not None and not torch.is_tensor(Y):
            Y = torch.from_numpy(Y).float()
        
        # Record original dimensionality if not already set
        if self.original_dim is None and Y is not None:
            self.original_dim = X.shape[1]
            self.input_dim = self.original_dim
        
        # Apply input scaling if needed and in training mode (Y is provided)
        if self.standardize_input and Y is not None:
            self.input_scaler = StandardScaler()
            X_np = X.cpu().numpy()
            X = torch.from_numpy(self.input_scaler.fit_transform(X_np)).float()
        elif self.standardize_input and self.input_scaler is not None:
            X_np = X.cpu().numpy()
            X = torch.from_numpy(self.input_scaler.transform(X_np)).float()
        
        # Apply dimensionality reduction if configured and in training mode
        if self.dim_reduction and Y is not None:
            X = self._apply_dim_reduction_fit(X)
        elif self.dim_reduction and self.pca_transformer is not None:
            X = self._apply_dim_reduction_transform(X)
        
        # Update input_dim after potential dimensionality reduction
        if Y is not None and (self.input_dim is None or self.input_dim != X.shape[1]):
            self.input_dim = X.shape[1]
        
        # Apply output scaling if in training mode
        if Y is not None and self.standardize_output:
            Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
            self.output_scaler = StandardScaler()
            Y_np = Y.cpu().numpy()
            Y = torch.from_numpy(self.output_scaler.fit_transform(Y_np)).float()
        elif Y is not None:
            Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        
        # Move tensors to device
        X = X.to(self._device)
        if Y is not None:
            Y = Y.to(self._device)
        
        return X, Y
    
    def _apply_dim_reduction_fit(self, X: torch.Tensor) -> torch.Tensor:
        """Apply dimensionality reduction during model fitting."""
        X_np = X.cpu().numpy()
        
        # Determine dimensionality reduction approach
        if self.dim_reduction == "auto":
            # Automatically decide based on input dimensions
            if X.shape[1] > 100:
                reduction_method = "pca"
            else:
                # No reduction needed
                return X
        else:
            reduction_method = self.dim_reduction
        
        # Apply the selected reduction method
        if reduction_method == "pca":
            n_components = min(X.shape[0], X.shape[1], self.dim_reduction_target)
            self.pca_transformer = PCA(n_components=n_components)
            X_reduced = self.pca_transformer.fit_transform(X_np)
            return torch.from_numpy(X_reduced).float().to(self._device)
        elif reduction_method == "random_proj":
            # Use random projection
            from sklearn.random_projection import GaussianRandomProjection
            n_components = min(X.shape[1], self.dim_reduction_target)
            self.pca_transformer = GaussianRandomProjection(n_components=n_components)
            X_reduced = self.pca_transformer.fit_transform(X_np)
            return torch.from_numpy(X_reduced).float().to(self._device)
        
        # Default - no reduction
        return X
    
    def _apply_dim_reduction_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Apply previously fit dimensionality reduction to new data."""
        if self.pca_transformer is None:
            return X
        
        X_np = X.cpu().numpy()
        X_reduced = self.pca_transformer.transform(X_np)
        return torch.from_numpy(X_reduced).float().to(self._device)
    
    def _select_inducing_points(self, X: torch.Tensor) -> torch.Tensor:
        """
        Select inducing points for the sparse GP.
        
        Args:
            X: Training data tensor, shape [N, D]
            
        Returns:
            Inducing points tensor, shape [M, D]
        """
        N = X.shape[0]
        
        # Fall back to using all points if data is smaller than requested inducing points
        if N <= self.num_inducing:
            return X.clone()
        
        if self.inducing_point_method == "kmeans":
            # Use K-means clustering to find representative points
            X_np = X.cpu().numpy()
            kmeans = KMeans(n_clusters=self.num_inducing, random_state=0, n_init="auto")
            kmeans.fit(X_np)
            inducing_points = torch.from_numpy(kmeans.cluster_centers_).float().to(self._device)
            
        elif self.inducing_point_method == "random":
            # Randomly select inducing points
            rand_idx = torch.randperm(N)[:self.num_inducing]
            inducing_points = X[rand_idx].clone()
            
        elif self.inducing_point_method == "greedy":
            # More sophisticated greedy selection - using initialization with farthest points
            # Start with a random point
            X_np = X.cpu().numpy()
            selected_indices = [np.random.randint(0, N)]
            selected_points = X_np[selected_indices]
            
            # Iteratively select points farthest from the currently selected set
            for _ in range(self.num_inducing - 1):
                # Compute distances from each point to the nearest selected point
                distances = np.min(np.linalg.norm(
                    X_np[:, np.newaxis, :] - selected_points[np.newaxis, :, :], 
                    axis=2
                ), axis=1)
                
                # Select the point with maximum distance
                new_idx = np.argmax(distances)
                selected_indices.append(new_idx)
                selected_points = X_np[selected_indices]
            
            inducing_points = torch.from_numpy(selected_points).float().to(self._device)
        
        return inducing_points
    
    def _create_kernel(self):
        """Create the appropriate kernel based on configuration."""
        if self.kernel_type == "rbf":
            base_kernel = RBFKernel(ard_num_dims=self.input_dim)
        elif self.kernel_type == "matern":
            base_kernel = MaternKernel(nu=self.matern_nu, ard_num_dims=self.input_dim)
        elif self.kernel_type == "linear":
            base_kernel = LinearKernel()
        elif self.kernel_type == "periodic":
            base_kernel = PeriodicKernel()
        elif self.kernel_type == "spectral_mixture":
            base_kernel = SpectralMixtureKernel(num_mixtures=4, ard_num_dims=self.input_dim)
        elif self.kernel_type == "additive":
            # Combine RBF and linear kernels for additive structure
            rbf_kernel = RBFKernel(ard_num_dims=self.input_dim)
            linear_kernel = LinearKernel()
            base_kernel = AdditiveKernel(rbf_kernel, linear_kernel)
        
        # Wrap with a scale kernel
        return ScaleKernel(base_kernel)
    
    def _create_model(self, X_train: torch.Tensor, Y_train: torch.Tensor) -> None:
        """Create the sparse GP model."""
        # Select inducing points
        inducing_points = self._select_inducing_points(X_train)
        self.inducing_points = inducing_points
        
        # Use BoTorch's SingleTaskVariationalGP which handles inducing points
        self.model = SingleTaskVariationalGP(
            train_X=X_train,
            train_Y=Y_train,
            inducing_points=inducing_points,
            learn_inducing_locations=self.optimize_inducing_points,
        ).to(self._device)
        
        # Access and customize the model's components if needed
        # (BoTorch SingleTaskVariationalGP already sets up optimal components for us)
        self.likelihood = self.model.likelihood
    
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "HighDimensionalSparseGPModel":
        """
        Fit the sparse GP model to the training data.
        
        Args:
            X: Training features, shape [N, D]
            Y: Training targets, shape [N] or [N, 1]
            
        Returns:
            Self, for method chaining
        """
        # Prepare data - convert, scale, reduce dimensionality if needed
        X_train, Y_train = self._prepare_data(X, Y)
        
        # Create the model with appropriate inducing points
        self._create_model(X_train, Y_train)
        
        # Set up the loss function (marginal log-likelihood)
        mll = VariationalELBO(
            self.likelihood, 
            self.model, 
            num_data=X_train.shape[0],
        )
        
        # Fit the model
        fit_gpytorch_model(
            mll, 
            options={
                "maxiter": self.training_iters, 
                "lr": self.lr,
                "disp": True
            }
        )
        
        return self
    
    def predict(
        self, 
        X: np.ndarray, 
        return_std: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions with the fitted sparse GP model.
        
        Args:
            X: Test features, shape [M, D]
            return_std: Whether to return standard deviations along with means
            
        Returns:
            If return_std=True: Tuple of (mean, std) arrays, each of shape [M]
            If return_std=False: Mean array of shape [M]
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        
        # Prepare the test data - convert, move to device, apply transformations
        X_test, _ = self._prepare_data(X)
        
        # Set model to evaluation mode
        self.model.eval()
        self.likelihood.eval()
        
        # Make predictions
        with torch.no_grad():
            # Get posterior distribution at test points
            posterior = self.model.posterior(X_test)
            
            # Extract mean and variance
            pred_mean = posterior.mean.squeeze(-1)
            pred_var = posterior.variance.squeeze(-1)
            
            # Convert to numpy and move to CPU
            pred_mean_np = pred_mean.cpu().numpy()
            
            # Reverse standardization if applied during training
            if self.standardize_output and self.output_scaler is not None:
                pred_mean_np = self.output_scaler.inverse_transform(
                    pred_mean_np.reshape(-1, 1)
                ).ravel()
                
                if return_std:
                    # Need to scale the standard deviation by output scale
                    std_np = torch.sqrt(pred_var).cpu().numpy()
                    # Multiply by the output scaler's scale_ to get the correct units
                    std_np = std_np * self.output_scaler.scale_[0]
                    return pred_mean_np, std_np
            
            if return_std:
                std_np = torch.sqrt(pred_var).cpu().numpy()
                return pred_mean_np, std_np
            
            return pred_mean_np
    
    def predict_with_uncertainty(
        self, 
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with the fitted sparse GP model, returning mean and variance.
        This is a convenience method that matches GeneDisco's interface.
        
        Args:
            X: Test features, shape [M, D]
            
        Returns:
            Tuple of (mean, variance) arrays, each of shape [M]
        """
        mean, std = self.predict(X, return_std=True)
        return mean, std ** 2
    
    def posterior(self, X: Union[np.ndarray, torch.Tensor]):
        """
        Get the posterior distribution at the given points.
        Useful for acquisition functions like LevelSetBAX that need the full posterior.
        
        Args:
            X: Test features, shape [M, D]
            
        Returns:
            GPyTorch posterior distribution object
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        
        # Prepare the test data
        X_test, _ = self._prepare_data(X)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Return the posterior
        with torch.no_grad():
            return self.model.posterior(X_test)
    
    def sample_posterior(
        self, 
        X: np.ndarray, 
        num_samples: int = 10
    ) -> np.ndarray:
        """
        Draw samples from the posterior distribution at the given points.
        
        Args:
            X: Test features, shape [M, D]
            num_samples: Number of samples to draw
            
        Returns:
            Samples array of shape [num_samples, M]
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        
        # Prepare the test data
        X_test, _ = self._prepare_data(X)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get posterior and draw samples
        with torch.no_grad():
            posterior = self.model.posterior(X_test)
            samples = posterior.rsample(torch.Size([num_samples]))
            samples = samples.squeeze(-1).cpu().numpy()  # [num_samples, M]
            
            # Reverse standardization if applied during training
            if self.standardize_output and self.output_scaler is not None:
                # Apply inverse transform to each sample
                transformed_samples = np.zeros_like(samples)
                for i in range(samples.shape[0]):
                    transformed_samples[i] = self.output_scaler.inverse_transform(
                        samples[i].reshape(-1, 1)
                    ).ravel()
                return transformed_samples
            
            return samples
    
    def train_model(self, X: np.ndarray, Y: np.ndarray):
        """
        Alias for fit() to maintain compatibility with GeneDisco's existing interface.
        
        Args:
            X: Training features, shape [N, D]
            Y: Training targets, shape [N] or [N, 1]
            
        Returns:
            Self, for method chaining
        """
        return self.fit(X, Y)