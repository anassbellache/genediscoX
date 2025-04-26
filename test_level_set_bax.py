#!/usr/bin/env python
"""
Test script for the two-stage "Uncertainty + Diversity" LevelSetBAXAcquisition function.

This script creates synthetic data and a mock GP model to demonstrate how
the acquisition function selects points using both uncertainty and diversity.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from slingpy import AbstractDataSource, AbstractBaseModel

from genedisco.active_learning_methods.acquisition_functions.level_set_bax import LevelSetBAXAcquisition


class MockGPModel(AbstractBaseModel):
    """Mock GP model that returns precomputed means and variances."""
    
    def __init__(self, means, variances):
        """
        Initialize with precomputed means and variances.
        
        Args:
            means: Array of mean values
            variances: Array of variance values
        """
        self.means = means
        self.variances = variances
    
    def posterior(self, X):
        """Return a posterior with the precomputed values."""
        class SimplePosterior:
            def __init__(self, mean, variance):
                self.mean = mean
                self.variance = variance
        
        # For realistic GP behavior, we'll simulate some correlation in the posterior
        # based on distances between points
        n = X.shape[0]
        if n <= len(self.means):
            mean_tensor = torch.tensor(self.means[:n]).unsqueeze(-1)
            var_tensor = torch.tensor(self.variances[:n]).unsqueeze(-1)
        else:
            # If we have more points than precomputed values, cycle through them
            mean_tensor = torch.tensor(np.tile(self.means, (n // len(self.means) + 1))[:n]).unsqueeze(-1)
            var_tensor = torch.tensor(np.tile(self.variances, (n // len(self.variances) + 1))[:n]).unsqueeze(-1)
        
        return SimplePosterior(mean_tensor, var_tensor)


class DummyDataSource(AbstractDataSource):
    """Dummy data source that wraps numpy arrays."""
    
    def __init__(self, features, targets=None, ids=None):
        """
        Initialize with features and optional targets.
        
        Args:
            features: Feature matrix [N, d]
            targets: Target values [N]
            ids: Sample IDs or indices
        """
        self.features = features
        self.targets = targets if targets is not None else np.zeros(len(features))
        self.ids = ids if ids is not None else [str(i) for i in range(len(features))]
    
    def get_data(self):
        """Return the data as a list [features, targets]."""
        return [self.features, self.targets]
    
    def get_shape(self):
        """Return shapes of features and targets."""
        return [self.features.shape, self.targets.shape]
    
    def subset(self, indices):
        """Return a subset of the data."""
        if isinstance(indices[0], str):
            # Convert string indices to integer positions
            pos = [int(idx) for idx in indices]
            return DummyDataSource(self.features[pos], self.targets[pos], indices)
        else:
            # Already integer indices
            return DummyDataSource(self.features[indices], self.targets[indices], 
                                  [self.ids[i] for i in indices])


def visualize_selection(features, selected_indices, stage1_indices=None, title="Selected Points"):
    """
    Visualize the selected points in a 2D plot (using PCA if dimensions > 2).
    
    Args:
        features: Feature matrix [N, d]
        selected_indices: Indices of selected points
        stage1_indices: Indices of points selected in Stage 1 (for two-stage method)
        title: Plot title
    """
    # If features are high-dimensional, use PCA to project to 2D
    if features.shape[1] > 2:
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
    else:
        features_2d = features[:, :2]
    
    plt.figure(figsize=(10, 7))
    
    # Plot all points
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c='lightgray', alpha=0.5, label='Candidates')
    
    # Plot Stage 1 selected points
    if stage1_indices is not None:
        plt.scatter(features_2d[stage1_indices, 0], features_2d[stage1_indices, 1], 
                   c='orange', s=60, alpha=0.6, label='Stage 1 (UCB Top-k)')
    
    # Plot final selected points
    plt.scatter(features_2d[selected_indices, 0], features_2d[selected_indices, 1], 
               c='red', s=100, marker='*', label='Selected (Stage 2)')
    
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    return plt.gcf()


def main():
    # Parameters
    n_samples = 200     # Total number of candidates in the pool
    batch_size = 5      # Number of points to select (B)
    beta = 10           # Stage 1 pool size multiplier (β·B)
    kappa = 2.0         # Exploration weight in UCB
    seed = 42
    threshold_value = 0.5  # Level-set threshold c
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create synthetic features - simulate gene embeddings
    n_dim = 20  # Feature dimension
    features = np.random.randn(n_samples, n_dim)
    
    # Add some cluster structure to make it more interesting
    n_clusters = 7
    cluster_centers = np.random.randn(n_clusters, n_dim) * 2
    for i in range(n_samples):
        cluster = i % n_clusters
        features[i] = features[i] * 0.3 + cluster_centers[cluster]
    
    # Create mock GP posterior values
    # Here we'll create a region of high means and a separate region of high variance
    means = np.zeros(n_samples)
    variances = np.ones(n_samples) * 0.1
    
    # Add a region of high means (higher function values)
    high_mean_indices = np.random.choice(n_samples, size=n_samples//10, replace=False)
    means[high_mean_indices] = np.random.uniform(0.5, 1.0, size=len(high_mean_indices))
    
    # Add a region of high uncertainty
    high_var_indices = np.random.choice(
        list(set(range(n_samples)) - set(high_mean_indices)), 
        size=n_samples//8, 
        replace=False
    )
    variances[high_var_indices] = np.random.uniform(0.5, 1.0, size=len(high_var_indices))
    
    # Create data source and model
    data_source = DummyDataSource(features)
    model = MockGPModel(means, variances)
    
    # Create acquisition function
    acq_func = LevelSetBAXAcquisition(
        threshold_value=threshold_value,
        beta=beta,
        kappa=kappa,
        use_gpu=torch.cuda.is_available()
    )
    
    # Create list of available indices
    available_indices = [str(i) for i in range(n_samples)]
    last_selected = []  # No previous selections
    
    # Run acquisition function
    selected_indices = acq_func(data_source, batch_size, available_indices, last_selected, model)
    selected_idx_int = [int(idx) for idx in selected_indices]
    
    # For visualization, we want to see both Stage 1 and Stage 2 selected points
    # Extract top β·B indices based on UCB scores
    with torch.no_grad():
        X = torch.from_numpy(features).float()
        posterior = model.posterior(X)
        ucb = posterior.mean.squeeze(-1) + kappa * torch.sqrt(posterior.variance.squeeze(-1))
        stage1_size = min(beta * batch_size, n_samples)
        _, top_indices = torch.topk(ucb, stage1_size)
        top_indices = top_indices.numpy()
    
    # Visualize results
    fig = visualize_selection(features, selected_idx_int, top_indices, 
                      title=f"Two-Stage Selection: Level-Set BAX (β={beta}, κ={kappa})")
    
    # Print selection details
    print(f"Pool size: {n_samples}")
    print(f"Batch size (B): {batch_size}")
    print(f"Stage 1 pool size (β·B): {beta * batch_size}")
    print(f"Selected indices: {selected_indices}")
    print(f"Mean values at selected points: {means[selected_idx_int]}")
    print(f"Variance values at selected points: {variances[selected_idx_int]}")
    
    # Save plot
    plt.savefig("level_set_bax_two_stage.png")
    plt.show()
    
    # Edge case: Single-point selection (should fall back to just UCB)
    single_batch = acq_func(data_source, 1, available_indices, last_selected, model)
    print(f"\nSingle-point selection (should use UCB only): {single_batch}")


if __name__ == "__main__":
    main()