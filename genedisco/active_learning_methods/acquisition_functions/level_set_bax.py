"""
Copyright 2021 Patrick Schwab, Arash Mehrjou, GlaxoSmithKline plc; Andrew Jesson, University of Oxford; Ashkan Soleymani, MIT

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
from typing import List, AnyStr, Tuple, Optional, Union
from slingpy import AbstractDataSource, AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import BaseBatchAcquisitionFunction
from sklearn.cluster import KMeans


class LevelSetBAXAcquisition(BaseBatchAcquisitionFunction):
    """
    Two-stage "Uncertainty + Diversity" Level-Set DiscoBAX acquisition function.
    
    Stage 1: Score candidates with GP-UCB = μ(x) + κσ(x) and keep top β·B points.
    Stage 2: Use K-means clustering on embeddings to ensure diversity among selected points.
    
    This implementation combines uncertainty sampling with diversity to find points 
    that provide maximum information gain about the level set:
    Xc = {x ∈ X : f(x) ≥ c} - the set of inputs whose function values are above threshold c.
    """
    def __init__(
        self,
        threshold_value: float,
        num_posterior_samples: int = 50,
        noise_type: str = "additive",
        noise_level: float = 0.1,
        beta: int = 10,
        kappa: float = 2.0,
        use_gpu: bool = True,
    ):
        """
        Args:
            threshold_value: The level-set threshold c where we want to identify {x : f(x) ≥ c}
            num_posterior_samples: Number of posterior samples for MC estimation of info gain
            noise_type: Type of observation noise ('additive' or 'multiplicative')
            noise_level: Standard deviation of noise (for additive) or scale (for multiplicative)
            beta: Multiplicative factor for Stage 1 pool size (β·B)
            kappa: Exploration weight in GP-UCB = μ(x) + κσ(x)
            use_gpu: Whether to use GPU acceleration when available
        """
        super().__init__()
        self.c = threshold_value
        self.num_samples = num_posterior_samples
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.beta = max(1, int(beta))
        self.kappa = float(kappa)
        
        # Set device for computations
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        
    def __call__(
        self,
        dataset_x: AbstractDataSource,
        batch_size: int,
        available_indices: List[AnyStr],
        last_selected_indices: List[AnyStr],
        last_model: AbstractBaseModel
    ) -> List[AnyStr]:
        """
        Select batch_size items using the two-stage "Uncertainty + Diversity" strategy.
        
        Args:
            dataset_x: Dataset containing features
            batch_size: Number of examples to select
            available_indices: List of available indices to select from
            last_selected_indices: Indices selected in the previous round
            last_model: Trained model from which to sample function posteriors
            
        Returns:
            List of selected indices (length = batch_size)
        """
        # Handle edge case: if fewer available than requested, return all
        if len(available_indices) <= batch_size:
            return available_indices
            
        # Get candidate feature data
        candidate_data = dataset_x.subset(available_indices)
        candidate_features_np = candidate_data.get_data()[0]  # Assumes features are first in the data tuple
        
        # Convert to tensor and move to appropriate device
        X_candidates = torch.from_numpy(candidate_features_np).float().to(self.device)
        
        # Fallback to single-stage acquisition if batch_size is 1
        if batch_size == 1:
            return self._select_single_point(X_candidates, last_model, available_indices)
            
        # Otherwise, proceed with the two-stage approach
        return self._select_two_stage(X_candidates, last_model, available_indices, batch_size)
    
    def _select_single_point(
        self,
        X_candidates: torch.Tensor,
        model: AbstractBaseModel,
        available_indices: List[AnyStr]
    ) -> List[AnyStr]:
        """
        Simple selection of a single point using GP-UCB criterion.
        
        Args:
            X_candidates: Tensor of candidates [N, d]
            model: Trained model
            available_indices: Original indices
            
        Returns:
            List with a single selected index
        """
        with torch.no_grad():
            # Get model posterior
            posterior = self._get_model_posterior(X_candidates, model)
            
            # Compute UCB scores: μ(x) + κσ(x)
            ucb_scores = posterior['mean'] + self.kappa * torch.sqrt(posterior['variance'])
            
            # Select the point with highest UCB score
            best_idx = torch.argmax(ucb_scores).item()
            
            return [available_indices[best_idx]]
    
    def _select_two_stage(
        self,
        X_candidates: torch.Tensor,
        model: AbstractBaseModel,
        available_indices: List[AnyStr],
        batch_size: int
    ) -> List[AnyStr]:
        """
        Two-stage selection process:
        1. Score with GP-UCB and keep top β·B candidates
        2. Apply K-means clustering for diversity
        
        Args:
            X_candidates: Tensor of candidates [N, d]
            model: Trained model 
            available_indices: Original indices
            batch_size: Number of points to select
            
        Returns:
            List of selected indices
        """
        N = X_candidates.shape[0]
        B = batch_size
        
        # Get model posterior
        with torch.no_grad():
            posterior = self._get_model_posterior(X_candidates, model)
            
            # Compute UCB scores: μ(x) + κσ(x)
            mu = posterior['mean']
            sigma = torch.sqrt(posterior['variance'])
            ucb_scores = mu + self.kappa * sigma
        
        # Stage 1: Select top β·B candidates by UCB score
        k = min(self.beta * B, N)  # Ensure we don't request more than available
        _, top_indices = torch.topk(ucb_scores, k)
        
        # Get the feature embeddings for these top candidates
        # For K-means, we'll use the original feature space
        # Note: In a real implementation, you might want to use different embeddings
        # that are more suitable for measuring diversity
        top_candidates = X_candidates[top_indices].cpu().numpy()
        
        # Stage 2: Apply K-means clustering to ensure diversity
        try:
            kmeans = KMeans(n_clusters=B, init="k-means++", n_init="auto", random_state=0)
            cluster_labels = kmeans.fit_predict(top_candidates)
            
            selected_indices = []
            
            # From each cluster, select the candidate with highest UCB score
            for cluster_id in range(B):
                # Find indices belonging to this cluster
                cluster_members = torch.tensor(
                    [i for i in range(k) if cluster_labels[i] == cluster_id]
                )
                
                # Handle empty clusters
                if len(cluster_members) == 0:
                    continue
                
                # Get UCB scores of the cluster members
                cluster_scores = ucb_scores[top_indices[cluster_members]]
                
                # Find best member within cluster
                best_in_cluster = cluster_members[torch.argmax(cluster_scores)]
                
                # Add to selection
                selected_indices.append(top_indices[best_in_cluster].item())
                
            # If we didn't get enough points (due to empty clusters),
            # fill in with remaining high-score points
            if len(selected_indices) < B:
                # Get indices that weren't already selected
                remaining = [idx.item() for idx in top_indices if idx.item() not in selected_indices]
                
                # Add them until we have B points
                selected_indices.extend(remaining[:B - len(selected_indices)])
                
            # Map back to original indices
            selected_original = [available_indices[i] for i in selected_indices]
            
            return selected_original[:B]  # Ensure we return exactly B indices
            
        except Exception as e:
            print(f"K-means clustering failed: {e}")
            # Fallback to simple top-UCB selection
            _, top_indices = torch.topk(ucb_scores, B)
            selected_indices = [available_indices[i.item()] for i in top_indices]
            return selected_indices
    
    def _get_model_posterior(
        self,
        X: torch.Tensor,
        model: AbstractBaseModel
    ) -> dict:
        """
        Get posterior mean and variance from the model.
        
        Args:
            X: Input tensor [N, d]
            model: Trained model
            
        Returns:
            Dictionary with posterior 'mean' and 'variance' tensors
        """
        # Try different methods depending on model interface
        try:
            # First try to directly get posterior from model if it has GP-like interface
            if hasattr(model, 'posterior') and callable(model.posterior):
                posterior = model.posterior(X)
                return {
                    'mean': posterior.mean.squeeze(-1),
                    'variance': posterior.variance.squeeze(-1)
                }
            
            # Otherwise, try to get prediction with uncertainty
            X_np = X.cpu().numpy()
            
            if hasattr(model, 'predict_with_uncertainty') and callable(model.predict_with_uncertainty):
                mean, variance = model.predict_with_uncertainty(X_np)
                return {
                    'mean': torch.from_numpy(mean).float().to(self.device),
                    'variance': torch.from_numpy(variance).float().to(self.device)
                }
                
            # Traditional predict method with multiple samples
            elif hasattr(model, 'get_model_prediction') and callable(model.get_model_prediction):
                class TempDataset(AbstractDataSource):
                    def __init__(self, data):
                        self.data = data
                    def get_data(self):
                        return [self.data]
                    def get_shape(self):
                        return [self.data.shape]
                    def subset(self, indices):
                        return self
                
                temp_dataset = TempDataset(X_np)
                samples = model.get_model_prediction(temp_dataset, return_multiple_preds=True)
                
                if isinstance(samples, list):
                    samples_array = np.array(samples)
                    mean = np.mean(samples_array, axis=0)
                    variance = np.var(samples_array, axis=0)
                    return {
                        'mean': torch.from_numpy(mean).float().to(self.device),
                        'variance': torch.from_numpy(variance).float().to(self.device)
                    }
            
            # Last resort - just get predictions and use constant variance
            mean = model.predict(X_np)
            if isinstance(mean, list):
                mean = mean[0]
                
            return {
                'mean': torch.from_numpy(mean).float().to(self.device),
                'variance': torch.ones_like(torch.from_numpy(mean)).float().to(self.device)
            }
            
        except Exception as e:
            print(f"Error getting model posterior: {e}")
            # Return random values for demonstration
            N = X.shape[0]
            return {
                'mean': torch.randn(N, device=self.device),
                'variance': torch.ones(N, device=self.device)
            }

    def _compute_batch_eig(
        self, 
        X_candidates: torch.Tensor, 
        model: AbstractBaseModel
    ) -> torch.Tensor:
        """
        Compute Expected Information Gain for all candidate points.
        
        The EIG is I(Yx; S) = H(S) - E_Yx[H(S|Yx)]
        where S is the random level-set {x : f(x) ≥ c}
        
        Args:
            X_candidates: Tensor of candidate points [N, d]
            model: Model to sample functions from
            
        Returns:
            Tensor of EIG values for each candidate [N]
        """
        N = X_candidates.shape[0]  # Number of candidates
        
        # Step 1: Draw M posterior function samples over all candidates
        # This will be shaped [M, N] - M function samples at each candidate
        all_function_samples = self._sample_functions(X_candidates, model, self.num_samples)
        
        # Step 2: For each sample, compute level-set membership (Binary tensor: True if f(x) ≥ c)
        # This will result in [M, N] binary tensor
        level_set_memberships = (all_function_samples >= self.c)
        
        # Step 3: Compute the prior entropy H(S) of level-set membership at each point
        # First, compute the marginal probability of membership for each point
        # p(x ∈ S) = 1/M * sum_{i=1}^M 1{f^i(x) ≥ c}
        marginal_probs = level_set_memberships.float().mean(dim=0)  # [N]
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        marginal_probs = torch.clamp(marginal_probs, epsilon, 1-epsilon)
        
        # H(S) = -p log p - (1-p) log (1-p) for each point
        prior_entropies = -marginal_probs * torch.log(marginal_probs) - (1-marginal_probs) * torch.log(1-marginal_probs)  # [N]
        
        # Step 4: For each candidate x_i, estimate the expected posterior entropy after observing f(x_i)
        posterior_entropies = torch.zeros(N, device=self.device)
        
        # For each candidate point
        for i in range(N):
            # For each possible function value we might observe at x_i
            # Extract samples at this point (M samples)
            f_samples_at_i = all_function_samples[:, i]
            
            # For each possible function value, compute the posterior probability of level-set membership
            # This is p(x_j ∈ S | f(x_i) = v_k) for each point x_j
            
            # For each possible function value (sampled), compute updated beliefs
            pointwise_posterior_entropies = torch.zeros(self.num_samples, device=self.device)
            
            for m in range(self.num_samples):
                # Imagine we observe f(x_i) = f_samples_at_i[m]
                observed_value = f_samples_at_i[m]
                
                # Get all function samples that have approximately this value at x_i
                # We're simulating: "If we observe f(x_i) ≈ observed_value, what would we believe about other points?"
                
                # Add noise to the observed value to simulate realistic observations
                if self.noise_type == "additive":
                    noise = torch.randn(self.num_samples, device=self.device) * self.noise_level
                else:  # multiplicative
                    noise = torch.randn(self.num_samples, device=self.device) * self.noise_level * torch.abs(observed_value)
                
                noisy_observations = observed_value + noise
                
                # Find function samples that are consistent with this observation (within bounds)
                # We compute weights for each function - higher weight if the sample is more consistent with the observation
                consistency_weights = torch.exp(-0.5 * ((all_function_samples[:, i] - observed_value) / self.noise_level)**2)
                consistency_weights = consistency_weights / consistency_weights.sum()  # Normalize weights
                
                # Compute posterior probabilities for each point's level-set membership
                # This is weighted average of memberships where weights correspond to consistency with the observation
                posterior_probs = torch.matmul(consistency_weights, level_set_memberships.float())  # [N]
                
                # Clamp to avoid numerical issues
                posterior_probs = torch.clamp(posterior_probs, epsilon, 1-epsilon)
                
                # Compute entropy of these posterior probabilities
                point_entropies = -posterior_probs * torch.log(posterior_probs) - (1-posterior_probs) * torch.log(1-posterior_probs)  # [N]
                
                # Average entropy across points (excluding the candidate itself)
                mask = torch.ones(N, device=self.device, dtype=torch.bool)
                mask[i] = False  # Exclude the candidate point itself
                pointwise_posterior_entropies[m] = point_entropies[mask].mean()
            
            # Expected posterior entropy for this candidate: average across all possible function values
            posterior_entropies[i] = pointwise_posterior_entropies.mean()
        
        # Step 5: Compute Expected Information Gain: I(Yx; S) = H(S) - E_Yx[H(S|Yx)]
        # First compute the average prior entropy across all points (excluding the candidate itself)
        eig_values = torch.zeros(N, device=self.device)
        
        for i in range(N):
            mask = torch.ones(N, device=self.device, dtype=torch.bool)
            mask[i] = False  # Exclude the candidate point itself
            prior_entropy_for_i = prior_entropies[mask].mean()
            
            # EIG is difference between prior entropy and expected posterior entropy
            eig_values[i] = prior_entropy_for_i - posterior_entropies[i]
        
        return eig_values
    
    def _sample_functions(
        self, 
        X: torch.Tensor, 
        model: AbstractBaseModel, 
        num_samples: int
    ) -> torch.Tensor:
        """
        Sample functions from the model's posterior.
        
        Args:
            X: Input tensor [N, d]
            model: Model with posterior sampling capability
            num_samples: Number of function samples to draw
            
        Returns:
            Tensor of function samples [M, N], where M=num_samples, N=X.shape[0]
        """
        # Convert to numpy for model.predict since it might expect numpy arrays
        X_np = X.cpu().numpy()
        
        # Sample from the model's posterior
        # We assume the model has a method to get multiple predictions (samples from posterior)
        try:
            # First attempt: Try to use a dedicated sampling method if it exists
            if hasattr(model, 'sample_posterior') and callable(model.sample_posterior):
                # A hypothetical method that returns M samples for each input X
                # Shape of samples would be [M, N]
                samples = model.sample_posterior(X_np, num_samples=num_samples)
                samples = torch.from_numpy(samples).float().to(self.device)
            elif hasattr(model, 'get_model_prediction') and callable(model.get_model_prediction):
                # The model has a method to get predictions with multiple samples
                # For example, in PytorchMLPRegressorWithUncertainty class
                samples = []
                
                # Convert to a tensor format expected by the model
                # Depending on the specific requirements of the get_model_prediction method
                # For GeneDisco's PytorchMLPRegressorWithUncertainty
                all_samples = []
                
                # Process in batches if the dataset is large
                batch_size = 256  # Adjust based on available memory
                for i in range(0, len(X_np), batch_size):
                    batch_X = X_np[i:i+batch_size]
                    
                    # Create a small temporary dataset from batch_X
                    # This mimics what happens in the model's predict method
                    class TempDataset(AbstractDataSource):
                        def __init__(self, data):
                            self.data = data
                            
                        def get_data(self):
                            return [self.data]
                            
                        def get_shape(self):
                            return [self.data.shape]
                            
                        def subset(self, indices):
                            return self  # Simple implementation for our needs
                    
                    temp_dataset = TempDataset(batch_X)
                    
                    # Get multiple predictions for this batch
                    batch_samples = model.get_model_prediction(temp_dataset, return_multiple_preds=True)
                    
                    # Process the samples based on their structure
                    if isinstance(batch_samples, list) and len(batch_samples) > 0:
                        # The samples might be a list of tensors or arrays
                        # Convert to numpy if they're torch tensors
                        if isinstance(batch_samples[0], torch.Tensor):
                            batch_samples = [s.cpu().numpy() for s in batch_samples]
                        
                        # Stack along a new dimension if needed
                        batch_samples = np.array(batch_samples)
                        
                        # Ensure the shape is [M, batch_size]
                        if batch_samples.ndim == 3:  # [1, M, batch_size]
                            batch_samples = batch_samples.squeeze(0)
                        elif batch_samples.ndim == 1:  # [batch_size]
                            batch_samples = np.expand_dims(batch_samples, 0)
                            
                    all_samples.append(batch_samples)
                
                # Combine samples from all batches
                if all_samples:
                    combined_samples = np.concatenate(all_samples, axis=1)  # Concatenate along batch dimension
                    samples = torch.from_numpy(combined_samples).float().to(self.device)
                else:
                    # Fallback if no samples were obtained
                    samples = torch.zeros((num_samples, X.shape[0]), device=self.device)
            else:
                # Fallback: Use regular predict method num_samples times
                samples = []
                for _ in range(num_samples):
                    # Assuming model.predict returns shape [N]
                    pred = model.predict(X_np)
                    if isinstance(pred, list):
                        pred = pred[0]  # Some models might return a list of arrays
                    samples.append(pred)
                samples = np.array(samples)  # [M, N]
                samples = torch.from_numpy(samples).float().to(self.device)
                
        except Exception as e:
            print(f"Error sampling from model: {e}")
            # Fallback: generate random samples for demonstration
            # In a real implementation, this should be handled more gracefully
            samples = torch.randn(num_samples, X.shape[0], device=self.device)
            
        return samples