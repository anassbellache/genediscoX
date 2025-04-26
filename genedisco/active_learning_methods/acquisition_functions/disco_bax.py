import numpy as np
from typing import AnyStr, List
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import BaseBatchAcquisitionFunction
import torch
import copy  # optional if you want to move a model clone to GPU
from genedisco.algorithms.subset_select import SubsetSelect

class DiscoBAXBatchAcquisitionFunction(BaseBatchAcquisitionFunction):
    """
    Example GPU-accelerated DiscoBAX acquisition for a CPU-based GeneDisco pipeline.

    This minimal illustration does:
      - Convert candidate features to GPU
      - Sample from model posterior (CPU or GPU approach, see below)
      - Perform submodular selection on GPU
      - Return top `batch_size` sample IDs on CPU
    """
    def __init__(self, 
                 subset_size: int = 10,
                 num_samples: int = 1,
                 num_noise_samples: int = 100,
                 noise_type: str = "additive"):
        """
        Args:
            subset_size: The number of points to pick in each batch (k).
            num_samples: # posterior function samples to approximate EIG (demo uses 1).
            num_noise_samples: # noise draws for each subset selection step.
            noise_type: 'additive' or 'multiplicative' (demo only shows 'additive').
        """
        self.subset_size = subset_size
        self.num_samples = num_samples
        self.num_noise_samples = num_noise_samples
        self.noise_type = noise_type

    def __call__(self,
                 dataset_x,
                 batch_size: int,
                 available_indices: List[AnyStr],
                 last_selected_indices: List[AnyStr],
                 last_model):
        """
        Select `batch_size` items via a BAX submodular approach.

        Args:
            dataset_x: AbstractDataSource holding all data
            batch_size: number of items to choose
            available_indices: list of item IDs we can select from
            last_selected_indices: the items chosen in the previous round (not used here, but required)
            last_model: a CPU-based model that we can call for posterior sampling

        Returns:
            List of selected item IDs (length = batch_size)
        """
        # If fewer available than requested, pick them all
        if len(available_indices) <= batch_size:
            return available_indices

        # 1) Subset the data to get candidate features on CPU (NumPy).
        candidate_data = dataset_x.subset(available_indices)  # typical approach in GeneDisco
        # Suppose candidate_data has a method .get_all_features() or we gather them manually
        candidate_features_np = candidate_data.get_all_features()  # shape [N, d] as NumPy
        N = candidate_features_np.shape[0]

        # 2) Choose device (GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 3) Convert candidate features to a GPU tensor
        candidate_features = torch.from_numpy(candidate_features_np).float().to(device)

        # 4) Build a small wrapper for submodular selection that can sample from model
        class _TempBAXAcqWrapper:
            def __init__(self, model, device):
                self.model = model
                self.device = device
                # We assume model has .likelihood.noise_covar.noise for the noise variance
                noise_var = self.model.likelihood.noise_covar.noise.detach()
                self.noise_dist = torch.distributions.Normal(
                    loc=torch.tensor(0.0, device=device),
                    scale=noise_var.sqrt().to(device)
                )

            def sample_function(self, X: torch.Tensor) -> torch.Tensor:
                """
                Return a single function sample of shape [N].
                
                Currently we do it by:
                  - CPU posterior sample
                  - move result to GPU
                Or we can do a partial approach if the model can do GPU inference.
                """
                # Option A: CPU sampling, then move to GPU
                X_cpu = X.cpu().numpy()  # shape [N, d]
                # We call a hypothetical 'model.posterior_predict(X_cpu)' 
                # or 'model.predict(...)' returning a single sample.
                # For demonstration, let's assume model has a .sample_posterior_once(...) method 
                # that returns shape [N] as a NumPy array. 
                # If your model doesn't, you'd adapt accordingly.

                f_sample_np = self.model.sample_posterior_once(X_cpu)  
                # This is purely conceptual. Replace with your actual method for sampling.
                
                # Move that sample to GPU
                f_samp = torch.from_numpy(f_sample_np).float().to(self.device)
                return f_samp

            def sample_noise(self, shape: torch.Size) -> torch.Tensor:
                """
                Sample noise from Normal(0, sigma^2).
                """
                return self.noise_dist.sample(shape)

        # 5) SubsetSelect to pick exactly 'batch_size' points
        bax_wrapper = _TempBAXAcqWrapper(last_model, device)
        subset_selector = SubsetSelect(bax_acquisition=bax_wrapper, 
                                       subset_size=batch_size, 
                                       num_noise_samples=self.num_noise_samples)

        # If we want multiple posterior draws (num_samples > 1), 
        # we might combine multiple runs. For simplicity, let's do one draw:
        #   selected_indices = subset_selector.select_subset(candidate_features)
        # However, the full DiscoBAX approach might combine sets from multiple draws. 
        # We'll keep it simple here:

        with torch.no_grad():
            selected_indices_gpu = subset_selector.select_subset(candidate_features)
        # [batch_size] on GPU

        # 6) Convert the chosen indices (integers) to CPU, then map to the original IDs
        selected_indices_cpu = selected_indices_gpu.cpu().numpy().tolist()  
        selected_ids = [available_indices[i] for i in selected_indices_cpu]

        return selected_ids
