import torch

class SubsetSelect:
    """
    Helper to perform greedy subset selection for a DiscoBAXAdditive-like objective.
    """
    def __init__(self, bax_acquisition, subset_size: int, num_noise_samples: int = 100):
        """
        Args:
            bax_acquisition: Object with .sample_function() & .sample_noise() 
                             that can operate on GPU Tensors.
            subset_size: k (# points to select).
            num_noise_samples: # of Monte Carlo draws for noise.
        """
        self.bax_acq = bax_acquisition
        self.k = subset_size
        self.num_noise = num_noise_samples

    def select_subset(self, candidate_X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            candidate_X: [N, d] float Tensor on GPU (candidate features).

        Returns:
            A [k] int Tensor of the chosen subset indices.
        """
        device = candidate_X.device
        N = candidate_X.shape[0]

        # 1) Sample function values for all candidates
        f_sample = self.bax_acq.sample_function(candidate_X)  # shape [N]
        f_vals = torch.clamp(f_sample, min=0.0)  # negative -> 0

        # 2) Sample noise for each candidate & each of the num_noise draws
        noise_samples = self.bax_acq.sample_noise((self.num_noise, N)).to(device)  
        outcomes = torch.clamp(f_vals.unsqueeze(0) + noise_samples, min=0.0)  # [num_noise, N]

        # 3) Greedy selection
        best_outcome_per_draw = torch.zeros(self.num_noise, device=device)
        selected_indices = []

        for _ in range(self.k):
            new_max_all = torch.max(best_outcome_per_draw.unsqueeze(1), outcomes)
            mask = torch.ones(N, dtype=torch.bool, device=device)
            if selected_indices:
                mask[torch.tensor(selected_indices, device=device)] = False

            mean_values = new_max_all.mean(dim=0)  # shape [N]
            mean_values_selected = mean_values.masked_fill(~mask, float('-inf'))
            best_idx = torch.argmax(mean_values_selected).item()
            selected_indices.append(best_idx)

            # Update best_outcome_per_draw
            best_outcome_per_draw = torch.max(best_outcome_per_draw, outcomes[:, best_idx])

        return torch.tensor(selected_indices, device=device)
