import torch
from typing import Callable, Dict

def _zero_out(tensor: torch.Tensor, severity: float, generator: torch.Generator) -> torch.Tensor:
    """Sets a fraction of tensor elements to zero."""
    mask = torch.rand(tensor.shape, generator=generator, device=tensor.device) > severity
    return tensor * mask

def _mean_out(tensor: torch.Tensor, severity: float, generator: torch.Generator) -> torch.Tensor:
    """Replaces a fraction of tensor elements with the tensor's mean."""
    ablated_tensor = tensor.clone()
    mask = torch.rand(tensor.shape, generator=generator, device=tensor.device) < severity
    ablated_tensor[mask] = tensor.mean()
    return ablated_tensor

def _shuffle_dim(tensor: torch.Tensor, dim: int, severity: float, generator: torch.Generator) -> torch.Tensor:
    """Shuffles a fraction of indices along a given dimension."""
    if tensor.dim() < 2: return tensor # Cannot shuffle 1D tensors meaningfully here
    
    n_indices = tensor.shape[dim]
    n_to_shuffle = int(n_indices * severity)
    if n_to_shuffle < 2: return tensor

    indices_to_shuffle = torch.randperm(n_indices, generator=generator, device=tensor.device)[:n_to_shuffle]
    shuffled_indices = indices_to_shuffle[torch.randperm(n_to_shuffle, generator=generator, device=tensor.device)]
    
    return torch.index_select(tensor, dim, shuffled_indices)

def _shuffle_y(tensor: torch.Tensor, severity: float, generator: torch.Generator) -> torch.Tensor:
    """Shuffles a fraction of rows (output dimension)."""
    return _shuffle_dim(tensor, 0, severity, generator)

def _shuffle_x(tensor: torch.Tensor, severity: float, generator: torch.Generator) -> torch.Tensor:
    """Shuffles a fraction of columns (input dimension)."""
    return _shuffle_dim(tensor, 1, severity, generator)

def _swap_dim(tensor: torch.Tensor, dim: int, severity: float, generator: torch.Generator) -> torch.Tensor:
    """Swaps pairs of indices along a given dimension, affecting a fraction of them."""
    if tensor.dim() < 2: return tensor
    
    n_indices = tensor.shape[dim]
    n_to_swap = int(n_indices * severity)
    if n_to_swap < 2: return tensor
    
    # Ensure even number for pairing
    if n_to_swap % 2 != 0:
        n_to_swap -=1

    indices_to_swap = torch.randperm(n_indices, generator=generator, device=tensor.device)[:n_to_swap]
    swapped_indices = torch.arange(n_indices, device=tensor.device)
    
    # Reshape to (pairs, 2) and swap columns
    pairs = indices_to_swap.view(-1, 2)
    swapped_pairs = torch.flip(pairs, dims=[1])
    
    # Apply the swaps
    swapped_indices[pairs.flatten()] = swapped_indices[swapped_pairs.flatten()]
    
    return torch.index_select(tensor, dim, swapped_indices)
    
def _swap_y(tensor: torch.Tensor, severity: float, generator: torch.Generator) -> torch.Tensor:
    """Swaps pairs of rows (output dimension)."""
    return _swap_dim(tensor, 0, severity, generator)

def _swap_x(tensor: torch.Tensor, severity: float, generator: torch.Generator) -> torch.Tensor:
    """Swaps pairs of columns (input dimension)."""
    return _swap_dim(tensor, 1, severity, generator)

def _global_shuffle(tensor: torch.Tensor, severity: float, generator: torch.Generator) -> torch.Tensor:
    """Flattens the tensor, shuffles a fraction of its elements, and reshapes."""
    original_shape = tensor.shape
    flat_tensor = tensor.flatten()
    
    n_elements = flat_tensor.numel()
    n_to_shuffle = int(n_elements * severity)
    if n_to_shuffle < 2: return tensor
    
    indices_to_shuffle = torch.randperm(n_elements, generator=generator, device=tensor.device)[:n_to_shuffle]
    shuffled_subset = flat_tensor[indices_to_shuffle]
    shuffled_subset = shuffled_subset[torch.randperm(n_to_shuffle, generator=generator, device=tensor.device)]
    
    ablated_tensor = flat_tensor.clone()
    ablated_tensor[indices_to_shuffle] = shuffled_subset
    
    return ablated_tensor.view(original_shape)


# Strategy registry
STRATEGIES: Dict[str, Callable] = {
    "zero_out": _zero_out,
    "mean_out": _mean_out,
    "shuffle_x": _shuffle_x,
    "shuffle_y": _shuffle_y,
    "swap_x": _swap_x,
    "swap_y": _swap_y,
    "global": _global_shuffle,
}

def apply_tensor_ablation(
    tensor: torch.Tensor,
    strategy_name: str,
    severity: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Applies a specified ablation strategy to a tensor.
    """
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown ablation strategy: '{strategy_name}'. Available: {list(STRATEGIES.keys())}")
    
    if severity == 0.0:
        return tensor
        
    strategy_func = STRATEGIES[strategy_name]
    return strategy_func(tensor, severity, generator)