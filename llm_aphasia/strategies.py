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
    if tensor.dim() < 2: return tensor
    
    ablated_tensor = tensor.clone()
    n_indices = tensor.shape[dim]
    n_to_shuffle = int(n_indices * severity)
    if n_to_shuffle < 2: return ablated_tensor

    # Select random indices to shuffle
    indices_to_shuffle = torch.randperm(n_indices, generator=generator, device=tensor.device)[:n_to_shuffle]
    
    # Create a permutation of these indices
    shuffled_order = indices_to_shuffle[torch.randperm(n_to_shuffle, generator=generator, device=tensor.device)]
    
    # Apply the shuffle by copying rows/columns
    if dim == 0:  # Shuffle rows
        ablated_tensor[indices_to_shuffle] = tensor[shuffled_order]
    else:  # Shuffle columns
        ablated_tensor[:, indices_to_shuffle] = tensor[:, shuffled_order]
    
    return ablated_tensor

def _shuffle_y(tensor: torch.Tensor, severity: float, generator: torch.Generator) -> torch.Tensor:
    """Shuffles a fraction of rows (output dimension)."""
    return _shuffle_dim(tensor, 0, severity, generator)

def _shuffle_x(tensor: torch.Tensor, severity: float, generator: torch.Generator) -> torch.Tensor:
    """Shuffles a fraction of columns (input dimension)."""
    return _shuffle_dim(tensor, 1, severity, generator)

def _swap_dim(tensor: torch.Tensor, dim: int, severity: float, generator: torch.Generator) -> torch.Tensor:
    """Swaps pairs of indices along a given dimension, affecting a fraction of them."""
    if tensor.dim() < 2: return tensor
    
    ablated_tensor = tensor.clone()
    n_indices = tensor.shape[dim]
    n_to_swap = int(n_indices * severity)
    if n_to_swap < 2: return ablated_tensor
    
    # Ensure even number for pairing
    if n_to_swap % 2 != 0:
        n_to_swap -= 1
    
    # Select random indices to swap
    indices_to_swap = torch.randperm(n_indices, generator=generator, device=tensor.device)[:n_to_swap]
    
    # Reshape to pairs and swap
    pairs = indices_to_swap.view(-1, 2)
    
    if dim == 0:  # Swap rows
        # Store the first of each pair
        temp = ablated_tensor[pairs[:, 0]].clone()
        # Copy second to first
        ablated_tensor[pairs[:, 0]] = ablated_tensor[pairs[:, 1]]
        # Copy temp (original first) to second
        ablated_tensor[pairs[:, 1]] = temp
    else:  # Swap columns
        # Store the first of each pair
        temp = ablated_tensor[:, pairs[:, 0]].clone()
        # Copy second to first
        ablated_tensor[:, pairs[:, 0]] = ablated_tensor[:, pairs[:, 1]]
        # Copy temp (original first) to second
        ablated_tensor[:, pairs[:, 1]] = temp
    
    return ablated_tensor
    
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
    
    # Select random indices to shuffle
    indices_to_shuffle = torch.randperm(n_elements, generator=generator, device=tensor.device)[:n_to_shuffle]
    
    # Get values at these indices and shuffle them
    values_to_shuffle = flat_tensor[indices_to_shuffle].clone()
    shuffled_order = torch.randperm(n_to_shuffle, generator=generator, device=tensor.device)
    shuffled_values = values_to_shuffle[shuffled_order]
    
    # Create ablated tensor and apply shuffled values
    ablated_tensor = flat_tensor.clone()
    ablated_tensor[indices_to_shuffle] = shuffled_values
    
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
