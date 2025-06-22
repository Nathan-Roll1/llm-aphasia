import torch
import torch.nn as nn
from typing import List, Dict
from .strategies import apply_tensor_ablation

def get_target_modules(model: nn.Module) -> List[nn.Module]:
    """
    Identifies target linear layers within the model's transformer blocks.
    This is designed for Llama/Mistral-style architectures but is broadly applicable.
    """
    target_modules = []
    for name, module in model.named_modules():
        is_transformer_layer = any(key in name for key in ['self_attn', 'mlp'])
        if isinstance(module, nn.Linear) and is_transformer_layer:
            target_modules.append(module)
    return target_modules

def save_weights(target_modules: List[nn.Module]) -> Dict[nn.Module, torch.Tensor]:
    """Saves the original weights of target modules."""
    return {module: module.weight.data.clone() for module in target_modules}

def restore_weights(saved_weights: Dict[nn.Module, torch.Tensor]):
    """Restores the original weights to the modules."""
    with torch.no_grad():
        for module, original_weight in saved_weights.items():
            module.weight.data = original_weight

def apply_ablation_to_model(
    target_modules: List[nn.Module],
    strategy: str,
    severity: float,
    generator: torch.Generator,
):
    """Applies a given ablation strategy to all target modules in the model."""
    with torch.no_grad():
        for module in target_modules:
            ablated_weight = apply_tensor_ablation(
                tensor=module.weight.data,
                strategy_name=strategy,
                severity=severity,
                generator=generator,
            )
            module.weight.data = ablated_weight

@torch.no_grad()
def generate_text(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
) -> str:
    """Generates text from a prompt using the provided model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)