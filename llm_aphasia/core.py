import torch
import torch.nn as nn
from typing import List, Dict
from .strategies import apply_tensor_ablation

def get_target_modules(model: nn.Module) -> List[nn.Module]:
    """
    Identifies target linear layers within the model's transformer blocks.
    This is designed to work with various transformer architectures.
    """
    target_modules = []
    
    # Common patterns for transformer layer names across different models
    transformer_patterns = [
        'transformer.h.',  # GPT-2 style
        'model.layers.',   # Llama style
        'transformer.blocks.',  # Some other models
        'encoder.layer.',  # BERT style
        'decoder.layers.',  # Some decoder models
    ]
    
    # Common component patterns
    component_patterns = ['self_attn', 'mlp', 'attention', 'feed_forward', 'fc', 'dense']
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this linear layer is part of a transformer block
            is_transformer_layer = any(pattern in name for pattern in transformer_patterns)
            has_component = any(comp in name.lower() for comp in component_patterns)
            
            # For GPT-2 style models, also include c_attn, c_proj, c_fc
            is_gpt2_style = any(x in name for x in ['c_attn', 'c_proj', 'c_fc'])
            
            if (is_transformer_layer and has_component) or is_gpt2_style:
                target_modules.append(module)
    
    # If no modules found with strict criteria, be more lenient
    if not target_modules:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'embed' not in name.lower() and 'head' not in name.lower():
                target_modules.append(module)
    
    print(f"Found {len(target_modules)} target modules for ablation")
    return target_modules

def save_weights(target_modules: List[nn.Module]) -> Dict[nn.Module, torch.Tensor]:
    """Saves the original weights of target modules."""
    return {module: module.weight.data.clone() for module in target_modules}

def restore_weights(saved_weights: Dict[nn.Module, torch.Tensor]):
    """Restores the original weights to the modules."""
    with torch.no_grad():
        for module, original_weight in saved_weights.items():
            module.weight.data = original_weight.clone()

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
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    try:
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    except Exception as e:
        print(f"Generation failed with error: {e}")
        print("Trying with reduced settings...")
        # Fallback with more conservative settings
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_new_tokens=min(max_new_tokens, 20),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
