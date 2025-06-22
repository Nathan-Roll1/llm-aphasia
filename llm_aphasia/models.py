import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Tuple

# Global cache for loaded models and tokenizers
_MODEL_CACHE: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}

def get_model_and_tokenizer(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads a model and tokenizer from Hugging Face, caching them for future use.

    Args:
        model_path (str): The path or name of the Hugging Face model.

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]

    print(f"Loading model and tokenizer for '{model_path}'... This may take a moment.")
    device_map = "auto"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    # Some models like Llama-3 require a specific pad token setup for open-ended generation
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    _MODEL_CACHE[model_path] = (model, tokenizer)
    return model, tokenizer