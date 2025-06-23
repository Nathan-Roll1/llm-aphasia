# LLM Aphasia

LLM Aphasia is a research toolkit for applying structured ablations to Hugging Face language models. It allows you to simulate neurological deficits or "lesion" specific aspects of a model's weights to study their impact on behavior and better understand model internals.

The library is designed for efficiency and ease of use, featuring both a high-level `ablate()` function for global ablations and a `targeted_ablate()` function for precise layer and component targeting.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

-   **Efficient Model Caching**: Models are loaded into memory once and reused across calls.
-   **Stateful Ablations**: The library tracks the current state of cached models, only applying ablations when necessary to avoid redundant computation.
-   **Diverse Ablation Strategies**: Implement various methods of "damaging" weight matrices, from zeroing out parameters to shuffling entire rows/columns.
-   **Targeted Ablations**: Apply ablations to specific layers and components (attention heads, MLP layers, etc.)
-   **Simple High-Level API**: Two main functions, `ablate()` and `targeted_ablate()`, serve as entry points.
-   **Reproducibility**: Use a `random_seed` to ensure that stochastic ablations are repeatable.

## Installation

To get started, clone the repository and install it in editable mode:

```bash
git clone https://github.com/NathanRoll1/llm-aphasia.git
cd llm-aphasia
pip install -e .
```

This will install the package and its dependencies (`torch`, `transformers`, `accelerate`).

## How to Use

### Basic Usage: Global Ablations

The `ablate()` function applies ablations globally to all transformer layers:

```python
from llm_aphasia import ablate

model_path = "gpt2"  # or any HuggingFace model
prompt = "The first person to walk on the moon was"

# 1. Get baseline output
unablated = ablate(
    model_path=model_path,
    ablation_strategy="unablated",
    severity=0.0,
    text_input=prompt
)
print(f"Baseline: {unablated}\n")

# 2. Apply 50% zero-out ablation
ablated = ablate(
    model_path=model_path,
    ablation_strategy="zero_out",
    severity=0.50,
    text_input=prompt
)
print(f"Ablated: {ablated}\n")
```

### Advanced Usage: Targeted Ablations

The `targeted_ablate()` function allows precise control over which layers and components to ablate:

```python
from llm_aphasia import targeted_ablate

# Ablate only attention components in layers 0-5
output = targeted_ablate(
    model_path="gpt2",
    ablation_strategy="zero_out",
    severity=0.5,
    text_input="The capital of France is",
    target_layers=[0, 1, 2, 3, 4, 5],
    target_components=['q', 'k', 'v', 'o']  # Only attention
)

# Ablate only MLP layers in the last 3 layers
output = targeted_ablate(
    model_path="gpt2",
    ablation_strategy="shuffle_y",
    severity=0.7,
    text_input="2 + 2 equals",
    target_layers=[-3, -2, -1],  # Last 3 layers
    target_components=['mlp']     # Only MLP/FFN
)

# Surgical ablation: only Key projections in middle layers
output = targeted_ablate(
    model_path="gpt2",
    ablation_strategy="zero_out",
    severity=0.9,
    text_input="Shakespeare wrote",
    target_layers=[5, 6, 7, 8],
    target_components=['k']  # Only key projections
)
```

### Convenience Functions

```python
from llm_aphasia.targeted import ablate_attention_only, ablate_mlp_only, ablate_layer_range

# Ablate only attention across all layers
output = ablate_attention_only("gpt2", "Hello world", severity=0.5)

# Ablate only MLPs in specific layers
output = ablate_mlp_only("gpt2", "Hello world", severity=0.7, target_layers=[0, 1, 2])

# Ablate layers 5 through 10
output = ablate_layer_range("gpt2", "Hello world", start_layer=5, end_layer=10)
```

## Ablation Strategies

| Strategy | Description |
|---|---|
| `unablated` | **No-op.** The baseline model without any changes. |
| `zero_out` | Randomly sets a fraction of weight parameters to zero. |
| `mean_out` | Randomly replaces a fraction of weights with the mean of the entire weight matrix. |
| `shuffle_x` | Randomly shuffles a fraction of the **columns** of weight matrices. This disrupts input feature mapping. |
| `shuffle_y` | Randomly shuffles a fraction of the **rows** of weight matrices. This disrupts output feature mapping. |
| `swap_x` | Swaps random pairs of **columns** in weight matrices. |
| `swap_y` | Swaps random pairs of **rows** in weight matrices. |
| `global` | Flattens weight matrices and shuffles a fraction of their elements globally, disrupting all spatial structure. |

## Target Components

When using `targeted_ablate()`, you can specify which components to ablate:

### Attention Components
- `q` - Query projection
- `k` - Key projection
- `v` - Value projection
- `o` - Output projection

### MLP/FFN Components
- `mlp` - All MLP layers (automatic handling for different architectures)
- For Llama models specifically: `gate_proj`, `up_proj`, `down_proj`
- For GPT models specifically: `mlp_in`, `mlp_out`

## Examples

### Example 1: Compare Different Ablation Strategies

```python
from llm_aphasia import ablate

model_path = "gpt2"
prompt = "The theory of relativity was developed by"
strategies = ['zero_out', 'mean_out', 'shuffle_y', 'global']

print(f"Baseline: {ablate(model_path, 'unablated', 0, prompt)}\n")

for strategy in strategies:
    output = ablate(model_path, strategy, 0.5, prompt)
    print(f"{strategy}: {output}")
```

### Example 2: Layer Importance Analysis

```python
from llm_aphasia import targeted_ablate

model_path = "gpt2"
prompt = "Paris is the capital of"

# Test each layer individually
for layer in range(12):  # GPT-2 has 12 layers
    output = targeted_ablate(
        model_path=model_path,
        ablation_strategy="zero_out",
        severity=0.75,
        text_input=prompt,
        target_layers=layer
    )
    print(f"Layer {layer}: {output}")
```

### Example 3: Component Importance Study

```python
from llm_aphasia import targeted_ablate

model_path = "gpt2"
prompt = "The sun is a"

components_to_test = [
    (['q'], "Query only"),
    (['k'], "Key only"),
    (['v'], "Value only"),
    (['q', 'k', 'v'], "All attention inputs"),
    (['o'], "Output projection"),
    (['mlp'], "MLP only"),
]

for components, description in components_to_test:
    output = targeted_ablate(
        model_path=model_path,
        ablation_strategy="zero_out",
        severity=0.8,
        text_input=prompt,
        target_components=components,
        target_layers=[5, 6, 7]  # Middle layers
    )
    print(f"{description}: {output}")
```

### Example 4: Progressive Ablation

```python
from llm_aphasia import ablate

model_path = "gpt2"
prompt = "Machine learning is"

for severity in [0.0, 0.25, 0.50, 0.75, 1.0]:
    output = ablate(model_path, "zero_out", severity, prompt)
    print(f"{int(severity*100)}% ablation: {output}")
```

## API Reference

### `ablate()`

```python
ablate(
    model_path: str,
    ablation_strategy: str,
    severity: float,
    text_input: str,
    random_seed: int = 42,
    max_new_tokens: int = 50,
) -> str
```

Apply global ablations to all transformer layers.

**Parameters:**
- `model_path`: HuggingFace model name or path
- `ablation_strategy`: One of the strategies listed above
- `severity`: Ablation strength from 0.0 (none) to 1.0 (complete)
- `text_input`: Input prompt
- `random_seed`: Seed for reproducible randomness
- `max_new_tokens`: Maximum tokens to generate

### `targeted_ablate()`

```python
targeted_ablate(
    model_path: str,
    ablation_strategy: str,
    severity: float,
    text_input: str,
    target_layers: Optional[Union[int, List[int]]] = None,
    target_components: Optional[List[str]] = None,
    random_seed: int = 42,
    max_new_tokens: int = 50,
) -> str
```

Apply ablations to specific layers and components.

**Parameters:**
- All parameters from `ablate()`, plus:
- `target_layers`: None (all), single int, or list of layer indices
- `target_components`: None (all) or list of component names

## Research Applications

This toolkit is useful for:
- Understanding which model components are critical for different tasks
- Studying how information flows through transformer layers
- Creating "lesion studies" similar to neuroscience research
- Analyzing model robustness and failure modes
- Investigating the role of attention vs. feed-forward layers

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
