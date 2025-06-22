
# LLM Aphasia

LLM Aphasia is a research toolkit for applying structured ablations to Hugging Face language models. It allows you to simulate neurological deficits or "lesion" specific aspects of a model's weights to study their impact on behavior and better understand model internals.

The library is designed for efficiency and ease of use, featuring a high-level `ablate()` function that handles model caching, in-place weight manipulation, and state management automatically.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

-   **Efficient Model Caching**: Models are loaded into memory once and reused across calls.
-   **Stateful Ablations**: The library tracks the current state of cached models, only applying ablations when necessary to avoid redundant computation.
-   **Diverse Ablation Strategies**: Implement various methods of "damaging" weight matrices, from zeroing out parameters to shuffling entire rows/columns.
-   **Simple High-Level API**: A single function, `ablate()`, serves as the main entry point for all operations.
-   **Reproducibility**: Use a `random_seed` to ensure that stochastic ablations are repeatable.

## Installation

To get started, clone the repository and install it in editable mode:

```bash
git clone https://github.com/your-username/llm-aphasia.git
cd llm-aphasia
pip install -e .
```

This will install the package and its dependencies (`torch`, `transformers`, `accelerate`).

## How to Use

The main function is `llm_aphasia.ablate()`. You can import and use it directly in your Python scripts.

### Basic Example

Here's how to compare the output of a regular model with an ablated version.

```python
from llm_aphasia import ablate

# Use a smaller model for quick tests, or a larger one for more complex research.
# model_path = "gpt2"
model_path = "meta-llama/Meta-Llama-3-8B-Instruct" # Note: Requires gated access and ~16GB VRAM

# You may need to log in to Hugging Face for gated models
# from huggingface_hub import login
# login()

prompt = "The first person to walk on the moon was"

# 1. Get the baseline output with no ablation
unablated_output = ablate(
    model_path=model_path,
    ablation_strategy="unablated",
    severity=0.0,
    text_input=prompt
)
print(f"✅ Unablated Output:\n{unablated_output}\n")

# 2. Apply a 'zero_out' ablation with 25% severity
# The library will automatically apply the new lesion.
zero_out_output = ablate(
    model_path=model_path,
    ablation_strategy="zero_out",
    severity=0.25,
    text_input=prompt
)
print(f"🔬 Ablated (Zero-Out 25%) Output:\n{zero_out_output}\n")

# 3. Apply a different ablation: shuffle 50% of weight matrix rows
# The library detects the change, restores the original weights, and applies the new lesion.
shuffle_y_output = ablate(
    model_path=model_path,
    ablation_strategy="shuffle_y",
    severity=0.50,
    text_input=prompt
)
print(f"🔬 Ablated (Shuffle Y 50%) Output:\n{shuffle_y_output}\n")

```

## Ablation Strategies

The `ablation_strategy` parameter can be one of the following:

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

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
