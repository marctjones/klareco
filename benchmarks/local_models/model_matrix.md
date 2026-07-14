# Local Model Matrix for Apple Silicon M5

Last refreshed: 2026-06-22.

This matrix focuses on open-weight or source-available models that are realistic
to run locally on a Mac M5 class machine and can be exercised from native
libraries in Python, Rust, or .NET.

## Runtime choices

| Language | Preferred library | Model format | Why |
| --- | --- | --- | --- |
| Python | `mlx-lm` | MLX 4-bit | Fastest Apple Silicon path for text generation. |
| Python images | `diffusers` | safetensors | Broadest local image-generation ecosystem. |
| Rust | `llama_cpp` crate | GGUF | Native Rust API over llama.cpp with Metal support. |
| .NET | `LLamaSharp` | GGUF | Native .NET API over llama.cpp; current NuGet path is CPU-first on macOS. |

## Recommended models and alternatives

Alternatives intentionally come from different producers where possible. At
least one US/EU producer is included for each text use case.

| Purpose | Primary recommendation | Alternative A | Alternative B | US/EU producer coverage |
| --- | --- | --- | --- | --- |
| General chat | `Qwen/Qwen3-4B` or `mlx-community/Qwen3-4B-Instruct-2507-4bit` | `meta-llama/Llama-3.2-3B-Instruct` | `google/gemma-3-4b-it` | Meta and Google are US. |
| Fast instruct | `meta-llama/Llama-3.2-3B-Instruct` | `Qwen/Qwen3-4B` | `HuggingFaceTB/SmolLM3-3B` | Meta and Hugging Face are US/EU. |
| Reasoning | `Qwen/Qwen3-4B` | `google/gemma-3-4b-it` | `mistralai/Ministral-8B-Instruct-2410` | Google is US; Mistral is EU. |
| Tool use / JSON | `Qwen/Qwen3-4B` | `meta-llama/Llama-3.2-3B-Instruct` | `HuggingFaceTB/SmolLM3-3B` | Meta and Hugging Face are US/EU. |
| Agentic coding / repo work | `Qwen/Qwen3.6-35B-A3B` if RAM allows | `Qwen/Qwen3-4B` | `mistralai/Ministral-8B-Instruct-2410` | Mistral is EU. |
| Longer local context | `mistralai/Mistral-Nemo-Instruct-2407` | `Qwen/Qwen3-4B` | `google/gemma-3-12b-it` | Mistral is EU; Google is US. |
| Lawful adult creative text flexibility | `Qwen/Qwen3-4B` | `meta-llama/Llama-3.2-3B-Instruct` | `mistralai/Ministral-8B-Instruct-2410` | Meta is US; Mistral is EU. |
| Image generation speed | `black-forest-labs/FLUX.1-schnell` | `stabilityai/stable-diffusion-3.5-medium` | `Qwen/Qwen-Image` | Black Forest Labs and Stability AI are EU/UK. |
| Image generation quality | `black-forest-labs/FLUX.1-dev` | `Qwen/Qwen-Image` | `stabilityai/stable-diffusion-3.5-large` | Black Forest Labs and Stability AI are EU/UK. |
| Flexible adult image workflows | SDXL community checkpoints | Pony/Illustrious-style SDXL checkpoints | Stable Diffusion 3.5 variants | Stability AI is UK. |

## Practical download targets

For `mlx-lm`, prefer MLX conversions:

- `mlx-community/Qwen3-4B-Instruct-2507-4bit`
- `mlx-community/Llama-3.2-3B-Instruct-4bit`
- `mlx-community/SmolLM3-3B-4bit`

For Rust and .NET, prefer GGUF files:

- Qwen3 4B GGUF, Q4_K_M or Q5_K_M
- Qwen3.6 35B-A3B GGUF, Q4_K_M if RAM allows
- Llama 3.2 3B Instruct GGUF, Q4_K_M or Q5_K_M
- Gemma 3 4B IT GGUF, Q4_K_M
- Ministral 8B Instruct GGUF, Q4_K_M
- Mistral Nemo Instruct GGUF, Q4_K_M if RAM allows

For images, start with Python `diffusers` before porting:

- `black-forest-labs/FLUX.1-schnell`
- `black-forest-labs/FLUX.1-dev`
- `Qwen/Qwen-Image`
- `stabilityai/stable-diffusion-3.5-medium`

## Notes

- Library choice does not remove model-level refusal behavior. For lawful adult
  creative use, test the actual checkpoint and prompt style you intend to use.
- Python/MLX and llama.cpp/GGUF results are not directly interchangeable; use the
  same prompts and output scoring, but compare each runtime family separately.
- Use release builds for Rust. Debug builds make llama.cpp performance numbers
  misleading.
