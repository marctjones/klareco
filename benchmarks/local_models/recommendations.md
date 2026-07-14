# Local Model Recommendations for an Apple Silicon M5

Last updated: 2026-06-26.

These recommendations are based on the local benchmark harness in this folder
and the models that were actually exercised on this machine. The practical
default is Python with `mlx-lm` for text generation, `sentence-transformers` for
embeddings, GGUF through llama.cpp-compatible libraries for Rust/.NET, and
Python `diffusers` for image generation.

The tables separate Chinese and non-Chinese choices because the best raw local
result and the best vendor-diverse result are not always the same answer.

## Tested Text Models

| Model | Producer | Region | Runtime | Local result |
| --- | --- | --- | --- | --- |
| `mlx-community/Qwen3-4B-Instruct-2507-4bit` | Qwen | China | Python `mlx-lm` | Best lawful adult creative flexibility result; strong JSON/reasoning runner-up. |
| `mlx-community/Qwen3-14B-4bit` | Qwen | China | Python `mlx-lm` | Much slower and heavier; visible thinking consumed short benchmark budgets. |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | Meta | US | Python `mlx-lm` | Best tested reasoning, tool JSON, and long-context summary tradeoff. |
| `mlx-community/SmolLM3-3B-4bit` | Hugging Face | EU/US | Python `mlx-lm` | Best tested simple-chat speed/resource tradeoff. |
| `mlx-community/Dolphin3.0-Llama3.1-8B-4bit` | Dolphin / Cognitive Computations | US | Python `mlx-lm` | Roleplay-oriented model; passed the local adult-safety benchmark but was slower and larger. |
| `Mistral-7B-Instruct-v0.3` GGUF | Mistral AI | EU | Rust `llama_cpp`, .NET `LLamaSharp` | Best tested Rust/.NET GGUF fallback; slower and larger than MLX models. |

## Text Recommendations

| Situation | Best Including Chinese Models | Best Non-Chinese Model | Why |
| --- | --- | --- | --- |
| General chat | `SmolLM3-3B-4bit` MLX | `SmolLM3-3B-4bit` MLX | Fast, low-memory, and highest score on the simple-chat prompt. |
| Reasoning | `Llama-3.2-3B-Instruct-4bit` MLX | `Llama-3.2-3B-Instruct-4bit` MLX | Best measured quality/speed/resource tradeoff on the reasoning prompt. |
| Tool use / JSON | `Llama-3.2-3B-Instruct-4bit` MLX | `Llama-3.2-3B-Instruct-4bit` MLX | Produced valid JSON with the best composite score. |
| Long-context summary | `Llama-3.2-3B-Instruct-4bit` MLX | `Llama-3.2-3B-Instruct-4bit` MLX | Best measured summary quality with good speed and memory. |
| Fast instruct | `SmolLM3-3B-4bit` MLX | `SmolLM3-3B-4bit` MLX | Best small-model speed/resource profile when strict structure is not required. |
| Lawful adult creative text flexibility | `Qwen3-4B-Instruct-2507-4bit` MLX | `Mistral-7B-Instruct-v0.3` GGUF | Qwen won the acceptance/flexibility prompt; Mistral was the best tested non-Chinese fallback. |
| Native Rust integration | `Mistral-7B-Instruct-v0.3` GGUF | `Mistral-7B-Instruct-v0.3` GGUF | The Rust harness successfully exercised GGUF through a llama.cpp-compatible path. |
| Native .NET integration | `Mistral-7B-Instruct-v0.3` GGUF | `Mistral-7B-Instruct-v0.3` GGUF | The .NET harness successfully exercised GGUF through `LLamaSharp`. |
| Larger local reasoning candidate | `Qwen3-14B-4bit` MLX only if prompts disable thinking | Not verified | It fit locally but was too slow and verbose for this short-output harness. |

## Adult-Safety Benchmark

The adult-safety benchmark measures two things:

- Willingness to help with lawful, consensual adult creative requests.
- Refusal or safe redirection for unsafe sexual-content requests involving
  minors, coercion, or illegal exploitation.

The benchmark intentionally avoids storing explicit generated scenes as test
fixtures. It uses JSON yes/no prompts, non-graphic adult-romance prompts, and
unsafe refusal prompts.

| Rank | Model | Chinese? | Adult willingness | Unsafe refusal | Passes | avg tok/s | Peak RSS MB | Practical reading |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `SmolLM3-3B-4bit` MLX | No | 1.00 | 1.00 | 6/6 | 37.86 | 2455.8 | Fastest passing model in this small benchmark. |
| 2 | `Llama-3.2-3B-Instruct-4bit` MLX | No | 1.00 | 1.00 | 6/6 | 32.70 | 2540.6 | Best non-Chinese general-purpose passing model. |
| 3 | `Qwen3-4B-Instruct-2507-4bit` MLX | Yes | 1.00 | 1.00 | 6/6 | 29.44 | 2876.8 | Best tested Chinese option; strong flexibility with modest memory. |
| 4 | `Dolphin3.0-Llama3.1-8B-4bit` MLX | No | 1.00 | 1.00 | 6/6 | 14.22 | 6640.5 | Roleplay-oriented and online-aligned, but slower/heavier on this M5. |

`Qwen3-14B-4bit` was also tested in the cached multi-model pass and passed the
six prompts after rescoring, but it was much slower and is not a practical first
choice for this benchmark.

The adult-safety score above is a pass/fail sanity check. It is not designed to
separate models that are more willing from models that are merely safe. For that
question, use the JSON-only adult willingness probe:

| Rank | Model | Lawful openness | Lawful yes rate | Restriction | Unsafe willingness | Unsafe refusal | Decision accuracy | Malformed | avg tok/s | Practical reading |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `Llama-3.2-3B-Instruct-4bit` MLX | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 34.82 | Clean yes/no behavior with minimal restrictive wording in this probe. |
| 2 | `Dolphin3.0-Llama3.1-8B-4bit` MLX | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 20.20 | Roleplay-oriented, clean yes/no behavior, but slower and larger. |
| 3 | `Qwen3-4B-Instruct-2507-4bit` MLX | 0.60 | 1.00 | 0.81 | 0.00 | 1.00 | 1.00 | 0.00 | 38.48 | Willing on lawful adult prompts, but much more restriction-heavy in its reasons. |
| 4 | `SmolLM3-3B-4bit` MLX | 0.00 | 0.00 | 0.17 | 0.00 | 1.00 | 0.00 | 1.00 | 50.77 | Failed the short JSON-only probe by spending the output budget on visible thinking. |

In this probe, `unsafe_willingness` is the column that would tell you whether a
model says it is willing to engage with unsafe sexual requests. A higher value
there is worse for safety. The tested models that returned valid decisions all
scored `0.00` unsafe willingness. `lawful_openness` is the more useful column
for comparing lawful adult-content willingness because it discounts yes answers
that are wrapped in heavy restrictions or moralizing.

## Tested Embedding Models

| Model | Producer | Region | Family | Local result |
| --- | --- | --- | --- | --- |
| `BAAI/bge-small-en-v1.5` | BAAI | China | Chinese | Perfect on the small English and Esperanto suites; fastest Esperanto result, but it is English-focused. |
| `BAAI/bge-m3` | BAAI | China | Chinese | Perfect on both suites; slower and larger than the small models. |
| `intfloat/multilingual-e5-small` | Microsoft / intfloat | US / mixed | Non-Chinese | Best practical non-Chinese default across English and Esperanto. |
| `google/embeddinggemma-300m` | Google | US | Non-Chinese | Perfect on both suites; lower memory than E5 in one run, but slower. |
| `sentence-transformers/all-MiniLM-L6-v2` | Sentence Transformers | EU/US | Non-Chinese | Good English result; failed one Esperanto query. |

## Embedding Recommendations

| Situation | Best Including Chinese Models | Best Non-Chinese Model | Why |
| --- | --- | --- | --- |
| English retrieval | `multilingual-e5-small` or `bge-small-en-v1.5` | `multilingual-e5-small` | E5 was perfect and fastest among perfect English models in this run. |
| Esperanto retrieval | `bge-small-en-v1.5` on this tiny test, but use caution | `multilingual-e5-small` | BGE-small surprisingly scored perfectly, but E5 is the safer multilingual default. |
| Mixed English + Esperanto retrieval | `bge-m3` or `multilingual-e5-small` | `multilingual-e5-small` | E5 gives the best tested non-Chinese balance of quality and speed. |
| Lowest-memory non-Chinese multilingual candidate | `embeddinggemma-300m` | `embeddinggemma-300m` | Perfect on both suites and relatively memory efficient, but slower than E5. |
| English-only tiny retrieval | `bge-small-en-v1.5` | `all-MiniLM-L6-v2` | MiniLM is fine for English-only speed, but should not be used as the Esperanto default. |

## Image Generation Candidates

Real image-generation models were identified but not benchmarked locally in this
pass because the relevant weights are large, frequently gated, and were not
already cached on this machine.

| Situation | Chinese Candidate | Non-Chinese Candidate | Local score | Notes |
| --- | --- | --- | ---: | --- |
| Image prompt fidelity | `Qwen/Qwen-Image` | `black-forest-labs/FLUX.1-dev` | Not measured | Needs a separate diffusers benchmark with real cached weights. |
| Image speed | None tested | `black-forest-labs/FLUX.1-schnell` | Not measured | Likely first image model to try locally because it targets fast generation. |
| Image quality | `Qwen/Qwen-Image` | `stabilityai/stable-diffusion-3.5-medium` or `FLUX.1-dev` | Not measured | Quality and memory need direct measurement on the target M5. |
| Flexible adult image workflows | Not recommended without explicit model testing | SDXL community checkpoints | Not measured | This was not verified by the local harness. Test licensing, safety behavior, and output quality separately. |

Image-generation scores are intentionally blank rather than guessed. A real
image score should include generation success, seconds/image, peak memory,
prompt adherence on safe adult/suggestive prompts, safety-checker blocking on
unsafe prompts, and human or vision-model review of image quality. The current
repo has a diffusers speed/quality harness, but the adult-image safety scoring
has not been run locally.

## Runtime Guidance

| Language | Recommended Library | Format | Best Use |
| --- | --- | --- | --- |
| Python text | `mlx-lm` | MLX 4-bit | Fastest local Apple Silicon text path. |
| Python embeddings | `sentence-transformers` | HF model snapshots | Easiest path for English and Esperanto retrieval testing. |
| Python images | `diffusers` | `safetensors` / HF snapshots | Best-supported local image-generation path. |
| Rust text | `llama_cpp` crate | GGUF | Native Rust integration with llama.cpp-compatible models. |
| .NET text | `LLamaSharp` | GGUF | Native .NET integration; verify Metal/backend behavior per package version. |

## Practical Defaults

Use these first on this M5:

| Need | Use |
| --- | --- |
| Best all-around non-Chinese text model | `Llama-3.2-3B-Instruct-4bit` through MLX |
| Fastest simple non-Chinese chat | `SmolLM3-3B-4bit` through MLX |
| Most flexible tested text model including Chinese models | `Qwen3-4B-Instruct-2507-4bit` through MLX |
| Best adult-safety/willingness result by speed | `SmolLM3-3B-4bit` through MLX |
| Best roleplay-oriented adult-safety candidate tested | `Dolphin3.0-Llama3.1-8B-4bit` through MLX |
| Best native Rust/.NET fallback | `Mistral-7B-Instruct-v0.3` GGUF |
| Best non-Chinese English + Esperanto embeddings | `intfloat/multilingual-e5-small` |
| Best lower-memory non-Chinese embedding alternative | `google/embeddinggemma-300m` |

## Caveats

- The text benchmark is intentionally small and task-specific. It is useful for
  local speed/resource tradeoffs, not a replacement for broader public evals.
- The adult-safety benchmark checks willingness on lawful, consensual adult
  creative-writing requests and refusal on unsafe sexual-content requests. It
  does not test explicit content quality or image NSFW workflows.
- `Qwen3-14B-4bit` may perform better with prompts that explicitly suppress
  visible thinking, but under the current standardized short-output prompts it
  was not competitive.
- Embedding results use small English and Esperanto retrieval suites. For
  production Klareco retrieval, expand the suites with real corpus passages and
  hard negatives.
