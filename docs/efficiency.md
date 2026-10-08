# Less text, less waiting

The first target is **end-to-end task time**, with correctness and successful tool execution held constant. A prettier or shorter serialization is useful only when it improves that result.

## What changed in ty 0.3–0.5

| Change | Why it helps | Tradeoff |
| --- | --- | --- |
| Changed-line composer updates | Reduces repainting and terminal flicker | Synchronized frames also depend on terminal support |
| Per-model latency logs in `/stats` | Identifies loading, prompt processing, and output generation separately | Real task prompts vary; benchmark separately for comparisons |
| Full history archive outside the prompt | Allows complete resume/export after compaction | Session files grow with the saved conversation |
| Batched native tool calls | Several independent actions can use one model response | Execution remains ordered for approval and steering |
| Live steering and cancellation | Stops unwanted work instead of waiting through another model generation | Completed mutations are preserved |
| Streamed compaction progress | Shows work and cancellation while context is summarized | Token counts are approximate; no progress percentage |
| Stable date/environment prefix within a session | Stops the clock from changing the prompt every model step | Does not guarantee prefix-cache reuse for hybrid models |
| `code` and `web` tool sets | Reduces schema overhead for focused tasks | Only those tools are available |
| Thinking off by default; 768 generated tokens per step | Limits long generations | Longer implementations need a larger output cap |
| A 1,200-character tool budget and pruning older tool outputs | Keeps large files and command logs from accumulating | Excerpts can omit needed context; raise the budget when necessary |
| Query-focused, cached page excerpts | Sends relevant source text instead of the start of every page | Lexical matching misses some paraphrases |
| HTML and lite search fallbacks; Jina only as a fallback | Avoids an extra intermediary when direct retrieval works | Public search pages can challenge or rate-limit clients |
| Duplicate-call refusal | Avoids repeated commands and writes when a model loops | Repeating an identical command after a deliberate state change within one task needs a new task |
| Estimated budgets include tool schemas | Makes context reports and compaction thresholds more useful | Character estimates still differ from the real tokenizer |
| Physical-core thread selection on Linux | Provides a conservative CPU baseline | Benchmark overrides for the machine and workload |

With an empty conversation, a `/workspace` directory, no project notes, and the default model, approximate overhead at v0.3 was:

| Tool set | Tools | Estimated tokens |
| --- | --- | --- |
| `core` | 7 | 492 |
| `code` | 5 | 375 |
| `web` | 2 | 213 |
| `all` | 9 | 610 |

These are `len(text)/4` estimates including the system prompt and schema JSON, not tokenizer measurements. The coding set is about 24% smaller than core; web is about 57% smaller. A shorter fixed prompt does not imply the same percentage improvement in total task time.

## This laptop

The development machine is an **Intel i7-3520M**, with **2 physical cores / 4 logical threads**, **about 8 GB RAM**, and CPU-only inference. It supports AVX, but not AVX2. Ollama 0.40.0 is installed with `qwen3.5:2b` and `qwen3.5:0.8b`.

At inspection, roughly 2.3 GiB of RAM was available and 1.1 GiB of swap was in use. Swap usage is not proof of active swapping. Check `vmstat 1` for sustained `si`/`so` activity and compare memory before and during a task.

The existing local Ollama logs showed prompt processing at roughly **11.5–13.2 tokens/s** as prompts grew from hundreds to thousands of tokens. At 12 tokens/s, another 1,000 prompt tokens costs roughly 83 seconds if the entire prompt is processed again. That arithmetic explains why trimming context can matter more than changing output punctuation.

The local server already sets a 4,096-token context, one concurrent request, one loaded model, flash attention, and a q8 KV cache. These are observations of this installation, not settings changed by ty. Whether KV quantization helps this particular hybrid model needs measurement; don't assume it halves the whole model's RAM.

Recommended starting points:

1. Use `--fast` for short questions and simple changes. Stay on one model for a work session to avoid reloads.
2. Use `--tools code` for coding or `--tools web` for research.
3. Keep thinking off. Try terse reasoning only when it improves task completion.
4. Start with `--threads 2`: the new comparison favored it for 0.8B and found no benefit from four threads for 2B. Repeat the measurement on the intended workload before changing it.
5. Keep the model resident during a session. Use `/unload` when you are finished, or `--unload-on-exit`. `--keep-alive 0` unloads after requests and can be expensive in a multi-step task.
6. Close memory-heavy applications if memory pressure rises. Do not increase the context merely to avoid pruning.
7. Use `/new` when old conversation stops helping. Current compaction summarizes older turns; it does not solve an oversized single turn.

### Model and thread comparison, October 7–8, 2026

```sh
python3 scripts/benchmark.py --models qwen3.5:0.8b qwen3.5:2b \
  --threads 2 4 --repeat 3 --tokens 32 \
  --output timings.csv --metadata timings.json
```

[Raw CSV](model-benchmark.csv) and [installed model/settings metadata](model-benchmark.json) contain 24 samples: two installed models, two thread counts, two prompt lengths, and three repetitions. Client settings were deterministic, thinking off, a 4,096-token context, and a 32-token output cap. The installed tags both use Q8_0; 0.8B also has `draft_num_predict = 2`, while 2B does not. This compares those installed configurations, rather than isolating parameter count or answer quality. Other applications and OS scheduling were not isolated.

Warm rows below are medians of repetitions 2 and 3.

| Model | Threads | Warm short task | Warm short generation | First 560-token prompt processing |
| --- | --- | --- | --- | --- |
| 0.8B | 2 | 2.89 s | 11.93 tokens/s | 18.34 s |
| 0.8B | 4 | 4.13 s | 8.34 tokens/s | 18.70 s |
| 2B | 2 | 4.83 s | 7.22 tokens/s | 42.85 s |
| 2B | 4 | 4.95 s | 7.03 tokens/s | 42.62 s |

At two threads, 0.8B generated about 1.65 times as many tokens per second and processed the first longer prompt about 2.34 times as fast. The repeated longer prompt took only 0.18–0.41 seconds to process on warm samples, showing a large cache effect in this workload. Reported prompt-token counts still describe the prompt, so warm prefill tokens/s must not be interpreted as fresh-token throughput.

The first request after changing model/thread settings included roughly 4.7–6.2 seconds of loading. Swap-out activity was concentrated around some model changes; warm samples had little swap activity. Keep one model loaded through a task and avoid switching for every page. Ollama documents the request-level [keep_alive setting and concurrency memory cost](https://docs.ollama.com/faq).

For routine work, try `ty --fast --tools code --threads 2`; for research, use `--tools web`. The larger model remains available in `/mode`. This keeps the model and tool choice explicit instead of changing behavior based on guessed task difficulty.

`/stats` now records the last 100 ordinary inference steps with model, thread count, tool set, prompt/output counts, first-output latency, load time, prompt-processing time, decode time, and wall time. The data remains in the session file, outside the prompt. `/ui verbose` prints latency components after each step. First output includes reasoning and native tool-call frames; it can precede the first rendered paragraph.

Both installed models use eight-bit weights. The official registry also offers a smaller [2B Q4_K_M build](https://ollama.com/library/qwen3.5:2b-q4_K_M), listed at 1.9 GB versus 2.7 GB for the [Q8_0 build](https://ollama.com/library/qwen3.5:2b-q8_0). This is the next candidate to benchmark for reduced memory traffic on this CPU. It is not installed or measured here; quantization may affect tool reliability, and speed depends on this processor’s kernels. Test it with read/search/edit tasks before selecting it as a default.

### Earlier thread comparison

```sh
python3 scripts/benchmark.py --model qwen3.5:2b --threads 2 4 --repeat 2
```

The earlier comparison used the same short prompt with a 4,096-token context, deterministic sampling, thinking off, and a 32-token output cap. It reports prompt/output counts, prefill and decode rates, total time, and load time. Changing thread settings may reload the runner. Compare warm runs separately from the first run after a change.

The comparison was rerun with other local model requests finished. [Raw measurements](laptop-benchmark.csv):

| Threads | First run total / load | Warm total | Warm decode |
| --- | --- | --- | --- |
| 2 | 12.58 s / 5.88 s | 4.81 s | 7.23 tokens/s |
| 4 | 12.52 s / 5.84 s | 4.80 s | 7.43 tokens/s |

The warm task times were effectively identical in this small sample. Two threads remain the conservative default for this two-core machine. The repeated 30-token prompt may benefit from caching; this does not measure a growing tool conversation or establish cache reuse for every task. More samples and longer workloads are needed for a general thread-count recommendation.

An earlier contended run took 92–284 seconds, mostly reported as load time. That illustrates why queueing and model reloads must be separated from decode speed. Keep other model clients idle when benchmarking.

A live two-step `read_file` task also passed: 68 seconds end to end, with 45 generated tokens. On its second step, the runner processed only 61 new prompt tokens in 4.76 seconds, with earlier context cached. This shows prefix reuse in that test after stabilizing the prompt; it is not a guarantee for longer conversations, pruning, or compaction.

## JSON, YAML, or a tiny custom tool format?

ty keeps **Ollama's native tool calls**. Its API accepts JSON schemas and returns structured tool arguments. See [Ollama's tool-calling documentation](https://docs.ollama.com/capabilities/tool-calling).

Changing the HTTP body's JSON spacing does not necessarily change the model's prompt: Ollama renders its own model template. To make the model generate YAML instead, we would need a textual protocol, custom parsing, history conversion, and validation. That could save tokens, but introduces problems with indentation, multiline code, escaped strings, ambiguous scalars, and unsupported tool calls. Small models can also output commentary around the intended call.

A narrow experiment is worthwhile before adopting it:

- Start with read-only calls such as `read_file(path=...)`, rather than file writes.
- Compare native calls against a short line protocol and strict YAML on the same 50–100 tasks.
- Include paths with spaces, quotes, Unicode, multiline strings, booleans, and malformed calls.
- Measure prompt tokens, generated tokens, parse failures, retry tokens, task success, and total wall time.
- Execute only allowlisted, validated calls. Do not recover a file mutation from loosely parsed text.

A useful break-even test is:

`tokens saved on successful calls > tokens spent on repair prompts and retries`

Even then, count the extra round-trip time and correctness failures. Human config is now TOML, which Python can read without adding a dependency. That change is independent of the model's tool protocol.

## A tiny local model to filter pages?

For relevance, a **cross-encoder reranker** is a promising experiment. It scores query/passage pairs in a forward pass instead of generating a summary. A small command-safety classifier and a passage-relevance model solve different tasks; one should not be assumed to work as the other.

One candidate is [Jina's tiny English reranker](https://huggingface.co/jinaai/jina-reranker-v1-tiny-en). Its model card describes a 33M-parameter cross-encoder. That is a candidate to evaluate, not a dependency or a tested integration in ty. Check its license, inference code, and runtime memory before distributing weights or enabling it.

A practical optional pipeline would be:

1. Clean the page and split it into bounded passages.
2. Use lexical scoring to select perhaps 12 candidate passages.
3. Run a quantized CPU reranker on those candidates in a batch.
4. Keep 3–5 passages with their source URL, title, and document order.
5. Cache passage scores by page hash, query, and model version.

Don't load a second generative Ollama model for each page. This laptop is already memory-constrained, and keeping only one model loaded means switching can evict the agent model. A small ONNX reranker in a separate optional process is a more plausible route, but it still competes for RAM and CPU.

If removing 1,000 input tokens saves ~83 seconds of full prefill, a relevance stage taking a few seconds could pay for itself. That is a hypothetical break-even calculation using the local prefill rate, not a measured reranker result. Evaluate on real pages with human-labeled relevant passages; track retained evidence as well as compression ratio.

The implemented lexical selector is instant by comparison and returns source excerpts. It is the baseline a neural model needs to beat in relevance and total latency. It intentionally marks omissions instead of presenting an excerpt as the complete page.

## Next experiments, in order

| Priority | Experiment | Success measure |
| --- | --- | --- |
| 1 | Add line-range reads and paginated grep output | Fewer follow-up reads without losing the relevant code |
| 2 | Compare the explicit 2B Q4_K_M tag and draft settings | Lower wall time and memory with unchanged tool success |
| 3 | Evaluate a small optional CPU reranker | More relevant evidence at lower total task time |
| 4 | Add a structured search-provider adapter | Fewer provider failures and cleaner snippets; credentials optional |
| 5 | Preserve complete sessions but budget older turns without a generation pass | Less compaction time with adequate task continuity |
| 6 | Benchmark an attention-only model on this CPU | Better warm prompt reuse at comparable task success |
| 7 | Trial a strict compact tool protocol | Lower wall time without higher parse or execution error rates |

Also compare Ollama releases and model templates with a recorded workload. Fixing the stable prefix helps remove one avoidable source of cache invalidation; it does not establish that hybrid/recurrent state can be reused in every runner. Ollama's [memory and concurrency guidance](https://docs.ollama.com/faq) is useful for general tuning, but actual measurements on the machine decide the settings.
