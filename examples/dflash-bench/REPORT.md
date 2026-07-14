# DFlash Speculative Decoding - Throughput Tuning Report

Date: 2026-07-13
Hardware: Mac Studio (Apple Silicon), macOS 15.7.7, Metal backend
Build: llama.cpp @ master + per-request speculative-override patch (see below)

## Summary: best throughput per model

Best decode t/s reached after tuning (single request, benchmark prompt, near-greedy
sampling). The benchmark prompt is predictable, so treat these as optimistic ceilings.

| Model | Type | Best drafter + config | t/s | Baseline |
|-------|------|-----------------------|-----|----------|
| Qwen 3.6 35B-A3B | MoE (3B active) | dflash, n_max=15, p_min=0.8 | **~89** | ~30 (dflash n=2) / ~80 (mtp) |
| Qwen 3.6 27B | dense | dflash, n_max=15, p_min=0.8 | **~28** | ~14 (mtp) |
| Gemma 4 26B-A4B | MoE (4B active) | dflash, n_max=2 (spec marginal) | **~53** | ~50 (no spec) / ~45 (mtp) |
| Gemma 4 31B | dense | dflash or mtp, n_max=4 | **~15.5** | ~15.5 (ceiling) |

Takeaways: Qwen wins big from dflash + adaptive `p_min` gating (35B ~3x, 27B ~2x).
Gemma is at its ceiling regardless of drafter (DFlash, MTP, and EAGLE-3 all converge;
EAGLE-3 was worst) - the limiter is the model, not the drafter. Lower temperature /
presence-penalty raises all these numbers (higher draft acceptance); see Sampling below.

## Goal

Find the draft-tuning settings that maximize decode throughput for DFlash
(diffusion-block) speculative decoding, across four target models and three
sampling regimes, and understand why the optimum differs by model.

## Method

- One server instance per model, loaded once. A driver (`sweep.py`) varies the
  draft parameters **per request** and reads `t/s` and draft acceptance straight
  from the `/completion` response (`timings.predicted_per_second`, `draft_n`,
  `draft_n_accepted`). No reloads between configs.
- Speculative decoding is exact: `n_max`/`p_min`/`n_min` change only speed, never
  the output distribution (with a fixed seed the text is identical). So these can
  be tuned freely; sampling (temp/top-k/...) defines the output and is treated as
  an exogenous scenario.
- Two sweeps per model:
  - `n_max` sweep at `p_min=0` (fixed block length),
  - `p_min` sweep at `n_max=15` (adaptive block: draft long when confident, cut
    short when not - `p_min` truncates the block when the draft's top-token
    probability drops below the threshold).
- Metrics are median of 2-3 repeats, fixed prompt + seed, warm-up discarded.
- `n_predict`: 128 (MoE), 96 (dense). `mean_len` in the `p_min` sweep is
  approximate (the cycle estimate assumes full-length blocks); `t/s` and
  `accept%` come directly from the server and are exact.

### The patch (benchmark accelerator only)

Upstream disables per-request speculative overrides (`#if 0` in
`server-schema.cpp`). To sweep without reloading, this branch:
- exposes `speculative.n_max`, `speculative.n_min`, `speculative.p_min` as
  per-request fields (`tools/server/server-schema.cpp`),
- caps the draft by the requested `n_max` in `get_n_draft_max()`
  (`tools/server/server-context.cpp`),
- threads `p_min`/`n_min` through the per-request draft-params override struct
  (`common/speculative.h`, `common/speculative.cpp`) and sets them from the
  request (`tools/server/server-context.cpp`).

Production does NOT need this patch: the winning configs deploy on the stock
server via the existing `--spec-draft-n-max` / `--spec-draft-p-min` launch flags.

## Results

All DFlash drafters here have `block_size=16` (max 15 draft tokens/step).
"real" uses each model's own models.ini sampling.

### Qwen 3.6 35B-A3B (MoE, ~3B active) - strong drafter

n_max sweep (t/s / accept%): peak at n_max=4.

| p_min @ n_max=15 | greedy | mid(0.7) | real(1.0, pp1.5) |
|------------------|--------|----------|------------------|
| 0.0 | 61.7 / 34% | 26.6 / 9% | 58.5 / 31% |
| 0.5 | 86.1 / 66% | 60.9 / 61% | 75.4 / 58% |
| 0.7 | 86.9 / 90% | 58.5 / 78% | 83.2 / 74% |
| 0.8 | 89.1 / 95% | 55.0 / 95% | 87.1 / 95% |
| 0.9 | 86.0 / 98% | 51.6 / 100% | 79.2 / 97% |

Winner: `n_max=15, p_min=0.8` -> ~87-89 t/s (greedy/real). Beats fixed n_max=4
on the real workload (78 -> 87). Baseline before tuning: ~30 t/s (~2.9x).

### Qwen 3.6 27B (dense) - strong drafter

n_max sweep: peak at n_max=4 (greedy 24.4, mid 24.6, real 18.6 t/s).

| p_min @ n_max=15 | greedy | mid(0.7) | real(1.0) |
|------------------|--------|----------|-----------|
| 0.5 | 25.9 / 72% | 25.3 / 71% | 20.5 / 65% |
| 0.7 | 27.8 / 99% | 28.1 / 99% | 22.2 / 84% |
| 0.8 | 27.2 / 99% | 28.7 / 100% | 23.3 / 97% |
| 0.9 | 25.8 / 100% | 26.6 / 100% | 22.7 / 100% |

Winner: `n_max=15, p_min=0.8` -> ~23-29 t/s. Dense verify is expensive, so long
accepted blocks pay off most here: p_min lifts the real workload 18.6 -> 23.3
(+25%), delivering ~18-22 accepted tokens per verify. Baseline (draft-mtp in
models.ini): ~14 t/s (~1.6-2x).

### Gemma 4 26B-A4B (MoE, ~4B active) - weak drafter

Greedy acceptance only ~38% at n_max=1 (vs 82-98% for the Qwen drafters).

| approach | greedy | mid(0.7) | real(1.0) |
|----------|--------|----------|-----------|
| fixed n_max=1 | 49.7 | 63.0 | 53.4 |
| fixed n_max=2 | 53.1 | 57.9 | 53.6 |
| p_min=0.8 @ n_max=15 | 44.5 | 48.7 | 46.1 |
| p_min=0.9 @ n_max=15 | 46.4 | 46.1 | 29.3 |

Winner: fixed `n_max=1-2` (~53-63 t/s). p_min gating HURTS here - a weak drafter
is wrong early, so no block-sizing strategy helps, and cheap MoE verify makes
extra draft compute net-negative.

### Gemma 4 31B (dense) - mediocre drafter

Greedy acceptance ~73% at n_max=1, maxing ~78% even fully gated.

| approach | greedy | mid(0.7) | real(1.0) |
|----------|--------|----------|-----------|
| fixed n_max=1 | 15.1 | 14.4 | 14.9 |
| p_min=0.8 @ n_max=15 | 15.5 | 14.6 | 14.7 |
| p_min=0.9 @ n_max=15 | 15.6 | 14.6 | 15.9 |

Winner: roughly tied between `n_max=1` and `p_min=0.9` (~15-16 t/s). Speculation
benefit is marginal - the drafter is not predictive enough.

## Cross-model analysis

1. Drafter quality dominates everything. Qwen drafters are strong (82-98% greedy
   acceptance); Gemma drafters are weak-to-mediocre (38-73%).
2. Adaptive `p_min` gating wins only with a strong drafter. Then long confident
   blocks get accepted and amortize the verify cost. With a weak drafter the
   block is wrong early, so gating cannot help and fixed minimal `n_max` is best.
3. Dense vs MoE sets the magnitude, not the rule. A dense target has an expensive
   verify, so long accepted blocks help more - but only if the drafter is strong
   (Qwen 27B: big p_min win; Gemma 31B: marginal).
4. Temperature erodes acceptance. Greedy is the ceiling; higher temp lowers
   acceptance and the optimal aggressiveness.

### Decision rule

- Strong drafter (greedy acceptance > ~80%): `n_max=15, p_min~0.8`.
- Weak drafter (greedy acceptance < ~50%): fixed `n_max=1-2`; reconsider whether
  a separate drafter beats self-speculation (MTP) or no speculation at all.

## Follow-up: does target quantization explain the weak Gemma drafter?

Tested by swapping the gemma-4-26B-A4B target from standard Q4_K_XL to the
quantization-aware-trained (QAT) Q4_K_XL variant (same 4-bit size, higher
fidelity), keeping the same drafter.

- Not corruption: the QAT model produces coherent output via its chat template -
  it is a reasoning model that emits `<|channel>thought` chains. On a raw
  `/completion` prompt it drifts into channel tokens, which deterministically
  tripped the Gemma4 chat-format parser (HTTP 500). Worked around with
  `--chat-template chatml` (bypasses the PEG_GEMMA4 parser).
- Greedy acceptance is the clean comparison (deterministic; temp>0 is confounded
  by the QAT model's reasoning behavior):

  | config | regular Q4 | QAT Q4 |
  |--------|-----------|--------|
  | greedy n_max=1 | 38.5% | 48.2% |
  | greedy n_max=2 | 34.9% | 35.4% |
  | greedy p_min=0.9 @ n_max=15 | 54.1% | 58.2% |

- QAT lifts greedy acceptance ~5-10 points. Real, and it confirms Gemma is
  quant-sensitive (the QAT variant exists for exactly this reason; Q4 perturbs
  the intermediate features DFlash conditions on). But it does NOT close the gap
  to the Qwen drafters (82-98%) - even with a quant-aware target the Gemma
  drafter tops out ~48-58%.

Conclusion: target quantization is a minor contributor, not the main cause. The
dominant limiter is the Gemma DFlash drafter itself (weaker checkpoint;
Gemma's local/global sliding-window attention is harder to draft for). A better
drafter, not a better target quant, is what would move the needle.

Caveat: temp>0 comparisons were unreliable (the QAT reasoning model + chatml
override change the token stream, e.g. real n_max=2 gave an anomalous 5.8%), so
the greedy numbers are the signal to trust.

## Alternative drafters for Gemma (MTP, EAGLE-3, classic)

Since DFlash was weak on Gemma, every other drafter type this build supports was
tested on the same targets. Best decode t/s and greedy acceptance at n_max=1:

26B-A4B (MoE):

| Drafter | best t/s | accept@1 |
|---------|----------|----------|
| DFlash  | ~53 (n=2) | 38% |
| MTP     | ~45 (n=4) | 69% |
| EAGLE-3 | ~45 (n=2) | 41% |

31B (dense):

| Drafter | best t/s | accept@1 |
|---------|----------|----------|
| DFlash  | ~15-16 | 73% |
| MTP     | ~15.5 (n=4) | 76% |
| EAGLE-3 | ~12 | 54% |

- MTP is a better *predictor* (higher acceptance) but nets no throughput win: on
  the fast MoE its per-step overhead makes it slower than DFlash; on the dense
  model it ties.
- EAGLE-3 (RedHatAI heads, GGUF) underperformed both, and was worst on the dense
  model. Caveat: llama.cpp's `draft-eagle3` support is new and these heads target
  vLLM; the weak result may partly reflect the llama.cpp implementation.
- Classic small-model draft (gemma-4-E2B) crashed (`GGML_ASSERT n_outputs_max` in
  `draft-simple`; E2B is a Gemma-3n MatFormer, an unusual arch).

Conclusion across four drafter types: Gemma's ceiling (~15.5 t/s dense, ~50 t/s
MoE) is a property of the model, not the drafter. Gemma yields short accepted
runs (mean_len ~1.7-2.9) regardless of drafter; Qwen + p_min reaches 18-22. Keep
Gemma simple (DFlash short fixed n_max, or no drafter).

## Sampling and throughput

Acceptance requires the target to re-sample the drafter's token, and the DFlash
draft sampler is a fixed greedy top-k=10. So the closer the target is to greedy,
the higher the acceptance -> the higher the throughput. Sampling is a speed knob,
not just a quality knob:

- Lower temperature -> sharper target distribution -> higher acceptance/t/s, at
  the cost of diversity. Greedy had the highest acceptance in every sweep.
- presence-penalty is the bigger lever: it reshapes the logits every step (even
  at temp=0 it shifts the argmax) away from the drafter's prediction, cutting
  acceptance sharply. High values (1.5) hurt both throughput and determinism.

Guidance:

- Deterministic work (code/extraction/agents): temp ~0-0.2, presence-penalty ~0.
  Best throughput AND the desired output. A tiny temp/penalty (~0.2) guards
  against rare greedy repetition loops at negligible acceptance cost.
- Creative/diverse: temp ~1.0, presence-penalty ~1.5 -- accept lower speculative
  throughput (the regime where DFlash helps least anyway).

Measured deltas (dflash, best config): 35B ~89 t/s greedy vs ~87 at temp=1.0/
pp=1.5; 27B ~27.8 greedy vs ~23.3 (+19%).

## Server / deployment settings (Apple Silicon / Metal)

Beyond `ngl=99` and `flash-attn` (essential, already on), the levers that matter,
ranked:

1. `ctx-size` -- the biggest memory lever. A 262144 context reserves a huge KV
   cache up front (tens of GB even at q8_0) whether used or not. Right-size to
   real usage (e.g. 131072 or less) to free memory for bigger models / more slots
   and faster startup.
2. `ubatch-size` (physical batch) -- default 512; raising to 2048 speeds prompt
   processing / TTFT on long inputs (more prefill memory, but transient).
   Requires `batch-size >= ubatch-size` (default batch is 2048).
3. KV cache quant (`ctk`/`ctv`) -- q8_0 is the safe default. q4_1 halves KV memory
   but costs long-context accuracy; keep q8_0 for deterministic/long-doc work.
4. `parallel` (slots) -- 1 for single-user (full context per request); higher only
   helps under concurrent load and splits the KV.

Minor: `--mlock` (pin model in RAM), `--spec-draft-type-k/v` (quantize draft KV).
Decode t/s is unaffected by these -- they trade memory and prefill/TTFT.

## Recommended models.ini

Global block: deterministic profile is set per-model (`temp = 0.2`,
`presence-penalty = 0.2`); Metal perf + right-sized context below.

```ini
[*]
ngl = 99
flash-attn = true
ubatch-size = 2048        ; faster prefill/TTFT (default 512)
ctk = q8_0
ctv = q8_0
# per model: ctx-size = 131072 (halved from 262144), temp = 0.2, presence-penalty = 0.2

[qwen-3.6-35B-dflash]
model = .../Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf
spec-draft-model = .../dflash/Qwen3.6-35B-A3B-DFlash-Q8_0.gguf
spec-type = draft-dflash
spec-draft-n-max = 15
spec-draft-p-min = 0.8
spec-draft-ngl = all

[qwen-3.6-27B]
model = .../Qwen3.6-27B-GGUF/Qwen3.6-27B-UD-Q4_K_XL.gguf
spec-draft-model = .../dflash/Qwen3.6-27B-DFlash-Q8_0.gguf
spec-type = draft-dflash
spec-draft-n-max = 15
spec-draft-p-min = 0.8
spec-draft-ngl = all

# Gemma drafters are weak: prefer a short fixed draft (or skip DFlash).
[gemma-4-a4b-dflash]
spec-draft-n-max = 2
spec-draft-p-min = 0.0

[gemma-4-31b-dflash]
spec-draft-n-max = 1
spec-draft-p-min = 0.0
```

## Caveats

- Acceptance is content- and context-dependent. The benchmark prompt is short
  and predictable, so it likely over-states acceptance vs long, varied
  production traffic. Re-run `sweep.py` with a representative `prompt.txt` (and
  the real context length) for production-accurate numbers.
- Numbers are single-request (no concurrency). Throughput under batched load
  will differ.

## Tooling

- `dflash-bench.cpp` - standalone CLI that loads once and sweeps configs
  (validated the DFlash loop; the server path is the one used for these numbers).
- `serve.sh` - launch one server for a model pair (override `BASE`/`DFLASH`).
- `sweep.py` - per-request sweep driver (`--mode nmax|pmin`).
- `prompt.txt` - fixed benchmark prompt.
