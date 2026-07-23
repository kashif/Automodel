# Kernels Hub Integration Investigation

**Branch:** `kashif/investigate/kernels-hub-integration`  
**Date:** 2026-07-23  
**Status:** Investigation / spike

## Summary

NeMo AutoModel currently depends on several **directly installed** native kernel packages (`flash-attn`, `liger-kernel`, `mamba-ssm`, `transformer-engine`, `tilelang`, etc.). Hugging Face's [`kernels`](https://huggingface.co/docs/kernels) library and the [`kernels-community`](https://huggingface.co/kernels-community) Hub organization provide **pre-built, versioned, dynamically loaded** replacements for many of these.

**Key finding:** AutoModel already pins `transformers==5.12.1`, which ships full Hub-kernel integration (`transformers.integrations.hub_kernels`). Partial Hub support is therefore **already available** for HF-model paths when the `kernels` package is installed — but AutoModel's own kernel layer does not expose or fully leverage it yet.

The integration opportunity is to **centralize kernel loading through the Hub** instead of maintaining parallel install/compile paths, while keeping NVIDIA-specific backends (TE, DeepEP, TileLang) on their existing paths.

---

## Background: The Kernels Ecosystem

| Component | Role |
|---|---|
| [`kernels`](https://github.com/huggingface/kernels) | Python loader — `get_kernel()`, `has_kernel()`, `kernelize()`, lockfiles |
| [`kernel-builder`](https://huggingface.co/docs/kernels) | Deterministic build/publish pipeline for Hub kernels |
| [`kernels-community`](https://huggingface.co/kernels-community) | Curated Hub kernels (flash-attn2/3/4, liger-kernels, mamba-ssm, activation, rotary, megablocks, …) |

### Upstream API (from `huggingface/kernels`)

The loader lives in [`kernels/src/kernels/`](https://github.com/huggingface/kernels/tree/main/kernels/src/kernels). Key entry points:

| API | Purpose |
|---|---|
| `get_kernel(repo_id, version=1)` | Download + import a Hub kernel module (cached on disk) |
| `has_kernel(repo_id, version=1)` | Cheap compatibility check for current PyTorch/CUDA |
| `get_kernel_variants(...)` | Full resolution trace when `has_kernel` returns False |
| `kernelize(model, mode=Mode.TRAINING)` | Replace layer `forward` with Hub kernel implementations |
| `load_kernel` / `get_locked_kernel` | Offline/reproducible loads from `kernels lock` lockfiles |
| `LOCAL_KERNELS=repo=path` | Dev override — redirect Hub id to a local `kernel-builder` build |

Usage pattern (from upstream [integration guide](https://github.com/huggingface/kernels/blob/main/kernel-builder/skills/cuda-kernels/references/huggingface-kernels-integration.md)):

```python
from kernels import get_kernel, has_kernel

if has_kernel("kernels-community/flash-attn2", version=1):
    fa2 = get_kernel("kernels-community/flash-attn2", version=1)
    out = fa2.flash_attn_func(q, k, v, causal=True)
    varlen = fa2.flash_attn_varlen_func  # same module, used by CP/packing paths
```

For container/offline builds, upstream recommends:

```bash
kernels lock kernels-community/flash-attn2 kernels-community/liger-kernels
kernels download
```

### AutoModel wrapper (this branch)

A thin spike module wraps the upstream API without replacing existing call sites yet:

- `nemo_automodel/components/kernels/hub.py` — `get_hub_kernel()`, `has_hub_kernel()`, `get_flash_attn_varlen_func()`, `has_flash_attn_available()`
- Unit tests: `tests/unit_tests/components/kernels/test_hub.py`

Phase 2 will migrate direct `flash_attn` imports (blockdiag CP, KimiVL, BAGEL) to `get_flash_attn_varlen_func()`.

Transformers 5.x extends this with:

- `attn_implementation="kernels-community/flash-attn2"` — Hub attention backends
- `use_kernels=True` — auto-replace RMSNorm, MLP, Linear, activations, RoPE, causal LM loss, etc.
- `KernelConfig` — per-layer / per-device kernel mapping
- **Automatic fallback:** `attn_implementation="flash_attention_2"` uses compiled `flash-attn` if present, else Hub `kernels-community/flash-attn2`

---

## Current AutoModel Kernel Architecture

### 1. HF wrapper path (`nemo_automodel/_transformers/`)

| File | What it does today |
|---|---|
| `kernel_patches.py` | Direct `safe_import("flash_attn")`; FA2/FA3/FA4 detection; fallback ladder; Liger patching via `liger_kernel.transformers`; SDPA patching |
| `auto_model.py` | Passes `attn_implementation` to transformers; applies Liger + SDPA patches post-init; **does not** pass `use_kernels` or `kernel_config` |

Default attention selection (`kernel_patches.py`):

```python
DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2" if HAS_FA else "sdpa"
```

Availability is gated on **compiled package presence**, not Hub kernel availability.

### 2. Native model path (`BackendConfig`)

NeMo-native models use `BackendConfig` (`components/models/common/utils.py`) with explicit backends:

- `attn`: `te | sdpa | flex | eager | tilelang`
- `linear`, `rms_norm`, `experts`, `dispatcher`, …

This path is **orthogonal** to the HF Hub-kernel system and does not use `kernels` today.

### 3. Direct `flash_attn` imports (bypass transformers)

These call `flash_attn` APIs directly and would **not** benefit from transformers' Hub fallback:

| Location | Usage |
|---|---|
| `components/distributed/blockdiag_cp/kernels.py` | `flash_attn_varlen_func` for CP varlen attention |
| `components/speculative/eagle/ring_attention.py` | Private `_flash_attn_forward/_backward` (pinned to FA 2.8.x ABI) |
| `components/models/kimivl/model.py` | `flash_attn_varlen_func` |
| `components/models/bagel/modeling_qwen2_packed.py` | `flash_attn_varlen_func` for packed inference |

### 4. Dependencies (`pyproject.toml`)

| Optional group | Packages |
|---|---|
| `fa` | `flash-attn<=2.8.3` |
| `cuda` | `mamba-ssm`, `causal-conv1d`, `transformer-engine`, `tilelang`, … |
| `diffusion_kernels` | `kernels` (declared but **no Python imports yet**) |
| (implicit) | `liger-kernel` via runtime import in `kernel_patches.py` |

Docker images compile `flash-attn` from source; this is a major install-time cost Hub kernels could eliminate.

---

## What Already Works (Minimal Change)

With `pip install kernels` and no code changes, these **should** work through the existing transformers passthrough:

```yaml
# Example recipe snippet
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-3B
  attn_implementation: kernels-community/flash-attn2
  # or rely on transformers fallback:
  # attn_implementation: flash_attention_2   # uses Hub when flash-attn pkg absent
```

Caveats discovered in AutoModel code:

1. **`_apply_preload_overrides`** — packed-sequence training checks `HAS_FA/HAS_FA3/HAS_FA4` (compiled packages), not Hub availability. Will fail or force SDPA even when Hub FA2 is loadable.
2. **`_get_next_fallback_attn`** — fallback ladder only knows `flash_attention_{2,3,4}`, `sdpa`, `eager`. Hub repo IDs like `kernels-community/flash-attn2` are treated as unknown → fallback to `eager`.
3. **`use_liger_kernel=True` (default)** — still imports `liger_kernel` directly instead of using `use_kernels=True` + `kernels-community/liger-kernels`.
4. **`use_kernels` / `kernel_config` not forwarded** — transformers kwargs are not explicitly plumbed through `NeMoAutoModel.from_pretrained`.

---

## Hub Kernels ↔ AutoModel Feature Map

| AutoModel feature | Current backend | Hub kernel available? | Notes |
|---|---|---|---|
| HF attention (FA2/3/4) | `flash-attn` pip package | ✅ `kernels-community/flash-attn{2,3,4}` | Transformers handles wrapper; AutoModel needs availability + fallback fixes |
| Liger (RMSNorm, MLP, Linear, loss) | `liger_kernel` pip | ✅ `kernels-community/liger-kernels` | Transformers `use_kernels=True` replaces `_patch_liger_kernel` |
| Mamba / GDN conv | `mamba-ssm`, `causal-conv1d` | ✅ `kernels-community/mamba-ssm` | Already in transformers default mapping |
| Activations (GELU, SiLU) | PyTorch / TE | ✅ `kernels-community/activation` | Inference + compile modes |
| RoPE | TE fused / torch | ✅ `kernels-community/rotary` | TE fused RoPE currently force-disabled (#3027) |
| MoE expert GEMM | TE / grouped_gemm / torch_mm | ⚠️ `kernels-community/megablocks` | Different API; not a drop-in for DeepEP path |
| CP blockdiag varlen | Direct `flash_attn_varlen_func` | ✅ via Hub FA2 module | Needs thin loader wrapper |
| EAGLE ring attention | Direct FA 2.8.x private API | ❓ | Uses `_flash_attn_forward` positional ABI; Hub FA2 may differ — needs parity test |
| TE attention / FP8 | `transformer-engine` | ❌ | NVIDIA proprietary; stays as direct dep |
| DeepEP / UCCL-EP | `deep_ep` | ❌ | Not in kernels-community |
| TileLang (DSV4, GLM-DSA) | `tilelang`, `tile-kernels` | ❌ | Custom NVIDIA/vendor kernels |
| FLA (linear attention) | `flash-linear-attention` | ❓ | Check Hub; not in transformers default map |
| FlexAttn / MagiAttention | PyTorch / custom | ❌ | Custom CP dispatch |

---

## Proposed Integration Plan

### Phase 0 — Validate passthrough (1–2 days)

**Goal:** Confirm Hub kernels work end-to-end through existing `NeMoAutoModel` without refactors.

- [ ] Install `kernels` in dev container
- [ ] Run a minimal finetune with `attn_implementation=kernels-community/flash-attn2` (no `flash-attn` pip)
- [ ] Run same with `attn_implementation=flash_attention_2` and no compiled FA — verify transformers Hub fallback
- [ ] Document working recipe YAML

### Phase 1 — Hub-aware availability & passthrough (small PR)

**Goal:** Make AutoModel's kernel layer Hub-aware without breaking existing installs.

1. **Add optional dependency group** (extend or alias `fa`):

   ```toml
   hub_kernels = ["kernels>=0.11.0"]
   fa = ["flash-attn<=2.8.3"]  # keep for TE-compat / ring-attn ABI pinning
   fa_or_hub = ["nemo_automodel[hub_kernels]"]  # Hub-only path
   ```

2. **Use `nemo_automodel/components/kernels/hub.py`** (added on this branch) — thin, cached wrapper around upstream `kernels.get_kernel` / `has_kernel`.

3. **Update `kernel_patches.py`:**
   - Extend `has_flash_attn()` to check Hub via `has_kernel()` / transformers `is_kernels_available()`
   - Teach `_apply_preload_overrides` that `kernels-community/flash-attn*` counts as flash for packed sequences
   - Extend `_get_next_fallback_attn` to handle Hub repo IDs (or delegate to transformers' resolver)

4. **Plumb `use_kernels` and `kernel_config`** through `NeMoAutoModel.from_pretrained` kwargs (pass-through, no new API surface beyond forwarding).

5. **Gate Liger:** when `use_kernels=True`, skip `_patch_liger_kernel` (transformers handles it).

### Phase 2 — Replace direct `flash_attn` imports (medium PR)

**Goal:** CP, speculative, and VLM paths load varlen kernels from Hub when pip package absent.

| File | Change |
|---|---|
| `blockdiag_cp/kernels.py` | Load `flash_attn_varlen_func` via `hub.py` |
| `kimivl/model.py`, `bagel/modeling_qwen2_packed.py` | Same |
| `ring_attention.py` | **High risk** — evaluate Hub FA2 private API compatibility; may keep pip pin for EAGLE |

### Phase 3 — Liger consolidation (medium PR)

- Default `use_kernels=True` for HF models when `kernels` installed
- Deprecate direct `liger_kernel` import path
- Add `kernels lock` to CI/container for reproducible builds

### Phase 4 — Native `BackendConfig` (optional, larger scope)

Extend `BackendConfig.attn` with Hub-backed options or a separate `hub_kernels: KernelConfig` field for NeMo-native model layers. This is a larger design decision — native models currently bypass transformers' attention dispatch entirely.

---

## Container / CI Implications

| Today | With Hub kernels |
|---|---|
| Docker builds compile `flash-attn` from source (slow, fragile) | `pip install kernels` + `kernels lock` / `kernels download` at build time |
| `fa` optional extra for users | `hub_kernels` extra; `fa` kept for TE version pinning |
| Per-CUDA wheel matrix in CI | Hub resolves compatible build variant at runtime |
| Air-gapped clusters | Pre-download with `kernels download` + lockfile in image |

Recommended container change:

```dockerfile
RUN uv pip install "kernels>=0.11.0" \
 && kernels lock kernels-community/flash-attn2 kernels-community/liger-kernels \
 && kernels download  # bake into image for offline use
```

---

## Risks & Blockers

| Risk | Severity | Mitigation |
|---|---|---|
| EAGLE ring attention pins FA 2.8.x private API | **High** | Keep `fa` dep for speculative; Hub path for standard HF attention only |
| TE requires specific `flash-attn` version | **High** | Document mutual exclusion; TE path keeps compiled FA |
| FSDP2 + Hub kernel autograd | **Medium** | Parity tests on 1-GPU and multi-GPU finetune |
| Hub download at training start (latency) | **Low** | `kernels lock` + pre-download in container |
| `USE_HUB_KERNELS=0` env disables all Hub loading | **Low** | Document; default YES in transformers |
| Packed sequence + CP override forces SDPA | **Medium** | Existing limitation; Hub FA varlen may enable CP+packing later |
| tilelang / FA4 / apache-tvm-ffi conflict | **High** | Unrelated to Hub; existing pyproject constraint remains |

---

## Recommended Next Steps

1. **Phase 0 spike** — run one L0 finetune on branch with `kernels` installed, no `flash-attn` pip, `attn_implementation=kernels-community/flash-attn2`.
2. **Phase 1 PR** — hub-aware availability checks + kwargs passthrough (minimal, backward compatible).
3. **Decision point** — after EAGLE ring-attn parity test, decide whether speculative stays on compiled FA permanently.
4. **Container experiment** — build image without `flash-attn` source compile, Hub-only FA2; measure build time and runtime parity.

---

## References

- [huggingface/kernels repo](https://github.com/huggingface/kernels) — loader source, CLI, lockfiles
- [Upstream integration guide](https://github.com/huggingface/kernels/blob/main/kernel-builder/skills/cuda-kernels/references/huggingface-kernels-integration.md)
- [Upstream example script](https://github.com/huggingface/kernels/blob/main/kernel-builder/skills/cuda-kernels/scripts/huggingface_kernels_example.py)
- [Kernels docs — Quickstart](https://huggingface.co/docs/kernels/en/basic-usage)
- [Transformers — Loading kernels](https://huggingface.co/docs/transformers/main/kernel_doc/loading_kernels)
- [Transformers `hub_kernels.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py)
- [kernels-community/flash-attn2](https://huggingface.co/kernels-community/flash-attn2)
- [kernels-community/liger-kernels](https://huggingface.co/kernels-community/liger-kernels)
- AutoModel: `nemo_automodel/components/kernels/hub.py`, `nemo_automodel/_transformers/kernel_patches.py`, `nemo_automodel/components/models/common/utils.py` (`BackendConfig`)
