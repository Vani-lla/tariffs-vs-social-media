# Design

## Context

See `proposal.md` (Why/What) and `specs/causal-discovery/spec.md` for behavioral requirements.

This repository already produces merged daily market/media data in `clean_out/merged/daily.csv` and Python modules under `merging_data/`. The new component must be production-usable in code and script form, while staying compatible with existing data conventions and OpenSpec workflow.

## Goals / Non-Goals

**Goals:**
- Provide a typed, reusable PCMCI analysis API built on Tigramite.
- Default to PCMCI+ with contemporaneous effects (`tau_min=0`) and support classic PCMCI (`tau_min=1`).
- Add an explicit adapter for mapping `daily.csv` columns into standardized causal variable names.
- Return significant links in a stable tabular schema and support optional graph export to PNG.

**Non-Goals:**
- Building a full ETL orchestration layer.
- Hyperparameter auto-tuning beyond exposed configuration fields.
- Enforcing domain semantics of each proxy series at design level.

## Decisions

### 1) Introduce a dedicated module `merging_data/causality.py`
- **Decision:** Keep causal discovery logic in one module with clear interfaces (`PCMCIConfig`, `DailyColumnAdapter`, `PCMCIAnalyzer`).
- **Why:** Enables notebook, script, and package-level reuse without coupling to one execution path.
- **Alternatives considered:**
  - Inline code in notebooks/scripts only: rejected due to poor testability and maintainability.
  - Embedding into existing merge pipeline functions: rejected to avoid mixing data prep and model logic.

### 2) Use `PCMCIConfig` as a single source of run-mode truth
- **Decision:** Keep `use_pcmciplus=True` and `tau_min=0` as default; enforce `tau_min=1` for classic PCMCI.
- **Why:** Makes the blocking policy explicit and validated before runtime.
- **Alternatives considered:**
  - Derive mode only from `tau_min`: rejected because intent is less explicit and easier to misuse.

### 3) Add explicit daily-column adapter with strict validation
- **Decision:** `DailyColumnAdapter.to_standard()` maps raw columns to standardized names and fails on missing inputs/NaN.
- **Why:** Avoids silent schema drift and gives predictable causal feature contract.
- **Alternatives considered:**
  - Implicit passthrough if columns already "look similar": rejected because it hides upstream quality issues.

### 4) Parse outputs into a canonical dataframe
- **Decision:** Convert Tigramite `val_matrix` + `p_matrix` into columns:
  `cause_var`, `effect_var`, `lag_tau`, `mci_val`, `p_value`, filtered by `alpha_level`, sorted by `|mci_val|`.
- **Why:** Makes downstream analysis/reporting reproducible and tool-agnostic.
- **Alternatives considered:**
  - Returning raw matrices only: rejected due to low usability for analysts.

### 5) Keep visualization optional and file-based
- **Decision:** `plot_causal_graph()` accepts result dict and writes PNG when requested.
- **Why:** Keeps batch workflows lightweight while still supporting exploratory inspection.
- **Alternatives considered:**
  - Always generating plots: rejected due to unnecessary runtime overhead.

### 6) Provide lightweight CLI runner in `scripts/run_pcmci.py`
- **Decision:** Add a small script for reproducible local runs on `clean_out/merged/daily.csv`.
- **Why:** Enables quick validation without notebook coupling.
- **Alternatives considered:**
  - No CLI runner: rejected because verification and onboarding become slower.

## Risks / Trade-offs

- [Proxy mismatch risk] Mapping `delta_vix` and `delta_yield_10y` uses available daily proxies by default. -> Mitigation: adapter mapping is explicit and overrideable.
- [Library API drift] Tigramite plotting API may change across versions. -> Mitigation: encapsulate plotting in one method and pin dependency in `requirements.txt`.
- [Data sparsity after cleaning] Dropping NaN rows can shorten series and reduce power. -> Mitigation: keep strict validation in module, and make CLI cleaning step explicit.
- [Interpretation risk] Significant links are statistical, not guaranteed structural causality. -> Mitigation: keep output transparent (`mci_val`, `p_value`, `lag_tau`) for domain review.

## Migration Plan

1. Install dependencies from `requirements.txt`.
2. Run unit tests for causal module.
3. Execute `scripts/run_pcmci.py` on `clean_out/merged/daily.csv`.
4. Review produced CSV and optional PNG.
5. If needed, override adapter mapping to project-specific columns and rerun.

Rollback is file-level: remove/disable `merging_data/causality.py`, `scripts/run_pcmci.py`, and related exports.

