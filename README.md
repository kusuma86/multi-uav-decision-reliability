# Decision Reliability in Multi-UAV Task Allocation under Imperfect Capability Estimation

**Version:** 1.0.0 release candidate — public repository URL and archival DOI not yet assigned.

Reproducibility materials for a simulation-based manuscript. The data are **synthetic**; no physical UAV flight experiments or measured battery discharge curves are included.

## Repository structure

| Path | Purpose |
|---|---|
| `experiments/01_initial_factorial.py` | Seven scenarios, eight policy combinations, 2,880 evaluations; initial allocation and factorial effects |
| `experiments/02_decision_reliability_audit.py` | Decision-level eligibility errors and missed feasible tasks, 2,880 evaluations |
| `experiments/03_initial_matched_oracle.py` | Supplementary initial-simulator matched-oracle audit, 1,440 evaluations |
| `experiments/04_resource_constrained.py` | Persistent battery and position states, 120 worlds, three policies, 360 evaluations, policy-matched perfect-information comparator |
| `notebooks/` | Equivalent cleaned Jupyter notebooks |
| `results/` | Archived generated outputs, one ZIP per experiment |
| `configs/seeds.json` | Base seed, scenario/world offsets, and bootstrap seeds |
| `docs/manuscript_numeric_audit.csv` | Comparison of 162 manuscript table values with rerun outputs |
| `docs/AUDIT_STATUS.json` | Scope and limitations of the independent CLI audit |

**Important:** Experiment 03's initial-simulator oracle is not the resource-constrained perfect-information comparator in Experiment 04. The initial oracle is an idealized per-step reference; Experiment 04 allows policy-dependent battery and position trajectories.

## Install

Use Python 3.11 or later in an isolated environment. These pinned package versions reflect the local audit environment; a fresh-environment installation has **not** been independently validated.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Reproduce the experiments

Run from the repository root, in this order:

```bash
python experiments/01_initial_factorial.py
python experiments/02_decision_reliability_audit.py
python experiments/03_initial_matched_oracle.py
python experiments/04_resource_constrained.py
python scripts/check_results.py
```

Scripts write to their own `*_Results/` folders in the current working directory (or `/content/` when running in Colab). Results ZIP archives are provided for comparison. Run times may be significant for the bootstrap analyses.

## Seed and scenario identifiers

The base seed is `20261025`. See `configs/seeds.json` for formulas and offsets; scenario iteration order must be preserved. The negative-bias scenario with bias -13 was renamed from `shift_overconfidence` to `moderate_pessimistic_shift` without changing parameters or scenario order. The severe-bias scenario retains the code identifier `severe_shift`, displayed as *Severe pessimistic shift* in the manuscript.

## Validation evidence and limitations

- Four extracted experiment scripts completed in local reruns.
- All 12 CSV files from experiments 01–03 matched earlier output numerically after scenario-label normalization.
- Resource-constrained experiment generated 360 policy-evaluation records.
- A separate table-level audit found 162/162 displayed values in manuscript Tables 4–8 consistent with rerun outputs after two-decimal rounding.
- **Not yet independently verified:** fresh-environment installation, all figure confidence-interval coordinates, and external reproducibility on another operating system.

The results do not establish operational UAV safety, physical energy consumption, or performance guarantees outside the synthetic model.

## Citation, license and release

See `CITATION.cff` and `LICENSE`. **Before making this repository public**, replace the placeholder repository URL in `CITATION.cff`, confirm copyright and coauthor approval of the proposed MIT License, and tag the final commit `v1.0.0`. An archival DOI can then be obtained through Zenodo. Do not add a DOI to the manuscript until it exists.
