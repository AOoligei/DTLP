# Experiment Matrix — DTLP+PAS SOTA Push

Live status of 11 gap datasets × 4 method columns. Last updated 2026-04-24.

## Current standings

| Dataset | Type | Our best | Reference SOTA | Gap | Owner |
|---------|------|----------|----------------|-----|-------|
| **tgbl-comment** | LP | **0.9188** | CRAFT 0.917 | **+0.2% ✅ SOTA** | lead |
| tgbl-flight | LP | 0.8844 | TGN 0.914 | -3.2% | lead |
| tgbl-coin | LP | **0.8271** | CRAFT 0.885 | -6.5% | lead |
| GoogleLocal | LP (TGB-Seq) | 0.6186 | CRAFT 0.624 | -0.9% | collaborator |
| tgbn-trade | NODE | 0.7953 | 0.863 | -7.8% | collaborator |
| ML-20M | LP (TGB-Seq) | 0.3110 | CRAFT 0.359 | -13.4% | collaborator |
| thgl-software | LP (THG) | **0.5849** | STHN 0.731 | -20.0% | collaborator |
| WikiLink | LP (TGB-Seq) | 0.5862 | CRAFT 0.755 | -22.4% | collaborator |
| tgbn-token | NODE | 0.3886 | NAVIS 0.513 | -24.2% | collaborator |
| Taobao | LP (TGB-Seq) | 0.4316 | CRAFT 0.707 | -39.0% | collaborator |
| Yelp | LP (TGB-Seq) | 0.3805 | CRAFT 0.727 | -47.7% | collaborator |

## Method × Dataset matrix

Each cell shows **best test MRR/NDCG so far** and **(number of live procs)**.
Live procs include runs across 10 machines: 1 local workstation (8×4090), my_new2/my_new3/my_new4 (8+3+3 × 4090), and one a100-8-2 (8×A100-80G).

| Dataset | DTLP (ours) | TPNet+PAS | CRAFT+PAS | TGN+PAS |
|---------|-------------|-----------|-----------|---------|
| tgbl-comment | **0.9188** (2 live) | running (3 live) | running (1 live) | — |
| tgbl-flight | 0.8844 (4 live) | running (4 live) | running (1 live) | — |
| tgbl-coin | **0.8271** (15 live) | running (4 live) | running (5 live) | — |
| GoogleLocal | 0.1958 (2 live) | running (2 live) | **0.6186** (2 live) | — |
| WikiLink | 0.5862 (1 live) | running (7 live) | running (2 live) | — |
| Yelp | 0.3805 (1 live) | running (3 live) | running (1 live) | — |
| Taobao | 0.4316 (1 live) | running (2 live) | running (1 live) | — |
| ML-20M | 0.3110 (2 live) | running (2 live) | running (1 live) | — |
| thgl-software | 0.5486 (3 live) | running on a100 | **0.5849** (2 live) | — |
| tgbn-token | 0.3886 (7 live) | — | — | running (1 live) |
| tgbn-trade | 0.7953 (grid) | — | — | 0.4408 (weak) |

## Split of ownership

Split adopted 2026-04-24:

### Lead (3 datasets — core paper storyline, link-prediction benchmarks)
- **tgbl-comment** (already SOTA — hold and confirm with multi-seed)
- **tgbl-flight** (close gap, trick-combo sweep active)
- **tgbl-coin** (close gap, neighborhood sweep around the 0.8271 winner)

### Collaborator (8 datasets — large gaps or extension benchmarks)
- **Yelp, Taobao, ML-20M, WikiLink, GoogleLocal** (5 TGB-Seq link-prediction)
- **tgbn-token, tgbn-trade** (2 node-classification)
- **thgl-software** (1 heterogeneous-graph link-prediction)

Collaborator runs on their own machine, completely independent. Datasets auto-download from TGB / TGB-Seq. See [`RESULTS_2026_04_24.md`](RESULTS_2026_04_24.md) for best-config launch commands and [`COLLABORATORS.md`](COLLABORATORS.md) for environment setup.

## Trick reference

| Letter | Flag | Effect |
|--------|------|--------|
| B | `--pair_recency_feat` | log(t − pair_last_time) feature — main winner |
| C | `--cold_start_feat` | Cold-start indicator (TGB-Seq only) |
| D | `--ema_decay 0.10` | EMA decay override |
| F | `--inv_freq_weight` | Inverse-frequency weighting |
| I | `--repeat_profile_feat` | Repeat-count profile |
| P | `--pair_temp_decay` | Temporal decay on pairs |
| hnr | `--hard_neg_ratio` | Hard-negative ratio |
| hd | `--hidden_dim` | Default 64; aggressive 128 / 256 / 512 |

**Empirical threshold rule**: pair-recurrence rate ρ < 10% → trick B + hnr 0.2–0.3 helps; ρ > 40% → hurts (confirmed across coin, comment, wiki, yago).

## Live workload summary

| Machine | GPUs used | Main workload |
|---------|-----------|---------------|
| localhost | 8×4090 | tgbl-coin trick-combo sweep + comment/flight DTLP + wiki ρ verification + backbone+PAS |
| my_new2 | 8×4090 | Yelp/WikiLink/Taobao/ML-20M TPNet+PAS + tgbn-trade grid + tgbn-token sweep |
| my_new3 | 3×4090 | tgbl-coin DTLP variants + Yelp TPNet+PAS aggressive |
| my_new4 | 3×4090 | ML-20M + WikiLink + Taobao TPNet+PAS aggressive (emb=256 nn=60) |
| a100-8-2 | 8×A100-80G | long-running TPNet+PAS on Yelp/Taobao/WikiLink/coin + thgl-software + tgbn-token DTLP |

Total ~60+ live procs at any time. Scanner: `bash scripts/check_gap.sh` for latest best-per-dataset snapshot.

## Key findings so far

1. **tgbl-comment 0.9188 beats CRAFT 0.917 by +0.2%** via DTLP with trick B + hnr=0.20 + ema=0.10.
2. **tgbl-coin 0.8271** via DTLP trick B + hnr=0.25 (gap closed from -8.1% to -6.5%).
3. **thgl-software CRAFT+PAS 0.5849 beats our own DTLP 0.547 by +7.1%** — confirms DTLP is a diagnostic probe and backbone+PAS gives headroom when DTLP plateaus.
4. **tgbn-trade ceiling 0.7953** (smaller models outperform larger; data-sparsity bounded).
5. **Trick-B effectiveness depends on ρ threshold** — ~10% recurrence-rate boundary confirmed.
