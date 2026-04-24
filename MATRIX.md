# Experiment Matrix — 2026-04-24

**Live status of the DTLP+PAS SOTA push across 11 gap datasets × 4 method columns.**

Legend:
- `✅ SOTA` — beats reference SOTA
- `🟡` — running, gap remaining
- `❌` — no active proc (dead or not yet started)
- `—` — N/A (method doesn't apply to task type)

Numbers below = **best Test MRR (link-pred) / Test NDCG (node-pred)** across all hosts.
Parenthesis shows alive procs count (local + 4 remote machines).

---

## The Matrix

| Dataset | Type | Reference SOTA | DTLP (our method) | TPNet+PAS | CRAFT+PAS | TGN+PAS |
|---------|------|----------------|--------------------|-----------|-----------|---------|
| **tgbl-comment** | LP | CRAFT 0.917 | **0.9188** ✅ SOTA (2 alive) | pending (3 alive) | pending (1 alive) | — |
| **tgbl-flight** | LP | TGN 0.914 | 0.8844 🟡 (4 alive) | pending (4 alive) | pending (1 alive) | — |
| **tgbl-coin** | LP | CRAFT 0.885 | **0.8271** 🟡 -6.5% (15 alive — aggressive) | pending (4 alive) | pending (5 alive) | — |
| **GoogleLocal** | LP (TGB-Seq) | CRAFT 0.624 | 0.1958 ❌ DTLP weak | pending (2 alive) | **0.6186** 🟡 -0.9% (2 alive) | — |
| **WikiLink** | LP (TGB-Seq) | CRAFT 0.755 | 0.5862 🟡 -22% | pending (7 alive) | pending (2 alive) | — |
| **Yelp** | LP (TGB-Seq) | CRAFT 0.727 | 0.3805 🟡 -48% | pending (3 alive) | pending (1 alive) | — |
| **Taobao** | LP (TGB-Seq) | CRAFT 0.707 | 0.4316 🟡 -39% (1 alive) | pending (2 alive) | pending (1 alive) | — |
| **ML-20M** | LP (TGB-Seq) | CRAFT 0.359 | 0.3110 🟡 -13% (2 alive) | pending (2 alive) | pending (1 alive) | — |
| **thgl-software** | LP (THG) | STHN 0.731 | 0.5486 🟡 -25% (3 alive) | pending (1 alive on a100, OOM-prone on 4090) | **0.5849** 🟡 -20% (2 alive) | — |
| **tgbn-token** | NODE | NAVIS 0.513 | 0.3886 🟡 -24% (7 alive) | — | — | pending (1 alive) |
| **tgbn-trade** | NODE | 0.863 | 0.7953 🟡 -7.8% (ceiling) | — | — | 0.4408 (≪ DTLP, worth multi-seed) |

---

## Currently running (total ~60+ procs across 10 machines)

| Machine | GPUs | Main workload |
|---------|------|---------------|
| localhost (8×4090) | 0-7 | tgbl-coin trickB sweep (hnr∈{0.22,0.25,0.28} + trickBI hd128/256), tgbl-comment CRAFT+PAS v4, tgbl-flight CRAFT+PAS, tgbl-comment trickBI, thgl-software CRAFT+PAS s42, wiki trickB verify (2 seeds) |
| my_new2 (8×4090) | 0-7 | Yelp/WikiLink TPNet+PAS jr+sg, Yelp CRAFT+PAS, ML-20M CRAFT+PAS, tgbn-token DTLP sweep, tgbn-trade grid2 (220 configs serial) |
| my_new3 (3×4090) | 0-2 | 6 tgbl-coin DTLP variants (ce+margin hnr∈{0, 0.1, 0.15}), coin hd128, Yelp TPNet+PAS aggressive emb=256 |
| my_new4 (3×4090) | 0-2 | ML-20M + WikiLink + Taobao TPNet+PAS aggressive (emb=256 nn=60) |
| a100-8-2 (8×A100-80G) | 0-7 | Yelp/Taobao/WikiLink/tgbl-coin TPNet+PAS 1.5-2 days; GoogleLocal TPNet aggr; GoogleLocal CRAFT+PAS s46; thgl-software TPNet+PAS (OOM-prone); coin emb=384 s43; tgbn-token DTLP a=0.9 h=1024 |

---

## What's missing (priority for new runs)

### 🔴 High priority — gaps we can't close alone

1. **Yelp -48%** — 我们 backbone+PAS + DTLP 都 weak 状态；需要新 backbone (TGN/DyGFormer/GraphMixer+PAS) or 大容量激进配置
2. **Taobao -39%** — 同上
3. **WikiLink -22%** — 多次 sweep 卡住
4. **tgbn-token -24%** — 小模型多 seed + 更大 alpha×hidden grid（跑得快）
5. **ML-20M -13%** — 可能 benefit from 大 GPU 上 emb=384+ 配置

### 🟡 Medium priority — 我们自己在推

- tgbl-coin (-6.5%)，tgbl-flight (-3.2%)，thgl-software (-20%)：trick-combo sweep 在跑
- GoogleLocal (-0.9%)：基本饱和

### 🟢 Already SOTA

- tgbl-comment 0.9188 (+0.2% over CRAFT 0.917)

---

## Trick combinations in use

| Letter | Flag | What it adds |
|--------|------|-------------|
| **B** | `--pair_recency_feat` | log(t - pair_last_time) as feature — **核心 winner** |
| **C** | `--cold_start_feat` | Cold-start indicator (TGB-Seq only) |
| **D** | `--ema_decay 0.10` | EMA decay override |
| **F** | `--inv_freq_weight` | Inverse-frequency weighting |
| **I** | `--repeat_profile_feat` | Repeat-count profile |
| **P** | `--pair_temp_decay` | Temporal decay on pairs |
| hnr | `--hard_neg_ratio` | Hard negative sampling ratio; **low-ρ datasets 0.2-0.3, high-ρ < 0.05** |
| hd | `--hidden_dim` | Default 64; aggressive 128/256/512 |

**Threshold rule confirmed**: pair-recurrence rate ρ < 10% → trick B + high hnr helps; ρ > 40% → hurts.

---

## Collaborator split

**Handed to external collaborator** (独立机器，不共享 GPU)：
- Yelp, Taobao, tgbn-token, WikiLink, ML-20M (5 大 gap 数据集)
- 自由发挥：换 backbone / 换 loss / 加新 feature
- Repo: clone `AOoligei/DTLP`, see `COLLABORATORS.md`

**Kept on our side**:
- tgbl-comment (SOTA, 保持+推高), tgbl-flight, tgbl-coin, thgl-software, GoogleLocal, tgbn-trade
- 加上 backbone+PAS 矩阵的全部监控

---

## Launch commands (reproducible)

See [`RESULTS_2026_04_24.md`](RESULTS_2026_04_24.md) for each best-config command.

Quick scan of current state: `bash scripts/check_gap.sh` (shows best per dataset across hosts with gap to SOTA).

---

_Auto-updated 2026-04-24 by monitoring harness._
