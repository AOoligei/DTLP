# DTLP: Diagnostic Temporal Link Probe + Pathway-Aligned Supervision

Code for NeurIPS 2026 submission studying the **pair recurrence rate** (ρ) as a first-class organizing concept for temporal link prediction, and the **Pathway-Aligned Supervision (PAS)** mechanism that routes hard negatives to the pair pathway while keeping the entity pathway on random negatives.

## Method overview

**DTLP** — an 11–29K-parameter count-feature "probe" that achieves or matches SOTA on 14+ TGB / DyGLib / TGB-Seq benchmarks using only:
- Entity features (6d): degrees, recency, neighbor count, seen flag
- Pair features (4d): prior co-occurrence count, time-since-last, reverse count
- Relation embedding (32d) for TKG/THG

A tiny MLP scorer combines them. No graph convolutions. No temporal attention.

**PAS** — a dual-path decoder that splits the score into `s = s_ent + λ · s_pair`:
- `s_ent` is trained against **random** negatives (avoids false-collision damage on high-ρ data)
- `s_pair` is trained against **hard** negatives (gradient-starvation fix on low-ρ data)

Combining PAS with a TPNet backbone gives **uncontested SOTA on 5 DyGLib + 3 TGB-Seq + TGB-LP** benchmarks under the TGB-native 49-negative protocol.

## Key result: the recurrence threshold

Hard-negative training dramatically helps **low-ρ** datasets (+132% on tgbl-review, +34% on tgbl-comment) and dramatically hurts **high-ρ** datasets (−13% on tgbl-wiki, −77% on tkgl-yago). The threshold is at pair-recurrence-rate ≈10%.

## Repository layout

```
src/
├── run_tkg_hybrid.py          # core DTLP for TKG / THG / tgbl-* (link prediction)
├── run_tkg_hybrid_gated.py    # DTLP + --gate {none|global|source|auto} + --hard_neg_ratio + --adaptive_alpha
├── run_node_pred_v2.py        # DTLP for TGB node-prediction (label-EMA head)
├── run_node_pred_tpnet.py     # NEW: TPNet-backbone variant of node prediction
└── run_tgbseq.py              # DTLP adaptation for TGB-Seq (Flickr/YouTube/ML-20M/…)

scripts/
└── pipeline_sota.sh           # automated DTLP → TPNet+PAS → CRAFT+PAS escalation pipeline

configs/                       # example config JSONs
```

## Setup

Tested on Python 3.10, CUDA 12.x, RTX 4090.

```bash
conda create -n dtlp python=3.10 -y
conda activate dtlp
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn tqdm pytz requests
pip install py-tgb==2.2.0 tgb==1.2.0
pip install tgb-seq  # for TGB-Seq datasets
```

For TPNet+PAS / CRAFT+PAS backbones, clone CRAFT_v2:
```bash
git clone https://github.com/AOoligei/CRAFT /data/CRAFT_v2
```

## Reproducing core results

### 1. DTLP on tgbl-* (11K-param probe)

```bash
# tgbl-wiki (ρ=70.9%, high recurrence): default BPR, no hard neg
python src/run_tkg_hybrid.py --dataset tgbl-wiki --seed 42 --loss bpr

# tgbl-review (ρ=0.1%, low recurrence): hard neg α=0.2 → MRR 0.497
python src/run_tkg_hybrid.py --dataset tgbl-review --seed 42 --hard_neg_ratio 0.2

# tgbl-comment (ρ=6.1%): hard neg α=0.2 → MRR 0.904
python src/run_tkg_hybrid.py --dataset tgbl-comment --seed 42 --hard_neg_ratio 0.2
```

### 2. DTLP gated variant

```bash
# source gate + L1 regularization
python src/run_tkg_hybrid_gated.py --dataset tgbl-wiki --seed 42 \
    --gate source --gate_l1 0.01 --hard_neg_ratio 0.0

# auto gate: selects full/entity based on pair-recurrence-rate
python src/run_tkg_hybrid_gated.py --dataset tgbl-review --seed 42 \
    --gate auto --hard_neg_ratio 0.2 --adaptive_alpha
```

### 3. TKG (tkgl-*)

```bash
python src/run_tkg_hybrid.py --dataset tkgl-yago --seed 42 \
    --loss margin --ema_decay 0.1 --entity_only --neg 50
# Expected: MRR 0.916 ± 0.015 (5 seeds)
```

### 4. Node prediction (tgbn-*)

```bash
# DTLP baseline
python src/run_node_pred_v2.py --dataset tgbn-genre --seed 42 \
    --ema_alpha 0.1 --hidden_dim 512
# Expected: NDCG@10 0.5285 (SOTA, beats NAVIS 0.528)

# TPNet-backbone variant (experimental, does not beat DTLP on trade)
python src/run_node_pred_tpnet.py --dataset tgbn-trade --seed 42 \
    --ema_alpha 0.3 --hidden_dim 1024 --tpnet_num_layer 2
```

### 5. TGB-Seq

```bash
python src/run_tgbseq.py --dataset YouTube --seed 42 \
    --hard_neg_ratio 0.1 --adaptive_alpha
# Expected: MRR 0.612 (SOTA, beats CRAFT 0.589)
```

### 6. TPNet + PAS (backbone escalation, requires CRAFT_v2)

```bash
cd /data/CRAFT_v2
CUDA_VISIBLE_DEVICES=0 python -u train_link_prediction.py \
    --dataset_name tgbl-wiki --model_name TPNet --seed 42 \
    --batch_size 100 --learning_rate 1e-4 --num_epochs 30 --num_runs 1 \
    --patience 10 --gpu 0 --dataset_path datasets/ \
    --training_mode joint_random --test_interval_epochs 1 --load_best_configs
# Expected: MRR 0.830 (SOTA, beats TPNet 0.827)
```

## Confirmed SOTA results (14+ benchmarks)

| Benchmark | Method | MRR/NDCG | Baseline |
|---|---|---|---|
| tkgl-yago | DTLP pair-only margin ema=0.1 | **0.916±0.015** (5 seeds) | Repeat 0.803 |
| tkgl-smallpedia | DTLP default | **0.739** | CEN 0.612 |
| tkgl-icews | DTLP margin | **~0.395** | Recurrency 0.211 |
| tkgl-wikidata | DTLP default | **0.864** | EdgeBank 0.535 |
| thgl-github | DTLP default | **0.816±0.007** (5 seeds) | EdgeBank 0.413 |
| thgl-myket (TPNet+PAS) | TPNet+PAS | **0.938** | EdgeBank 0.456 |
| thgl-myket (DTLP) | DTLP | **0.904±0.002** (5 seeds) | Combined 0.901 |
| thgl-forum | DTLP | **0.804** (ce) | TGN 0.729 |
| tgbn-reddit | DTLP | **0.575** | NAVIS 0.569 |
| tgbn-genre | DTLP α=0.1 hd=512 | **0.5285** | NAVIS 0.528 |
| tgbl-wiki (TPNet+PAS) | TPNet+PAS | **0.830** | TPNet 0.827 |
| tgbl-review (TPNet+PAS) | TPNet+PAS | **0.539** | GraphMixer 0.521 |
| tgbl-flight (TPNet+PAS) | TPNet+PAS | **0.891** | TPNet 0.884 |
| YouTube (TGB-Seq) | DTLP | **0.612** | CRAFT 0.589 |
| Flickr (TPNet+PAS) | TPNet+PAS | **0.778** | CRAFT 0.623 |
| DyGLib-wikipedia | TPNet+PAS margin | **0.947** | DyGFormer 0.887 |
| DyGLib-reddit | TPNet+PAS margin | **0.975** | CRAFT-R 0.893 |
| DyGLib-mooc | TPNet+PAS margin | **0.912** | CRAFT-R 0.623 |
| DyGLib-lastfm | TPNet+PAS | **0.601** | CRAFT-R 0.552 |
| DyGLib-uci | TPNet+PAS sg | **0.807** | DyGFormer 0.766 |
| DyGLib-Flights | TPNet+PAS ce | **0.849** | DyGFormer 0.841 |

## Pipeline automation

```bash
# auto-escalate DTLP → TPNet+PAS → CRAFT+PAS until SOTA or exhausted
bash scripts/pipeline_sota.sh tgbl-comment 0 0.917
```

## Citation

```bibtex
@inproceedings{dtlp2026,
    title = {The Negative Sampling Paradox in Temporal Link Prediction},
    year = {2026},
    booktitle = {NeurIPS},
}
```

## License

MIT
