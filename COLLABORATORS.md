# Collaborator Guide — DTLP+PAS Experiments

For NeurIPS 2026 deadline. ALL code in this one repo (DTLP + backbones).

## Assignments

| Collaborator | Datasets | Baseline | Our current gap | GPU needed | Time budget | Backbone needed |
|---|---|---|---|---|---|---|
| **C1** | Yelp | CRAFT 0.727 | -45% | 24GB (4090/A100) | 3 days | TPNet+PAS |
| **C1** | WikiLink | CRAFT 0.755 | -14% | 24GB | 2 days | TPNet+PAS |
| **C2** | ML-20M | CRAFT 0.359 | -14% | 12-24GB | 1 day | TPNet+PAS + CRAFT+PAS |
| **C2** | Taobao | CRAFT 0.707 | -8% | 24GB | 2 days | TPNet+PAS |
| **C3** | tgbn-token | NAVIS 0.513 | unknown | any 8GB+ | 1 day | DTLP only (no backbone needed) |
| (Lead) | coin/comment/flight/GoogleLocal/thgl-software | close-gap | -0.9% to -25% | 4090 | hours each | various |

## One-time setup

```bash
# 1. Clone. ALL code (DTLP + TPNet/CRAFT backbone) included.
git clone https://github.com/AOoligei/DTLP /data/DTLP
cd /data/DTLP

# 2. Python env
conda create -n dtlp python=3.10 -y
conda activate dtlp
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn tqdm pytz requests
pip install py-tgb==2.2.0 tgb==1.2.0 tgb-seq
```

## Run your dataset

### C1 — Yelp (TPNet+PAS)
```bash
cd /data/DTLP/external/craft_v2
mkdir -p logs data
# jr mode (default best)
CUDA_VISIBLE_DEVICES=0 python -u train_link_prediction.py \
  --dataset_name Yelp --model_name TPNet --seed 42 \
  --batch_size 100 --learning_rate 1e-4 --num_epochs 30 --num_runs 1 \
  --patience 10 --gpu 0 --dataset_path ./data/ \
  --training_mode joint_random --test_interval_epochs 1 --load_best_configs \
  > logs/Yelp_jr_s42.log 2>&1 &
# sg mode (alternative)
CUDA_VISIBLE_DEVICES=1 python -u train_link_prediction.py \
  --dataset_name Yelp --model_name TPNet --seed 42 \
  --batch_size 100 --learning_rate 1e-4 --num_epochs 30 --num_runs 1 \
  --patience 10 --gpu 0 --dataset_path ./data/ \
  --training_mode decoupled_stopgrad --test_interval_epochs 1 --load_best_configs \
  > logs/Yelp_sg_s42.log 2>&1 &
```
**Target**: test MRR > 0.727. Data auto-downloads on first run.

### C1 — WikiLink (TPNet+PAS)
```bash
cd /data/DTLP/external/craft_v2
CUDA_VISIBLE_DEVICES=0 python -u train_link_prediction.py \
  --dataset_name WikiLink --model_name TPNet --seed 42 \
  --batch_size 100 --learning_rate 1e-4 --num_epochs 30 --num_runs 1 \
  --patience 10 --gpu 0 --dataset_path ./data/ \
  --training_mode joint_random --test_interval_epochs 1 --load_best_configs \
  > logs/WikiLink_jr_s42.log 2>&1 &
```
**Target**: test MRR > 0.755.

### C2 — ML-20M (TPNet+PAS AND CRAFT+PAS)
```bash
cd /data/DTLP/external/craft_v2
# TPNet+PAS
CUDA_VISIBLE_DEVICES=0 python -u train_link_prediction.py \
  --dataset_name ML-20M --model_name TPNet --seed 42 \
  --batch_size 100 --learning_rate 1e-4 --num_epochs 30 --num_runs 1 \
  --patience 10 --gpu 0 --dataset_path ./data/ \
  --training_mode joint_random --test_interval_epochs 1 --load_best_configs \
  > logs/ML-20M_TPNet_jr.log 2>&1 &
# CRAFT+PAS
CUDA_VISIBLE_DEVICES=1 python -u train_link_prediction.py \
  --dataset_name ML-20M --model_name CRAFT --seed 42 \
  --batch_size 200 --learning_rate 1e-4 --num_epochs 100 --num_runs 1 \
  --patience 10 --gpu 0 --dataset_path ./data/ \
  --training_mode joint_random --load_best_configs \
  > logs/ML-20M_CRAFT_jr.log 2>&1 &
```
**Target**: test MRR > 0.359.

### C2 — Taobao (TPNet+PAS)
```bash
cd /data/DTLP/external/craft_v2
CUDA_VISIBLE_DEVICES=0 python -u train_link_prediction.py \
  --dataset_name Taobao --model_name TPNet --seed 42 \
  --batch_size 100 --learning_rate 1e-4 --num_epochs 30 --num_runs 1 \
  --patience 10 --gpu 0 --dataset_path ./data/ \
  --training_mode joint_random --test_interval_epochs 1 --load_best_configs \
  > logs/Taobao_jr_s42.log 2>&1 &
```
**Target**: test MRR > 0.707.

### C3 — tgbn-token (DTLP only — no backbone)
```bash
cd /data/DTLP
for a in 0.1 0.3 0.5 0.7 0.9; do
  python -u src/run_node_pred_v2.py \
    --dataset tgbn-token --seed 42 --ema_alpha $a \
    --hidden_dim 512 --gpu 0 --epochs 100 \
    > logs/tgbn-token_a${a}.log 2>&1 &
done
```
**Target**: NDCG@10 > 0.513.

## Report

Please share:
- Final test MRR (or NDCG for tgbn-token)
- GPU used (card + hours)
- Best config (training_mode, seed)
- Log file

Post in team chat: `<dataset>_<model>_<mode>_s<seed>: best_test=0.XXX (N hours on <GPU>)`

## Troubleshooting

1. **EOFError on first run** (data download) → run once interactively and answer `y`, or wrap: `echo y | python ...`
2. **OOM on Yelp/WikiLink** → reduce `--batch_size` to 50 or 25
3. **TPNet+PAS "stuck" at model init** → waiting on neighbor sampler construction for 50M+ edge datasets (30-60 min first time, cached after).
4. **thgl-software errors** → use relative `--dataset_path datasets/` (tgb prepends PROJ_DIR to absolute paths — breaks them).
5. **"No such test sample"** for thgl-* → ensure your CRAFT_v2 has our fix (lookup key `(time, src, edge_type)`). Already applied in this repo.

## Known close-gap status (lead is pushing these)

| Dataset | Current | Target | Gap | ETA |
|---|---|---|---|---|
| GoogleLocal | CRAFT+PAS 0.6186 | 0.624 | -0.9% | hrs (sg mode training) |
| tgbl-comment | DTLP 0.904 | 0.917 | -1.3% | hrs (Trick B running) |
| tgbl-flight | TPNet+PAS 0.891 | 0.914 | -2.3% | hrs |
| tgbl-coin | TPNet+PAS 0.786 | 0.885 | -11% | hrs |
| thgl-software | DTLP 0.547 | 0.731 | -25% | hrs (just unblocked TPNet+PAS) |
| tgbn-trade | DTLP 0.79 | 0.863 | -8% | ceiling confirmed — possibly paper limitation |

