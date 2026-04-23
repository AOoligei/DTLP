# Collaborator Guide — DTLP+PAS Experiments

For NeurIPS 2026 deadline reproduction. Each collaborator gets a subset of datasets to train.

## Assignment (suggested)

| Collaborator | Dataset | Baseline | Our gap | Expected time |
|---|---|---|---|---|
| C1 | Yelp | CRAFT 0.727 | -45% | 3 days |
| C1 | WikiLink | CRAFT 0.755 | -14% | 2 days |
| C2 | ML-20M | CRAFT 0.359 | -14% | 1 day |
| C2 | Taobao | CRAFT 0.707 | -8% | 2 days |
| C3 | tgbn-token | NAVIS 0.513 | unknown | 1 day |
| (self) | coin / comment / flight / GoogleLocal / thgl-software | close-gap | small | hours |

## Setup

```bash
# Clone both repos
git clone https://github.com/AOoligei/DTLP /data/DTLP
git clone https://github.com/AOoligei/CRAFT /data/CRAFT_v2

# Python env
conda create -n dtlp python=3.10 -y
conda activate dtlp
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn tqdm pytz requests
pip install py-tgb==2.2.0 tgb==1.2.0 tgb-seq
```

## Download data

TGB-Seq (Taobao/Yelp/WikiLink/ML-20M/YouTube/Flickr/GoogleLocal/Patent) — auto-downloaded by `tgb-seq` on first use. Cache at `$dataset_path/<dataset>/`.

```bash
# Test download (just run once, answer y to prompt)
cd /data/CRAFT_v2 && python -c "from tgb_seq.LinkPred.dataloader import TGBSeqLoader; TGBSeqLoader('Yelp', '/data/CRAFT_v2/data/')"
```

TGB-LP (tgbl-coin/comment/flight/wiki/review) and TGB-NodePred (tgbn-trade/genre/reddit/token) and TGB-THG (thgl-*) auto-download via pip `tgb`.

```bash
# Python one-liner to pre-download thgl-software (adapt for other datasets)
python -c "
import sys; sys.path.insert(0, '/data/CRAFT_v2')
import builtins
builtins.input = lambda *a, **k: 'y'
from tgb.linkproppred.dataset import LinkPropPredDataset
ds = LinkPropPredDataset(name='thgl-software', root='datasets/', preprocess=True)
"
```

## Launch commands per dataset (close-gap datasets not listed — self)

### Yelp (TPNet+PAS, large)
```bash
cd /data/CRAFT_v2
for mode in joint_random decoupled_stopgrad; do
  CUDA_VISIBLE_DEVICES=0 python -u train_link_prediction.py \
    --dataset_name Yelp --model_name TPNet --seed 42 \
    --batch_size 100 --learning_rate 1e-4 --num_epochs 30 --num_runs 1 \
    --patience 10 --gpu 0 --dataset_path /data/CRAFT_v2/data/ \
    --training_mode $mode --test_interval_epochs 1 --load_best_configs \
    > logs/Yelp_${mode}.log 2>&1 &
done
```
Target: > CRAFT 0.727.

### WikiLink (TPNet+PAS)
```bash
cd /data/CRAFT_v2
CUDA_VISIBLE_DEVICES=0 python -u train_link_prediction.py \
  --dataset_name WikiLink --model_name TPNet --seed 42 \
  --batch_size 100 --learning_rate 1e-4 --num_epochs 30 --num_runs 1 \
  --patience 10 --gpu 0 --dataset_path /data/CRAFT_v2/data/ \
  --training_mode joint_random --test_interval_epochs 1 --load_best_configs \
  > logs/WikiLink_jr.log 2>&1 &
```
Target: > CRAFT 0.755.

### ML-20M (CRAFT+PAS, try both modes)
```bash
cd /data/CRAFT_v2
for mode in joint_random decoupled_stopgrad; do
  CUDA_VISIBLE_DEVICES=0 python -u train_link_prediction.py \
    --dataset_name ML-20M --model_name CRAFT --seed 42 \
    --batch_size 200 --learning_rate 1e-4 --num_epochs 100 --num_runs 1 \
    --patience 10 --gpu 0 --dataset_path /data/CRAFT_v2/data/ \
    --training_mode $mode --load_best_configs \
    > logs/ML-20M_CRAFT_${mode}.log 2>&1 &
done
```
Target: > CRAFT 0.359.

### Taobao (TPNet+PAS)
```bash
cd /data/CRAFT_v2
CUDA_VISIBLE_DEVICES=0 python -u train_link_prediction.py \
  --dataset_name Taobao --model_name TPNet --seed 42 \
  --batch_size 100 --learning_rate 1e-4 --num_epochs 30 --num_runs 1 \
  --patience 10 --gpu 0 --dataset_path /data/CRAFT_v2/data/ \
  --training_mode joint_random --test_interval_epochs 1 --load_best_configs \
  > logs/Taobao_jr.log 2>&1 &
```
Target: > CRAFT 0.707.

### tgbn-token (DTLP α sweep)
```bash
cd /data/DTLP
for a in 0.1 0.3 0.5 0.7 0.9; do
  python -u src/run_node_pred_v2.py \
    --dataset tgbn-token --seed 42 --ema_alpha $a \
    --hidden_dim 512 --gpu 0 --epochs 100 \
    > logs/tgbn-token_a${a}.log 2>&1 &
done
```
Target: > NAVIS 0.513.

## Reporting results

Please share:
- Best test MRR per config
- Full log file
- GPU used (model, hours)

Upload to Google Drive / shared folder: <TBD by team lead>.

## If issues

1. **"EOFError: EOF when reading a line"** on data download → set `builtins.input = lambda *a, **k: 'y'` before import
2. **CUDA OOM on large datasets** → reduce `--batch_size` to 50 or 25
3. **thgl-software protocol mismatch** → use relative `--dataset_path datasets/` (CRAFT_v2's tgb prepends PROJ_DIR which breaks absolute paths)
4. **TPNet+PAS stuck at model init** → wait — neighbor sampler construction for 50M+ edge datasets takes 30-60 min before first log output

## Hardware recommendations

- **Yelp/WikiLink/Taobao**: need 24GB VRAM (4090 or A100). 2-3 days training.
- **ML-20M**: fits on 12GB (3090). 1 day.
- **tgbn-token**: CPU-bound warmup (51M edges). Any GPU works, but 1-2 days total.
- **Close-gap LP datasets**: 4090 ~3-6 hours each.
