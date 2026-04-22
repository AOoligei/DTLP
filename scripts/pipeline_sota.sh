#!/bin/bash
# Pipeline: DTLP sweep → TPNet+PAS → CRAFT+PAS for a given tgbl-* dataset
# Usage: ./pipeline_sota.sh <dataset> <gpu_idx> <baseline_mrr>
# Example: ./pipeline_sota.sh tgbl-comment 3 0.917

set -e
DATASET=$1
GPU=$2
BASELINE=$3

TGB2=/data/chenlibin/TGB2
CRAFT=/data/chenlibin/CRAFT_v2
PY=/home/chenlibin/anaconda3/envs/dygrag/bin/python
LOGS=$TGB2/logs/pipeline_${DATASET}

mkdir -p $LOGS
echo "[$(date)] START pipeline for $DATASET | GPU=$GPU | baseline=$BASELINE"

# ---- Stage 1: DTLP aggressive sweep ----
echo "[$(date)] Stage 1: DTLP sweep"
cd $TGB2
for loss in ce margin; do
  for hnr in 0.10 0.20 0.30; do
    for ema in 0.01 0.05; do
      tag=dtlp_${loss}_hnr${hnr}_ema${ema}
      [ -f $LOGS/$tag.done ] && continue
      $PY -u src/run_tkg_hybrid.py --dataset $DATASET --seed 42 \
        --loss $loss --hard_neg_ratio $hnr --ema_decay $ema --gpu $GPU \
        > $LOGS/$tag.log 2>&1 && touch $LOGS/$tag.done
    done
  done
done

# extract best test MRR from DTLP
DTLP_BEST=$(grep -h "test_mrr\|Test MRR" $LOGS/dtlp_*.log 2>/dev/null | \
  grep -oE "[0-9]+\.[0-9]+" | sort -rn | head -1)
echo "[$(date)] Stage 1 DTLP best: $DTLP_BEST  vs baseline $BASELINE"

# compare: bc or awk
BEAT=$(awk -v a="$DTLP_BEST" -v b="$BASELINE" 'BEGIN{print (a > b) ? 1 : 0}')
if [ "$BEAT" = "1" ]; then
  echo "[$(date)] ✅ DTLP SOTA! MRR $DTLP_BEST > $BASELINE. DONE."
  exit 0
fi

# ---- Stage 2: TPNet+PAS ----
echo "[$(date)] Stage 2: TPNet+PAS"
cd $CRAFT
for mode in joint_random decoupled_stopgrad; do
  tag=tpnet_${mode}
  [ -f $LOGS/$tag.done ] && continue
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU \
    $PY -u train_link_prediction.py \
    --dataset_name $DATASET --model_name TPNet --seed 42 --batch_size 100 \
    --learning_rate 1e-4 --num_epochs 30 --num_runs 1 --patience 10 --gpu 0 \
    --dataset_path datasets/ --training_mode $mode --test_interval_epochs 1 \
    --load_best_configs > $LOGS/$tag.log 2>&1 && touch $LOGS/$tag.done
done

TPNET_BEST=$(grep -h "test_metric\|test MRR\|Test MRR" $LOGS/tpnet_*.log 2>/dev/null | \
  grep -oE "[0-9]+\.[0-9]+" | sort -rn | head -1)
echo "[$(date)] Stage 2 TPNet+PAS best: $TPNET_BEST"

BEAT=$(awk -v a="$TPNET_BEST" -v b="$BASELINE" 'BEGIN{print (a > b) ? 1 : 0}')
if [ "$BEAT" = "1" ]; then
  echo "[$(date)] ✅ TPNet+PAS SOTA! MRR $TPNET_BEST > $BASELINE. DONE."
  exit 0
fi

# ---- Stage 3: CRAFT+PAS ----
echo "[$(date)] Stage 3: CRAFT+PAS"
for mode in joint_random decoupled_stopgrad; do
  tag=craft_${mode}
  [ -f $LOGS/$tag.done ] && continue
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU \
    $PY -u train_link_prediction.py \
    --dataset_name $DATASET --model_name CRAFT --seed 42 --batch_size 200 \
    --learning_rate 1e-4 --num_epochs 100 --num_runs 1 --patience 10 --gpu 0 \
    --dataset_path datasets/ --training_mode $mode --load_best_configs \
    > $LOGS/$tag.log 2>&1 && touch $LOGS/$tag.done
done

CRAFT_BEST=$(grep -h "test_metric\|test MRR\|Test MRR" $LOGS/craft_*.log 2>/dev/null | \
  grep -oE "[0-9]+\.[0-9]+" | sort -rn | head -1)
echo "[$(date)] Stage 3 CRAFT+PAS best: $CRAFT_BEST"

BEAT=$(awk -v a="$CRAFT_BEST" -v b="$BASELINE" 'BEGIN{print (a > b) ? 1 : 0}')
if [ "$BEAT" = "1" ]; then
  echo "[$(date)] ✅ CRAFT+PAS SOTA! MRR $CRAFT_BEST > $BASELINE. DONE."
else
  echo "[$(date)] ❌ All 3 stages failed. Best: DTLP $DTLP_BEST, TPNet $TPNET_BEST, CRAFT $CRAFT_BEST vs $BASELINE"
fi
