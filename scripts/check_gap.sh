#!/bin/bash
# Auto-scan 11 gap datasets best TEST MRR/NDCG across 4 machines + A100.
# Usage: bash /data/chenlibin/TGB2/scripts/check_gap.sh

declare -A BASELINE=(
  [tgbl-comment]=0.917
  [tgbl-flight]=0.914
  [tgbl-coin]=0.885
  [tgbn-trade]=0.863
  [tgbn-token]=0.513
  [Taobao]=0.707
  [Yelp]=0.727
  [WikiLink]=0.755
  [ML-20M]=0.359
  [GoogleLocal]=0.624
  [thgl-software]=0.731
)

scan_test() {
  local host=$1
  local ds=$2
  local paths=$3
  local pattern="RESULT.*hybrid MRR|Test MRR:|Test NDCG:|average test mrr|test mrr,[^,]*\$"
  local cmd="grep -h -E '$pattern' $paths 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | sort -rn | head -1"
  if [ "$host" = "localhost" ]; then
    eval "$cmd"
  else
    ssh -o ConnectTimeout=5 $host "$cmd" 2>/dev/null
  fi
}

get_best_test() {
  local ds=$1
  local best=0
  local best_src=""

  # Short alias (strip tgbl-/tgbn-/thgl- prefix) — many logs use short names e.g. coin_* / comment_*
  local ds_short="${ds#tgbl-}"; ds_short="${ds_short#tgbn-}"; ds_short="${ds_short#thgl-}"
  declare -A LOG_PATHS=(
    [localhost]="/data/chenlibin/TGB2/logs/*${ds}*.log /data/chenlibin/TGB2/logs/*${ds_short}_*.log /data/chenlibin/CRAFT_v2/logs/*${ds}*.log /data/chenlibin/TGB2/logs/tgbseq_${ds_short}_*.log /data/chenlibin/TGB2/logs/tgbseq_${ds}_*.log"
    [my_new2]="/data/chenlibin/TGB2/logs/*${ds}*.log /data/chenlibin/TGB2/logs/*${ds_short}_*.log /data/chenlibin/CRAFT_v2/logs/*${ds}*.log"
    [my_new3]="/data/clb/logs/*${ds}*.log /data/clb/logs/*${ds_short}_*.log /data/clb/CRAFT_v2/logs/*${ds}*.log"
    [my_new4]="/data/clb/logs/*${ds}*.log /data/clb/logs/*${ds_short}_*.log /data/clb/CRAFT_v2/logs/*${ds}*.log"
    [a100-8-2]="/data/chenlibin/TGB2/logs/*${ds}*.log /data/chenlibin/TGB2/logs/*${ds_short}_*.log /data/chenlibin/CRAFT_v2/logs/*${ds}*.log"
  )
  # Fallback: scan local audit dirs (pulled from remote)
  declare -A AUDIT_PATHS=(
    [my_new2]="/data/chenlibin/TGB2/logs_audit/my_new2/*${ds}*.log /data/chenlibin/TGB2/logs_audit/my_new2_craft/*${ds}*.log"
    [my_new3]="/data/chenlibin/TGB2/logs_audit/my_new3/*${ds}*.log"
    [my_new4]="/data/chenlibin/TGB2/logs_audit/my_new4/*${ds}*.log"
    [a100-8-2]="/data/chenlibin/TGB2/logs_audit/a100/*${ds}*.log"
  )

  for host in localhost my_new2 my_new3 my_new4 a100-8-2; do
    local t=$(scan_test "$host" "$ds" "${LOG_PATHS[$host]}")
    t=${t:-0}
    if (( $(awk "BEGIN {print ($t > $best)}") )); then best=$t; best_src="$host"; fi
    # also try audit fallback
    if [ "${AUDIT_PATHS[$host]:-}" ]; then
      local ta=$(scan_test "localhost" "$ds" "${AUDIT_PATHS[$host]}")
      ta=${ta:-0}
      if (( $(awk "BEGIN {print ($ta > $best)}") )); then best=$ta; best_src="${host}(audit)"; fi
    fi
  done

  echo "${best}|${best_src}"
}

printf "%-16s | %-7s | %-12s | %-11s | %-7s\n" "Dataset" "SOTA" "Best TEST" "Source" "Gap"
printf -- "-----------------+---------+--------------+-------------+--------\n"

for ds in "${!BASELINE[@]}"; do
  r=$(get_best_test $ds)
  b=${r%%|*}
  src=${r##*|}
  s=${BASELINE[$ds]}

  if [ "$b" = "0" ]; then
    gap="N/A"
    marker="(no test)"
    src="-"
  else
    gap=$(awk "BEGIN {printf \"%+.1f%%\", ($b - $s) / $s * 100}")
    if (( $(awk "BEGIN {print ($b >= $s)}") )); then marker="✅ SOTA"; else marker=""; fi
  fi

  printf "%-16s | %-7s | %-12s | %-11s | %-7s %s\n" "$ds" "$s" "$b" "$src" "$gap" "$marker"
done
echo ""
echo "Only final TEST MRR/NDCG shown. Run bash $0 to refresh."
