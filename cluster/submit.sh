#!/usr/bin/env bash
# Submit a chain of pre-training jobs for one config; each link starts when the previous one ends
# for any reason, so a run longer than the walltime resumes from its last checkpoint.
#
#   bash cluster/submit.sh configs/lpl.yaml            # 2 links (default)
#   LINKS=1 RUN_NAME=calib_lpl OVERRIDE=pretrain.max_steps=300 bash cluster/submit.sh configs/lpl.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:?usage: submit.sh <config.yaml>}
NAME=$(basename "$CONFIG" .yaml)
LINKS=${LINKS:-2}
VARS="CONFIG=$CONFIG"
[ -n "${RUN_NAME:-}" ] && VARS="$VARS,RUN_NAME=$RUN_NAME"
[ -n "${OVERRIDE:-}" ] && VARS="$VARS,OVERRIDE=$OVERRIDE"
WALLTIME=${WALLTIME:-}
prev=""
for i in $(seq 1 "$LINKS"); do
    dep=${prev:+-W depend=afterany:$prev}
    prev=$(qsub -N "pw_${RUN_NAME:-$NAME}" -v "$VARS" ${WALLTIME:+-l walltime=$WALLTIME} $dep \
        cluster/pretrain.cmd)
    echo "link $i: $prev"
done
