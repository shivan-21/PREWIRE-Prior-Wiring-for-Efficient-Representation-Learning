#!/usr/bin/env bash
# Submit a chain of pre-training jobs for one config; each link starts when the previous one ends
# for any reason, so a run longer than the walltime resumes from its last checkpoint.
#
#   bash cluster/submit.sh configs/lpl.yaml            # 2 links (default)
#   LINKS=1 RUN_NAME=calib_lpl OVERRIDE=pretrain.max_steps=300 bash cluster/submit.sh configs/lpl.yaml
#   Prints each link's job id, then "last: <id>" (the final link) for chaining.
#   AFTER=afterany:2576655.hn1 bash cluster/submit.sh configs/simclr.yaml  # any dependency type
#   AFTER=2576655.hn1 bash cluster/submit.sh configs/simclr.yaml   # first link waits for that job
#                                                                  # to succeed (afterok)
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
    if [ -n "$prev" ]; then
        dep="-W depend=afterany:$prev"
    else
        case "${AFTER:-}" in
            "") dep="" ;;
            after*) dep="-W depend=$AFTER" ;;           # e.g. afterany:<jobid>
            *) dep="-W depend=afterok:$AFTER" ;;
        esac
    fi
    prev=$(qsub -N "pw_${RUN_NAME:-$NAME}" -v "$VARS" ${WALLTIME:+-l walltime=$WALLTIME} $dep \
        cluster/pretrain.cmd)
    echo "link $i: $prev"
done
echo "last: $prev"
