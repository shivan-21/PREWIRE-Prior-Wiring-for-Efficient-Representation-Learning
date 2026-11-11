#!/usr/bin/env bash
# Queue the remaining pre-training runs (temporal test + seeds 1 and 2) in three lanes. Within a
# lane each run starts after the previous one ends for any reason (afterany), so at most three
# GPU jobs run at once and one failure does not block the rest.
#
#   bash cluster/queue_remaining.sh
set -euo pipefail
cd "$(dirname "$0")/.."

A="" B="" C=""
submit() {  # lane config run_name [override]
    local lane=$1 config=$2 name=$3 override=${4:-}
    local after=${!lane}
    local out
    out=$(AFTER=${after:+afterany:$after} RUN_NAME=$name OVERRIDE=$override \
        bash cluster/submit.sh "$config")
    echo "$out" | sed "s/^/[$lane $name] /"
    printf -v "$lane" '%s' "$(echo "$out" | awk '/^last:/ {print $2}')"
}

# Lane A: temporal test, main contrast and its controls (all rho 0.8 unless set).
submit A configs/lpl_temporal.yaml lpl_temporal_rho0p8_s0
submit A configs/lpl_temporal.yaml lpl_temporal_rho0_s0 data.sequence.rho=0
submit A configs/lpl_temporal.yaml lpl_temporal_ordershuf_s0 data.sequence.order=shuffled
submit A configs/lpl_temporal.yaml lpl_temporal_crossclip_s0 pretrain.shuffle_positives=true

# Lane B: rho sweep, two-view shuffled-positives ablation, one extra LPL seed.
submit B configs/lpl_temporal.yaml lpl_temporal_rho0p5_s0 data.sequence.rho=0.5
submit B configs/lpl_temporal.yaml lpl_temporal_rho0p95_s0 data.sequence.rho=0.95
submit B configs/lpl_shuffled.yaml lpl_shuffled_s0
submit B configs/lpl.yaml lpl_s1 seed=1

# Lane C: remaining extra seeds.
submit C configs/lpl.yaml lpl_s2 seed=2
submit C configs/simclr.yaml simclr_s1 seed=1
submit C configs/simclr.yaml simclr_s2 seed=2
submit C configs/oja.yaml oja_s1 seed=1
submit C configs/oja.yaml oja_s2 seed=2
