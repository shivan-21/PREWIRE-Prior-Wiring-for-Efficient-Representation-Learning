#!/usr/bin/env bash
# Run the full M4 evaluation on the pre-trained encoders in results_and_figs/checkpoints/.
#
#   bash scripts/evaluate_all.sh [probe|finetune]   # one lane, or both lanes in parallel
#
# Linear probes: all 20 encoders (4 conditions x 3 seeds, the 7 ablation runs, Zenke's reference);
# every layer, except the 7 ablation runs, which are probed on the last layer only.
# Fine-tuning: 4 conditions x 3 seeds, plus the rho 0.8 and rho 0 clip runs (14 encoders).
# Seed s pairs the seed-s encoders with the seed-s labelled subsets. The scripts append to
# results_and_figs/tables/*_s<seed>.csv; plot_results.py keeps the last row per (model, n).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}
CK=results_and_figs/checkpoints

main() {  # encoder specs for one seed: name=path
    local s=$1
    echo "lpl=$CK/lpl_s$s simclr=$CK/simclr_s$s oja=$CK/oja_s$s random=$CK/random_init_s$s"
}
ABLATIONS="lpl_shuffled=$CK/lpl_shuffled_s0"
for run in rho0p8 rho0 ordershuf crossclip rho0p5 rho0p95; do
    ABLATIONS="$ABLATIONS lpl_temporal_$run=$CK/lpl_temporal_${run}_s0"
done

probe() {
    # shellcheck disable=SC2046
    $PY scripts/linear_probe.py --seed 0 --encoders $(main 0) "zenke_reference=$CK/zenke_reference"
    # shellcheck disable=SC2086
    $PY scripts/linear_probe.py --seed 0 --layers 8 --encoders $ABLATIONS
    for s in 1 2; do $PY scripts/linear_probe.py --seed $s --encoders $(main $s); done
}

finetune() {
    # shellcheck disable=SC2046
    $PY scripts/finetune.py --seed 0 --encoders $(main 0) \
        "lpl_temporal_rho0p8=$CK/lpl_temporal_rho0p8_s0" "lpl_temporal_rho0=$CK/lpl_temporal_rho0_s0"
    for s in 1 2; do $PY scripts/finetune.py --seed $s --encoders $(main $s); done
}

case "${1:-both}" in
    probe) probe ;;
    finetune) finetune ;;
    both) probe & finetune & wait ;;
esac
