#!/bin/bash
# Reference run of upstream LPL (vendor/lpl_zenke, unmodified) in the legacy env, to check our
# re-implementation's learning curve. Upstream cannot resume across jobs, so this is one job; it
# saves a Lightning checkpoint every epoch, so a run cut short can still be evaluated.
#
#   qsub -N pw_upstream_lpl -v EPOCHS=160,WARMUP_EPOCHS=2 cluster/upstream_lpl.cmd
#
# Compare with our LPL on the same schedule: EPOCHS x 45k images = steps x 1024 images, warmup
# 1.25% (upstream trains on 45k images, holding out 5k for validation).
#PBS -q gpuq
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=24:00:00

cd "$PBS_O_WORKDIR"
mkdir -p logs
exec > "logs/${PBS_JOBNAME}.${PBS_JOBID%%.*}.log" 2>&1
source "$PBS_O_WORKDIR/cluster/gpu_env.sh"

# Upstream reads ~/data/datasets and writes ~/data/lpl; point HOME at a project folder.
OUT=${PREWIRE_OUT:-$HOME/prewire_out}
UP_HOME=$OUT/upstream_home
mkdir -p "$UP_HOME/data/datasets"
ln -sfn "${PREWIRE_DATA:-$HOME/scratch/prewire}/cifar10" "$UP_HOME/data/datasets/cifar10"
PY=$HOME/envs/prewire-upstream/bin/python

echo "host $(hostname)  gpu $CUDA_VISIBLE_DEVICES  epochs ${EPOCHS:-200}  start $(date)"
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "ABORT: no CUDA on $(hostname)"; exit 1; }

cd vendor/lpl_zenke
HOME="$UP_HOME" $PY lpl_main.py --max_epochs "${EPOCHS:-200}" \
    --warmup_epochs "${WARMUP_EPOCHS:-10}" --num_workers "${NCPUS:-8}" \
    --experiment_name "reference_e${EPOCHS:-200}"
rc=$?
echo "end $(date)  exit $rc"
exit $rc
