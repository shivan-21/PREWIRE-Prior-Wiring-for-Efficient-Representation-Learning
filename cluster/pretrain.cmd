#!/bin/bash
# One pre-training job. Submit through cluster/submit.sh, which passes CONFIG (and optionally
# RUN_NAME and OVERRIDE, a single section.key=value) with qsub -v.
#PBS -q gpuq
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=24:00:00

cd "$PBS_O_WORKDIR"
mkdir -p logs
exec > "logs/${PBS_JOBNAME}.${PBS_JOBID%%.*}.log" 2>&1   # PBS .o/.e delivery is unreliable here
source "$PBS_O_WORKDIR/cluster/gpu_env.sh"

export PREWIRE_DATA=${PREWIRE_DATA:-$HOME/scratch/prewire}
export PREWIRE_OUT=${PREWIRE_OUT:-$HOME/prewire_out}
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-file://$PREWIRE_OUT/mlruns}
export MLFLOW_ALLOW_FILE_STORE=true
export MLFLOW_DISABLE_AGENT_HINT=1
export GIT_PYTHON_REFRESH=quiet   # mlflow-skinny probes for git; the cluster git is too old
mkdir -p "$PREWIRE_OUT"
PY=$HOME/envs/prewire/bin/python

echo "host $(hostname)  gpu $CUDA_VISIBLE_DEVICES  config $CONFIG  start $(date)"
# NEVER `module load cuda*`: its libcuda stub hides the driver and torch silently runs on CPU.
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "ABORT: no CUDA on $(hostname)"; exit 1; }

$PY scripts/pretrain.py --config "$CONFIG" ${RUN_NAME:+--run_name "$RUN_NAME"} \
    ${OVERRIDE:+--set "$OVERRIDE"}
rc=$?
echo "end $(date)  exit $rc"
exit $rc
