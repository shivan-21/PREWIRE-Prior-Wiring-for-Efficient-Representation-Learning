# Pick the least-used GPU on the node: PBS does not set CUDA_VISIBLE_DEVICES and nodes are shared.
CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t, -k2 -n | head -1 | tr -d ' ' | cut -d, -f1)
export CUDA_VISIBLE_DEVICES
echo "gpu: chose least-used card -> $CUDA_VISIBLE_DEVICES"
