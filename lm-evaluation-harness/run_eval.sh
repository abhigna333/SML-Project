BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

#!/bin/bash
export CUDA_HOME=$CONDA_PREFIX
export NCCL_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

# 🔧 Add this line to enable importing your custom model
export PYTHONPATH="$(dirname "$(realpath "$0")")/../src:$PYTHONPATH"


run_name="DTRNet_360M"
model_path=/work/aman/outputs/DTRNet_360M_fineweb_15B/checkpoint-10

export CUDA_VISIBLE_DEVICES="0,1"
NUM_GPUs=$(python -c "import os; print(len(os.getenv('CUDA_VISIBLE_DEVICES', '').split(',')))")

date;pwd

export MASTER_ADDR="$(hostname --fqdn)"
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
export RDVZ_ID=$RANDOM
echo "RDZV Endpoint $MASTER_ADDR:$MASTER_PORT"

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=WARN
# export NCCL_DEBUG_SUBSYS=WARN
# export TORCH_CPP_LOG_LEVEL=INFO
# export LOGLEVEL=INFO

# export WANDB_PROJECT="pretrain_fused_smollm"
# export WANDB_MODE="offline"
export ACCELERATE_LOG_LEVEL="info"

LOG_DIR="eval_logs"
LOG_PATH="${LOG_DIR}/log_${run_name}.log"
# Make logging directories.
mkdir -p "${LOG_DIR}"
echo "Placing logs in: ${LOG_DIR}"

accelerate launch \
    --multi-gpu \
    --num_machines=1 \
    --num_processes=$NUM_GPUs \
    --machine_rank=0 \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --rdzv_backend=static \
    lm_eval --model hf \
    --model_args pretrained=${model_path},dtype=float16 \
    --tasks winogrande\
    --batch_size 1 \
    --num_fewshot 0 \
    --trust_remote_code \
    --gen_kwargs max_new_tokens=1024,do_sample=False \
    --output_path ${model_path}/lm_harness_output > ${LOG_PATH} 2>&1
 