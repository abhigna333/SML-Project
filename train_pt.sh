BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$BASE_DIR/.env"

export CUDA_VISIBLE_DEVICES="4,5,6,7" # Provide GPU IDs here, e.g., "0,1,2,3"
NUM_DEVICES=$(python -c "import os; print(len(os.getenv('CUDA_VISIBLE_DEVICES', '').split(',')))")

export ACCELERATE_LOG_LEVEL="info"

EXPERIMENT_YAML_FILE="$BASE_DIR/experiments/smollm_360M_fineweb_15B_PT.yaml"
EXPERIMENT_NAME="Experiment_name" # Provide name here, e.g., "smollm_360M_fineweb_15B_PT"
export ACCELERATE_CONFIG_FILE="$BASE_DIR/configs/ddp.yaml"
export OUTPUT_DIR="$BASE_DIR/outputs/$EXPERIMENT_NAME" # Provide output directory here, e.g., "/outputs/smollm_360M_fineweb_15B_PT"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
echo "BASE_DIR: $BASE_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EXPERIMENT_YAML_FILE: $EXPERIMENT_YAML_FILE"
echo "LOG_FILE: $OUTPUT_DIR/log_experiment.txt"
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "ACCELERATE_CONFIG_FILE: $ACCELERATE_CONFIG_FILE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NUM_DEVICES: $NUM_DEVICES"
mkdir -p "$BASE_DIR/logs/$EXPERIMENT_NAME/"
accelerate launch \
    --num_processes $NUM_DEVICES \
    --main_process_port 29600 \
    --config_file $ACCELERATE_CONFIG_FILE \
    src/pipeline/sft/workflow.py \
    --config $EXPERIMENT_YAML_FILE \
    --run_name=$EXPERIMENT_NAME \
    --output_dir=$OUTPUT_DIR > $BASE_DIR/logs/$EXPERIMENT_NAME/log_experiment.log 2>&1 