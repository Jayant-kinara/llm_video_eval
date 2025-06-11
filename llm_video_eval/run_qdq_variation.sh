#!/bin/bash

DATASET=MVBench
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
SAMPLES=1000
ATN_IMPL=eager
ROTATE=--rotate

run_and_log() {
    DESC=$1
    shift
    CMD="$@"
    LOG_FILE="${DESC}.log"
    echo "=== Running: $DESC ==="
    echo "Logging to: $LOG_FILE"
    echo "======================"
    echo "$CMD" | tee -a "$LOG_FILE"
    eval "$CMD" 2>&1 | tee -a "$LOG_FILE"
    echo "=== Finished: $DESC ==="
    echo ""
}

# # Vision QDQ full
# run_and_log "results_output_visionQDQ" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_visionQDQ $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --vision_qdq --weights_qdq --hooks_qdq

sleep 5

# Lang QDQ full
run_and_log "results_output_langQDQ" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_langQDQ $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --lang_qdq --weights_qdq --hooks_qdq

sleep 5

# weights QDQ only
run_and_log "results_output_weightsonlyQDQ" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_weightsonlyQDQ $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --vision_qdq  --lang_qdq --weights_qdq

sleep 5

# hooks QDQ only
run_and_log "results_output_hooksonlyQDQ" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_hooksonlyQDQ $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --vision_qdq  --lang_qdq --hooks_qdq

sleep 5

# Vision weights only
run_and_log "results_output_visionQDQ_weightsonly" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_visionQDQ_weightsonly $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --vision_qdq --weights_qdq

sleep 5

# Lang weights only
run_and_log "results_output_langQDQ_weightsonly" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_langQDQ_weightsonly $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --lang_qdq --weights_qdq

sleep 5

# Vision hooks only
run_and_log "results_output_visionQDQ_hooksonly" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_visionQDQ_hooksonly $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --vision_qdq --hooks_qdq

# Lang hooks only
run_and_log "results_output_langQDQ_hooksonly" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_langQDQ_hooksonly $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --lang_qdq --hooks_qdq

# all qdq
run_and_log "results_output_all_qdq" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_all_qdq $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --vision_qdq --lang_qdq --hooks_qdq --weights_qdq