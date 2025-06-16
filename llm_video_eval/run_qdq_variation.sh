#!/bin/bash

DATASET=MVBench
MODEL_PATH=/auto/work/sw/kapil/hub/Qwen2.5-VL-7B-Instruct-GPTQ-Int4_Dequantized
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

# # # Jayanth Env Float - without Rotate
# run_and_log "results_output_MVBench_JayanthFloatWithouOutRotated" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_MVBench_JayanthFloatWithouOutRotated --samples $SAMPLES --attn_implementation $ATN_IMPL

# sleep 5

# # # Jayanth Env Float - with Rotate
# run_and_log "results_output_MVBench_JayanthFloatRotated" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_MVBench_JayanthFloatRotated $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL

# sleep 5

# # # Vision QDQ full
# run_and_log "results_output_MVBench_visionQDQ" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_MVBench_visionQDQ $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --vision_qdq --weights_qdq --hooks_qdq

# sleep 5

# Lang QDQ full
run_and_log "results_output_MVBench_langQDQ" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_MVBench_langQDQ $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --lang_qdq --weights_qdq --hooks_qdq
sleep 5

# weights QDQ only
run_and_log "results_output_MVBench_weightsonlyQDQ" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_MVBench_weightsonlyQDQ $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --vision_qdq  --lang_qdq --weights_qdq

sleep 5

# hooks QDQ only
run_and_log "results_output_MVBench_hooksonlyQDQ" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_MVBench_hooksonlyQDQ $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --vision_qdq  --lang_qdq --hooks_qdq

sleep 5

# Vision weights only
run_and_log "results_output_MVBench_visionQDQ_weightsonly" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_MVBench_visionQDQ_weightsonly $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --vision_qdq --weights_qdq

sleep 5

# Lang weights only
run_and_log "results_output_MVBench_langQDQ_weightsonly" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_MVBench_langQDQ_weightsonly $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --lang_qdq --weights_qdq

sleep 5

# Vision hooks only
run_and_log "results_output_MVBench_visionQDQ_hooksonly" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_MVBench_visionQDQ_hooksonly $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --vision_qdq --hooks_qdq

# Lang hooks only
run_and_log "results_output_MVBench_langQDQ_hooksonly" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_MVBench_langQDQ_hooksonly $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --lang_qdq --hooks_qdq

# all qdq
run_and_log "results_output_MVBench_all_qdq" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_MVBench_all_qdq $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --vision_qdq --lang_qdq --hooks_qdq --weights_qdq




# ----------------------------------------------------------------------------------------------------

DATASET=PerceptionTest

# # Jayanth Env Float - without Rotate
run_and_log "results_output_PerceptionTest_JayanthFloatWithouOutRotated" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_PerceptionTest_JayanthFloatWithouOutRotated --samples $SAMPLES --attn_implementation $ATN_IMPL

sleep 5

# # Jayanth Env Float - with Rotate
run_and_log "results_output_PerceptionTest_JayanthFloatRotated" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_PerceptionTest_JayanthFloatRotated $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL

sleep 5

# # Vision QDQ full
run_and_log "results_output_PerceptionTest_visionQDQ" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_PerceptionTest_visionQDQ $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --vision_qdq --weights_qdq --hooks_qdq

sleep 5

# Lang QDQ full
run_and_log "results_output_PerceptionTest_langQDQ" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_PerceptionTest_langQDQ $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --lang_qdq --weights_qdq --hooks_qdq

sleep 5

# weights QDQ only
run_and_log "results_output_PerceptionTest_weightsonlyQDQ" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_PerceptionTest_weightsonlyQDQ $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --vision_qdq  --lang_qdq --weights_qdq

sleep 5

# hooks QDQ only
run_and_log "results_output_PerceptionTest_hooksonlyQDQ" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_PerceptionTest_hooksonlyQDQ $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --vision_qdq  --lang_qdq --hooks_qdq

sleep 5

# Vision weights only
run_and_log "results_output_PerceptionTest_visionQDQ_weightsonly" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_PerceptionTest_visionQDQ_weightsonly $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --vision_qdq --weights_qdq

sleep 5

# Lang weights only
run_and_log "results_output_PerceptionTest_langQDQ_weightsonly" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_PerceptionTest_langQDQ_weightsonly $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --lang_qdq --weights_qdq

sleep 5

# Vision hooks only
run_and_log "results_output_PerceptionTest_visionQDQ_hooksonly" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_PerceptionTest_visionQDQ_hooksonly $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --vision_qdq --hooks_qdq

# Lang hooks only
run_and_log "results_output_PerceptionTest_langQDQ_hooksonly" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_PerceptionTest_langQDQ_hooksonly $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --lang_qdq --hooks_qdq

# all qdq
run_and_log "results_output_PerceptionTest_all_qdq" python3 run_evaluation.py --dataset $DATASET --model_path $MODEL_PATH --output_path results_output_PerceptionTest_all_qdq $ROTATE --samples $SAMPLES --attn_implementation $ATN_IMPL --vision_qdq --lang_qdq --hooks_qdq --weights_qdq