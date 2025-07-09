python3 run_evaluation.py --dataset MVBench --model_path /auto/work/sw/kapil/EfficientQAT/dequantized_real_rotated_2048 --output_path results_output_PerceptionTest_EQAT_2048_visionfull+langfull_64 --samples 1000 --attn_implementation eager --weights_vision_qdq --hooks_vision_qdq --weights_lang_qdq --hooks_lang_qdq


# Video-LLM Accuracy Runner – README  
*(quick-start & reference guide)*

NOTE: Currently PARTIAL ROT is TRUE as default

---

## 1. What is this llm_video_eval?

A lightweight harness to **benchmark video-LLM checkpoints** (Qwen2.5-VL family, but easily extendible) on  
- **MVBench**  
- **MMBench-Video**  
- **PerceptionTest MC-VQA**  

with optional **QuaRot** rotation and a menu of **QDQ (Quantise-DeQuantise)** experiments for the vision and language towers.

---

## 2. Directory layout you should have

```
.
├─ run_evaluation.py             # entry-point CLI
├─ infer_qwen2_5_vl.py           # wraps Qwen2.5-VL model
├─ dataset/
│   ├─ MMBenchVideoDataset.py
│   ├─ MVBenchDataset.py
│   └─ PerceptionTestMCVQADataset.py
├─ vlm_accuracy_runner/
│   └─ ... (util modules)
└─ datasets/
    ├─ MMBench-Video/            # unpacked official files
    ├─ MVBench/
    └─ PerceptionTest-MC_VQA/
```

---

## 3. Python environment

```bash
unset PYTHONPATH
source source /auto/regrt/sw/jayant/.new_env/bin/activate
export PYTHONPATH=/auto/regrt/sw/jayant/.new_env/lib/python3.11/site-packages

or 
unset PYTHONPATH
source /auto/regrt/sw/vpothula/vlmEvalKit_test1/VLMEvalKit/env_vlmevalkit/bin/activate
```

GPU driver ≥ 520, CUDA ≥ 11.8 is assumed.

---

## 4. Running an evaluation

### 4.1 Minimal float run

```bash
python run_evaluation.py \
  --dataset MVBench \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --output_path results_mvbench_float
```

### 4.2 Recommended options

| Flag | Description | Default |
|------|-------------|---------|
| `--samples N` | Evaluate only **N** random samples (deterministic via `SEED`). | all |
| `--continue_run path/` | Resume & append to an earlier run folder. | — |
| `--attn_implementation {sdpa,eager}` | Use PyTorch 2.1 SDPA or legacy eager attention. | `sdpa` |
| `--rotate` | Apply **QuaRot** full rotation (vision+LLM) before eval. | off |
| QDQ Vision | `--vision_qdq` + (`--weights_qdq`, `--hooks_qdq`) |
| QDQ Lang   | `--lang_qdq`   + (`--weights_qdq`, `--hooks_qdq`) |

### 4.3 Full EQAT example (vision & language QDQ, hooks + weights, custom model)

```bash
python3 run_evaluation.py \
  --dataset PerceptionTest \
  --model_path /auto/work/sw/kapil/EfficientQAT/dequantized_real_rotated_2048 \
  --output_path results_output_PerceptionTest_EQAT_2048_visionfull+langfull_64 \
  --samples 1000 \
  --attn_implementation eager \
  --vision_qdq --weights_qdq --hooks_qdq \
  --lang_qdq  --weights_qdq --hooks_qdq \
  --rotate
```

---

## 5. Outputs

Each run creates

```
output_<DATASET>_runX/
├─ results.json          # one line per sample
├─ run_meta.json         # live progress + aggregate stats
└─ category_summary.json # per-task accuracy (if applicable)
```

Key fields per sample:

| key | note |
|-----|------|
| `question_id` | unique id / resume guard |
| `prediction`  | model raw answer |
| `is_correct`  | Boolean via Ollama self-grader |
| `soft_score`  | ROUGE-L against GT (MMBench‐style) |
| `fps_used` / `duration` | profiling |

---
