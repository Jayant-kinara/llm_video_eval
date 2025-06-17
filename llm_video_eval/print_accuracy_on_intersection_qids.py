import os
import json
from glob import glob

# List your base folders here
base_dirs = [
    "/auto/regrt/sw/vpothula/llm_eval_repo/llm_eval/llm_video_eval/results_output_MVBench_EQAT",
    "/auto/regrt/sw/vpothula/llm_eval_repo/llm_eval/llm_video_eval/results_output_MVBench_EQAT_1024",
    "/auto/regrt/sw/vpothula/llm_eval_repo/llm_eval/llm_video_eval/results_output_MVBench_EQAT_2048",
    "/auto/regrt/sw/vpothula/llm_eval_repo/llm_eval/llm_video_eval/results_output_MVBench_EQAT_2048_visionfull+lang_hooks_qdq_64",
    "/auto/regrt/sw/vpothula/llm_eval_repo/llm_eval/llm_video_eval/results_output_MVBench_EQAT_1024_visionfull+lang_hooks_qdq_32"
]

results_per_folder = {}

for folder in base_dirs:
    run_dirs = sorted(glob(os.path.join(folder, "output_MVBench_run*")))
    if not run_dirs:
        print(f"[SKIP] No run folders found in: {folder}")
        continue

    last_run = run_dirs[-1]
    result_path = os.path.join(last_run, "results.json")
    if not os.path.exists(result_path):
        print(f"[SKIP] No results.json in: {last_run}")
        continue

    with open(result_path) as f:
        data = json.load(f)

    qid_map = {entry["question_id"]: entry["is_correct"] for entry in data}
    results_per_folder[folder] = qid_map
    print(f"[OK] {folder}: Loaded {len(qid_map)} questions from {os.path.basename(last_run)}")

# Compute mutually inclusive question_ids
if not results_per_folder:
    print("\n[FAIL] No valid folders with results.json found.")
else:
    sets = [set(qids.keys()) for qids in results_per_folder.values()]
    common_qids = set.intersection(*sets) if sets else set()
    print(f"\n[INFO] Common Question IDs across folders: {len(common_qids)}\n")

    # Compute accuracy for each
    for folder, qid_map in results_per_folder.items():
        correct = sum(1 for qid in common_qids if qid_map[qid])
        acc = correct / len(common_qids) if common_qids else 0
        print(f"[RESULT] {os.path.basename(folder)} Accuracy on common QIDs: {acc*100:.2f}%")
