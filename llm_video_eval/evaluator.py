import torch
import random
import json
import os
import datetime
from tqdm import tqdm
from typing import List
from infer_qwen2_5_vl import Qwen2_5_VL_Inferer
from dataset import MMBenchVideoDataset, MVBenchDataset, PerceptionTestMCVQADataset
from eval_utils import print_summary, compute_soft_score, calculate_accuracy
import ollama
import gc


OLLAMA_MODEL = "qwen2.5:32b"
os.environ["OLLAMA_HOST"] = "http://10.10.20.16:11434"

# Dataset mapping
# /auto/share/sw/common/data/llm_accuracy_datasets/video_datasets
DATASET_PATH_MAP = {
    "MMBench-Video": "/auto/regrt/sw/vpothula/llm_eval_repo/llm_eval/llm_video_eval/datasets/MMBench-Video",
    "MVBench": "/auto/regrt/sw/vpothula/llm_eval_repo/llm_eval/llm_video_eval/datasets/MVBench",
    "PerceptionTest": "/auto/regrt/sw/vpothula/llm_eval_repo/llm_eval/llm_video_eval/datasets/PerceptionTest-MC_VQA"
}

DATASET_MAP = {
    "MMBench-Video": MMBenchVideoDataset(),
    "MVBench": MVBenchDataset(),
    "PerceptionTest": PerceptionTestMCVQADataset()

}

SEED = 121222

class VideoEvaluator:
    def __init__(self, model_path, output_path, rotate, attn_implementation, 
                weights_vision_qdq=False, hooks_vision_qdq=False,
                weights_lang_qdq=False,hooks_lang_qdq=False):
        print(f"Loading model from: {model_path}")
        self.model_path = model_path
        self.inferer = Qwen2_5_VL_Inferer(model_id=model_path, rotate=rotate, attn_implementation=attn_implementation, weights_vision_qdq=weights_vision_qdq,
        hooks_vision_qdq=hooks_vision_qdq, weights_lang_qdq=weights_lang_qdq, hooks_lang_qdq=hooks_lang_qdq)
        self.output_path = output_path
        self.dataset = None

    def generate(self, video_path, question, bound, video_type="video"):
        response, duration, fps = "", None, None
        if video_type == "video":
            response, duration, fps = self.inferer.infer_video(
                video_path=video_path,
                question=question,
                dynamic=True,
                dataset=self.dataset
            )
        elif video_type == "frame":
            response, duration, fps = self.inferer.infer_frames(
                video_path,
                question=question,
                num_frames=8,
                bound=bound,
                dataset=self.dataset
            )
        else:
            raise ValueError(f"Unsupported video_type: {video_type}")

        return response, duration, fps

    def generate_batch(self,
                    items: List[dict]):
        """items is a list of dataset.get_item() dicts, all video_type=='video'."""
        video_paths  = [it["video_path"] for it in items]
        questions    = [it["question"]   for it in items]
        responses, durations, fps_list = self.inferer.infer_video_batch(
            video_paths, questions, dataset=self.dataset, dynamic=True
        )
        return list(zip(responses, durations, fps_list))


    def evaluate_with_ollama(self, question, model_response, ground_truth, options=None):
        if not options:
            ollama_prompt = (
                "You are an AI evaluator that determines the output from an AI model with some reference groundtruth values\n"
                f"Question: {question}\n"
                f"Ground Truth Answer: {ground_truth}\n"
                f"Model Response: {model_response}\n\n"
                "Is the model response correct according to the ground truth reference? need not be exact word to word match, but is it correct?"
                "Return only '1' for correct or '0' for incorrect."
            )
        else:
            ollama_prompt = (
                "You are an AI evaluator that determines if the selected option from an AI model matches the correct answer/option.\n"
                f"Question with Options:\n{question}\n\n"
                f"Ground Truth Option: {ground_truth}\n"
                f"Model Response: {model_response}\n\n"
                "Does the model response select the correct option? "
                "Return only '1' for correct or '0' for incorrect."
            )
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": ollama_prompt}],
            options={"temperature": 0}
        )

        content = response.get('message', {}).get('content', '0').strip()
        return content, content == '1'

    def _get_run_folder(self, dataset_name):
        base_dir = self.output_path
        os.makedirs(base_dir, exist_ok=True)

        run_id = 1
        while True:
            run_folder = os.path.join(base_dir, f"output_{dataset_name}_run{run_id}")
            if not os.path.exists(run_folder):
                os.makedirs(run_folder)
                return run_folder
            run_id += 1

    def evaluate(self, dataset_name, samples=None, continue_run_folder=None):
        if dataset_name not in DATASET_PATH_MAP:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        dataset_path = DATASET_PATH_MAP[dataset_name]

        # Load dataset
        self.dataset = DATASET_MAP[dataset_name]
        self.dataset.init(dataset_path)
        total_samples = len(self.dataset)

        # Determine sample indices
        sample_indices = list(range(total_samples))
        random.seed(SEED)
        random.shuffle(sample_indices)
        if samples is not None:
            sample_indices = sample_indices[:int(samples)]

        print(f"Running evaluation on {len(sample_indices)} samples out of {total_samples}")

        # Create run folder
        run_folder = self._get_run_folder(dataset_name)
        output_path = os.path.join(run_folder, "results.json")
        meta_path = os.path.join(run_folder, "run_meta.json")

        # Initialize meta
        run_meta = {
            "dataset": dataset_name,
            "run_folder": run_folder,
            "model": self.model_path,
            "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "total_samples": len(sample_indices),
            "completed": 0,
            "accuracy": 0,
            "total_soft_score": 0,
            "soft_score": 0,
            "fps_usage_counts": {},
            "total_duration": 0,
            "average_duration": 0,
            "pending": len(sample_indices),
            "completed_qids": [],
            "status": "running"
        }

        # Save meta initially
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2)

        # Load previous results if continue_run
        if continue_run_folder:
            prev_results_path = os.path.join(continue_run_folder, "results.json")
            if os.path.exists(prev_results_path):
                with open(prev_results_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                existing_qids = set(item["question_id"] for item in results)
                print(f"[Continue Run] Loaded {len(existing_qids)} previous results from {continue_run_folder}")
            else:
                print(f"[Warning] No previous results found in {continue_run_folder}. Starting fresh.")
                results = []
                existing_qids = set()
        else:
            results = []
            existing_qids = set()

        for idx in tqdm(sample_indices):
            # idx = 1411
            item = self.dataset.get_item(idx)
            qid = item["question_id"]

            if qid in existing_qids:
                continue

            try:
                model_response, duration, fps_used = self.generate(item["video_path"], item["question"], bound=item['bound'], video_type=item.get("video_type", "video"))
                model_response = self.inferer.postprocess_response(model_response)
                #ollama_output, is_correct = self.evaluate_with_ollama(
                #    item["question"], model_response, item["answer"], options=item.get("options", None)
                #)
                print("model_response: ", model_response)
                print("ground_truth: ", item["answer"])
                soft_score = compute_soft_score(model_response, item["answer"])
                print(f"Soft Score: {soft_score:.2f}")

                result_item = {
                    "question_id": qid,
                    "task_type": item.get("task_type", None),
                    "video_path": item["video_path"],
                    "duration": duration,
                    "fps_used": fps_used,
                    "question": item["question"],
                    "ground_truth": item["answer"],
                    "prediction": model_response,
                    #"is_correct": is_correct,
                    "soft_score": soft_score,
                    #"ollma_output": ollama_output,
                    "continued_from": continue_run_folder if continue_run_folder else None
                }
                fps_value = str(fps_used)

                results.append(result_item)

                # LIVE DUMP
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                # Update meta
                if "fps_usage_counts" not in run_meta:
                    run_meta["fps_usage_counts"] = {}
                if fps_value not in run_meta["fps_usage_counts"]:
                    run_meta["fps_usage_counts"][fps_value] = 0
                run_meta["fps_usage_counts"][fps_value] += 1
                run_meta["completed"] += 1
                run_meta["pending"] = run_meta["total_samples"] - run_meta["completed"]
                run_meta['accuracy'] = calculate_accuracy(results)
                run_meta["total_soft_score"] += soft_score
                run_meta["soft_score"] = run_meta["total_soft_score"] / run_meta["completed"]
                run_meta["total_duration"] += duration
                run_meta["average_duration"] = run_meta["total_duration"] / run_meta["completed"]
                run_meta["last_update_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                run_meta["completed_qids"].append(qid)

                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(run_meta, f, indent=2)

                print(f"[Saved] QID: {qid} | Correct: {is_correct}")

            except Exception as e:
                print(f"Error on sample idx {idx} ({qid}): {str(e)}")

            torch.cuda.empty_cache()
            gc.collect()

            # if idx % 3 == 0 and idx != 0:
            #     print("Reloading inferer to prevent memory accumulation...")
            #     del self.inferer
            #     gc.collect()
            #     torch.cuda.empty_cache()
            #     self.inferer = Qwen2_5_VL_Inferer(model_id=self.model_path)

        run_meta["end_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_meta["status"] = "completed"

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2)

        print_summary(results)

        if any("task_type" in item for item in results):
            task_correct = {}
            task_total = {}

            for item in results:
                task = item.get("task_type")
                if not task:
                    continue
                if task not in task_total:
                    task_total[task] = 0
                    task_correct[task] = 0
                task_total[task] += 1
                if item["is_correct"]:
                    task_correct[task] += 1

            print("\n=== Per-Category Accuracy ===")
            category_summary = {}

            for task in sorted(task_total.keys()):
                acc = (task_correct[task] / task_total[task]) * 100
                print(f"{task}: {task_correct[task]}/{task_total[task]} ({acc:.2f}%)")
                category_summary[task] = {
                    "correct": task_correct[task],
                    "total": task_total[task],
                    "accuracy": acc
                }

            # Save category summary
            category_summary_path = os.path.join(run_folder, "category_summary.json")
            with open(category_summary_path, "w", encoding="utf-8") as f:
                json.dump(category_summary, f, indent=2)

            print(f"\nCategory summary saved to {category_summary_path}")

