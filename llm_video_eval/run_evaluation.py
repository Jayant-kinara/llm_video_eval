import argparse, os
from dataset import MMBenchVideoDataset
from evaluator import VideoEvaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", default="results_output")
    parser.add_argument("--samples", default=None)
    parser.add_argument("--continue_run", default=None)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--rotate", action="store_true", help="Enable QuaRot rotation")
    parser.add_argument("--weights_vision_qdq", action="store_true", help="Enable weights_vision_qdq")
    parser.add_argument("--hooks_vision_qdq", action="store_true", help="Enable hooks_vision_qdq")
    parser.add_argument("--weights_lang_qdq", action="store_true", help="Enable weights_lang_qdq")
    parser.add_argument("--hooks_lang_qdq", action="store_true", help="Enable hooks_lang_qdq")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="How many *video* samples to send to the model at once. "
                            "If a batch contains any frame-directory sample or "
                            "batch_size==1 we fall back to single-query mode.")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    evaluator = VideoEvaluator(args.model_path, args.output_path, rotate=args.rotate,
                               attn_implementation=args.attn_implementation,
                               weights_vision_qdq=args.weights_vision_qdq, hooks_vision_qdq=args.hooks_vision_qdq,
                               weights_lang_qdq=args.weights_lang_qdq,hooks_lang_qdq=args.hooks_lang_qdq)
    evaluator.evaluate(args.dataset, args.samples, continue_run_folder=args.continue_run)

if __name__ == "__main__":
    main()
