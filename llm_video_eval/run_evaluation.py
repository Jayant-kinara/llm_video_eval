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
    parser.add_argument("--vision_qdq", action="store_true", help="Enable vision_qdq")
    parser.add_argument("--lang_qdq", action="store_true", help="Enable lang_qdq")
    parser.add_argument("--weights_qdq", action="store_true", help="Enable weights_qdq")
    parser.add_argument("--hooks_qdq", action="store_true", help="Enable hooks_qdq")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    evaluator = VideoEvaluator(args.model_path, args.output_path, rotate=args.rotate,
                               attn_implementation=args.attn_implementation,
                               vision_qdq=args.vision_qdq, lang_qdq=args.lang_qdq,
                               weights_qdq=args.weights_qdq,hooks_qdq=args.hooks_qdq)
    evaluator.evaluate(args.dataset, args.samples, continue_run_folder=args.continue_run)

if __name__ == "__main__":
    main()
