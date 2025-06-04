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
    parser.add_argument("--rotate", action="store_true", help="Enable QuaRot rotation")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    evaluator = VideoEvaluator(args.model_path, args.output_path, rotate=args.rotate)
    evaluator.evaluate(args.dataset, args.samples, continue_run_folder=args.continue_run)

if __name__ == "__main__":
    main()
