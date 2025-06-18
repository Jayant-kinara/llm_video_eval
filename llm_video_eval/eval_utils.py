import evaluate
import decord 
import torchvision.io as tvio


# Load once — cache
_bleu = evaluate.load("bleu")
_rouge = evaluate.load("rouge")
_meteor = evaluate.load("meteor")

def compute_soft_score(pred, gt):
    # MMBench uses BLEU-4
    bleu_score = _bleu.compute(predictions=[pred], references=[[gt]])['bleu']
    
    # ROUGE-L
    rouge_score = _rouge.compute(predictions=[pred], references=[gt])['rougeL']
    
    # METEOR
    meteor_score = _meteor.compute(predictions=[pred], references=[gt])['meteor']
    
    # Final softscore (this is exactly how MMBench leaderboard computes)
    soft_score = bleu_score + rouge_score + meteor_score
    
    return soft_score


def calculate_accuracy(results):
    correct = sum(r["is_correct"] for r in results)
    return correct / len(results) if results else 0

def print_summary(results):
    accuracy = calculate_accuracy(results)
    print(f"Accuracy: {accuracy*100:.2f}% ({len(results)} examples)")



def get_video_duration(video_path):
    vr = decord.VideoReader(video_path)
    num_frames = len(vr)
    fps = vr.get_avg_fps()
    duration = num_frames / fps
    return duration

def get_video_duration_torchvision(video_path):
    video, _, info = tvio.read_video(video_path, pts_unit='sec')
    duration = info['video_duration']
    return duration
