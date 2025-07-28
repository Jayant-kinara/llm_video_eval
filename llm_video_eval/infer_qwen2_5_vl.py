import torch, os
import decord
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
from rotated.rotation_utils import rotate_llm_model, rotate_visual_model, partially_rotate_visual_model
import numpy as np
import torchvision.io as tvio
from qdq_util import *
import qdq_util
# decord.bridge.set_bridge("torch")

def get_video_duration(video_path):
    vr = decord.VideoReader(video_path)
    height, width = vr[0].shape[:2]
    num_frames = len(vr)
    fps = vr.get_avg_fps()
    duration = num_frames / fps
    return duration, height, width

def get_video_duration_torchvision(video_path):
    video, _, info = tvio.read_video(video_path, pts_unit='sec')
    height, width = video.shape[1:3]
    duration = info['video_duration']
    return duration, height, width

class Qwen2_5_VL_Inferer:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda",
                rotate=False, attn_implementation="sdpa",
                weights_vision_qdq=False, hooks_vision_qdq=False,
                weights_lang_qdq=False,hooks_lang_qdq=False,
                partial_rotate=False):
        print("using attn_implementation: ", attn_implementation)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            # max_memory={
            #     0: "3GB",    # allow only 5 GB of model weights on GPU 0
            #     1: "22GB",   # allow up to 30 GB on GPU 1
            # },
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            # cache_dir="/auto/worka/vpothula/models",
            attn_implementation = attn_implementation
            # attn_implementation = "sdpa"
        )
        print("Model load done")

        if rotate:
            print("QuaRot START")
            rotate_visual_model(self.model)
            # rotate_llm_model(self.model)
            print("QuaRot DONE")
        if partial_rotate and (not rotate):
            print("Partial Rotate START")
            partially_rotate_visual_model(self.model)
            print("Partial Rotate DONE")
            
        # Apply QDQ logic
        self._apply_vision_qdq(weights_vision_qdq, hooks_vision_qdq)
        qdq_util.LANG_HOOK_GROUP_SIZE = 64 # configurable
        self._apply_lang_qdq(weights_lang_qdq, hooks_lang_qdq)
        # else:
        #     print("No QDQ chosen. Use --vision_qdq or --lang_qdq with --weights_qdq and/or --hooks_qdq to enable QDQ.")

        MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.processor = Qwen2_5_VLProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            use_fast=True
        )

        self.device = torch.device(device)


    def _apply_vision_qdq(self, weights_qdq, hooks_qdq):
        print("Performing Vision QDQ")
        if weights_qdq:
            print(" - Applying Vision Weights QDQ")
            quantize_model_weights_vision(self.model.visual)
        if hooks_qdq:
            print(" - Adding Vision Hooks QDQ")
            self.model.visual, output_hooks, hook_ref = add_output_hooks_visual(self.model.visual, {})

    def _apply_lang_qdq(self, weights_qdq, hooks_qdq):
        print("Performing Language QDQ")
        if weights_qdq:
            print(" - Applying Language Weights QDQ")
            quantize_model_weights_language(self.model)
        if hooks_qdq:
            print(" - Adding Language Hooks QDQ")
            self.model.language_model, output_hooks, hook_ref = add_output_hooks_language(self.model.language_model, {})

    def infer_video(self, video_path, question, dataset, max_new_tokens=512, fps=0.5, resized_height=12*28, resized_width=12*28, dynamic=False, ):
        # If dynamic FPS is enabled → auto select fps
        print("question")
        print(question)
        if dynamic:
            try:
                duration_sec, height_orig, width_orig = get_video_duration(video_path)
                
            except Exception as e:
                print(f"[Fallback] Error with decord on {video_path}, using torchvision: {e}")
                duration_sec, height_orig, width_orig = get_video_duration_torchvision(video_path)

            
            # if duration_sec < 10:
            #     fps = 3.0
            # elif duration_sec < 20:
            #     fps = 2.0
            # elif duration_sec < 60:
            #     fps = 1.0
            # elif duration_sec < 120:
            #     fps = 0.5
            # else:
            #     fps = 0.25

            if duration_sec < 5:
                fps = 3.0
            elif duration_sec < 10:
                fps = 2.0
            elif duration_sec < 30:
                fps = 1.0
            elif duration_sec < 60:
                fps = 0.5
            else:
                fps = 0.25


            print(f"[Video] {video_path} | Duration: {duration_sec:.1f} sec | Using FPS: {fps}")

        messages = [
            {
                "role": "system",
                "content": dataset.SYS
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        # "fps": fps,
                        # "resized_height": resized_height,
                        # "resized_width": resized_width,
                    },
                    {"type": "text", "text": question}
                ]
            }
        ]
        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        model_inputs = self.processor(
            text=[chat_text],
            videos=video_inputs,
            return_tensors="pt",
            **video_kwargs
        )
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        torch.cuda.empty_cache()

        with torch.no_grad():
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens
            )

        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return response, duration_sec, fps

    def infer_frames(self, frame_folder, question, dataset, max_new_tokens=512, num_frames=32, bound=None, fps=3, resized_height=12*28, resized_width=12*28):
        # All frame files
        frame_files = sorted([
            os.path.join(frame_folder, f)
            for f in os.listdir(frame_folder)
            if f.endswith(".jpg")
        ])
        max_frame = len(frame_files)
        
        def get_index(bound, fps, max_frame, first_idx=1):
            if bound:
                start, end = bound[0], bound[1]
            else:
                start, end = -100000, 100000
            start_idx = max(first_idx, round(start * fps))
            end_idx = min(round(end * fps), max_frame)
            seg_size = float(end_idx - start_idx) / num_frames
            frame_indices = np.array([
                int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
                for idx in range(num_frames)
            ])
            return frame_indices

        # Sample indices
        frame_indices = get_index(bound, fps, max_frame, first_idx=1)

        # Clip indices to valid range
        frame_indices = np.clip(frame_indices, 1, max_frame)
        selected_frames = [
            frame_files[idx-1]  # because first_idx=1
            for idx in frame_indices
        ]

        frame_uris = [f"file://{f}" for f in selected_frames]

        print(f"[infer_frames] Using Num frames:{len(frame_uris)}")

        messages = [
            {
                "role": "system",
                "content": dataset.SYS
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frame_uris,
                        # "resized_height": resized_height,
                        # "resized_width": resized_width,
                    },
                    {"type": "text", "text": question}
                ]
            }
        ]

        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        model_inputs = self.processor(
            text=[chat_text],
            videos=video_inputs,
            return_tensors="pt",
            **video_kwargs
        )
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        torch.cuda.empty_cache()

        with torch.no_grad():
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens
            )
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        return response, bound[1]-bound[0], fps



    def postprocess_response(self, response):
        postprocessed_response = response.split("assistant\n")[-1]
        return postprocessed_response




# run_infer_test.py

# import Qwen2_5_VL_Inferer

# Load model once
# inferer = Qwen2_5_VL_Inferer()

# # Infer on a video
# video_path = "/auto/regrt/sw/vpothula/vlmEvalKit_test1/vlm_accuracy_runner/datasets/MMBench-Video/video/_3CvF9fk7Bc.mp4"
# question = "Describe what is happening in the video."

# response = inferer.infer_video(video_path, question)

# print("Model Response:", response)
