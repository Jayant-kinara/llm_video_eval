import json
import os

class MMBenchVideoDataset:

    SYS = 'You are an AI assistant responsible for answering questions about videos. Be concise and only output the answer to the question.'
    def __init__(self):
        pass

    def init(self,dataset_path):
        self.dataset_path = dataset_path
        self.questions = self._load_json("MMBench-Video_q.json")
        self.answers = self._load_json("MMBench-Video_a.json")
        
        # Mapping answers by question_id for fast lookup
        self.answer_dict = {a["question_id"]: a["answer"] for a in self.answers}

    def _load_json(self, filename):
        filepath = os.path.join(self.dataset_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def __len__(self):
        return len(self.questions)
    

    def get_item(self, idx):
        question_entry = self.questions[idx]
        question_id = question_entry["question_id"]
        video_id = question_entry["video_name"]
        question = question_entry["question"]
        question_text = f"Question: {question}\n"
        answer_text = self.answer_dict.get(question_id, "Answer not found.")
        
        video_path = os.path.join(self.dataset_path, "video", f"{video_id}.mp4")

        return {
            "question_id": question_id,
            "video_path": video_path,
            "question": question_text,
            "answer": answer_text,
            "dimensions": question_entry["dimensions"],
            "video_type": "video",
            "task_type": question_entry["video_type"]
        }


class MVBenchDataset:

    SYS = """Carefully watch the video and pay attention to the cause and sequence of events, \
    the detail and movement of objects, and the action and pose of persons. \
    Based on your observations, select the best option that accurately addresses the question.
    """

    def __init__(self):
        self.data_list_config = {
        "Action Sequence": ("action_sequence.json", "star/Charades_v1_480", "video", True),
        "Action Prediction": ("action_prediction.json", "star/Charades_v1_480", "video", True),
        "Action Antonym": ("action_antonym.json", "ssv2_video", "video", False),
        "Fine-grained Action": ("fine_grained_action.json", "Moments_in_Time_Raw/videos", "video", False),
        "Unexpected Action": ("unexpected_action.json", "FunQA_test/test", "video", False),
        "Object Existence": ("object_existence.json", "clevrer/video_validation", "video", False),
        "Object Interaction": ("object_interaction.json", "star/Charades_v1_480", "video", True),
        "Object Shuffle": ("object_shuffle.json", "perception/videos", "video", False),
        "Moving Direction": ("moving_direction.json", "clevrer/video_validation", "video", False),
        "Action Localization": ("action_localization.json", "sta/sta_video", "video", True),
        "Scene Transition": ("scene_transition.json", "scene_qa/video", "video", False),
        "Action Count": ("action_count.json", "perception/videos", "video", False),
        "Moving Count": ("moving_count.json", "clevrer/video_validation", "video", False),
        "Moving Attribute": ("moving_attribute.json", "clevrer/video_validation", "video", False),
        "State Change": ("state_change.json", "perception/videos", "video", False),
        # "Fine-grained Pose": ("fine_grained_pose.json", "nturgbd", "video", False),
        "Character Order": ("character_order.json", "perception/videos", "video", False),
        "Egocentric Navigation": ("egocentric_navigation.json", "vlnqa", "video", False),
        "Episodic Reasoning": ("episodic_reasoning.json", "tvqa/frames_fps3_hq", "frame", True),
        "Counterfactual Inference": ("counterfactual_inference.json", "clevrer/video_validation", "video", False),
        }

        pass

    def init(self, dataset_path):
        self.dataset_path = dataset_path
        self.data_list = []

        for task_name, v in self.data_list_config.items():
            json_path = os.path.join(dataset_path, "json", v[0])
            video_prefix = os.path.join(dataset_path, "video_unzipped", v[1])
            data_type = v[2]
            has_bound = v[3]

            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            for entry in json_data:
                self.data_list.append({
                    "task_type": task_name,
                    "json_path": json_path,
                    "video_prefix": video_prefix,
                    "data_type": data_type,
                    "has_bound": has_bound,
                    "data": entry
                })

    def __len__(self):
        return len(self.data_list)

    def get_item(self, idx):
        entry = self.data_list[idx]
        video_path = os.path.join(entry["video_prefix"], entry["data"]["video"])

        # If frames (tvqa), then it's a directory — else it's a video file
        if entry["data_type"] == "frame":
            video_path = os.path.join(entry["video_prefix"], entry["data"]["video"])

        # Compose QA prompt (with options)
        subtitle = entry["data"].get("subtitle", "").strip()

        question_text = ""
        if subtitle:
            question_text += f"Subtitle for the Video:\n{subtitle}\n\n"

        question_text += f"Question: {entry['data']['question']}\n"
        question_text += "Options:\n"

        answer_text = entry["data"]["answer"]
        answer_idx = -1

        for idx_opt, c in enumerate(entry["data"]["candidates"]):
            question_text += f"({chr(ord('A') + idx_opt)}) {c}\n"
            if c == answer_text:
                answer_idx = idx_opt

        question_text = question_text.rstrip()
        formatted_answer = f"({chr(ord('A') + answer_idx)}) {answer_text}"

        return {
            "question_id": f"{entry['task_type']}__{entry['data']['video']}__{idx}",
            "video_path": video_path,
            "question": question_text,
            "answer": formatted_answer,
            "task_type": entry["task_type"],
            "options": entry["data"]["candidates"],
            "bound": (entry["data"]["start"], entry["data"]["end"]) if entry["has_bound"] else None,
            "video_type": entry["data_type"] 
        }