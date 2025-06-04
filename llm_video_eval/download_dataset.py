from huggingface_hub import snapshot_download

snapshot_download(repo_id="OpenGVLab/MVBench", repo_type="dataset", local_dir="datasets/MVBench")
