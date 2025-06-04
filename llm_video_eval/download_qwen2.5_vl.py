from huggingface_hub import snapshot_download

# Target repo
REPO_ID = "Qwen/Qwen-VL-Chat"
TARGET_DIR = "models/Qwen2.5-VL-7B-Instruct"

# Download
print(f"Downloading {REPO_ID} to {TARGET_DIR}...")
snapshot_download(
    repo_id=REPO_ID,
    local_dir=TARGET_DIR,
    local_dir_use_symlinks=False,
    revision=None  # you can pin revision if needed
)
print("Download completed!")
