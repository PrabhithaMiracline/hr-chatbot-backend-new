from huggingface_hub import snapshot_download

model_name = "sentence-transformers/all-MiniLM-L6-v2"
save_dir = r"D:\hr-assistant-chatbot\hugging_face"

# Download the whole model repo
snapshot_download(repo_id=model_name, local_dir=save_dir)

print(f"Model downloaded to: {save_dir}")
