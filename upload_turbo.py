from huggingface_hub import HfApi, create_repo

api = HfApi()

# Upload turbo decoder model
repo_id = "prompterminal/eidetic-turbo-fdm-decoder"

print(f"Uploading to {repo_id}...")

try:
    create_repo(repo_id, repo_type="model", exist_ok=True)
except Exception as e:
    print(f"Repo note: {e}")

# Upload the model weights
api.upload_file(
    path_or_fileobj="turbo_fdm_decoder_best.pt",
    path_in_repo="turbo_fdm_decoder_best.pt",
    repo_id=repo_id,
    repo_type="model",
)

# Upload the encoder and decoder scripts
api.upload_file(
    path_or_fileobj="turbo_fdm_v2.py",
    path_in_repo="turbo_fdm_v2.py",
    repo_id=repo_id,
    repo_type="model",
)

api.upload_file(
    path_or_fileobj="turbo_fdm_decoder.py",
    path_in_repo="turbo_fdm_decoder.py",
    repo_id=repo_id,
    repo_type="model",
)

print(f"✅ Done: https://huggingface.co/{repo_id}")
