from huggingface_hub import HfApi, create_repo

api = HfApi()
repo_id = "prompterminal/eidetic-turbo-high-capacity-decoder"

print(f"Uploading to {repo_id}...")
try:
    create_repo(repo_id, repo_type="model", exist_ok=True)
except Exception as e:
    print(f"Repo note: {e}")

api.upload_file(
    path_or_fileobj="turbo_high_capacity_decoder_best.pt",
    path_in_repo="turbo_high_capacity_decoder_best.pt",
    repo_id=repo_id,
    repo_type="model",
)
api.upload_file(
    path_or_fileobj="turbo_fdm_high_capacity.py",
    path_in_repo="turbo_fdm_high_capacity.py",
    repo_id=repo_id,
    repo_type="model",
)
api.upload_file(
    path_or_fileobj="turbo_fdm_high_capacity_decoder.py",
    path_in_repo="turbo_fdm_high_capacity_decoder.py",
    repo_id=repo_id,
    repo_type="model",
)
print(f"✅ Done: https://huggingface.co/{repo_id}")
