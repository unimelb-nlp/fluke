from huggingface_hub import HfApi, create_repo

# Initialize API
api = HfApi()

# Create repository (replace with your username)
repo_id = "joey234/fluke"
create_repo(repo_id, repo_type="dataset", private=False)

# Upload all files
api.upload_folder(
    folder_path="./fluke_dataset_standard",
    repo_id=repo_id,
    repo_type="dataset"
)