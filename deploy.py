import os
from huggingface_hub import HfApi, login

login(token=os.environ["HF_TOKEN"])
api = HfApi()

api.upload_folder(
    folder_path="./backend",
    repo_id="sas1ru/neuroheaven-backend",
    repo_type="space",
    ignore_patterns=["__pycache__", "*.pyc", ".env"]
)

# api.upload_folder(
#     folder_path="./frontend",
#     repo_id="sas1ru/neuroheaven-frontend",
#     repo_type="space",
#     ignore_patterns=["__pycache__", "*.pyc", ".env"]
# )

print("Deployment complete!")