from huggingface_hub import HfApi, Repository, login
import shutil
import os

HF_USERNAME = ""
REPO_NAME = "lexclipr-two_tower__bert-base-uncased__translated"
LOCAL_REPO_DIR = f"{REPO_NAME}"

QUERY_MODEL_DIR = "path/to/query/model"
CTX_MODEL_DIR = "path/to/ctx/model"

login()

api = HfApi()
repo_url = api.create_repo(repo_id=f"{HF_USERNAME}/{REPO_NAME}", exist_ok=True)

if os.path.exists(LOCAL_REPO_DIR):
    shutil.rmtree(LOCAL_REPO_DIR)

os.makedirs(f"{LOCAL_REPO_DIR}/query_model", exist_ok=True)
os.makedirs(f"{LOCAL_REPO_DIR}/ctx_model", exist_ok=True)

shutil.copytree(QUERY_MODEL_DIR, f"{LOCAL_REPO_DIR}/query_model", dirs_exist_ok=True)
shutil.copytree(CTX_MODEL_DIR, f"{LOCAL_REPO_DIR}/ctx_model", dirs_exist_ok=True)

api.upload_folder(
    folder_path=LOCAL_REPO_DIR,
    repo_id=f"rohit-upadhya/{REPO_NAME}",
    repo_type="model",
)


print("Ecoder uploaded")
