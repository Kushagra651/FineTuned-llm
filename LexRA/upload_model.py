from huggingface_hub import HfApi
import os
import dotenv

dotenv.load_dotenv()

api = HfApi()
api.upload_folder(
    folder_path="G:/My Drive/LexRA/model/best_model",
    repo_id="aapnakaamkar/LexRA-TinyLlama-Legal",
    repo_type="model",
    token=os.getenv("HF_TOKEN")
)
print("Upload complete!")