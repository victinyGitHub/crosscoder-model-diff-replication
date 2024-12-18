from utils import *
from crosscoder import CrossCoder
from huggingface_hub import HfApi, create_repo
import tempfile
import os

def upload_checkpoint_to_hf(
    version_dir: str, 
    checkpoint_version: str,
    repo_id: str = "victiny1223/crosscoder-checkpoints",
    commit_message: str = None
):
    """
    Upload a specific checkpoint to HuggingFace.
    
    Args:
        version_dir: Directory name (e.g., 'version_0')
        checkpoint_version: Checkpoint number to upload
        repo_id: HuggingFace repo ID
        commit_message: Optional commit message
    """
    # Load the checkpoint
    save_dir = Path(os.path.expanduser("~/.cache/huggingface/crosscoder-model-diff-replication/checkpoints")) / version_dir
    cfg_path = save_dir / f"{checkpoint_version}_cfg.json"
    weight_path = save_dir / f"{checkpoint_version}.pt"
    
    if not cfg_path.exists() or not weight_path.exists():
        raise FileNotFoundError(f"Checkpoint files not found in {save_dir}")
    
    # Create repo if needed
    try:
        create_repo(repo_id, exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload files
    api = HfApi()
    
    # Upload weights
    print(f"Uploading weights from {weight_path}")
    api.upload_file(
        path_or_fileobj=str(weight_path),
        path_in_repo=f"checkpoints/{version_dir}/{checkpoint_version}.pt",
        repo_id=repo_id,
        commit_message=commit_message or f"Upload checkpoint {checkpoint_version} from {version_dir}"
    )
    
    # Upload config
    print(f"Uploading config from {cfg_path}")
    api.upload_file(
        path_or_fileobj=str(cfg_path),
        path_in_repo=f"checkpoints/{version_dir}/{checkpoint_version}_cfg.json",
        repo_id=repo_id,
        commit_message=commit_message or f"Upload checkpoint {checkpoint_version} from {version_dir}"
    )
    
    print(f"Successfully uploaded checkpoint {checkpoint_version} from {version_dir} to {repo_id}")

if __name__ == "__main__":
    # Example usage:
    upload_checkpoint_to_hf(
        version_dir="version_13",
        checkpoint_version="2"
    )
