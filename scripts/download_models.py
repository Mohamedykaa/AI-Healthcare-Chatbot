import os
from huggingface_hub import hf_hub_download

# Configuration
# Switching to MaziyarPanahi as TheBloke's failing
REPO_ID = "MaziyarPanahi/BioMistral-7B-GGUF"
FILENAME = "BioMistral-7B.Q4_K_M.gguf" 
# We need to rename it to what the app expects if we want it to work out of box
# But hf_hub_download downloads the exact filename.
# We will rename it after download.
DEST_DIR = "models"
TARGET_FILENAME = "biomistral-7b.gguf"

def main():
    print(f"⬇️ Downloading {FILENAME} from {REPO_ID} using huggingface_hub...")
    print("   This handles authentication and resuming automatically.")
    
    try:
        # Download (will cache in ~/.cache/huggingface and link, or we can move it)
        # We want it in our local 'models' folder.
        # hf_hub_download doesn't strictly download "to" a specific custom path easily without cache 
        # unless we assume cache usage. 
        # Actually 'local_dir' parameter exists in newer versions.
        
        file_path = hf_hub_download(
            repo_id=REPO_ID, 
            filename=FILENAME, 
            local_dir=DEST_DIR,
            local_dir_use_symlinks=False,
            force_download=False,
            token=False # Explicitly disable authentication
        )
        
        print(f"✅ Download Complete! File saved at: {file_path}")
        
        # Rename/Move to expected location
        import shutil
        target_path = os.path.join(DEST_DIR, TARGET_FILENAME)
        if file_path != target_path:
             shutil.copy(file_path, target_path)
             print(f"✅ Renamed/Copied to: {target_path}")

        print("\n🎉 You can now run the app in PRO mode.")
        
    except Exception as e:
        print(f"\n❌ Download Failed: {e}")
        print("Try running: pip install huggingface_hub --upgrade")

if __name__ == "__main__":
    main()
