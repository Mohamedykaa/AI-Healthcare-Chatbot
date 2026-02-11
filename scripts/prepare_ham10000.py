# scripts/prepare_ham10000.py
import os
import shutil
import pandas as pd

# Configuration
DATA_DIR = r"d:\disease_prediction_project\data\skin_images"
METADATA_FILE = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
PART1_DIR = os.path.join(DATA_DIR, "HAM10000_images_part_1")
PART2_DIR = os.path.join(DATA_DIR, "HAM10000_images_part_2")
OUTPUT_DIR = os.path.join(DATA_DIR, "all_images")
LABELS_FILE = os.path.join(OUTPUT_DIR, "labels.csv")

# Diagnosis mapping (abbreviation -> full name)
DX_MAP = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

def main():
    print("Preparing HAM10000 dataset...")
    
    if not os.path.exists(METADATA_FILE):
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read metadata
    df = pd.read_csv(METADATA_FILE)
    print(f"Loaded metadata with {len(df)} records.")
    
    # Prepare new labels dataframe
    new_labels = []
    
    # Helper to find image
    def find_image(image_id):
        filename = f"{image_id}.jpg"
        p1 = os.path.join(PART1_DIR, filename)
        p2 = os.path.join(PART2_DIR, filename)
        if os.path.exists(p1):
            return p1
        if os.path.exists(p2):
            return p2
        return None

    print("Consolidating images and creating labels...")
    count = 0
    total = len(df)
    for _, row in df.iterrows():
        image_id = row['image_id']
        dx = row['dx']
        label = DX_MAP.get(dx, dx)
        
        src_path = find_image(image_id)
        if src_path:
            dst_filename = f"{image_id}.jpg"
            dst_path = os.path.join(OUTPUT_DIR, dst_filename)
            
            # Copy file if not exists (to save time on re-runs)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
            
            new_labels.append({
                'filename': dst_filename,
                'label': label
            })
        else:
            print(f"Warning: Image {image_id} not found.")
        
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{total} images...", end='\r')
            
    # Save labels.csv
    labels_df = pd.DataFrame(new_labels)
    labels_df.to_csv(LABELS_FILE, index=False)
    print(f"\nSuccessfully prepared {len(labels_df)} images in {OUTPUT_DIR}")
    print(f"Labels saved to {LABELS_FILE}")
    print("\nRun to run training:")
    print(f"python src/train_image_model.py --data_dir {OUTPUT_DIR} --use_csv --epochs 10")

if __name__ == "__main__":
    main()
