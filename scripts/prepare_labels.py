# scripts/prepare_labels.py
import os
import pandas as pd

def main():
    data_dir = "data/skin_images"
    metadata_path = os.path.join(data_dir, "HAM10000_metadata.csv")
    output_path = os.path.join(data_dir, "labels.csv")
    
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    print(f"Reading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    
    # We need to map image_id to filename
    # The images are likely in 'all_images' folder or split into parts.
    # Based on previous exploration, they seem to be in 'all_images' (we saw ISIC_0024306.jpg there)
    # Let's assume they are in 'all_images' for now, or check where they are.
    
    # Check if 'all_images' exists
    all_images_dir = os.path.join(data_dir, "all_images")
    if not os.path.exists(all_images_dir):
        # Fallback: check if they are in part_1 and part_2 and maybe we need to look there?
        # But the user seems to have an 'all_images' folder from previous `ls` command.
        print(f"Warning: {all_images_dir} does not exist. Checking for parts...")
        # If all_images doesn't exist, we might need to look into part_1 and part_2
        # For now, let's assume the user has consolidated them or we can find them.
        pass

    # Construct filename
    # We will use relative paths to data_dir in labels.csv to be safe, or absolute?
    # The training script handles relative paths if they are not absolute.
    # Let's try to find where each image is.
    
    image_paths = {}
    
    # Scan directories
    dirs_to_scan = ["all_images", "HAM10000_images_part_1", "HAM10000_images_part_2"]
    for d in dirs_to_scan:
        full_d = os.path.join(data_dir, d)
        if os.path.exists(full_d):
            print(f"Scanning {d}...")
            for f in os.listdir(full_d):
                if f.endswith(".jpg"):
                    img_id = os.path.splitext(f)[0]
                    image_paths[img_id] = os.path.join(d, f)
    
    print(f"Found {len(image_paths)} images.")
    
    # Create new dataframe
    # Columns: filename, label
    new_rows = []
    found_count = 0
    missing_count = 0
    
    for idx, row in df.iterrows():
        img_id = row['image_id']
        dx = row['dx']
        
        if img_id in image_paths:
            new_rows.append({
                "filename": image_paths[img_id],
                "label": dx
            })
            found_count += 1
        else:
            missing_count += 1
            
    print(f"Matched {found_count} images. Missing {missing_count}.")
    
    out_df = pd.DataFrame(new_rows)
    out_df.to_csv(output_path, index=False)
    print(f"Saved labels to {output_path}")
    
    # Print label distribution
    print("Label distribution:")
    print(out_df['label'].value_counts())

if __name__ == "__main__":
    main()
