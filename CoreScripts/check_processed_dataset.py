import os

processed_dir = 'processed_dataset'

print("=" * 60)
print("PROCESSED DATASET STRUCTURE")
print("=" * 60)

for split in ['train', 'valid', 'val', 'test']:
    split_path = os.path.join(processed_dir, split)
    
    if os.path.exists(split_path):
        images_path = os.path.join(split_path, 'images')
        masks_path = os.path.join(split_path, 'masks')
        labels_path = os.path.join(split_path, 'labels')
        
        img_count = len(os.listdir(images_path)) if os.path.exists(images_path) else 0
        mask_count = len(os.listdir(masks_path)) if os.path.exists(masks_path) else 0
        label_count = len(os.listdir(labels_path)) if os.path.exists(labels_path) else 0
        
        print(f"\n{split}:")
        print(f"  Images: {img_count}")
        print(f"  Masks: {mask_count}")
        print(f"  Labels: {label_count}")
    else:
        print(f"\n{split}: NOT FOUND")

print("\n" + "=" * 60)