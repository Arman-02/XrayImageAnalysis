"""
Create validation split from training data (80-20 split)
"""

import os
import shutil
import random
from pathlib import Path

def create_validation_split(dataset_dir='processed_dataset', split_ratio=0.2):
    """Create validation split from training data"""
    
    train_images = os.path.join(dataset_dir, 'train/images')
    train_masks = os.path.join(dataset_dir, 'train/masks')
    train_labels = os.path.join(dataset_dir, 'train/labels')
    
    val_images = os.path.join(dataset_dir, 'val/images')
    val_masks = os.path.join(dataset_dir, 'val/masks')
    val_labels = os.path.join(dataset_dir, 'val/labels')
    
    print("=" * 60)
    print("CREATING VALIDATION SPLIT")
    print("=" * 60)
    
    # Get all training files
    image_files = sorted([f for f in os.listdir(train_images) if f.endswith(('.jpg', '.png'))])
    print(f"\nTotal training images: {len(image_files)}")
    
    # Randomly select validation samples
    num_val = int(len(image_files) * split_ratio)
    print(f"Moving {num_val} images to validation ({split_ratio*100}%)")
    print(f"Keeping {len(image_files) - num_val} images for training")
    
    # Random seed for reproducibility
    random.seed(42)
    val_files = random.sample(image_files, num_val)
    
    # Create validation directories if they don't exist
    os.makedirs(val_images, exist_ok=True)
    os.makedirs(val_masks, exist_ok=True)
    os.makedirs(val_labels, exist_ok=True)
    
    # Move files
    print("\nMoving files...")
    for img_file in val_files:
        # Image
        src = os.path.join(train_images, img_file)
        dst = os.path.join(val_images, img_file)
        shutil.move(src, dst)
        
        # Mask
        mask_file = img_file.rsplit('.', 1)[0] + '_mask.png'
        src = os.path.join(train_masks, mask_file)
        dst = os.path.join(val_masks, mask_file)
        shutil.move(src, dst)
        
        # Label
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        src = os.path.join(train_labels, label_file)
        dst = os.path.join(val_labels, label_file)
        shutil.move(src, dst)
    
    # Verify
    train_count = len(os.listdir(train_images))
    val_count = len(os.listdir(val_images))
    
    print("\n" + "=" * 60)
    print("VALIDATION SPLIT CREATED")
    print("=" * 60)
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Total: {train_count + val_count}")
    print("=" * 60)

if __name__ == '__main__':
    create_validation_split(dataset_dir='processed_dataset', split_ratio=0.2)