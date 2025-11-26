"""
Debug script to inspect dataset structure and label format
Run this BEFORE running preprocessing
"""

import os
import cv2
from pathlib import Path

def inspect_dataset():
    """Inspect dataset structure"""
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    
    # Find dataset
    dataset_path = None
    for name in ['bone_fracture_detection.v4-v4.yolov8', 'bone_fracture_detection', '.']:
        if os.path.exists(name):
            if os.path.exists(os.path.join(name, 'data.yaml')):
                dataset_path = name
                print(f"✓ Found dataset at: {dataset_path}")
                break
    
    if dataset_path is None:
        print("❌ Dataset not found!")
        return
    
    # List directory structure
    print("\nDirectory Structure:")
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        
        # Only go 3 levels deep
        if level < 3:
            subindent = ' ' * 2 * (level + 1)
            for f in files[:5]:  # Show first 5 files
                print(f'{subindent}{f}')
            if len(files) > 5:
                print(f'{subindent}... and {len(files) - 5} more files')
    
    # Check train split
    train_images = os.path.join(dataset_path, 'train', 'images')
    train_labels = os.path.join(dataset_path, 'train', 'labels')
    
    print(f"\n{'=' * 60}")
    print("TRAIN SPLIT ANALYSIS")
    print(f"{'=' * 60}")
    
    if not os.path.exists(train_images):
        print(f"❌ Train images directory not found: {train_images}")
        return
    
    image_files = [f for f in os.listdir(train_images) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(image_files)} training images")
    
    if len(image_files) == 0:
        print("❌ No images found!")
        return
    
    # Check first image
    first_image = image_files[0]
    img_path = os.path.join(train_images, first_image)
    img = cv2.imread(img_path)
    print(f"\nFirst image: {first_image}")
    print(f"  Size: {img.shape if img is not None else 'ERROR'}")
    
    # Check first label
    label_file = first_image.split('.')[0] + '.txt'
    label_path = os.path.join(train_labels, label_file)
    
    print(f"\nCorresponding label: {label_file}")
    
    if not os.path.exists(label_path):
        print(f"❌ Label file not found: {label_path}")
        return
    
    with open(label_path, 'r') as f:
        content = f.read()
        print(f"Label content (first 500 chars):")
        print(content[:500])
        
        lines = content.strip().split('\n')
        print(f"\nTotal lines: {len(lines)}")
        
        if len(lines) > 0:
            first_line = lines[0].split()
            print(f"First line breakdown:")
            print(f"  Class ID: {first_line[0]}")
            print(f"  Coordinates count: {len(first_line) - 1}")
            print(f"  Full line: {' '.join(first_line[:10])}...")
            
            # Check if coordinates look normalized (0-1)
            try:
                coords = list(map(float, first_line[1:]))
                print(f"  Min coord: {min(coords):.4f}")
                print(f"  Max coord: {max(coords):.4f}")
                print(f"  Format: {'NORMALIZED (0-1) ✓' if min(coords) >= 0 and max(coords) <= 1 else 'PIXEL COORDINATES'}")
            except:
                print("  ⚠️  Could not parse coordinates")
    
    # Count total files
    print(f"\n{'=' * 60}")
    print("FULL DATASET COUNT")
    print(f"{'=' * 60}")
    
    for split in ['train', 'val', 'test']:
        split_images = os.path.join(dataset_path, split, 'images')
        split_labels = os.path.join(dataset_path, split, 'labels')
        
        if os.path.exists(split_images):
            img_count = len([f for f in os.listdir(split_images) if f.endswith(('.jpg', '.png', '.jpeg'))])
            label_count = len([f for f in os.listdir(split_labels) if f.endswith('.txt')]) if os.path.exists(split_labels) else 0
            print(f"{split:5s}: {img_count:4d} images, {label_count:4d} labels")
        else:
            print(f"{split:5s}: NOT FOUND")
    
    print(f"\n{'=' * 60}")
    print("✓ Inspection Complete")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    inspect_dataset()