"""
Check the actual format of label files
"""

import os

dataset_path = 'bone_fracture_detection.v4-v4.yolov8'
train_labels = os.path.join(dataset_path, 'train', 'labels')
train_images = os.path.join(dataset_path, 'train', 'images')

print("=" * 70)
print("LABEL FILE FORMAT INSPECTION")
print("=" * 70)

# Get all label files
label_files = sorted([f for f in os.listdir(train_labels) if f.endswith('.txt')])
print(f"\nTotal label files: {len(label_files)}")
print(f"Sample label files:")
for f in label_files[:5]:
    print(f"  - {f}")

# Get all image files
image_files = sorted([f for f in os.listdir(train_images) if f.endswith(('.jpg', '.png', '.jpeg'))])
print(f"\nTotal image files: {len(image_files)}")
print(f"Sample image files:")
for f in image_files[:5]:
    print(f"  - {f}")

# Check if they match
print("\n" + "=" * 70)
print("CHECKING IMAGE-LABEL CORRESPONDENCE")
print("=" * 70)

matched = 0
unmatched_images = []

for img_file in image_files[:20]:  # Check first 20
    base_name = img_file.rsplit('.', 1)[0]
    
    # Try different label file naming conventions
    possible_labels = [
        base_name + '.txt',  # Simple
        img_file.rsplit('.', 1)[0] + '.txt',  # With .rf part
    ]
    
    found = False
    for label_name in possible_labels:
        if os.path.exists(os.path.join(train_labels, label_name)):
            found = True
            matched += 1
            print(f"✓ MATCHED: {img_file}")
            print(f"           → {label_name}")
            break
    
    if not found:
        unmatched_images.append(img_file)
        print(f"✗ NOT FOUND: {img_file}")

print(f"\nMatched: {matched}/{len(image_files[:20])}")

# Read a sample label file
print("\n" + "=" * 70)
print("SAMPLE LABEL FILE CONTENT")
print("=" * 70)

sample_label = label_files[0]
sample_path = os.path.join(train_labels, sample_label)

print(f"File: {sample_label}\n")
with open(sample_path, 'r') as f:
    content = f.read()
    lines = content.strip().split('\n')
    
    print(f"Total lines: {len(lines)}\n")
    
    for i, line in enumerate(lines[:3]):  # Show first 3 lines
        coords = line.split()
        print(f"Line {i+1}:")
        print(f"  Class ID: {coords[0]}")
        print(f"  Coordinates: {len(coords) - 1} values")
        print(f"  Full: {line[:100]}...")
        
        # Check if normalized
        try:
            values = list(map(float, coords[1:]))
            print(f"  Min value: {min(values):.4f}")
            print(f"  Max value: {max(values):.4f}")
            print(f"  Appears normalized (0-1): {min(values) >= 0 and max(values) <= 1}")
        except:
            print(f"  Could not parse as numbers")
        print()

print("=" * 70)