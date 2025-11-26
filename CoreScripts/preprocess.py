import os
import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class BoneFracturePreprocessor:
    def __init__(self, dataset_path, output_path='processed_dataset'):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.class_names = {}
        self.statistics = {
            'total_images': 0,
            'class_distribution': {},
            'image_sizes': [],
            'corrupted_files': [],
            'empty_masks': []
        }
        
    def create_output_structure(self):
        """Create processed dataset directory structure"""
        os.makedirs(self.output_path, exist_ok=True)
        for split in ['train', 'val', 'test']:
            os.makedirs(f'{self.output_path}/{split}/images', exist_ok=True)
            os.makedirs(f'{self.output_path}/{split}/labels', exist_ok=True)
            os.makedirs(f'{self.output_path}/{split}/masks', exist_ok=True)
        print(f"✓ Created output directory structure at {self.output_path}")
    
    def extract_classes(self):
        """Extract class information from data.yaml"""
        yaml_path = os.path.join(self.dataset_path, 'data.yaml')
        if os.path.exists(yaml_path):
            try:
                import yaml
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                    names = data.get('names', {})
                    # Handle both dict and list formats
                    if isinstance(names, list):
                        self.class_names = {i: name for i, name in enumerate(names)}
                    else:
                        self.class_names = names
                    print(f"✓ Found classes: {self.class_names}")
            except Exception as e:
                print(f"⚠ Could not parse data.yaml: {e}")
        else:
            print("⚠ data.yaml not found")
    
    def polygon_to_mask(self, image_shape, polygon_coords):
        """Convert polygon coordinates to binary segmentation mask"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        try:
            # Normalize polygon coordinates to pixel coordinates
            polygon = np.array(polygon_coords, dtype=np.float32).reshape(-1, 2)
            
            if len(polygon) < 3:
                return None
            
            # Convert from normalized (0-1) to pixel coordinates
            polygon[:, 0] = polygon[:, 0] * w
            polygon[:, 1] = polygon[:, 1] * h
            
            # Clip to image bounds
            polygon = np.clip(polygon, 0, [w-1, h-1])
            polygon = polygon.astype(np.int32)
            
            # Fill polygon
            cv2.fillPoly(mask, [polygon], 1)
            
            # Check if mask has any pixels
            if mask.sum() > 10:  # At least 10 pixels
                return mask
            else:
                return None
        except Exception as e:
            return None
    
    def parse_label_file(self, label_path, image_shape):
        """Parse polygon label file and generate segmentation mask"""
        masks = []
        classes = []
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    coords = list(map(float, line.split()))
                    
                    if len(coords) < 3:
                        continue
                    
                    class_id = int(coords[0])
                    polygon_coords = coords[1:]
                    
                    # Need at least 6 coordinates (3 points, 2 coords each)
                    if len(polygon_coords) >= 6 and len(polygon_coords) % 2 == 0:
                        mask = self.polygon_to_mask(image_shape, polygon_coords)
                        if mask is not None:
                            masks.append(mask)
                            classes.append(class_id)
        except Exception as e:
            pass
        
        return masks, classes
    
    def process_image(self, image_path, label_path, split):
        """Process single image with augmentation and validation"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                self.statistics['corrupted_files'].append(image_path)
                return False
            
            # Resize to uniform size (640x640 for efficiency on M4 Pro)
            original_h, original_w = image.shape[:2]
            target_size = 640
            
            # Maintain aspect ratio with padding
            scale = min(target_size / original_w, target_size / original_h)
            new_w, new_h = int(original_w * scale), int(original_h * scale)
            
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Add padding
            padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            x_offset = (target_size - new_w) // 2
            y_offset = (target_size - new_h) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            # Generate masks from label file
            masks, classes = self.parse_label_file(label_path, (new_h, new_w))
            
            if len(masks) == 0:
                self.statistics['empty_masks'].append(image_path)
                return False
            
            # Combine masks and create padded version
            combined_mask = np.zeros((target_size, target_size), dtype=np.uint8)
            for mask in masks:
                padded_mask = np.zeros((target_size, target_size), dtype=np.uint8)
                padded_mask[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = mask
                combined_mask = np.maximum(combined_mask, padded_mask)
            
            # Save processed files
            base_name = os.path.basename(image_path).rsplit('.', 1)[0]
            cv2.imwrite(f'{self.output_path}/{split}/images/{base_name}.jpg', padded)
            cv2.imwrite(f'{self.output_path}/{split}/masks/{base_name}_mask.png', combined_mask * 255)
            
            # Save classes info
            with open(f'{self.output_path}/{split}/labels/{base_name}.txt', 'w') as f:
                f.write(','.join(map(str, classes)))
            
            self.statistics['image_sizes'].append((new_h, new_w))
            for cls in classes:
                self.statistics['class_distribution'][cls] = self.statistics['class_distribution'].get(cls, 0) + 1
            
            return True
        
        except Exception as e:
            self.statistics['corrupted_files'].append(image_path)
            return False
    
    def find_label_file(self, img_file, labels_dir):
        """Find corresponding label file (handles different naming conventions)"""
        # For this dataset, simply replace .jpg/.png with .txt
        base_name = img_file.rsplit('.', 1)[0]
        label_file = base_name + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        if os.path.exists(label_path):
            return label_path
        return None
    
    def preprocess_split(self, split_name):
        """Process all images in a split"""
        split_path = os.path.join(self.dataset_path, split_name)
        images_dir = os.path.join(split_path, 'images')
        labels_dir = os.path.join(split_path, 'labels')
        
        if not os.path.exists(images_dir):
            print(f"⚠ {split_name} split not found at {images_dir}")
            return 0
        
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        if len(image_files) == 0:
            print(f"⚠ No images found in {split_name}")
            return 0
        
        print(f"\nProcessing {split_name} split ({len(image_files)} images)...")
        
        success_count = 0
        for img_file in tqdm(image_files, desc=f"{split_name}"):
            img_path = os.path.join(images_dir, img_file)
            label_path = self.find_label_file(img_file, labels_dir)
            
            if label_path is None:
                continue
            
            if self.process_image(img_path, label_path, split_name):
                success_count += 1
        
        print(f"✓ {split_name}: {success_count}/{len(image_files)} images processed successfully")
        return success_count
    
    def run(self):
        """Execute full preprocessing pipeline"""
        print("=" * 60)
        print("BONE FRACTURE DATASET PREPROCESSING")
        print("=" * 60)
        
        self.create_output_structure()
        self.extract_classes()
        
        total_processed = 0
        for split in ['train', 'valid', 'test']:  # Note: dataset uses 'valid' not 'val'
            count = self.preprocess_split(split)
            if count:
                total_processed += count
        
        self.statistics['total_images'] = sum(self.statistics['class_distribution'].values())
        
        # Print statistics
        print("\n" + "=" * 60)
        print("PREPROCESSING STATISTICS")
        print("=" * 60)
        print(f"Total images processed: {self.statistics['total_images']}")
        print(f"Corrupted files: {len(self.statistics['corrupted_files'])}")
        print(f"Empty masks: {len(self.statistics['empty_masks'])}")
        print(f"\nClass Distribution:")
        
        if self.statistics['class_distribution']:
            for cls_id in sorted(self.statistics['class_distribution'].keys()):
                count = self.statistics['class_distribution'][cls_id]
                cls_name = self.class_names.get(cls_id, f"Class {cls_id}")
                print(f"  {cls_name}: {count} images")
        else:
            print("  No classes found")
        
        print(f"\nOutput directory: {self.output_path}")
        print("=" * 60)
    
    def visualize_samples(self, num_samples=6):
        """Visualize preprocessed samples"""
        print(f"\nVisualizing {num_samples} random samples...")
        
        train_images = os.path.join(self.output_path, 'train/images')
        train_masks = os.path.join(self.output_path, 'train/masks')
        
        if not os.path.exists(train_images):
            print("⚠ No train images to visualize")
            return
        
        images = sorted(os.listdir(train_images))[:num_samples]
        
        if len(images) == 0:
            print("⚠ No images found to visualize")
            return
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for idx, img_file in enumerate(images):
            img_path = os.path.join(train_images, img_file)
            mask_file = img_file.split('.')[0] + '_mask.png'
            mask_path = os.path.join(train_masks, mask_file)
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            axes[idx, 0].imshow(img)
            axes[idx, 0].set_title(f'Image: {img_file}', fontsize=10)
            axes[idx, 0].axis('off')
            
            if mask is not None:
                axes[idx, 1].imshow(mask, cmap='gray')
                axes[idx, 1].set_title(f'Mask', fontsize=10)
            axes[idx, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('visualization_samples.png', dpi=100, bbox_inches='tight')
        print("✓ Visualization saved as 'visualization_samples.png'")
        plt.close()

if __name__ == '__main__':
    # Find dataset
    dataset_path = None
    for name in ['bone_fracture_detection.v4-v4.yolov8', 'bone_fracture_detection', '.']:
        if os.path.exists(name) and os.path.exists(os.path.join(name, 'data.yaml')):
            dataset_path = name
            print(f"✓ Found dataset at: {dataset_path}")
            break
    
    if dataset_path is None:
        print("ERROR: Could not find dataset!")
        print("Make sure the dataset is extracted in the current directory")
        exit(1)
    
    preprocessor = BoneFracturePreprocessor(dataset_path)
    preprocessor.run()
    preprocessor.visualize_samples(num_samples=6)