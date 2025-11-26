"""
Train a fracture type classifier on top of the segmentation model
Uses frozen segmentation features to classify fracture location/type
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from tqdm import tqdm
import json
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from model import BoneFractureSegmentationNet

# Fracture type mapping
FRACTURE_CLASSES = {
    'elbow positive': 'Elbow',
    'fingers positive': 'Fingers',
    'forearm fracture': 'Forearm',
    'humerus fracture': 'Humerus',
    'humerus': 'Humerus',
    'shoulder fracture': 'Shoulder',
    'wrist positive': 'Wrist'
}

# Create reverse mapping: class_id -> class_name
CLASS_NAMES = list(set(FRACTURE_CLASSES.values()))  # Remove duplicates
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

print("Fracture Classes:")
for idx, name in IDX_TO_CLASS.items():
    print(f"  {idx}: {name}")

class FractureClassifierDataset(Dataset):
    """Dataset for fracture type classification with augmentation"""
    def __init__(self, dataset_dir, split='train'):
        self.split = split
        self.images_dir = f'{dataset_dir}/{split}/images'
        self.masks_dir = f'{dataset_dir}/{split}/masks'
        self.labels_dir = f'{dataset_dir}/{split}/labels'
        
        # Get valid images with masks and class labels
        self.samples = []
        for f in sorted(os.listdir(self.images_dir)):
            if f.endswith(('.jpg', '.png')):
                mask_file = f.rsplit('.', 1)[0] + '_mask.png'
                label_file = f.rsplit('.', 1)[0] + '.txt'
                
                mask_path = os.path.join(self.masks_dir, mask_file)
                label_path = os.path.join(self.labels_dir, label_file)
                
                if os.path.exists(mask_path) and os.path.exists(label_path):
                    self.samples.append((f, label_file))
        
        print(f"Dataset {split}: Found {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def _augment(self, image):
        """Aggressive augmentation for training"""
        if self.split != 'train':
            return image
        
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
        
        # Random vertical flip
        if np.random.rand() > 0.5:
            image = np.flipud(image).copy()
        
        # Random rotation (-15 to 15 degrees)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Random brightness
        if np.random.rand() > 0.5:
            brightness = np.random.uniform(0.7, 1.3)
            image = np.clip(image * brightness, 0, 1)
        
        # Random contrast
        if np.random.rand() > 0.5:
            contrast = np.random.uniform(0.7, 1.3)
            image = np.clip((image - 0.5) * contrast + 0.5, 0, 1)
        
        # Random Gaussian blur
        if np.random.rand() > 0.7:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Random noise
        if np.random.rand() > 0.7:
            noise = np.random.normal(0, 0.05, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        # Random elastic deformation
        if np.random.rand() > 0.7:
            alpha = 30
            sigma = 5
            random_state = np.random.RandomState()
            shape = image.shape[:2]
            
            dx = random_state.randn(*shape) * sigma
            dy = random_state.randn(*shape) * sigma
            
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            x = (x + alpha * dx).astype(np.float32)
            y = (y + alpha * dy).astype(np.float32)
            
            for c in range(image.shape[2]):
                image[:, :, c] = cv2.remap(image[:, :, c], x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return image
    
    def __getitem__(self, idx):
        img_file, label_file = self.samples[idx]
        
        img_path = os.path.join(self.images_dir, img_file)
        label_path = os.path.join(self.labels_dir, label_file)
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        image = image.astype(np.float32) / 255.0
        image = np.ascontiguousarray(image)
        
        # Augmentation
        image = self._augment(image)
        
        # Read class label
        with open(label_path, 'r') as f:
            class_id = int(f.readline().split(',')[0])
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return image, class_id

class FractureClassifier(nn.Module):
    """Lightweight classifier for fracture types"""
    def __init__(self, num_classes=7, frozen_backbone=True):
        super().__init__()
        self.num_classes = num_classes
        
        # Load pretrained segmentation model as backbone
        self.backbone = BoneFractureSegmentationNet(pretrained=True)
        
        # Freeze backbone
        if frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classification head - more sophisticated
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Combine avg and max pooling features
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 2, 1024),  # Doubled from 2048 due to concat
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Extract features from backbone encoder
        enc0 = self.backbone.encoder0(x)
        enc1 = self.backbone.encoder1(enc0)
        enc2 = self.backbone.encoder2(enc1)
        enc3 = self.backbone.encoder3(enc2)
        enc4 = self.backbone.encoder4(enc3)
        
        # Global average and max pooling
        avg_features = self.avgpool(enc4)
        avg_features = avg_features.view(avg_features.size(0), -1)
        
        max_features = self.maxpool(enc4)
        max_features = max_features.view(max_features.size(0), -1)
        
        # Concatenate features
        features = torch.cat([avg_features, max_features], dim=1)
        
        # Classification
        logits = self.classifier(features)
        return logits

class ClassifierTrainer:
    def __init__(self, num_classes, device, class_weights=None):
        self.device = device
        self.num_classes = num_classes
        
        self.model = FractureClassifier(num_classes=num_classes, frozen_backbone=True)
        self.model = self.model.to(device)
        
        # Use class weights if provided
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using class weights: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / total
        return avg_loss, avg_acc
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_acc = correct / total if total > 0 else 0
        return avg_loss, avg_acc
    
    def train(self, train_loader, val_loader, epochs=50):
        print("\n" + "="*60)
        print("TRAINING FRACTURE CLASSIFIER")
        print("="*60)
        
        best_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch [{epoch+1}/{epochs}]")
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(checkpoint, 'checkpoints/best_classifier.pth')
                print(f"  ✓ Best model saved (Acc: {val_acc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⚠️  Early stopping triggered (no improvement for {patience} epochs)")
                    break
            
            self.scheduler.step()
        
        print("\n" + "="*60)
        print("CLASSIFIER TRAINING COMPLETED")
        print("="*60)

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = FractureClassifierDataset('processed_dataset', split='train')
    val_dataset = FractureClassifierDataset('processed_dataset', split='val')
    
    if len(train_dataset) == 0:
        print("❌ No training data found!")
        return
    
    # Calculate class weights
    print("\nCalculating class weights...")
    class_counts = Counter()
    for _, label in train_dataset.samples:
        with open(os.path.join(train_dataset.labels_dir, label), 'r') as f:
            class_id = int(f.readline().split(',')[0])
            class_counts[class_id] += 1
    
    # Weight = total / (num_classes * class_count)
    total = sum(class_counts.values())
    class_weights = {}
    for class_id in range(len(IDX_TO_CLASS)):
        if class_id in class_counts:
            class_weights[class_id] = total / (len(IDX_TO_CLASS) * class_counts[class_id])
        else:
            class_weights[class_id] = 1.0
    
    print("Class weights:")
    for class_id in sorted(class_weights.keys()):
        class_name = IDX_TO_CLASS.get(str(class_id), f"Class {class_id}")
        weight = class_weights[class_id]
        print(f"  {class_name}: {weight:.2f}x")
    
    # Convert to list format for PyTorch
    class_weights_list = [class_weights[i] for i in range(len(IDX_TO_CLASS))]
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Train classifier with class weights
    trainer = ClassifierTrainer(num_classes=len(IDX_TO_CLASS), device=device, class_weights=class_weights_list)
    trainer.train(train_loader, val_loader, epochs=50)
    
    # Save class mapping
    mapping = {
        'class_to_idx': CLASS_TO_IDX,
        'idx_to_class': IDX_TO_CLASS,
        'class_weights': class_weights,
        'history': trainer.history
    }
    with open('checkpoints/classifier_mapping.json', 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print("✓ Classifier training completed!")
    print(f"✓ Best model saved to: checkpoints/best_classifier.pth")
    print(f"✓ Class mapping saved to: checkpoints/classifier_mapping.json")

if __name__ == '__main__':
    main()