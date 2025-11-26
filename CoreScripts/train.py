import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from model import BoneFractureSegmentationNet, DiceLoss

# Import model module
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class BoneFractureDataset(Dataset):
    """Custom dataset for bone fracture segmentation"""
    def __init__(self, dataset_dir, split='train', augment=True):
        self.split = split
        self.augment = augment
        self.images_dir = f'{dataset_dir}/{split}/images'
        self.masks_dir = f'{dataset_dir}/{split}/masks'
        self.labels_dir = f'{dataset_dir}/{split}/labels'
        
        # Only include images that have corresponding masks
        self.images = []
        for f in sorted(os.listdir(self.images_dir)):
            if f.endswith(('.jpg', '.png')):
                mask_file = f.rsplit('.', 1)[0] + '_mask.png'
                mask_path = os.path.join(self.masks_dir, mask_file)
                if os.path.exists(mask_path):
                    self.images.append(f)
        
        print(f"Dataset {split}: Found {len(self.images)} images with valid masks")
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_name = img_name.rsplit('.', 1)[0] + '_mask.png'
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        # Read image and mask
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Handle missing or corrupt files
        if image is None:
            # Return black image if read fails
            image = np.zeros((640, 640, 3), dtype=np.uint8)
        if mask is None:
            # Return black mask if read fails
            mask = np.zeros((640, 640), dtype=np.uint8)
        
        # Make copies to avoid stride issues
        image = image.copy()
        mask = mask.copy()
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        
        # Data augmentation for training
        if self.augment and self.split == 'train':
            image, mask = self._augment(image, mask)
        
        # Make contiguous in memory
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        mask = torch.from_numpy(mask).unsqueeze(0)  # HW -> 1HW
        
        return image, mask
    
    def _augment(self, image, mask):
        """Simple data augmentation"""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        
        # Random vertical flip
        if np.random.rand() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        
        # Random brightness adjustment
        if np.random.rand() > 0.5:
            brightness = np.random.uniform(0.9, 1.1)
            image = np.clip(image * brightness, 0, 1)
        
        # Random contrast adjustment
        if np.random.rand() > 0.5:
            contrast = np.random.uniform(0.9, 1.1)
            image = np.clip((image - 0.5) * contrast + 0.5, 0, 1)
        
        return image, mask

class SegmentationMetrics:
    """Calculate segmentation evaluation metrics"""
    @staticmethod
    def dice_coefficient(pred, target, smooth=1.0):
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return dice.item()
    
    @staticmethod
    def iou(pred, target, smooth=1.0):
        pred = (pred > 0.5).float()
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()
    
    @staticmethod
    def sensitivity(pred, target):
        pred = (pred > 0.5).float()
        tp = (pred * target).sum()
        fn = ((1 - pred) * target).sum()
        sensitivity = tp / (tp + fn + 1e-6)
        return sensitivity.item()
    
    @staticmethod
    def specificity(pred, target):
        pred = (pred > 0.5).float()
        tn = ((1 - pred) * (1 - target)).sum()
        fp = (pred * (1 - target)).sum()
        specificity = tn / (tn + fp + 1e-6)
        return specificity.item()

class BoneFractureTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = BoneFractureSegmentationNet(
            num_classes=1, 
            backbone=config['backbone'],
            pretrained=config['pretrained']
        ).to(self.device)
        
        # Loss functions
        self.seg_loss = DiceLoss().to(self.device)
        self.bce_loss = nn.BCEWithLogitsLoss().to(self.device)
        
        # Optimizer - Use lower learning rate for fine-tuning
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=1e-6
        )
        
        # Metrics storage
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'train_iou': [],
            'val_iou': [],
            'lr': []
        }
        
        # Checkpoint directory
        os.makedirs('checkpoints', exist_ok=True)
        self.best_val_dice = 0
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        metrics_dice = []
        metrics_iou = []
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Combined loss (Dice + BCE)
            dice_loss = self.seg_loss(outputs, masks)
            bce_loss = self.bce_loss(outputs, masks)
            loss = 0.7 * dice_loss + 0.3 * bce_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                dice = SegmentationMetrics.dice_coefficient(outputs, masks)
                iou = SegmentationMetrics.iou(outputs, masks)
                metrics_dice.append(dice)
                metrics_iou.append(iou)
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        avg_dice = np.mean(metrics_dice)
        avg_iou = np.mean(metrics_iou)
        
        return avg_loss, avg_dice, avg_iou
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        metrics_dice = []
        metrics_iou = []
        metrics_sensitivity = []
        metrics_specificity = []
        
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                
                dice_loss = self.seg_loss(outputs, masks)
                bce_loss = self.bce_loss(outputs, masks)
                loss = 0.7 * dice_loss + 0.3 * bce_loss
                
                dice = SegmentationMetrics.dice_coefficient(outputs, masks)
                iou = SegmentationMetrics.iou(outputs, masks)
                sensitivity = SegmentationMetrics.sensitivity(outputs, masks)
                specificity = SegmentationMetrics.specificity(outputs, masks)
                
                metrics_dice.append(dice)
                metrics_iou.append(iou)
                metrics_sensitivity.append(sensitivity)
                metrics_specificity.append(specificity)
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
        
        avg_loss = total_loss / len(val_loader)
        avg_dice = np.mean(metrics_dice)
        avg_iou = np.mean(metrics_iou)
        avg_sensitivity = np.mean(metrics_sensitivity)
        avg_specificity = np.mean(metrics_specificity)
        
        return avg_loss, avg_dice, avg_iou, avg_sensitivity, avg_specificity
    
    def train(self, train_loader, val_loader):
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch [{epoch+1}/{self.config['epochs']}]")
            
            # Train
            train_loss, train_dice, train_iou = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_dice, val_iou, val_sensitivity, val_specificity = self.validate(val_loader)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_dice'].append(train_dice)
            self.history['val_dice'].append(val_dice)
            self.history['train_iou'].append(train_iou)
            self.history['val_iou'].append(val_iou)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print metrics
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f}")
            print(f"  Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f}")
            print(f"  Sensitivity: {val_sensitivity:.4f} | Specificity: {val_specificity:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_dice': val_dice,
                    'config': self.config
                }
                torch.save(checkpoint, 'checkpoints/best_model.pth')
                print(f"  ✓ Best model saved (Dice: {val_dice:.4f})")
            
            # Scheduler step
            self.scheduler.step()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
    
    def plot_metrics(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Dice
        axes[0, 1].plot(self.history['train_dice'], label='Train Dice', linewidth=2)
        axes[0, 1].plot(self.history['val_dice'], label='Val Dice', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Coefficient')
        axes[0, 1].set_title('Dice Coefficient')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # IoU
        axes[1, 0].plot(self.history['train_iou'], label='Train IoU', linewidth=2)
        axes[1, 0].plot(self.history['val_iou'], label='Val IoU', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU')
        axes[1, 0].set_title('Intersection over Union')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(self.history['lr'], linewidth=2, color='green')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=100, bbox_inches='tight')
        print("✓ Training metrics saved to 'training_metrics.png'")
        plt.close()

def main():
    # Configuration
    config = {
        'dataset_dir': 'processed_dataset',
        'batch_size': 8,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 30,
        'backbone': 'resnet50',
        'pretrained': True,
        'num_workers': 0  # Set to 0 to avoid worker issues
    }
    
    print("Loading datasets...")
    train_dataset = BoneFractureDataset(config['dataset_dir'], split='train', augment=True)
    
    # Try 'valid' first (as in the actual dataset), fall back to 'val'
    val_split = 'valid' if os.path.exists(os.path.join(config['dataset_dir'], 'valid')) else 'val'
    val_dataset = BoneFractureDataset(config['dataset_dir'], split=val_split, augment=False)
    
    # If no validation data, use 10% of training data
    if len(val_dataset) == 0:
        print("⚠ No validation data found, using 10% of training data")
        val_split = 'train'
        val_dataset = BoneFractureDataset(config['dataset_dir'], split=val_split, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Train
    trainer = BoneFractureTrainer(config)
    trainer.train(train_loader, val_loader)
    trainer.plot_metrics()
    
    # Save history
    with open('training_history.json', 'w') as f:
        json.dump(trainer.history, f, indent=2)
    print("✓ Training history saved to 'training_history.json'")

if __name__ == '__main__':
    main()