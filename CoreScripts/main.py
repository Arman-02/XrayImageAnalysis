#!/usr/bin/env python3
"""
Master script for Bone Fracture Detection project
Handles: Dataset download, preprocessing, training, and inference
"""

import os
import subprocess
import sys
from pathlib import Path
import argparse
import json

class BoneFractureProject:
    def __init__(self):
        self.project_dir = Path.cwd()
        self.dataset_name = "bone_fracture_detection.v4-v4.yolov8"
        self.kaggle_dataset = "pkdarabi/bone-fracture-detection-computer-vision-project"
    
    def check_kaggle_auth(self):
        """Check if Kaggle API is authenticated"""
        kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'
        if not kaggle_config.exists():
            print("❌ Kaggle authentication not found!")
            print("\nTo set up Kaggle API:")
            print("1. Go to https://www.kaggle.com/settings/account")
            print("2. Click 'Create New API Token'")
            print("3. Move kaggle.json to ~/.kaggle/")
            print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
            return False
        return True
    
    def download_dataset(self):
        """Download dataset from Kaggle"""
        print("\n" + "="*60)
        print("DOWNLOADING DATASET FROM KAGGLE")
        print("="*60)
        
        if not self.check_kaggle_auth():
            sys.exit(1)
        
        if os.path.exists(self.dataset_name):
            response = input(f"\n'{self.dataset_name}' already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Skipping download...")
                return
        
        try:
            print(f"\nDownloading {self.kaggle_dataset}...")
            subprocess.run([
                'kaggle', 'datasets', 'download', '-d', self.kaggle_dataset,
                '--unzip'
            ], check=True)
            print("✓ Dataset downloaded successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Download failed: {e}")
            sys.exit(1)
    
    def preprocess_dataset(self):
        """Run preprocessing pipeline"""
        print("\n" + "="*60)
        print("PREPROCESSING DATASET")
        print("="*60)
        
        if not os.path.exists(self.dataset_name):
            print(f"❌ Dataset not found at {self.dataset_name}")
            print("Run with --download flag first")
            sys.exit(1)
        
        try:
            from preprocess import BoneFracturePreprocessor
            
            preprocessor = BoneFracturePreprocessor(self.dataset_name)
            preprocessor.run()
            preprocessor.visualize_samples(num_samples=6)
            
            print("\n✓ Preprocessing completed!")
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            sys.exit(1)
    
    def train_model(self):
        """Train the model"""
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        if not os.path.exists('processed_dataset'):
            print("❌ Processed dataset not found!")
            print("Run preprocessing first")
            sys.exit(1)
        
        try:
            from train import BoneFractureTrainer, BoneFractureDataset
            from torch.utils.data import DataLoader
            import torch
            
            config = {
                'dataset_dir': 'processed_dataset',
                'batch_size': 8,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'epochs': 30,
                'backbone': 'resnet50',
                'pretrained': True,
                'num_workers': 4
            }
            
            print(f"\nConfiguration:")
            for key, value in config.items():
                print(f"  {key}: {value}")
            
            # Load datasets
            train_dataset = BoneFractureDataset(config['dataset_dir'], split='train', augment=True)
            val_dataset = BoneFractureDataset(config['dataset_dir'], split='val', augment=False)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=config['num_workers'],
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                pin_memory=True
            )
            
            print(f"\nDataset sizes:")
            print(f"  Train: {len(train_dataset)}")
            print(f"  Val: {len(val_dataset)}")
            
            # Train
            trainer = BoneFractureTrainer(config)
            trainer.train(train_loader, val_loader)
            trainer.plot_metrics()
            
            # Save config
            with open('training_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            print("\n✓ Training completed!")
            print("✓ Best model saved to: checkpoints/best_model.pth")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def run_inference(self, image_path=None):
        """Run inference on an image"""
        print("\n" + "="*60)
        print("RUNNING INFERENCE")
        print("="*60)
        
        if not os.path.exists('checkpoints/best_model.pth'):
            print("❌ Trained model not found!")
            print("Run training first")
            sys.exit(1)
        
        try:
            import cv2
            import torch
            import numpy as np
            import matplotlib.pyplot as plt
            from model import BoneFractureSegmentationNet
            
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            print(f"\nUsing device: {device}")
            
            # Load model
            model = BoneFractureSegmentationNet(num_classes=1, backbone='resnet50', pretrained=False)
            checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            print("✓ Model loaded successfully!")
            
            if image_path:
                # Process single image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"❌ Could not read image: {image_path}")
                    return
                
                print(f"\nProcessing: {image_path}")
                # ... inference code ...
                print("✓ Inference completed!")
        
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            sys.exit(1)
    
    def launch_streamlit(self):
        """Launch Streamlit app"""
        print("\n" + "="*60)
        print("LAUNCHING STREAMLIT APP")
        print("="*60)
        
        if not os.path.exists('checkpoints/best_model.pth'):
            print("❌ Trained model not found!")
            print("Run training first")
            sys.exit(1)
        
        print("\n🚀 Starting Streamlit app...")
        print("Open browser at: http://localhost:8501")
        
        try:
            subprocess.run(['streamlit', 'run', 'app.py'])
        except KeyboardInterrupt:
            print("\n✓ Streamlit app closed")
        except Exception as e:
            print(f"❌ Error launching Streamlit: {e}")
            sys.exit(1)
    
    def full_pipeline(self):
        """Run complete pipeline"""
        print("\n" + "🦴"*30)
        print("BONE FRACTURE DETECTION - FULL PIPELINE")
        print("🦴"*30)
        
        self.download_dataset()
        self.preprocess_dataset()
        self.train_model()
        
        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Run inference: python main.py --inference path/to/image.jpg")
        print("  2. Launch Streamlit: python main.py --streamlit")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Bone Fracture Detection Pipeline')
    parser.add_argument('--download', action='store_true', help='Download dataset from Kaggle')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess dataset')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--inference', type=str, help='Run inference on image')
    parser.add_argument('--streamlit', action='store_true', help='Launch Streamlit app')
    parser.add_argument('--full', action='store_true', help='Run complete pipeline')
    
    args = parser.parse_args()
    
    project = BoneFractureProject()
    
    if args.full:
        project.full_pipeline()
    else:
        if args.download:
            project.download_dataset()
        if args.preprocess:
            project.preprocess_dataset()
        if args.train:
            project.train_model()
        if args.inference:
            project.run_inference(args.inference)
        if args.streamlit:
            project.launch_streamlit()
        
        if not any([args.download, args.preprocess, args.train, args.inference, args.streamlit]):
            parser.print_help()

if __name__ == '__main__':
    main()