"""
Comprehensive Dataset Analysis Script
Analyzes fracture dataset structure, class distribution, and quality
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class DatasetAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.stats = {
            'total_images': 0,
            'total_masks': 0,
            'total_labels': 0,
            'class_distribution': {},
            'image_sizes': [],
            'mask_sizes': [],
            'corrupted_files': [],
            'missing_labels': [],
            'missing_masks': [],
            'mask_coverage': [],  # Percentage of image that is mask
            'empty_masks': [],
            'splits': {}
        }
        
        # Fracture classes from dataset
        self.class_names = {
            0: 'elbow positive',
            1: 'fingers positive',
            2: 'forearm fracture',
            3: 'humerus fracture',
            4: 'humerus',
            5: 'shoulder fracture',
            6: 'wrist positive'
        }
    
    def analyze_split(self, split_name):
        """Analyze a single split (train, val, test)"""
        print(f"\n{'='*60}")
        print(f"ANALYZING {split_name.upper()} SPLIT")
        print(f"{'='*60}")
        
        images_dir = os.path.join(self.dataset_path, split_name, 'images')
        masks_dir = os.path.join(self.dataset_path, split_name, 'masks')
        labels_dir = os.path.join(self.dataset_path, split_name, 'labels')
        
        if not os.path.exists(images_dir):
            print(f"⚠️  {split_name} split not found")
            return
        
        # Get all images
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
        print(f"\nFound {len(image_files)} images")
        
        split_stats = {
            'total_images': len(image_files),
            'valid_images': 0,
            'corrupted': 0,
            'missing_mask': 0,
            'missing_label': 0,
            'empty_masks': 0,
            'class_dist': Counter(),
            'image_sizes': [],
            'mask_coverage': []
        }
        
        # Analyze each image
        for img_file in tqdm(image_files, desc=f"Analyzing {split_name}"):
            img_path = os.path.join(images_dir, img_file)
            mask_file = img_file.rsplit('.', 1)[0] + '_mask.png'
            label_file = img_file.rsplit('.', 1)[0] + '.txt'
            mask_path = os.path.join(masks_dir, mask_file)
            label_path = os.path.join(labels_dir, label_file)
            
            # Check image
            img = cv2.imread(img_path)
            if img is None:
                split_stats['corrupted'] += 1
                self.stats['corrupted_files'].append(img_path)
                continue
            
            # Check mask
            if not os.path.exists(mask_path):
                split_stats['missing_mask'] += 1
                self.stats['missing_masks'].append(img_path)
                continue
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                split_stats['corrupted'] += 1
                continue
            
            # Check label
            if not os.path.exists(label_path):
                split_stats['missing_label'] += 1
                self.stats['missing_labels'].append(img_path)
                continue
            
            # Parse label
            with open(label_path, 'r') as f:
                try:
                    class_id = int(f.readline().split(',')[0])
                    split_stats['class_dist'][class_id] += 1
                except:
                    continue
            
            # Image stats
            split_stats['image_sizes'].append(img.shape)
            
            # Mask stats
            mask_pixels = np.sum(mask > 0)
            total_pixels = mask.shape[0] * mask.shape[1]
            coverage = (mask_pixels / total_pixels) * 100
            split_stats['mask_coverage'].append(coverage)
            
            if mask_pixels == 0:
                split_stats['empty_masks'] += 1
                self.stats['empty_masks'].append(img_path)
            
            split_stats['valid_images'] += 1
            self.stats['class_distribution'][class_id] = self.stats['class_distribution'].get(class_id, 0) + 1
            self.stats['total_images'] += 1
        
        self.stats['splits'][split_name] = split_stats
        
        # Print split analysis
        print(f"\n{split_name.upper()} SPLIT SUMMARY:")
        print(f"  Total images: {split_stats['total_images']}")
        print(f"  Valid images: {split_stats['valid_images']}")
        print(f"  Corrupted: {split_stats['corrupted']}")
        print(f"  Missing masks: {split_stats['missing_mask']}")
        print(f"  Missing labels: {split_stats['missing_label']}")
        print(f"  Empty masks: {split_stats['empty_masks']}")
        print(f"\n  Class distribution:")
        for class_id in sorted(split_stats['class_dist'].keys()):
            class_name = self.class_names.get(class_id, f"Unknown {class_id}")
            count = split_stats['class_dist'][class_id]
            percentage = (count / split_stats['valid_images']) * 100
            print(f"    {class_name}: {count} ({percentage:.1f}%)")
        
        # Image size analysis
        if split_stats['image_sizes']:
            heights = [s[0] for s in split_stats['image_sizes']]
            widths = [s[1] for s in split_stats['image_sizes']]
            print(f"\n  Image dimensions:")
            print(f"    Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.0f}")
            print(f"    Width - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.0f}")
        
        # Mask coverage analysis
        if split_stats['mask_coverage']:
            coverage = split_stats['mask_coverage']
            print(f"\n  Mask coverage (% of image):")
            print(f"    Min: {min(coverage):.2f}%, Max: {max(coverage):.2f}%, Mean: {np.mean(coverage):.2f}%")
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*60)
        print("DATASET ANALYSIS REPORT")
        print("="*60)
        
        for split in ['train', 'val', 'test']:
            self.analyze_split(split)
        
        # Overall statistics
        print(f"\n{'='*60}")
        print("OVERALL STATISTICS")
        print(f"{'='*60}")
        print(f"Total valid images: {self.stats['total_images']}")
        print(f"Total corrupted files: {len(self.stats['corrupted_files'])}")
        print(f"Total missing masks: {len(self.stats['missing_masks'])}")
        print(f"Total missing labels: {len(self.stats['missing_labels'])}")
        print(f"Total empty masks: {len(self.stats['empty_masks'])}")
        
        print(f"\nGLOBAL CLASS DISTRIBUTION:")
        total = sum(self.stats['class_distribution'].values())
        for class_id in sorted(self.stats['class_distribution'].keys()):
            class_name = self.class_names.get(class_id, f"Unknown {class_id}")
            count = self.stats['class_distribution'][class_id]
            percentage = (count / total) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Class imbalance analysis
        print(f"\nCLASS IMBALANCE ANALYSIS:")
        if self.stats['class_distribution']:
            counts = list(self.stats['class_distribution'].values())
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count
            print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
            print(f"  Most common: {max_count} samples")
            print(f"  Least common: {min_count} samples")
            
            if imbalance_ratio > 2:
                print(f"  ⚠️  HIGH IMBALANCE - Consider class weighting or data augmentation")
            elif imbalance_ratio > 1.5:
                print(f"  ⚠️  MODERATE IMBALANCE - May benefit from augmentation")
            else:
                print(f"  ✅ BALANCED - Good class distribution")
        
        # Recommendations
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")
        
        recommendations = []
        
        if len(self.stats['corrupted_files']) > 0:
            recommendations.append(f"• Remove/fix {len(self.stats['corrupted_files'])} corrupted files")
        
        if len(self.stats['missing_masks']) > 0:
            recommendations.append(f"• Generate missing masks for {len(self.stats['missing_masks'])} images")
        
        if len(self.stats['empty_masks']) > 0:
            recommendations.append(f"• Remove {len(self.stats['empty_masks'])} images with empty masks")
        
        # Check for rare classes
        for class_id, count in self.stats['class_distribution'].items():
            if count < 50:
                class_name = self.class_names.get(class_id, f"Class {class_id}")
                recommendations.append(f"• Collect more data for {class_name} ({count} samples is too few)")
        
        # Check for class imbalance
        if self.stats['class_distribution']:
            counts = list(self.stats['class_distribution'].values())
            if max(counts) / min(counts) > 2:
                recommendations.append("• Use class weighting or data augmentation to handle imbalance")
        
        if not recommendations:
            recommendations.append("✅ Dataset looks good!")
        
        for rec in recommendations:
            print(rec)
    
    def visualize_analysis(self):
        """Create visualization plots"""
        print("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Class distribution
        if self.stats['class_distribution']:
            classes = [self.class_names.get(k, f"Class {k}") for k in sorted(self.stats['class_distribution'].keys())]
            counts = [self.stats['class_distribution'][k] for k in sorted(self.stats['class_distribution'].keys())]
            
            axes[0, 0].bar(range(len(classes)), counts, color='steelblue')
            axes[0, 0].set_xticks(range(len(classes)))
            axes[0, 0].set_xticklabels(classes, rotation=45, ha='right')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Class Distribution (All Data)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Split distribution
        split_names = []
        split_counts = []
        for split_name in ['train', 'val', 'test']:
            if split_name in self.stats['splits']:
                split_names.append(split_name.upper())
                split_counts.append(self.stats['splits'][split_name]['valid_images'])
        
        if split_names:
            axes[0, 1].bar(split_names, split_counts, color=['green', 'blue', 'red'])
            axes[0, 1].set_ylabel('Number of Images')
            axes[0, 1].set_title('Images per Split')
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Data quality
        quality_labels = ['Valid', 'Corrupted', 'Missing Mask', 'Missing Label', 'Empty Mask']
        quality_counts = [
            self.stats['total_images'],
            len(self.stats['corrupted_files']),
            len(self.stats['missing_masks']),
            len(self.stats['missing_labels']),
            len(self.stats['empty_masks'])
        ]
        
        colors = ['green', 'red', 'orange', 'orange', 'yellow']
        axes[1, 0].barh(quality_labels, quality_counts, color=colors)
        axes[1, 0].set_xlabel('Count')
        axes[1, 0].set_title('Data Quality Overview')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Mask coverage (if available)
        all_coverage = []
        for split_name in self.stats['splits'].keys():
            if 'mask_coverage' in self.stats['splits'][split_name]:
                all_coverage.extend(self.stats['splits'][split_name]['mask_coverage'])
        
        if all_coverage:
            axes[1, 1].hist(all_coverage, bins=30, color='purple', alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Mask Coverage (%)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Mask Coverage Distribution')
            axes[1, 1].axvline(np.mean(all_coverage), color='red', linestyle='--', label=f'Mean: {np.mean(all_coverage):.1f}%')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dataset_analysis.png', dpi=100, bbox_inches='tight')
        print("✓ Visualization saved to 'dataset_analysis.png'")
        plt.close()
        
        # Save report to JSON
        report = {
            'total_images': self.stats['total_images'],
            'corrupted_files': len(self.stats['corrupted_files']),
            'missing_masks': len(self.stats['missing_masks']),
            'missing_labels': len(self.stats['missing_labels']),
            'empty_masks': len(self.stats['empty_masks']),
            'class_distribution': self.stats['class_distribution'],
            'splits': {k: {'valid_images': v.get('valid_images', 0), 
                          'class_dist': dict(v.get('class_dist', {}))} 
                      for k, v in self.stats['splits'].items()}
        }
        
        with open('dataset_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print("✓ Report saved to 'dataset_analysis_report.json'")

def main():
    dataset_path = 'processed_dataset'
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        return
    
    analyzer = DatasetAnalyzer(dataset_path)
    analyzer.generate_report()
    analyzer.visualize_analysis()
    
    print("\n" + "="*60)
    print("✓ ANALYSIS COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()