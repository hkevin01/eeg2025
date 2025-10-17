#!/usr/bin/env python3
"""
Challenge 2: Psychopathology Prediction from RestingState EEG
==============================================================

Train model to predict clinical scores from RestingState EEG:
- p_factor (general psychopathology)
- attention (attention problems)
- internalizing (anxiety/depression)
- externalizing (conduct/aggression)
"""
import os
import sys
from pathlib import Path
import time

# Force CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HIP_VISIBLE_DEVICES'] = ''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import mne

print("="*80)
print("🎯 CHALLENGE 2: PSYCHOPATHOLOGY PREDICTION")
print("="*80)
print("Device: CPU")
print("="*80)


class ClinicalEEGDataset(Dataset):
    """EEG dataset with clinical scores"""
    
    def __init__(self, data_dir, segment_length=512):
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        
        # Load participants with clinical scores
        print("\n📋 Loading clinical data...")
        participants_file = self.data_dir / "participants.tsv"
        self.participants_df = pd.read_csv(participants_file, sep='\t')
        
        # Filter for subjects with EEG and clinical scores
        clinical_columns = ['p_factor', 'attention', 'internalizing', 'externalizing']
        self.participants_df = self.participants_df.dropna(subset=clinical_columns)
        
        print(f"   Participants with clinical scores: {len(self.participants_df)}")
        
        # Find subjects with RestingState EEG
        print("🔍 Finding RestingState EEG files...")
        self.segments = []
        self.clinical_scores = []
        
        for _, row in self.participants_df.iterrows():
            subject_id = row['participant_id']
            subject_dir = self.data_dir / subject_id / "eeg"
            
            if not subject_dir.exists():
                continue
            
            # Find RestingState EEG
            eeg_files = list(subject_dir.glob("*RestingState*.set"))
            if not eeg_files:
                continue
            
            try:
                # Load EEG
                raw = mne.io.read_raw_eeglab(eeg_files[0], preload=True, verbose=False)
                data = raw.get_data()
                
                # Standardize
                data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
                
                # Create segments
                n_samples = data.shape[1]
                n_segments = n_samples // segment_length
                
                # Clinical scores
                scores = np.array([
                    row['p_factor'],
                    row['attention'],
                    row['internalizing'],
                    row['externalizing']
                ], dtype=np.float32)
                
                for i in range(n_segments):
                    start = i * segment_length
                    end = start + segment_length
                    segment = data[:, start:end]
                    
                    self.segments.append(torch.FloatTensor(segment))
                    self.clinical_scores.append(scores)
                
                print(f"   ✅ {subject_id}: {n_segments} segments")
                
            except Exception as e:
                print(f"   ⚠️  {subject_id}: {e}")
                continue
        
        print(f"\n📊 Total segments: {len(self.segments)}")
        
        # Normalize clinical scores
        self.scores_array = np.array(self.clinical_scores)
        self.scores_mean = self.scores_array.mean(axis=0)
        self.scores_std = self.scores_array.std(axis=0)
        self.scores_normalized = (self.scores_array - self.scores_mean) / (self.scores_std + 1e-8)
        
        print(f"   Clinical scores normalized (mean=0, std=1)")
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        scores = torch.FloatTensor(self.scores_normalized[idx])
        return segment, scores
    
    def denormalize_scores(self, scores_normalized):
        """Convert normalized scores back to original scale"""
        return scores_normalized * self.scores_std + self.scores_mean


class ClinicalCNN(nn.Module):
    """CNN for clinical score prediction"""
    
    def __init__(self, n_channels=129, n_outputs=4):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_outputs)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output


def train_model(model, train_loader, val_loader, dataset, epochs=30):
    """Train the clinical prediction model"""
    print("\n" + "="*80)
    print("🔥 Training Clinical Prediction Model")
    print("="*80)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    clinical_names = ['P-Factor', 'Attention', 'Internalizing', 'Externalizing']
    
    best_mae = float('inf')
    best_correlations = None
    patience_counter = 0
    patience = 8
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.append(outputs.detach().numpy())
            train_labels.append(labels.numpy())
        
        # Validate
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.append(outputs.numpy())
                val_labels.append(labels.numpy())
        
        scheduler.step()
        
        # Compute metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_preds = np.vstack(train_preds)
        train_labels = np.vstack(train_labels)
        val_preds = np.vstack(val_preds)
        val_labels = np.vstack(val_labels)
        
        # Denormalize for interpretable metrics
        val_preds_real = dataset.denormalize_scores(val_preds)
        val_labels_real = dataset.denormalize_scores(val_labels)
        
        # Per-factor metrics
        val_correlations = []
        val_maes = []
        
        for i, name in enumerate(clinical_names):
            corr, _ = pearsonr(val_preds_real[:, i], val_labels_real[:, i])
            mae = mean_absolute_error(val_labels_real[:, i], val_preds_real[:, i])
            val_correlations.append(corr)
            val_maes.append(mae)
        
        mean_mae = np.mean(val_maes)
        mean_corr = np.mean(val_correlations)
        
        print(f"\nEpoch {epoch+1:2d}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val MAE:    {mean_mae:.3f}")
        print(f"  Val Corr:   {mean_corr:.4f}")
        print(f"  Individual:")
        for i, name in enumerate(clinical_names):
            print(f"    {name:<15} Corr={val_correlations[i]:>6.3f}, MAE={val_maes[i]:>6.3f}")
        
        # Save best model
        if mean_mae < best_mae:
            best_mae = mean_mae
            best_correlations = val_correlations
            patience_counter = 0
            
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'mae': best_mae,
                'correlations': best_correlations,
                'scores_mean': dataset.scores_mean,
                'scores_std': dataset.scores_std
            }, checkpoint_dir / "challenge2_clinical.pth")
            print(f"  💾 New best model! MAE={best_mae:.3f}")
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping after {epoch+1} epochs")
            break
    
    return best_mae, best_correlations


def main():
    """Main training function"""
    start_time = time.time()
    
    print("\n📂 Loading dataset...")
    data_dir = Path("data/raw/hbn")
    
    dataset = ClinicalEEGDataset(data_dir=data_dir)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"   Train segments: {train_size}")
    print(f"   Val segments: {val_size}")
    
    # Train model
    model = ClinicalCNN()
    best_mae, best_correlations = train_model(model, train_loader, val_loader, dataset, epochs=30)
    
    total_time = time.time() - start_time
    
    # Final results
    clinical_names = ['P-Factor', 'Attention', 'Internalizing', 'Externalizing']
    
    print("\n" + "="*80)
    print("📊 FINAL RESULTS - Challenge 2")
    print("="*80)
    print(f"\nBest Validation MAE: {best_mae:.3f}")
    print(f"Mean Correlation: {np.mean(best_correlations):.4f}")
    print(f"\nPer-Factor Results:")
    for i, name in enumerate(clinical_names):
        print(f"  {name:<15} Correlation: {best_correlations[i]:.4f}")
    
    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    results_df = pd.DataFrame({
        'factor': clinical_names,
        'correlation': best_correlations
    })
    results_df.to_csv(results_dir / "challenge2_results.csv", index=False)
    
    print("\n💾 Results saved to: results/challenge2_results.csv")
    print("💾 Model saved to: checkpoints/challenge2_clinical.pth")
    
    print("\n" + "="*80)
    print("✅ CHALLENGE 2 TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
