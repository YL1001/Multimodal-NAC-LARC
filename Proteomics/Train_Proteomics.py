from types import SimpleNamespace
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from Proteomics.Model_Proteomics import BioGAT
from Prepare_Proteomics import prepare_data
from utils import find_optimal_cutoff, calculate_metrics_train




def train_model(train_data, val_data, model_para):
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1)
    
    # Initialize model
    model = BioGAT(func_dim=100, gene_n=model_para.gene_n).to(model_para.gpu_id)
    criterion = nn.BCELoss()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    
    # Initialize best metrics tracking
    best_metrics = {
        'auc': 0,
        'accuracy': 0,
        'f1': 0
    }
    
    epochs_no_impro = 0
    
    # Training loop
    for epoch in range(1000):
        # Training phase
        model.train()
        for batch in train_loader:
            batch = batch.to(model_para.gpu_id)
            optimizer.zero_grad()
            out = model(batch).view(-1)
            loss = criterion(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            
        # Validation phase
        model.eval()
        metrics = {
            'auc': 0,
            'accuracy': 0,
            'f1': 0
        }
        
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(model_para.gpu_id)
                preds.extend(model(batch).view(-1).cpu().numpy())
                labels.extend(batch.cpu().y.numpy())
        
        # Calculate metrics
        cutoff = find_optimal_cutoff(labels, preds)
        metrics = calculate_metrics_train(labels, preds, threshold=cutoff)
        
        # Check for improvement
        if metrics['auc'] > best_metrics['auc']:
            best_cutoff = cutoff
            best_epoch = epoch
            best_metrics = metrics
            torch.save(model.state_dict(), 'best_Proteomics.pth')
            
            epochs_no_impro = 0
            
            print('\nBest Metrics--------------------------------------')
            print(f"Best epoch: {best_epoch}")
            for metric, value in best_metrics.items():
                print(f'{metric}: {value:.4f}')
            print(f"Best cutoff: {best_cutoff}")
        else :
            epochs_no_impro += 1
            if epochs_no_impro >= 10:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break


if __name__ == "__main__":

    # Prepare datasets
    train_data = prepare_data('train')
    val_data = prepare_data('val')

    # Model configuration
    model_para = SimpleNamespace()
    model_para.gene_n = train_data.gene_n
    model_para.gpu_id = 'cuda:0'

    # Train Model
    train_model(train_data, val_data, model_para)