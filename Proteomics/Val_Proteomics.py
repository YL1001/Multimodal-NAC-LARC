import os
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from Proteomics.Model_Proteomics import BioGAT
from Prepare_Proteomics import prepare_data
from utils import find_optimal_cutoff, calculate_metrics_val


if __name__ == "__main__":

    # Prepare datasets
    train_data = prepare_data('train')
    val_data = prepare_data('val')
    test_data = prepare_data('test')

    # Model configuration
    model_para = SimpleNamespace()
    model_para.gene_n = train_data.gene_n
    model_para.gpu_id = 'cuda:0'

    # Create data loaders
    train_loader1 = DataLoader(train_data, batch_size=1)
    val_loader = DataLoader(val_data, batch_size=1)
    test_loader = DataLoader(test_data, batch_size=1)

    # Initialize model
    model = BioGAT(func_dim=100, gene_n=model_para.gene_n).to(model_para.gpu_id)

    # Load trained model checkpoint
    checkpoint = torch.load("best_Proteomics.pth", map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(model_para.gpu_id)
    model.eval()

    # Initialize metrics dictionary
    metrics = {
        'train': {'auc': 0, 'accuracy': 0, 'Sensitivity': 0, 'specificity': 0, 'ppv': 0, 'npv': 0, 'f1': 0},
        'val': {'auc': 0, 'accuracy': 0, 'Sensitivity': 0, 'specificity': 0, 'ppv': 0, 'npv': 0, 'f1': 0},
        'test': {'auc': 0, 'accuracy': 0, 'Sensitivity': 0, 'specificity': 0, 'ppv': 0, 'npv': 0, 'f1': 0},
        'epoch': 0
    }

    # Evaluate on all phases
    for phase, loader in [('val', val_loader), ('train', train_loader1), ('test', test_loader)]:
        preds, labels, m_ids = [], [], []
        with torch.no_grad():
            for batch in loader:
                batch_ids = batch.id
                del batch.id
                batch = batch.to(model_para.gpu_id)
                pred = model(batch).view(-1)  
                preds.append(pred.squeeze().cpu().numpy())
                labels.append(batch.y[1].cpu().numpy())
                m_ids.append(batch_ids[0])
                
        preds = np.array(preds)
        labels = np.array(labels)
        m_ids = np.array(m_ids)
        
        if phase == "val":
            best_cutoff = find_optimal_cutoff(labels, preds)
            
        # Save predictions
        results_df = pd.DataFrame({
            'm_id': m_ids,
            'prediction': preds,
            'label': labels,
            'best_cutoff': best_cutoff, 
            'phase': phase
        })
        
        results_df.to_csv('Proteomics_predictions.csv', mode='a', 
                         header=not os.path.exists('Proteomics_predictions.csv'), 
                         index=False)
        
        # Calculate metrics
        current_metrics = calculate_metrics_val(labels, preds, threshold=best_cutoff)
        metrics[phase].update(current_metrics)