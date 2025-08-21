import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from MRI.Setting_MRI import sets
from utils import find_optimal_cutoff, calculate_metrics_val
from MRI.Model_MRI import generate_model
from Prepare_MRI import prepare_data


if __name__ == "__main__":
    # Load data
    train_ds = prepare_data(sets, "train")
    val_ds = prepare_data(sets, "val")
    test_ds = prepare_data(sets, "test")

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=sets.num_workers
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=sets.num_workers
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=sets.num_workers
    )
    
    # Initialize model
    model = generate_model(
        152,
        conv1_t_size=sets.model_conv1_t_size,
        conv1_t_stride=sets.model_conv1_t_stride,
        no_max_pool=sets.model_no_max_pool,
        shortcut_type=sets.model_shortcut_type,
        widen_factor=sets.model_widen_factor,
        SA_ks=sets.SA_ks,
        sigma=sets.sigma
    )

    # Load trained model checkpoint
    checkpoint = torch.load("best_Proteomics.pth", map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(sets.gpu_id)
    model.eval()
    
    # Initialize metrics dictionary
    metrics = {
        'train': {
            'auc': 0, 'accuracy': 0, 'Sensitivity': 0, 'specificity': 0,
            'ppv': 0, 'npv': 0, 'f1': 0
        },
        'val': {
            'auc': 0, 'accuracy': 0, 'Sensitivity': 0, 'specificity': 0,
            'ppv': 0, 'npv': 0, 'f1': 0
        },
        'test': {
            'auc': 0, 'accuracy': 0, 'Sensitivity': 0, 'specificity': 0,
            'ppv': 0, 'npv': 0, 'f1': 0
        },
        'epoch': 0
    }

    # Evaluate on all phases
    for phase, loader in [('val', val_loader), ('train', train_loader), ('test', test_loader)]:
        preds, labels, m_ids = [], [], []
        
        with torch.no_grad():
            for batch_data in loader:
                m_id, image_array, y = batch_data
                image_array = image_array.to(sets.gpu_id)
                preds.append(model(image_array).squeeze().cpu().numpy())
                labels.append(y[0].numpy())
                m_ids.append(m_id[0])
                
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
        
        results_df.to_csv(
            'MRI_predictions.csv',
            mode='a',
            header=not os.path.exists('MRI_predictions.csv'),
            index=False
        )
        
        # Calculate metrics
        current_metrics = calculate_metrics_val(preds, labels, threshold=best_cutoff)
        metrics[phase].update(current_metrics)
