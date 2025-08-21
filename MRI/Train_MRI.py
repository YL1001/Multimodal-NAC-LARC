import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from Prepare_MRI import prepare_data
from MRI.Model_MRI import generate_model
from utils import find_optimal_cutoff, calculate_metrics_train
from MRI.Setting_MRI import sets


def train_model(sets):
    # Load datasets
    train_ds = prepare_data(sets, "train")
    val_ds = prepare_data(sets, "val")

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=sets.batch_size,
        shuffle=True,
        num_workers=sets.num_workers
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=sets.num_workers
    )
    
    model_dict = {
        'r3d18_KM_200ep.pth': (18, 1039),
        'r3d34_KM_200ep.pth': (34, 1039),
        'r3d50_KMS_200ep.pth': (50, 1139),
        'r3d101_KM_200ep.pth': (101, 1039),
        'r3d152_KM_200ep.pth': (152, 1039),
        'r3d200_KM_200ep.pth': (200, 1039)
    }
    layer_num, param_num = model_dict[sets.pretrain_m]

    # Initialize model
    model = generate_model(
        layer_num,
        conv1_t_size=sets.model_conv1_t_size,
        conv1_t_stride=sets.model_conv1_t_stride,
        no_max_pool=sets.model_no_max_pool,
        shortcut_type=sets.model_shortcut_type,
        widen_factor=sets.model_widen_factor,
        SA_ks=sets.SA_ks,
        sigma=sets.sigma
    )
    
    model = model.to(sets.gpu_id)
    criterion = nn.BCELoss()
    
    # Initialize best metrics tracking
    best_metrics = {
        'auc': 0,
        'accuracy': 0,
        'f1': 0
    }

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=sets.learning_rate,
        weight_decay=sets.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    epochs_no_impro = 0

    # Training loop
    for epoch in range(sets.n_epochs):
        # Training phase
        model.train()
        for batch_data in train_loader:
            image_array, y = [data.to(sets.gpu_id) for data in batch_data]
            optimizer.zero_grad()
            p_01 = model(image_array).squeeze()
            loss_bce = criterion(p_01, y)
            loss_bce.backward()
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
            for batch_data in val_loader:
                image_array, y = [data.to(sets.gpu_id) for data in batch_data]
                preds.append(model(image_array).squeeze().cpu().numpy())
                labels.append(y.cpu().numpy())
        
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
        else:
            epochs_no_impro += 1
            if epochs_no_impro >= sets.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        scheduler.step()
        
if __name__ == '__main__':
    train_model(sets)