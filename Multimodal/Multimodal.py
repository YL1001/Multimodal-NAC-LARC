import pandas as pd
import numpy as np
import lightgbm as lgb
from utils import find_optimal_cutoff, calculate_metrics_val
from Prepare_Multimodal import prepare_datasets


def train_lightgbm(datasets):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = datasets
    
    # Prepare LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Model parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'seed': 42,
        'learning_rate': 0.01,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.6,
        'lambda_l1': 0,
        'lambda_l2': 1,
    }
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(50)
        ]
    )

    val_preds = model.predict(X_val)
    best_threshold = find_optimal_cutoff(y_val, val_preds)
    
    # Calculate metrics for all datasets
    train_metrics = calculate_metrics_val(y_train, model.predict(X_train), best_threshold)
    val_metrics = calculate_metrics_val(y_val, val_preds, best_threshold)
    test_metrics = calculate_metrics_val(y_test, model.predict(X_test), best_threshold)
    

    # Save predictions
    results_df = pd.DataFrame({
        'dataset': ['train'] * len(y_train) + ['val'] * len(y_val) + ['test'] * len(y_test),
        'true_label': np.concatenate([y_train, y_val, y_test]),
        'prediction': np.concatenate([
            model.predict(X_train),
            val_preds,
            model.predict(X_test)
        ]),
        'best_threshold': best_threshold
    })
    
    results_df.to_csv('lightgbm_predictions.csv', index=False)
    

if __name__ == "__main__":
    datasets = prepare_datasets()
    train_lightgbm(datasets)