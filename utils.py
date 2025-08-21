import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix


def find_optimal_cutoff(labels, preds):
    fpr, tpr, thresholds = roc_curve(labels, preds)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    return best_thresh

def calculate_metrics_train(labels, preds, threshold=0.5):
    preds = np.array(preds)
    labels = np.array(labels)
    preds_binary = (preds >= threshold).astype(int)
    
    metric = {
        'accuracy': accuracy_score(labels, preds_binary),
        'f1': f1_score(labels, preds_binary, zero_division=0),
    }
    
    metric['auc'] = roc_auc_score(labels, preds)

    return metric


def calculate_metrics_val(labels, preds, threshold=0.5):
    preds = np.array(preds)
    labels = np.array(labels)
    preds_binary = (preds >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(labels, preds_binary).ravel()
    
    accuracy = accuracy_score(labels, preds_binary)
    precision = precision_score(labels, preds_binary, zero_division=0)
    recall = recall_score(labels, preds_binary, zero_division=0)
    f1 = f1_score(labels, preds_binary, zero_division=0)
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    auc = roc_auc_score(labels, preds)

    metric = {
        'auc': auc,
        'accuracy': accuracy,
        'Sensitivity': recall,
        'specificity': specificity,
        'ppv': precision, 
        'npv': npv,
        'f1': f1
    }
    return metric