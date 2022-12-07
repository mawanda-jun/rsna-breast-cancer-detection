import numpy as np


def pfbeta(labels, predictions, beta):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0
    
def best_pfbeta(labels: np.ndarray, predictions: np.ndarray, trick_threshold:float = 0.1):
    # Try competition metric
    predictions = np.copy(predictions)
    predictions[predictions < trick_threshold] = 0
    pfbs = np.array([pfbeta(labels, predictions, beta) for beta in np.linspace(0, 4, 100)])
    max_idx = np.argmax(pfbs)
    beta = np.linspace(0, 4, 100)[max_idx]
    pf1_score = pfbs[max_idx]
    return beta, pf1_score


def precision_recall(labels: np.ndarray, predictions: np.ndarray, threshold: float=0.5):
    predictions = np.copy(predictions)
    predictions[predictions >= threshold] = 1
    predictions[predictions < threshold] = 0
    
    tp = np.logical_and(predictions, labels).sum()
    fp = predictions.sum() - tp
    tn = np.logical_not(np.logical_or(predictions, labels)).sum()  # NOR between predictions and labels so we take only those elements in which both are 0
    fn = np.logical_not(predictions).sum() - tn

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall
