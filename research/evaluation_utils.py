import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar

def compute_ks_statistic(y_true, y_score):
    """
    Computes the Kolmogorov-Smirnov (KS) statistic.
    Uses scipy.stats.ks_2samp to compute max separation between score CDFs of y_true==1 and y_true==0.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    scores_1 = y_score[y_true == 1]
    scores_0 = y_score[y_true == 0]
    ks_stat, _ = ks_2samp(scores_1, scores_0)
    return ks_stat

def mcnemar_test(y_true, y_pred_model1, y_pred_model2):
    """
    Computes McNemar's chi-squared test to test if two models' error patterns differ significantly.
    Returns the test statistic and p-value.
    """
    y_true = np.array(y_true)
    y_pred1 = np.array(y_pred_model1)
    y_pred2 = np.array(y_pred_model2)
    
    # Contingency table
    #           Model 2 Correct | Model 2 Incorrect
    # Model 1 Correct      n00           n01
    # Model 1 Incorrect    n10           n11
    
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    
    n00 = np.sum(correct1 & correct2)
    n01 = np.sum(correct1 & ~correct2)
    n10 = np.sum(~correct1 & correct2)
    n11 = np.sum(~correct1 & ~correct2)
    
    table = [[n00, n01], [n10, n11]]
    result = mcnemar(table, exact=False, correction=True)
    return result.statistic, result.pvalue

def compute_gini(y_true, y_score):
    """
    Returns 2 * roc_auc_score(y_true, y_score) - 1.
    """
    return 2 * roc_auc_score(y_true, y_score) - 1
