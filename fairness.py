"""
Utility function to produce fairness metric
"""
import numpy as np
from retention import *
from regression_retention import *

def acc_disparity(X,y,Z,theta):
    """
    return the accuracy disparity 
    """
    acc = calc_metric(X,y,Z,theta)
    return np.max(acc) - np.min(acc)

def l_disparity(X,y,Z,theta):
    """
    return the loss disparity 
    """
    l = calc_lprev(X,y,Z,theta,0)
    return np.max(l) - np.min(l)

def l_disparity_reg(X,y,Z,theta):
    """
    return loss disparity for regression model
    """
    l = calc_MSE_prev(X,y,Z,theta)
    return np.max(l) - np.min(l)

def dp_disparity(X,y,Z,theta):
    """
    return the demographic parity disparity
    """
    metric = []
    for z in np.unique(Z):
        X_z = X[Z==z]
        y_z = y[Z==z]
        h = 1/(1+np.exp(-np.matmul(X_z,theta)))
        y_pred = np.where(h > 0.5, 1, 0)
        metric.append(len(y_pred[y_pred == 1])/len(y_pred))
    
    return np.max(metric) - np.min(metric)


def eo_disparity(X,y,Z,theta):
    """
    return the equal opportunity disparity
    """
    metric = []
    for z in np.unique(Z):
        X_z = X[Z==z]
        y_z = y[Z==z]
        h = 1/(1+np.exp(-np.matmul(X_z,theta)))
        y_pred = np.where(h > 0.5, 1, 0)
        tp = np.where((y_pred == 1) & (y_z == 1), 1, 0)
        tp_cnt = len(tp[tp==1])
        p_cnt = len(y_z[y_z==1])
        metric.append(tp_cnt/p_cnt)
    return np.max(metric) - np.min(metric)

