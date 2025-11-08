"""
retention rate dynamics
specifically, p_s^{t+1} is proportional to the accuracy at t on group s
"""
import numpy as np
from sklearn.metrics import accuracy_score

def calc_metric(X,y,Z,theta):
    """
    calcultate group-wise accuracy for logistic regression
    """
    metric = []
    for z in np.unique(Z):
        X_z = X[Z==z]
        y_z = y[Z==z]
        h = 1/(1+np.exp(-np.matmul(X_z,theta)))
        y_pred = np.where(h > 0.5, 1, 0)
        acc = accuracy_score(y_z, y_pred)
        metric.append(acc)
    
    return np.array(metric)


def calc_lprev(X,y,Z,theta,lam=0):
    """
    calculate expected loss wrt each group
    """
    lprev = []
    for z in np.unique(Z):
        X_z = X[Z==z]
        y_z = y[Z==z]
        n = len(y_z)
        # compute new loss
        t1 = 1.0/n * np.sum(-1 * np.multiply(y_z, X_z @ theta) +
                            np.log(1 + np.exp(X_z @ theta)))
        t2 = (lam / 2.0) * np.linalg.norm(theta[:-1])**2
        loss = t1 + t2
        lprev.append(loss)
    return np.array(lprev)

# def calc_fair_lprev(X,Y,Z,theta,rho):
#     lprev = []
#     for z in np.unique(Z):
#         X_z = X[Z==z]
#         y_z = Y[Z==z]
#         n = len(y_z)
#         pz = len(Z[Z==z])/len(Z)
#         l = 1.0/n * np.sum(-1 * np.multiply(y_z, X_z @ theta) +
#                             np.log(1 + np.exp(X_z @ theta)))
#         loss = (l + rho*l**2)
#         lprev.append(loss)
#     return np.array(lprev)


def calc_curr_acc(pt,metric):
    return np.sum(pt*metric)


def calc_curr_l(pt,lprev):
    return np.sum(pt*lprev)


def participation_map(metric, pt, pmin, groups = [0,1]):
    """
    given a metric, output the participation rates of next round for all groups
    """
    psum = 0
    pnext = np.zeros(len(pt))
    if len(pt) == 2:
        for g in groups:
            psum += (0.5-sum(pmin)/2)*(pt[g] + metric[1-g]/(metric[g]+metric[1-g])) + pmin[g]
        for g in groups:
            pnext[g] = ((0.5-sum(pmin)/2)*(pt[g] + metric[1-g]/(metric[g]+metric[1-g])) + pmin[g])/psum
        return pnext
    else:
        groups = list(range(len(pt)))
        # k groups: the group with loss ranking i, will multiply with the group with loss ranking k+1-i
        metric_sort = np.sort(metric)[::-1]
        # First argsort: get indices that would sort the array
        sorted_indices = np.argsort(metric)
        # Second argsort: get the ranking
        ranks = np.argsort(sorted_indices)
        metric_sort = metric_sort[ranks] # metric sort has the reranked loss
        for g in groups:
            psum += (0.5-sum(pmin)/len(pt))*(pt[g] + metric_sort[g]/(metric[g]+metric_sort[g])) + pmin[g]
        for g in groups:
            pnext[g] = ((0.5-sum(pmin)/len(pt))*(pt[g] + metric_sort[g]/(metric[g]+metric_sort[g])) + pmin[g])/psum
        return pnext


def simple_map(metric, pt, pmin, groups = [0,1]):
    """
    given a metric, output the participation rates of next round for all groups
    """
    pnext = np.zeros(len(pt))
    amount = (metric[1] - metric[0])*0.02
    if abs(amount) <= 0.001:
        amount = 0
    if abs(amount) >= 0.01:
        amount = 0.01*np.sign(amount)
    pnext[0] = max(pt[0] + amount, pmin[0])
    pnext[0] = min(pnext[0], 1-pmin[1])
    pnext[1] = 1 - pnext[0]
    return pnext


def DRO_map(pnow, lprev, rho = 0.1):
    """
    DRO mapping: given participation rates now, and the group loss last time, find qnow
    """
    S = len(pnow)
    qnow = pnow + (rho*lprev/S)
    qnow = qnow/np.linalg.norm(qnow, ord=1)
    return qnow



def best_response(X, theta, epsilon, strat_features):
    """
    Add strategic behaviors into retention
    """
    n = X.shape[0]

    X_strat = np.copy(X)

    for i in range(n):
        # move everything by epsilon in the direction towards better classification
        theta_strat = theta[strat_features]
        X_strat[i, strat_features] += -epsilon * theta_strat

    return X_strat

