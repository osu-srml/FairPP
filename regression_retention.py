"""Linear regression model"""

import numpy as np
from retention import *

def evaluate_MSE(X, Y, theta):
    """
    Get MSE
    """
    n = X.shape[0]
    return 1/n*np.sum((Y-X@theta)**2)

def calc_MSE_prev(X,Y,Z,theta):
    """
    Get group-wise MSE
    """
    lprev = []
    for z in np.unique(Z):
        X_z = X[Z==z]
        Y_z = Y[Z==z]
        n = len(Y_z)
        loss = 1/n*np.sum((Y_z-X_z@theta)**2)
        lprev.append(loss)
    return np.array(lprev)

# def calc_fair_MSE_prev(X,Y,Z,theta,rho):
#     """
#     Get group-wise fair MSE
#     """
#     lprev = []
#     for z in np.unique(Z):
#         X_z = X[Z==z]
#         Y_z = Y[Z==z]
#         n = len(Y_z)
#         l = evaluate_MSE(X_z,Y_z,theta)
#         # fair regularized loss aggregated by group
#         lprev.append(l + rho*l**2)
#     return np.array(lprev)

def calc_MSE_agg(pt, lprev):
    return np.sum(pt*lprev)

def evaluate_fair_MSE(X,Y,Z,theta,rho):
    """
    fair regularized MSE loss
    """
    loss = 0
    for z in np.unique(Z):
        pz = len(Z[Z==z])/len(Z)
        X_z = X[Z==z]
        Y_z = Y[Z==z]
        l = evaluate_MSE(X_z,Y_z,theta)
        # fair regularized loss aggregated by group
        loss += (pz*(l + rho*l**2))
    return loss

def evaluate_e5_MSE(X,Y,Z,theta,rho):
    """
    counter example of fair MSE in example 5
    """
    loss = 0
    # total loss
    l_all = evaluate_MSE(X,Y,theta)
    loss += l_all
    for z in np.unique(Z):
        pz = len(Z[Z==z])/len(Z)
        X_z = X[Z==z]
        Y_z = Y[Z==z]
        l = evaluate_MSE(X_z,Y_z,theta)
        # fair variance regularized loss aggregated by group
        loss += (rho*pz*(l - l_all)**2)
    return loss

def calc_fair_reg_grad_e5(X,Y,Z,theta,rho,n):
    """
    fair gradient for example 5
    """
    # take regular gradients
    grad_a= -2.0/n * X.T@(Y-X@theta)
    # total loss
    loss_a = np.sum((Y-X@theta)**2)
    gradient = grad_a

    for z in np.unique(Z):
        X_z = X[Z==z]
        Y_z = Y[Z==z]
        n_z = len(Y_z)
        grad_z = -2.0/n_z * X_z.T@(Y_z-X_z@theta)
        loss_z = 1.0/n_z * np.sum((Y_z-X_z@theta)**2)
        # chain rule
        gradient += (rho*n_z/n*2*(loss_z - loss_a)*(grad_z - grad_a))
    
    return gradient

def calc_fair_reg_grad(X,Y,Z,theta,rho,n):
    # take regular gradients
    gradient = -2.0/n * X.T@(Y-X@theta)

    for z in np.unique(Z):
        X_z = X[Z==z]
        Y_z = Y[Z==z]
        n_z = len(Y_z)
        grad_z = -2.0/n_z * X_z.T@(Y_z-X_z@theta)
        loss_z = np.sum((Y_z-X_z@theta)**2)
        gradient += ((rho/n)*grad_z*2*loss_z)
    
    return gradient

def evaluate_fair_MSE_demo(X,Y,theta,rho):
    n = X.shape[0]
    return 1/n*(np.sum((Y-X@theta)**2) + rho*np.sum((Y-X@theta)**4))

def calc_fair_reg_grad_demo(X,Y,theta,rho,n):
    gradient = -2.0/n * X.T@(Y-X@theta) - 4.0/n*rho*X.T@((Y - X@theta)**3)
    return gradient

def linear_regression_ana(X,Y):
    """
    return optimal theta for linear regression
    use analytical solution
    """
    return np.array(np.linalg.inv(X.T@X)@X.T@Y)

def weighted_lr_ana(X,Y,Z,reweight=[]):
    ratio = len(Z[Z == 0])/len(Z)
    factor_0, factor_1 = reweight[0]/ratio, reweight[1]/(1-ratio)
    dro_ratio = np.where(Z == 0,factor_0,factor_1)
    W = np.diag(dro_ratio)
    return np.array(np.linalg.inv(X.T@W@X)@X.T@W@Y)


# def linear_regression(X_orig, Y_orig, Z_orig, method, tol=1e-30, theta_init=None, eta_init = 0.1, reweight = []):
#     """
#     linear regression with gradient update
#     """
#     # assumes that the last coordinate is the bias term
#     X = np.copy(X_orig)
#     Y = np.copy(Y_orig)
#     n, d = X.shape

#     if theta_init is not None:
#         theta = np.copy(theta_init)
#     else:
#         theta = np.zeros(d)

#     # evaluate initial loss
#     prev_loss = evaluate_MSE(X, Y, theta)

#     loss_list = [prev_loss]
#     is_gd = False
#     i = 0
#     gap = 1e30
#     eta = eta_init

#     while gap > tol and not is_gd:
#         # new gradient
#         if len(reweight) == 0:
#             gradient = -2.0/n * X.T@(Y-X@theta)
#         else:
#             ratio = len(Z_orig[Z_orig == 0])/len(Z_orig)
#             factor_0, factor_1 = reweight[0]/ratio, reweight[1]/(1-ratio)
#             dro_ratio = np.where(Z_orig == 0,factor_0,factor_1)
#             W = np.diag(dro_ratio)
#             # reweighted least square
#             gradient =  -2.0/n * X.T@W@(Y-X@theta)

#         new_theta = theta - eta * gradient

#         # compute new loss
#         loss = evaluate_MSE(X,Y,new_theta)

#         # do backtracking line search
#         if loss > prev_loss and method == 'Exact':
#             eta = eta * .1
#             gap = 1e30
#             continue
#         else:
#             eta = eta_init

#         theta = np.copy(new_theta)

#         loss_list.append(loss)
#         gap = prev_loss - loss
#         prev_loss = loss

#         if method == 'GD':
#             is_gd = True

#         i += 1

#     return theta, loss_list


def fair_linear_regression(X_orig, Y_orig, Z_orig, method, tol=1e-7, theta_init=None, rho=0.1, eta_init = 0.1, demo=True):
    """
    linear regression with fair regularized loss
    """
    # assumes that the last coordinate is the bias term
    X = np.copy(X_orig)
    Y = np.copy(Y_orig)
    Z = np.copy(Z_orig)
    n, d = X.shape

    if theta_init is not None:
        theta = np.copy(theta_init)
    else:
        theta = np.zeros(d)

    # evaluate initial loss
    prev_loss = evaluate_fair_MSE(X, Y, Z, theta, rho)

    loss_list = [prev_loss]
    is_gd = False
    i = 0
    gap = 1e30
    eta = eta_init

    while gap > tol and not is_gd:
        if demo:
            # new gradient
            gradient = calc_fair_reg_grad(X,Y,Z,theta,rho,n)

            new_theta = theta - eta * gradient

            # compute new loss
            loss = evaluate_fair_MSE(X,Y,Z,new_theta,rho)
        else:
            # new gradient
            gradient = calc_fair_reg_grad_demo(X,Y,theta,rho,n)

            new_theta = theta - eta * gradient

            # compute new loss
            loss = evaluate_fair_MSE_demo(X,Y,new_theta,rho)

        # do backtracking line search
        if loss > prev_loss and method == 'Exact':
            eta = eta * .1
            gap = 1e30
            continue
        else:
            eta = eta_init

        theta = np.copy(new_theta)

        loss_list.append(loss)
        gap = prev_loss - loss
        prev_loss = loss

        if method == 'GD':
            is_gd = True

        i += 1

    return theta, loss_list


def e5_regression(X_orig, Y_orig, Z_orig, method, tol=1e-7, theta_init=None, rho=0.1, eta_init = 0.1):
    """
    regression code for example 5
    """
    # assumes that the last coordinate is the bias term
    X = np.copy(X_orig)
    Y = np.copy(Y_orig)
    Z = np.copy(Z_orig)
    n, d = X.shape

    if theta_init is not None:
        theta = np.copy(theta_init)
    else:
        theta = np.zeros(d)

    # evaluate initial loss
    prev_loss = evaluate_e5_MSE(X, Y, Z, theta, rho)

    loss_list = [prev_loss]
    is_gd = False
    i = 0
    gap = 1e30
    eta = eta_init

    while gap > tol and not is_gd:
        # new gradient
        gradient = calc_fair_reg_grad_e5(X,Y,Z,theta,rho,n)

        new_theta = theta - eta * gradient

        # compute new loss
        loss = evaluate_e5_MSE(X,Y,Z,new_theta,rho)

        # do backtracking line search
        if loss > prev_loss and method == 'Exact':
            eta = eta * .1
            gap = 1e30
            continue
        else:
            eta = eta_init

        theta = np.copy(new_theta)

        loss_list.append(loss)
        gap = prev_loss - loss
        prev_loss = loss

        if method == 'GD':
            is_gd = True

        i += 1

    return theta, loss_list