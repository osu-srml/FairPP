"""
utility functions to simulate: 1. regular RRM; 2. fair RRM; 3. DRO
"""
import numpy as np
from fairness import *
from data_prep import *
from retention import *
from opt_retention import *
from regression_retention import *


# global parameters
num_iter = 30
lam = 0.2
syn_p_list = [0.15,0.25,0.6]
syn_mu_list = [0.3,0.5,0.7]
syn_num = 10000
pmin = [0.1,0.1,0.1]
sd = 0.05

def Regular_RRM_reg(seeds=[0,1,2,3,4,5,6]):
    """
    regular RRM for regression
    """
    k, g = len(seeds), len(syn_p_list)
    rrm_p_list, rrm_l_list = np.zeros((k,num_iter,g)),np.zeros((k,num_iter))
    rrm_theta_list, rrm_ldisp_list = np.zeros((k,num_iter)),np.zeros((k,num_iter))

    for i in range(len(seeds)):
        seed = seeds[i]
        X, Y, Z = Gaussian_mean_data_multi(mu_list=syn_mu_list,sd=sd,p_list=syn_p_list,num_samples=syn_num,seed=seed)  

        # initial round
        n,d = X.shape[0], X.shape[1]
        theta_initial = linear_regression_ana(X, Y)
        rrm_lprev = calc_MSE_prev(X,Y,Z,theta_initial)
        rrm_theta = np.copy(theta_initial)
        rrm_p_list[i][0] = syn_p_list
        rrm_theta_list[i][0] = rrm_theta
        rrm_ldisp_list[i][0] = l_disparity_reg(X,Y,Z,rrm_theta)
        rrm_l_list[i][0] = calc_curr_l(syn_p_list, rrm_lprev)

        # RRM process
        for t in range(1,num_iter):
        # first get the participation rate induced by theta_t
            rrm_p_t = participation_map(rrm_lprev,rrm_p_list[i][t-1],pmin)
            rrm_p_list[i][t] = rrm_p_t

            # generate new features
            X, Y, Z = Gaussian_mean_data_multi(mu_list=syn_mu_list,sd=sd,p_list=rrm_p_t,num_samples=syn_num,seed=seed)
            
            # get the corresponding expected accuracy/loss
            rrm_l_list[i][t] = calc_curr_l(rrm_p_t, rrm_lprev)

            # perform ERM with the new X, Y, Z parametrized by p_t, get theta
            rrm_theta_new = linear_regression_ana(X, Y)

            # get the new acc and loss with respect to the original sample X, Y, Z. Not the resampled ones
            rrm_lprev = calc_MSE_prev(X,Y,Z,rrm_theta_new)
            rrm_theta_list[i][t] = np.copy(rrm_theta_new)
            rrm_ldisp_list[i][t] = l_disparity_reg(X,Y,Z,rrm_theta_new)
    
    return rrm_p_list, rrm_l_list, rrm_theta_list, rrm_ldisp_list


def fair_RRM_reg(seeds = [0,1,2,3,4,5,6], rho=0.1, demo=True):
    """
    fair RRM for regression
    """
    k,g = len(seeds),len(syn_p_list)
    rrm_p_list, rrm_l_list = np.zeros((k,num_iter,g)),np.zeros((k,num_iter))
    rrm_theta_list, rrm_ldisp_list = np.zeros((k,num_iter)),np.zeros((k,num_iter))

    for i in range(len(seeds)):
        seed = seeds[i]
        X, Y, Z = Gaussian_mean_data_multi(mu_list=syn_mu_list,sd=sd,p_list=syn_p_list,num_samples=syn_num, seed=seed) 

        # initial round
        n,d = X.shape[0], X.shape[1]
        theta_initial, _ = fair_linear_regression(X, Y, Z, method='Exact',tol=1e-10,rho=rho,demo=demo)
        rrm_lprev = calc_MSE_prev(X,Y,Z,theta_initial)
        rrm_theta = np.copy(theta_initial)
        rrm_p_list[i][0] = syn_p_list
        rrm_theta_list[i][0] = rrm_theta
        rrm_ldisp_list[i][0] = l_disparity_reg(X,Y,Z,rrm_theta)
        rrm_l_list[i][0] = calc_curr_l(syn_p_list, rrm_lprev)

        # RRM process
        for t in range(1,num_iter):
        # first get the participation rate induced by theta_t
            rrm_p_t = participation_map(rrm_lprev,rrm_p_list[i][t-1],pmin)
            rrm_p_list[i][t] = rrm_p_t

            # generate new features
            X, Y, Z = Gaussian_mean_data_multi(mu_list=syn_mu_list,sd=sd,p_list=rrm_p_t,num_samples=syn_num, seed=seed)
            
            # get the corresponding expected accuracy/loss
            rrm_l_list[i][t] = calc_curr_l(rrm_p_t, rrm_lprev)

            # perform ERM with the new X, Y, Z parametrized by p_t, get theta
            rrm_theta_new, _ = fair_linear_regression(X, Y, Z, method='Exact',tol=1e-10,rho=rho,demo=demo)

            # get the new acc and loss with respect to the original sample X, Y, Z. Not the resampled ones
            rrm_lprev = calc_MSE_prev(X,Y,Z,rrm_theta_new)
            rrm_theta_list[i][t] = np.copy(rrm_theta_new)
            rrm_ldisp_list[i][t] = l_disparity_reg(X,Y,Z,rrm_theta_new)
    
    return rrm_p_list, rrm_l_list, rrm_theta_list, rrm_ldisp_list

def DRO_reg(seeds=[0,1,2,3,4,5,6],rho=0.1):
    """
    DRO for regression
    """
    k,g = len(seeds),len(syn_p_list)
    rrm_p_list, rrm_q_list, rrm_l_list = np.zeros((k,num_iter,g)),np.zeros((k,num_iter,g)),np.zeros((k,num_iter))
    rrm_theta_list, rrm_ldisp_list = np.zeros((k,num_iter)),np.zeros((k,num_iter))

    for i in range(len(seeds)):
        seed = seeds[i]
        X, Y, Z, = Gaussian_mean_data_multi(mu_list=syn_mu_list,sd=sd,p_list=syn_p_list,num_samples=syn_num,seed=seed) 

        # initial round
        n,d = X.shape[0], X.shape[1]
        theta_initial = linear_regression_ana(X, Y)
        rrm_lprev = calc_MSE_prev(X,Y,Z,theta_initial)
        rrm_theta = np.copy(theta_initial)
        rrm_p_list[i][0] = syn_p_list
        rrm_theta_list[i][0] = rrm_theta
        rrm_ldisp_list[i][0] = l_disparity_reg(X,Y,Z,rrm_theta)
        rrm_l_list[i][0] = calc_curr_l(syn_p_list, rrm_lprev)

        # RRM process
        for t in range(1,num_iter):
        # first get the participation rate induced by theta_t
            rrm_p_t = participation_map(rrm_lprev,rrm_p_list[i][t-1],pmin)
            rrm_p_list[i][t] = rrm_p_t

            # generate new features
            X, Y, Z = Gaussian_mean_data_multi(mu_list=syn_mu_list,sd=sd,p_list=rrm_p_t,num_samples=syn_num,seed=seed)
            
            # get the corresponding expected accuracy/loss
            rrm_l_list[i][t] = calc_curr_l(rrm_p_t, rrm_lprev)
            rrm_q_t = DRO_map(rrm_p_t,rrm_lprev,rho=rho)
            rrm_q_list[i][t] = rrm_q_t

            # perform ERM with the new X, Y, Z parametrized by p_t, get theta
            rrm_theta_new = weighted_lr_ana(X,Y,Z,reweight=rrm_q_t)

            # get the new acc and loss with respect to the original sample X, Y, Z. Not the resampled ones
            rrm_lprev = calc_MSE_prev(X,Y,Z,rrm_theta_new)
            rrm_theta_list[i][t] = np.copy(rrm_theta_new)
            rrm_ldisp_list[i][t] = l_disparity_reg(X,Y,Z,rrm_theta_new)

    return rrm_p_list, rrm_q_list, rrm_l_list, rrm_theta_list, rrm_ldisp_list
