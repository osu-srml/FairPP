import numpy as np
from fairness import *
from data_prep import *
from retention import *
from opt_retention import *
from regression_retention import *


# global parameters
num_iter = 30
lam = 0.1
syn_p0 = 0.3
syn_p1 = 0.7
syn_num = 1002
pmin = [0.02,0.02]
mu_1 = 0.3
mu_2 = 0.7
sd = 0.05
strat_features = np.array([0, 4, 6])
K = 3

def Regular_RRM_clf(data = 'credit',seeds = [0,1,2,3,4,5,6],eps=0, K=K):
    """
    regular RRM for classification
    """

    # all metrics/results we want to solve: k seeds * num_iter iterations
    k = len(seeds)
    if data == 'credit':
        d = Credit_data()[0].shape[1]
    else:
        d = 3
    rrm_p0_list, rrm_p1_list, rrm_acc_list, rrm_l_list = np.zeros((k,num_iter)),np.zeros((k,num_iter)),np.zeros((k,num_iter)),np.zeros((k,num_iter))
    rrm_theta_list, rrm_a_list, rrm_dp_list, rrm_eo_list, rrm_ldisp_list = np.zeros((k,num_iter,d)),np.zeros((k,num_iter)),np.zeros((k,num_iter)),np.zeros((k,num_iter)),np.zeros((k,num_iter))

    for i in range(len(seeds)):
        seed = seeds[i]
        # read data
        if data == 'credit':
            X, Y, Z, _ = Credit_data(seed=seed)
            p_0 = len(Z[Z==0])/len(Z)
            p_1 = 1 - p_0

        elif data == 'syn':
            X, Y, Z, _ = Gaussian_data(p_0=syn_p0,p_1=syn_p1,num_samples=syn_num,seed=seed)
            p_0 = syn_p0
            p_1 = syn_p1
    
        else:
            print('wrong data!')
            return

        # initial round
        n,d = X.shape[0], X.shape[1]
        theta_initial, _, _ = logistic_regression(X, Y, Z, lam, 'Exact')
        rrm_acc, rrm_lprev = calc_metric(X,Y,Z,theta_initial), calc_lprev(X,Y,Z,theta_initial,lam)
        rrm_theta = np.copy(theta_initial)
        rrm_p0_list[i][0] = p_0
        rrm_p1_list[i][0] = p_1
        rrm_theta_list[i][0] = rrm_theta
        rrm_a_list[i][0] = acc_disparity(X,Y,Z,rrm_theta)
        rrm_ldisp_list[i][0] = l_disparity(X,Y,Z,rrm_theta)
        rrm_dp_list[i][0] = dp_disparity(X,Y,Z,rrm_theta)
        rrm_eo_list[i][0] = eo_disparity(X,Y,Z,rrm_theta)
        rrm_l_list[i][0] = calc_curr_l([p_0,p_1], rrm_lprev)


        # RRM process
        for t in range(1,num_iter):
        # first get the participation rate induced by theta_t
            rrm_p_t = participation_map(rrm_lprev,[rrm_p0_list[i][t-1], rrm_p1_list[i][t-1]],pmin)
            rrm_p0_list[i][t] = rrm_p_t[0]
            rrm_p1_list[i][t] = rrm_p_t[1]

            # generate new features
            if data == 'credit':
                # strategic behaviors with K delay
                if eps > 0:
                    X, Y, Z = [],[],[]
                    for l in range(K):
                        if t - l >= 0:
                            X_piece, Y_piece, Z_piece = sample_credit_data(p_0=rrm_p0_list[i][t-l],p_1=rrm_p1_list[i][t-l], num_samples = n//K, seed=seed)
                            X_piece = best_response(X_piece,rrm_theta_list[i][t-l],eps,strat_features)
                        else:
                            X_piece, Y_piece, Z_piece = sample_credit_data(p_0=rrm_p0_list[i][0],p_1=rrm_p1_list[i][0], num_samples = n//K, seed=seed)
                            X_piece = best_response(X_piece,rrm_theta_list[i][t],eps,strat_features)   
                        X.append(X_piece)
                        Y.append(Y_piece)
                        Z.append(Z_piece)
                    X, Y, Z = np.concatenate(X), np.concatenate(Y), np.concatenate(Z)


            else:
                X, Y, Z, _ = Gaussian_data(p_0=rrm_p_t[0],p_1=rrm_p_t[1],num_samples=syn_num, seed=seed)
            
            # get the corresponding expected accuracy/loss
            rrm_phat_t = [len(Z[Z == 0])/len(Z), len(Z[Z==1])/len(Z)]
            rrm_acc_list[i][t] = calc_curr_acc(rrm_phat_t,rrm_acc)
            rrm_l_list[i][t] = calc_curr_l(rrm_phat_t, rrm_lprev)

            # perform ERM with the new X, Y, Z parametrized by p_t, get theta
            rrm_theta_new,_,_ = logistic_regression(X, Y, Z, lam, 'Exact', tol=1e-7)


            # get the new acc and loss with respect to the original sample X, Y, Z. Not the resampled ones
            rrm_acc = calc_metric(X,Y,Z,rrm_theta_new)
            rrm_lprev = calc_lprev(X,Y,Z,rrm_theta_new,lam)
            rrm_theta_list[i][t] = np.copy(rrm_theta_new)
            rrm_a_list[i][t] = acc_disparity(X,Y,Z,rrm_theta_new)
            rrm_ldisp_list[i][t] = l_disparity(X,Y,Z,rrm_theta_new)
            rrm_dp_list[i][t] = dp_disparity(X,Y,Z,rrm_theta_new)
            rrm_eo_list[i][t] = eo_disparity(X,Y,Z,rrm_theta_new)
    
    return rrm_p0_list, rrm_p1_list, rrm_acc_list, rrm_l_list,  rrm_theta_list, rrm_a_list, rrm_dp_list, rrm_eo_list, rrm_ldisp_list


def DRO_clf(data = 'credit',seeds = [0,1,2,3,4,5,6],rho=0.1,eps=0):
    """
    DRO for classification
    """
    # all metrics/results we want to solve: k seeds * num_iter iterations
    k = len(seeds)
    if data == 'credit':
        d = Credit_data()[0].shape[1]
    else:
        d = 3
    rrm_p0_list, rrm_p1_list, rrm_acc_list, rrm_l_list, rrm_q0_list, rrm_q1_list = np.zeros((k,num_iter)),np.zeros((k,num_iter)),np.zeros((k,num_iter)),np.zeros((k,num_iter)),np.zeros((k,num_iter)),np.zeros((k,num_iter))
    rrm_theta_list, rrm_a_list, rrm_dp_list, rrm_eo_list, rrm_ldisp_list = np.zeros((k,num_iter,d)),np.zeros((k,num_iter)),np.zeros((k,num_iter)),np.zeros((k,num_iter)),np.zeros((k,num_iter))

    for i in range(len(seeds)):
        seed = seeds[i]
        # read data
        if data == 'credit':
            X, Y, Z, _ = Credit_data(seed=seed)
            p_0 = len(Z[Z==0])/len(Z)
            p_1 = 1 - p_0

        elif data == 'syn':
            X, Y, Z, _ = Gaussian_data(p_0=syn_p0,p_1=syn_p1,num_samples=syn_num,seed=seed)
            p_0 = syn_p0
            p_1 = syn_p1
    
        else:
            print('wrong data!')
            return

        # initial round
        n,d = X.shape[0], X.shape[1]
        theta_initial, _, _ = logistic_regression(X, Y, Z, lam, 'Exact')
        rrm_acc, rrm_lprev = calc_metric(X,Y,Z,theta_initial), calc_lprev(X,Y,Z,theta_initial,lam)
        rrm_theta = np.copy(theta_initial)
        rrm_p0_list[i][0] = p_0
        rrm_p1_list[i][0] = p_1
        rrm_theta_list[i][0] = rrm_theta
        rrm_a_list[i][0] = acc_disparity(X,Y,Z,rrm_theta)
        rrm_ldisp_list[i][0] = l_disparity(X,Y,Z,rrm_theta)
        rrm_dp_list[i][0] = dp_disparity(X,Y,Z,rrm_theta)
        rrm_eo_list[i][0] = eo_disparity(X,Y,Z,rrm_theta)
        rrm_l_list[i][0] = calc_curr_l([p_0,p_1], rrm_lprev)
        
        # RRM process
        for t in range(1,num_iter):
        # first get the participation rate induced by theta_t
            rrm_p_t = participation_map(rrm_lprev,[rrm_p0_list[i][t-1], rrm_p1_list[i][t-1]],pmin)
            rrm_p0_list[i][t] = rrm_p_t[0]
            rrm_p1_list[i][t] = rrm_p_t[1]

            # generate new features
            if data == 'credit':
                # strategic behaviors with K delay
                if eps > 0:
                    X, Y, Z = [],[],[]
                    for l in range(K):
                        if t - l >= 0:
                            X_piece, Y_piece, Z_piece = sample_credit_data(p_0=rrm_p0_list[i][t-l],p_1=rrm_p1_list[i][t-l], num_samples = n//K, seed=seed)
                            X_piece = best_response(X_piece,rrm_theta_list[i][t-l],eps,strat_features)
                        else:
                            X_piece, Y_piece, Z_piece = sample_credit_data(p_0=rrm_p0_list[i][0],p_1=rrm_p1_list[i][0], num_samples = n//K, seed=seed)
                            X_piece = best_response(X_piece,rrm_theta_list[i][t],eps,strat_features)   
                        X.append(X_piece)
                        Y.append(Y_piece)
                        Z.append(Z_piece)
                    X, Y, Z = np.concatenate(X), np.concatenate(Y), np.concatenate(Z)
            else:
                X, Y, Z, _ = Gaussian_data(p_0=rrm_p_t[0],p_1=rrm_p_t[1],num_samples=syn_num, seed=seed)
            
            # get the corresponding expected accuracy/loss
            rrm_phat_t = [len(Z[Z == 0])/len(Z), len(Z[Z==1])/len(Z)]
            rrm_acc_list[i][t] = calc_curr_acc(rrm_phat_t,rrm_acc)
            rrm_l_list[i][t] = calc_curr_l(rrm_phat_t, rrm_lprev)
            rrm_q_t = DRO_map(rrm_p_t,rrm_lprev,rho=rho)
            rrm_q0_list[i][t] = rrm_q_t[0]
            rrm_q1_list[i][t] = rrm_q_t[1]          

            # perform ERM with the new X, Y, Z parametrized by p_t, get theta
            rrm_theta_new,_,_ = logistic_regression(X, Y, Z,lam, 'Exact', tol=1e-7,reweight=rrm_q_t)

            # get the new acc and loss with respect to the original sample X, Y, Z. Not the resampled ones
            rrm_acc = calc_metric(X,Y,Z,rrm_theta_new)
            rrm_lprev = calc_lprev(X,Y,Z,rrm_theta_new,lam)
            rrm_theta_list[i][t] = np.copy(rrm_theta_new)
            rrm_a_list[i][t] = acc_disparity(X,Y,Z,rrm_theta_new)
            rrm_ldisp_list[i][t] = l_disparity(X,Y,Z,rrm_theta_new)
            rrm_dp_list[i][t] = dp_disparity(X,Y,Z,rrm_theta_new)
            rrm_eo_list[i][t] = eo_disparity(X,Y,Z,rrm_theta_new)
    
    return rrm_p0_list, rrm_p1_list, rrm_q0_list, rrm_q1_list,rrm_acc_list, rrm_l_list,  rrm_theta_list, rrm_a_list, rrm_dp_list, rrm_eo_list, rrm_ldisp_list



def fair_RRM_clf(data = 'credit',seeds = [0,1,2,3,4,5,6], rho=0.1, demo=True,eps=0):
    """
    fair RRM for classification
    """

    # all metrics/results we want to solve: k seeds * num_iter iterations
    k = len(seeds)
    if data == 'credit':
        d = Credit_data()[0].shape[1]
    else:
        d = 3
    rrm_p0_list, rrm_p1_list, rrm_acc_list, rrm_l_list = np.zeros((k,num_iter)),np.zeros((k,num_iter)),np.zeros((k,num_iter)),np.zeros((k,num_iter))
    rrm_theta_list, rrm_a_list, rrm_dp_list, rrm_eo_list, rrm_ldisp_list = np.zeros((k,num_iter,d)),np.zeros((k,num_iter)),np.zeros((k,num_iter)),np.zeros((k,num_iter)),np.zeros((k,num_iter))

    for i in range(len(seeds)):
        seed = seeds[i]
        # read data
        if data == 'credit':
            X, Y, Z, _ = Credit_data(seed=seed)
            p_0 = len(Z[Z==0])/len(Z)
            p_1 = 1 - p_0

        elif data == 'syn':
            X, Y, Z, _ = Gaussian_data(p_0=syn_p0,p_1=syn_p1,num_samples=syn_num,seed=seed)
            p_0 = syn_p0
            p_1 = syn_p1
    
        else:
            print('wrong data!')
            return

        # initial round
        n,d = X.shape[0], X.shape[1]
        theta_initial, _, _ = fair_logistic_regression(X, Y, Z, 'Exact', rho=rho, demo=demo, lam=lam)
        rrm_acc, rrm_lprev = calc_metric(X,Y,Z,theta_initial), calc_lprev(X,Y,Z,theta_initial,lam)
        rrm_theta = np.copy(theta_initial)
        rrm_p0_list[i][0] = p_0
        rrm_p1_list[i][0] = p_1
        rrm_theta_list[i][0] = rrm_theta
        rrm_a_list[i][0] = acc_disparity(X,Y,Z,rrm_theta)
        rrm_ldisp_list[i][0] = l_disparity(X,Y,Z,rrm_theta)
        rrm_dp_list[i][0] = dp_disparity(X,Y,Z,rrm_theta)
        rrm_eo_list[i][0] = eo_disparity(X,Y,Z,rrm_theta)
        rrm_l_list[i][0] = calc_curr_l([p_0,p_1], rrm_lprev)

        # RRM process
        for t in range(1,num_iter):
        # first get the participation rate induced by theta_t
            rrm_p_t = participation_map(rrm_lprev,[rrm_p0_list[i][t-1], rrm_p1_list[i][t-1]],pmin)
            rrm_p0_list[i][t] = rrm_p_t[0]
            rrm_p1_list[i][t] = rrm_p_t[1]

            # generate new features
            if data == 'credit':
                # strategic behaviors with K delay
                if eps > 0:
                    X, Y, Z = [],[],[]
                    for l in range(K):
                        if t - l >= 0:
                            X_piece, Y_piece, Z_piece = sample_credit_data(p_0=rrm_p0_list[i][t-l],p_1=rrm_p1_list[i][t-l], num_samples = n//K, seed=seed)
                            X_piece = best_response(X_piece,rrm_theta_list[i][t-l],eps,strat_features)
                        else:
                            X_piece, Y_piece, Z_piece = sample_credit_data(p_0=rrm_p0_list[i][0],p_1=rrm_p1_list[i][0], num_samples = n//K, seed=seed)
                            X_piece = best_response(X_piece,rrm_theta_list[i][t],eps,strat_features)   
                        X.append(X_piece)
                        Y.append(Y_piece)
                        Z.append(Z_piece)
                    X, Y, Z = np.concatenate(X), np.concatenate(Y), np.concatenate(Z)
            else:
                X, Y, Z, _ = Gaussian_data(p_0=rrm_p_t[0],p_1=rrm_p_t[1],num_samples=syn_num, seed=seed)
            
            # get the corresponding expected accuracy/loss
            rrm_phat_t = [len(Z[Z == 0])/len(Z), len(Z[Z==1])/len(Z)]
            rrm_acc_list[i][t] = calc_curr_acc(rrm_phat_t,rrm_acc)
            rrm_l_list[i][t] = calc_curr_l(rrm_phat_t, rrm_lprev)

            # perform ERM with the new X, Y, Z parametrized by p_t, get theta
            rrm_theta_new,_,_ = fair_logistic_regression(X, Y, Z, 'Exact', rho=rho, demo=demo, lam=lam)

            # get the new acc and loss with respect to the original sample X, Y, Z. Not the resampled ones
            rrm_acc = calc_metric(X,Y,Z,rrm_theta_new)
            rrm_lprev = calc_lprev(X,Y,Z,rrm_theta_new,lam)
            rrm_theta_list[i][t] = np.copy(rrm_theta_new)
            rrm_a_list[i][t] = acc_disparity(X,Y,Z,rrm_theta_new)
            rrm_ldisp_list[i][t] = l_disparity(X,Y,Z,rrm_theta_new)
            rrm_dp_list[i][t] = dp_disparity(X,Y,Z,rrm_theta_new)
            rrm_eo_list[i][t] = eo_disparity(X,Y,Z,rrm_theta_new)
    
    return rrm_p0_list, rrm_p1_list, rrm_acc_list, rrm_l_list,  rrm_theta_list, rrm_a_list, rrm_dp_list, rrm_eo_list, rrm_ldisp_list
