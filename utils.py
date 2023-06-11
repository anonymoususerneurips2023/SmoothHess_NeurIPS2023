import torch 
from cvxpy import * 
import numpy as np 
import torch 
from torch.autograd.functional import hessian
import pandas as pd
import os
import json
import sys 
import time 
from math import sqrt 
from sklearn.metrics import auc 


# Local Imports
from Models import * 
from Datasets import * 

############################ Functions ###################################
def TaylorSecondOrder(Grad, Hess, x_perturbed_minus_x, use_hess = True):
    with torch.no_grad(): 
        if use_hess:
            return x_perturbed_minus_x @ Grad + 1/2 * torch.diag((x_perturbed_minus_x @ Hess @ x_perturbed_minus_x.T)) 
        else: 
            return x_perturbed_minus_x @ Grad  

def SampleFromBall(d, n1, r):
    MVN_Unit_Samples          = torch.normal(mean = torch.zeros(d * n1), std = torch.ones(d * n1 ))
    MVN_Unit_Samples          = MVN_Unit_Samples.view(n1, d)
    Sphere_Uniform_Samples    = MVN_Unit_Samples / MVN_Unit_Samples.norm(dim = 1).unsqueeze(dim = 1) 
    Radii_Uniform_Samples     = torch.rand((n1,1)) * r
    Ball_Uniform_Samples      = Sphere_Uniform_Samples * Radii_Uniform_Samples
    Ball_Uniform_Samples      = Ball_Uniform_Samples.cuda() 
    return Ball_Uniform_Samples
#############################################################################



######## Optimization #######



# def eigen_topk_convert_grad_hess(Grad, Hess):

    
#     return Grad_New, Hess_New

######################### Inputs: #########################
# (1) Grad : Gradient, d dimensional vector
# (2) Hessian: Hessian, d x d dimension matrix
# (3) r: Neighorhood to consider in optimization, scalar
# (4) min: boolean indicating if we aim to minimize or maximize taylor expansion in r
# (5) include_grad: boolean indicating use of only hessian or both hessian and gradient 
######################### Outputs: #########################
# (1) optimval_val : the optimal value attained by the optimization, scalar
# (2) delta.value : the perturbation which achieves the optimal value, d diomensional vecotr 
############################################################
def FindOptimalDelta(Grad, Hess, r, min = True, include_grad = True, savedir = None, savename = None, V_tilde = None):
    if min == False:
        Hess *= -1
        Grad *= -1 

    d = Hess.shape[0]
    sq_r = r ** 2 
    one = np.ones((1,1))

    X = Variable((d, d), symmetric=True)
    delta = Variable((d,1))
    if include_grad:
        cost = (trace(1/2 * Hess @ X) + Grad @ delta)
    else:
        cost = (trace(1/2 * Hess @ X)) 

    cmat = bmat([[X, delta], [transpose(delta), one]]) #, [transpose(delta), 1]])


    # if B is None:
    constraints = [trace(X) - sq_r <=0, cmat >> 0] 
    # else:
    #     constraints = [trace(B @ X) - sq_r <=0, cmat >> 0] 

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    optimal_val = prob.solve()
    

    optimal_delta = delta.value
    optimal_delta = np.squeeze(optimal_delta, axis = 1)

    if V_tilde is not None:
        optimal_delta = np.matmul(V_tilde, optimal_delta)

    if not (savedir is None or savename is None):
        np.save(savedir + savename + ".npy" , optimal_delta)

    return optimal_val, optimal_delta 