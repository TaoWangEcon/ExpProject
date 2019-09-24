# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### GMM Estimation of Model Parameters
#
# - This notebook includes functions that estimate the parameter of rigidity for different models
# - It allows for flexible choices of moments to be used, forecast error, disagreement, and uncertainty, etc. 
# - It includes 
#   - A general function that implements the estimation using the minimum distance algorithm. 
#   - Model-specific functions that take real-time data and process parameters as inputs and produces forecasts as outputs. It is model-specific because different models of expectation formation bring about different forecasts. 
#   - Auxiliary functions that compute the moments as well as the difference of data and model prediction, which will be used as inputs for GMM estimator. 

from scipy.optimize import minimize
import numpy as np


# + {"code_folding": [0]}
# a estimating function of the parameter
def Estimator(moments,method):
    """
    Inputs
    ------
    - moments: a function of the rigidity model parameter  
    - method: method of optimization 
    
    Outputs
    -------
    - parameter: an array of estimated parameter
    """
    
    parameter = minimize(moments,method=method)
    return parameter 


# + {"code_folding": [0]}
# a function that prepares moment conditions
def PrepMom(model_moments,data_moments):
    """
    Inputs
    -----
    model_moments: an array of moments from a certain model, i.e. forecast error, disagreement and uncertainty. 
    data_moments: an array of moments computed from the survey data
    
    Outputs
    ------
    diff: an array of distances of data and model 
    
    """
    diff = model_moments - data_moments
    return diff


# + {"code_folding": [0]}
## some parameters 
rho = 0.95
sigma = 0.1
process_para = {'rho':rho,
                'sigma':sigma}


# + {"code_folding": []}
## auxiliary functions 
def hstepvar(h,sigma,rho):
    return sum([ rho**(2*i)*sigma**2 for i in range(h)] )

def hstepfe(h,sigma,rho):
    return sum([rho**i*sigma*np.random.randn(h)[i] for i in range(h)])
## This is not correct. 



# + {"code_folding": [0]}
# a function that generates population moments according to FIRE 
def FIREForecaster(real_time,horizon =10,process_para = process_para):
    n = len(real_time)
    FE = np.zeros(n) 
    Disg =np.zeros(n)
    rho = process_para['rho']
    sigma = process_para['sigma']
    infoset = real_time
    nowcast = infoset
    forecast = rho**horizon*nowcast
    Var = hstepvar(horizon,sigma,rho)* np.ones(n)
    return {"Forecast":forecast,
            "FE":FE,
           "Disg":Disg,
           "Var":Var}


# + {"code_folding": [0]}
## test
FIREtest = FIREForecaster(np.array([1,2]))
FIREtest

# + {"code_folding": [0]}
## SE parameters

SE_para ={'lambda':0.75}


# + {"code_folding": []}
# a function that generates population moments according to SE 
def SEForecaster(real_time,horizon =10,process_para = process_para,exp_para = SE_para):
    n = len(real_time)
    rho = process_para['rho']
    sigma = process_para['sigma']
    lbd = exp_para['lambda']
    max_back = 10 # need to avoid magic numbers 
    FE = sum( [lbd*(1-lbd)**tau*hstepfe(horizon+tau,sigma,rho) for tau in range(max_back)] ) * np.ones(n) # a function of lambda, real-time and process_para 
    Disg = 0 # same as above
    Var = sum([ lbd*(1-lbd)**tau*hstepvar(horizon+tau,sigma,rho) for tau in range(max_back)] ) * np.ones(n)  
    # same as above 
    nowcast = sum([ lbd*(1-lbd)**tau*(rho**tau)*np.roll(real_time,tau) for tau in range(max_back)]) 
    # the first tau needs to be burned
    forecast = rho**horizon*nowcast
    return {"Forecast":forecast, 
            "FE":FE,
            "Disg":Disg,
            "Var":Var}


# + {"code_folding": [0]}
## test 
xxx = np.random.rand(10)
SEForecaster(xxx,horizon=1)


# + {"code_folding": [0]}
# a function that generates population moments according to NI 
def NIForecaster(real_time,horizon =10,process_para = process_para,exp_para = exp_para):
    rho = process_para['rho']
    sigma = process_para['sigma']
    sigma_pr = exp_para['sigma_pr']
    sigma_pb = exp_para['sigma_pb']
    
    NIFE = # a function of lambda, real-time and process_para 
    NIDisg = 0 # same as above
    NIVar = # same as above 
    return {"FE":NIFE,
           "Disg":NIDisg,
           "Var":NIVar}
