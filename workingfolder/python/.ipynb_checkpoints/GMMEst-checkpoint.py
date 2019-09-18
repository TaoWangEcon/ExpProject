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


# + {"code_folding": [0]}
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


# + {"code_folding": []}
def FIREForecaster(real_time,horizon =10,process_para = process_para):
    FIREFE = 0 
    FIREDisg = 0
    rho = process_para['rho']
    sigma = process_para['sigma']
    FIREVar = sum([rho**i * sigma**2 for i in range(horizon)])
    return {"FE":FIREFE,
           "Disg":FIREDisg,
           "Var":FIREVar}


# + {"code_folding": []}
def SEForecaster(real_time,horizon =10,process_para = process_para,exp_para = exp_para):
    rho = process_para['rho']
    sigma = process_para['sigma']
    lambda = exp_para['lambda']
    FIREFE = # a function of lambda, real-time and process_para 
    FIREDisg = 0 # same as above
    FIREVar = # same as above 
    return {"FE":SEFE,
           "Disg":SEDisg,
           "Var":SEVar}
