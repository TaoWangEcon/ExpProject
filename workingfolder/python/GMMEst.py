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
#   - Model-specific functions that take real-time data and process parameters as inputs and produces forecasts and moments as outputs. It is model-specific because different models of expectation formation bring about different forecasts. 
#   - Auxiliary functions that compute the moments as well as the difference of data and model prediction, which will be used as inputs for GMM estimator. 

from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import arma_generate_sample


# + {"code_folding": [0]}
# a estimating function of the parameter
def Estimator(obj_func,para_guess,method='CG'):
    """
    Inputs
    ------
    - moments: a function of the rigidity model parameter  
    - method: method of optimization 
    
    Outputs
    -------
    - parameter: an array of estimated parameter
    """
    
    parameter = minimize(obj_func,x0 = para_guess,method=method)['x']
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
    diff: the Euclidean distance of two arrays of data and model 
    
    """
    diff = np.linalg.norm(model_moments - data_moments)
    return diff


# + {"code_folding": [0]}
## some parameters 
rho = 0.95
sigma = 0.1
process_para = {'rho':rho,
                'sigma':sigma}


# + {"code_folding": [0]}
## auxiliary functions 
def hstepvar(h,sigma,rho):
    return sum([ rho**(2*i)*sigma**2 for i in range(h)] )

np.random.seed(12345)
def hstepfe(h,sigma,rho):
    return sum([rho**i*(np.random.randn(1)*simga)*np.random.randn(h)[i] for i in range(h)])
## This is not correct. 


def AR1_simulator(rho,sigma,nobs):
    xxx = np.zeros(nobs+1)
    shocks = np.random.randn(nobs+1)*sigma
    xxx[0] = 0 
    for i in range(nobs):
        xxx[i+1] = rho*xxx[i] + shocks[i+1]
    return xxx[1:]


def ForecastPlot(test):
    plt.figure(figsize=([3,13]))
    for i,val in enumerate(test):
        plt.subplot(4,1,i+1)
        plt.plot(test[val],label=val)
        plt.legend(loc=1)


# + {"code_folding": []}
## AR1 series for testing 
nobs = 100
rho = process_para['rho']
sigma = process_para['sigma']
xxx = AR1_simulator(rho,sigma,nobs)


# + {"code_folding": [0]}
# a function that generates population moments according to FIRE 
def FIREForecaster(real_time,horizon =10,process_para = process_para):
    n = len(real_time)
    rho = process_para['rho']
    sigma = process_para['sigma']
    Disg =np.zeros(n)
    FE = np.random.rand(n)*sigma  ## forecast errors depend on realized shocks 
    infoset = real_time
    nowcast = infoset
    forecast = rho**horizon*nowcast
    Var = hstepvar(horizon,sigma,rho)* np.ones(n)
    return {"Forecast":forecast,
            "FE":FE,
           "Disg":Disg,
           "Var":Var}


# + {"code_folding": [0]}
## simulate a AR1 series for testing 
FIREtest = FIREForecaster(xxx,horizon=1)

# + {"code_folding": [0]}
# plot different moments
ForecastPlot(FIREtest)

# + {"code_folding": [0]}
## SE parameters

SE_para ={'lambda':0.75}


# + {"code_folding": [0, 1]}
# a function that generates population moments according to SE 
def SEForecaster(real_time,horizon =10,process_para = process_para,exp_para = SE_para):
    n = len(real_time)
    rho = process_para['rho']
    sigma = process_para['sigma']
    lbd = exp_para['lambda']
    max_back = 10 # need to avoid magic numbers 
    FE = sum( [lbd*(1-lbd)**tau*hstepfe(horizon+tau,sigma,rho) for tau in range(max_back)] ) * np.ones(n) # a function of lambda, real-time and process_para 
    Var = sum([ lbd*(1-lbd)**tau*hstepvar(horizon+tau,sigma,rho) for tau in range(max_back)] ) * np.ones(n)  
    # same as above 
    nowcast = sum([ lbd*(1-lbd)**tau*(rho**tau)*np.roll(real_time,tau) for tau in range(max_back)]) 
    # the first tau needs to be burned
    forecast = rho**horizon*nowcast
    Disg =  sum([ lbd*(1-lbd)**tau*(rho**(tau+horizon)*np.roll(real_time,tau)-forecast)**2 for tau in range(max_back)] )
    return {"Forecast":forecast, 
            "FE":FE,
            "Disg":Disg,
            "Var":Var}


# + {"code_folding": [0]}
## test 

SEtest = SEForecaster(xxx,horizon=1)
ForecastPlot(SEtest)

# + {"code_folding": [0]}
## prepare inputs for the estimation

horizon = 1
real_time = xxx 
process_para = process_para
data_moms_dct = SEtest


# + {"code_folding": [0]}
## a function estimating SE model parameter only 

def SE_EstObjfunc(lbd):
    """
    input
    -----
    lbd: the parameter of SE model to be estimated
    
    output
    -----
    the objective function to minmize
    """

    SE_para = {"lambda":lbd}
    SE_moms_dct = SEForecaster(real_time,horizon=horizon,process_para = process_para,exp_para = SE_para)
    SE_moms = np.array([val for key,val in SE_moms_dct.items()] )
    data_moms = np.array([val for key,val in data_moms_dct.items()] ) + np.random.rand(4,nobs)
    obj_func = PrepMom(SE_moms,data_moms)
    return obj_func 


# + {"code_folding": []}
## invoke the estimation of SE 

lbd_est = Estimator(SE_EstObjfunc,para_guess =0.8,method='CG')
lbd_est


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
