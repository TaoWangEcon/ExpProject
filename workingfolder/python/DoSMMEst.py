# -*- coding: utf-8 -*-
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

# ## SMM Estimation of Theories of Expectation Formation on Inflation 
#
# - The codes are organized in following ways
#
#   1. Each pair of a theory of expectation formation (re, se, ni, de,etc) and an assumed process of inflation process (ar1 or sv)  are encapsulated in a specific python class. 
#     - the class initilizes corresponding parameters of inflation process and expectation formation 
#     - and embodies a specific function that generates all the SMM moments of both inflation and expectations 
#     
#   2. A generaly written objective function that computes the distances in moments as a function of parameters specific to the chosen model, moments and the data. 
#   3.  The general function is to be used to compute the specific objective function that takes parameter as the only input for the minimizor to work
#   4.  Then a general function that does optimization algorithm takes the specific objective function and estimates the parameters

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numba as nb
from numba import jit, njit, jitclass, float64, int64
#import statsmodels.api as sm
#from numba.typed import List

# + {"code_folding": [2, 11, 35, 48]}
## auxiliary functions 
@njit
def hstepvar(h,
             ρ,
             σ):
    var = 0
    for i in range(h):
        var += ρ**(2*i)*σ**2 
    return var 

@njit
def hstepvarSV(h,
               σs_now,
               γ):
    '''
    inputs
    ------
    h: forecast horizon
    σs_now, 2 x 1 vector [sigma_eta, sigma_eps]
    γ, volatility, scalar. 
    
    outputs
    -------
    scalar: h-step-forward variance 
    '''
    ratio = 0
    for k in range(h):
        ratio += np.exp(-0.5*(k+1)*γ)
    var_eta = σs_now[0]**2*ratio
    var_eps = σs_now[1]**2*np.exp(-0.5*h*γ)
    var = var_eta + var_eps
    return var

### AR1 simulator 
@njit
def SimAR1(ρ,
           σ,
           T):
    xxx = np.zeros(T+1)
    shocks = np.random.randn(T+1)*σ
    xxx[0] = 0 
    for i in range(T):
        xxx[i+1] = ρ*xxx[i] + shocks[i+1]
    return xxx[1:]


### UC-SV simulator 
@njit
def SimUCSV(γ,
            nobs,
            p0 = 0,
            seed = False):
    """
    p: permanent 
    t: transitory 
    
    output
    ======
    y: the draw of series
    p: the permanent component
    svols_p: permanent volatility 
    svols_t: transitory volatility 
    """
    if seed == True:
        np.random.seed(12344)
    else:
        pass
    svols_p_shock = np.random.randn(nobs+1)*γ
    svols_t_shock = np.random.randn(nobs+1)*γ
    
    svols_p = np.zeros(nobs+1)
    svols_p[0] = 0.001
    svols_t = np.zeros(nobs+1)
    svols_t[0] = 0.001
    for i in range(nobs):
        svols_p[i+1] = np.sqrt( np.exp(np.log(svols_p[i]**2) + svols_p_shock[i+1]) ) 
        svols_t[i+1] = np.sqrt( np.exp(np.log(svols_t[i]**2) + svols_t_shock[i+1]) ) 
    shocks_p = np.multiply(np.random.randn(nobs+1),svols_p)  
    shocks_t = np.multiply(np.random.randn(nobs+1),svols_t)
    
    p = np.zeros(nobs+1)
    t = np.zeros(nobs+1)
    
    ## initial level of eta, 0 by default
    p[0] = p0
    
    for i in range(nobs):
        p[i+1] = p[i] + shocks_p[i+1]
        t[i+1] = shocks_t[i+1]
        
    y = p + t
    return y, p, svols_p, svols_t


# + {"code_folding": [1]}
@njit
def ObjGen(model,
           paras,
           data_mom_dict,
           moment_choice,
           how = 'expectation',
           n_exp_paras = 0):
    if how =='expectation':
        model.exp_para = paras
    elif how=='process':
        model.process_para = paras
    elif how=='joint':
        model.exp_para = paras[0:n_exp_paras]
        model.process_para = paras[n_exp_paras:]
        
    # simulated moments 
    model_mom_dict = model.SMM()
    diff = np.array([model_mom_dict[mom] - data_mom_dict[mom] for mom in moment_choice]) 
    distance = np.linalg.norm(diff)
    
    return distance


# + {"code_folding": [1]}
## parameter estimation non-jitted because jit does not support scipy.optimize
def ParaEst(ObjSpec,
            para_guess,
            method = 'Nelder-Mead',
            bounds = None,
            options = {'disp': True}):
    
    parameter = minimize(ObjSpec,
                         x0 = para_guess,
                         method = method,
                         bounds = bounds,
                         options = options)['x']
    return parameter 


# -

# ### Rational Expectation (RE) + AR1

# + {"code_folding": [0]}
model_data = [
    ('exp_para', float64[:]),             # parameters for expectation formation, empty for re
    ('process_para', float64[:]),         # parameters for inflation process, 2 entries for AR1 
    ('horizon', int64),                   # forecast horizons 
    ('real_time',float64[:]),             # real time data on inflation 
    ('history',float64[:]),               # a longer history of inflation 
    ('realized',float64[:])               # realized inflation 
]


# + {"code_folding": [1, 2, 14, 18, 39]}
@jitclass(model_data)
class RationalExpectationAR:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
        
    def SimForecasts(self):
        ## parameters
        n = len(self.real_time)
        ρ,σ = self.process_para
        horizon = self.horizon

        ## information set 
        real_time = self.real_time
        
        ## forecast moments 
        Disg = np.zeros(n)
        nowcast = real_time
        forecast = ρ**horizon*nowcast
        Var = hstepvar(horizon,ρ,σ)* np.ones(n)
        FE = forecast - self.realized           ## forecast errors depend on realized shocks
        FEATV = np.zeros(n)
        forecast_moments = {"FE":FE,
                            "Disg":Disg,
                            "Var":Var}
        return forecast_moments
    
    def SMM(self):
        
        ρ,σ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = 0.0
        InfVar = σ**2/(1-ρ**2)
        InfATV = ρ**2*InfVar
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim}
        return SMMMoments


# + {"code_folding": []}
model_sv_data = [
                ('exp_para', float64[:]),             # parameters for expectation formation, empty for re
                ('process_para', float64[:]),         # parameters for inflation process, 1 entry for SV 
                ('horizon', int64),                   # forecast horizons 
                ('real_time',float64[:,:]),           # 2d array, 4 x T, real time permanent component of inf 
                ('history',float64[:,:]),             # 2d array, 4 x T+n_burn longer history of permanent component of inf
                ('realized',float64[:])               # realized inflation 
]


# -

# ### Rational Expectation (RE) + SV

# + {"code_folding": [2, 14, 49, 81]}
@jitclass(model_sv_data)
class RationalExpectationSV:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
        
    def SimForecasts(self):
        ## parameters
        n = len(self.real_time[0,:])
        γ = self.process_para
            
        ## inputs
        real_time = self.real_time
        horizon = self.horizon
        
        ## forecast moments 
        ## now the informationset needs to contain differnt components seperately.
        ## therefore, real_time fed in the begining is a tuple, not just current eta, but also current sigmas. 
        
        infoset = real_time 
        y_real_time = infoset[0,:]  ## permanent income componenet 
        nowcast = infoset[1,:]  ## permanent income componenet 
        forecast = nowcast
        σs_now = infoset[2:3,:]  ## volatility now 
        
        Var = np.zeros(n)
        for i in range(n):
            Var[i] = hstepvarSV(horizon,
                                σs_now = σs_now[:,i],
                                γ = γ[0]) ## γ[0] instead of γ is important make sure the input is a float
        FE = forecast - self.realized ## forecast errors depend on realized shocks 
        Disg = np.zeros(n)
        
        forecast_moments = {"FE":FE,
                            "Disg":Disg,
                            "Var":Var}
        return forecast_moments
        
    def SMM(self):
        
        γ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = np.nan
        InfVar = np.nan
        InfATV = np.nan
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim}
        return SMMMoments


# + {"code_folding": []}
### create a RESV instance 

p0_fake = 0
γ_fake = 0.1
σs_now_fake = [0.2,0.3]

ucsv_fake = SimUCSV(γ_fake,
                    nobs = 200,
                    p0 = p0_fake,
                    ) 

xx_real_time,xx_p_real_time,vol_p_real_time,vol_t_real_time = ucsv_fake  

xx_realized = xx_real_time[1:-1]

xx_real_time= np.array([xx_real_time,
                        xx_p_real_time,
                        vol_p_real_time,
                        vol_t_real_time]
                      )[:,0:-2]


## initialize 
resv = RationalExpectationSV(exp_para = np.array([]),
                             process_para = np.array([0.1]),
                             real_time = xx_real_time,
                             history = xx_real_time) ## history does not matter here, 

## get the realization 

resv.GetRealization(xx_realized)
# -

resv.SMM()

# #### Estimation inflation only 

# + {"code_folding": [0]}
## generate an instance of the model

ρ0,σ0 = 0.95,0.1

history0 = SimAR1(ρ0,
                  σ0,
                  200)
real_time0 = history0[11:-2]

realized0 = history0[12:-1]


## initialize an re instance 

rear0 = RationalExpectationAR(exp_para = np.array([]),
                              process_para = np.array([ρ0,σ0]),
                              real_time = real_time0,
                              history = history0,
                              horizon = 1)

rear0.GetRealization(realized0)


## fake data moments dictionary 

data_mom_dict_re = rear0.SMM()


## specific objective function for estimation 
moments0 = ['InfAV',
           'InfVar',
           'InfATV']


## specific objective function 

def Objrear(paras):
    scalor = ObjGen(rear0,
                    paras= paras,
                    data_mom_dict = data_mom_dict_re,
                    moment_choice = moments0,
                    how = 'process')
    return scalor


# + {"code_folding": [0]}
## invoke estimation 

ParaEst(Objrear,
        para_guess = (0.8,0.1)
       )


# -

# ### Sticky Expectation (SE) + AR1

# + {"code_folding": [0, 3]}
### Some jitted functions that are needed (https://github.com/numba/numba/issues/1269)

@njit
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result

@njit
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)

@njit
def np_std(array, axis):
    return np_apply_along_axis(np.std, axis, array)

@njit
def np_var(array, axis):
    return np_apply_along_axis(np.var, axis, array)

@njit
def np_max(array, axis):
    return np_apply_along_axis(np.max, axis, array)


@njit
def np_min(array, axis):
    return np_apply_along_axis(np.min, axis, array)


# + {"code_folding": [2, 14, 18, 81]}
@jitclass(model_data)
class StickyExpectationAR:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
        
    def SimForecasts(self,
                     n_sim = 500):        
        ## parameters and inputs 
        real_time = self.real_time
        history  = self.history
        
        n = len(real_time)
        ρ,σ = self.process_para
        lbd = self.exp_para
        horizon = self.horizon
        
        n_history = len(history) # of course equal to len(history)
        n_burn = len(history) - n
        
        ## simulation
        np.random.seed(12345)
        update_or_not_val = np.random.uniform(0,
                                              1,
                                              size = (n_sim,n_history))
        update_or_not_bool = update_or_not_val>=1-lbd
        update_or_not = update_or_not_bool.astype(np.int64)
        most_recent_when = np.empty((n_sim,n_history),dtype = np.int64)
        nowcasts_to_burn = np.empty((n_sim,n_history),dtype = np.float64)
        Vars_to_burn = np.empty((n_sim,n_history),dtype = np.float64)
        
        # look back for the most recent last update for each point of time  
        for i in range(n_sim):
            for j in range(n_history):
                most_recent = j 
                for x in range(j):
                    if update_or_not[i,j-x]==1 and most_recent<=x:
                        most_recent = most_recent
                    elif update_or_not[i,j-x]==1 and most_recent>x:
                        most_recent = x
                most_recent_when[i,j] = most_recent
                nowcasts_to_burn[i,j] = history[j - most_recent_when[i,j]]*ρ**most_recent_when[i,j]
                Vars_to_burn[i,j]= hstepvar((most_recent_when[i,j]+horizon),
                                            ρ,
                                            σ)
        
        ## burn initial forecasts since history is too short 
        nowcasts = nowcasts_to_burn[:,n_burn:] 
        forecasts = ρ**horizon*nowcasts
        Vars = Vars_to_burn[:,n_burn:]
        FEs = forecasts - self.realized
        
        ## compuate population moments
        forecasts_mean = np_mean(forecasts,axis = 0)
        forecasts_var = np_var(forecasts,axis = 0)
        FEs_mean = forecasts_mean - self.realized
            
        Vars_mean = np_mean(Vars,axis = 0) ## need to change 
        
        forecasts_vcv = np.cov(forecasts.T)
        forecasts_atv = np.array([forecasts_vcv[i+1,i] for i in range(n-1)])
        FEs_vcv = np.cov(FEs.T)
        FEs_atv = np.array([FEs_vcv[i+1,i] for i in range(n-1)]) ## this is no longer needed
        
        forecast_moments_sim = {"FE":FEs_mean,
                                "Disg":forecasts_var,
                                "Var":Vars_mean}
        return forecast_moments_sim
            
    def SMM(self):
        
        ρ,σ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = 0.0
        InfVar = σ**2/(1-ρ**2)
        InfATV = ρ**2*InfVar
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim}
        return SMMMoments


# -

# ### Sticky Expectation (SE) + SV

# + {"code_folding": [2, 14, 18, 88]}
@jitclass(model_sv_data)
class StickyExpectationSV:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
        
    def SimForecasts(self,
                     n_sim = 502):        
        ## inputs 
        real_time = self.real_time
        history  = self.history
        n = len(real_time[0,:])
        horizon = self.horizon
        n_history = len(history[0,:]) # of course equal to len(history)
        n_burn = n_history - n
        
        ## get the information set 
        infoset = history 
        y_now,p_now, sigmas_p_now, sigmas_t_now= infoset[0,:],infoset[1,:],infoset[2,:],infoset[3,:]
        sigmas_now = np.concatenate((sigmas_p_now,sigmas_t_now),axis=0).reshape((2,-1))
        
        ## parameters 
        γ = self.process_para
        lbd = self.exp_para
       
        ## simulation
        np.random.seed(12345)
        update_or_not_val = np.random.uniform(0,
                                              1,
                                              size = (n_sim,n_history))
        update_or_not_bool = update_or_not_val>=1-lbd
        update_or_not = update_or_not_bool.astype(np.int64)
        most_recent_when = np.empty((n_sim,n_history),dtype = np.int64)
        nowcasts_to_burn = np.empty((n_sim,n_history),dtype = np.float64)
        Vars_to_burn = np.empty((n_sim,n_history),dtype = np.float64)
        
        # look back for the most recent last update for each point of time  
        for i in range(n_sim):
            for j in range(n_history):
                most_recent = j 
                for x in range(j):
                    if update_or_not[i,j-x]==1 and most_recent<=x:
                        most_recent = most_recent
                    elif update_or_not[i,j-x]==1 and most_recent>x:
                        most_recent = x
                most_recent_when[i,j] = most_recent
                ###########################################################################
                nowcasts_to_burn[i,j] = p_now[j - most_recent_when[i,j]]
                Vars_to_burn[i,j]= hstepvarSV((most_recent_when[i,j]+horizon),
                                              sigmas_now[:,j-most_recent_when[i,j]],
                                              γ[0])
                ###############################################################
        
        ## burn initial forecasts since history is too short 
        nowcasts = nowcasts_to_burn[:,n_burn:] 
        forecasts = nowcasts
        Vars = Vars_to_burn[:,n_burn:]
        FEs = forecasts - self.realized
        
        ## compuate population moments
        forecasts_mean = np_mean(forecasts,axis = 0)
        forecasts_var = np_var(forecasts,axis = 0)
        FEs_mean = forecasts_mean - self.realized
            
        Vars_mean = np_mean(Vars,axis = 0) ## need to change 
        
        forecasts_vcv = np.cov(forecasts.T)
        forecasts_atv = np.array([forecasts_vcv[i+1,i] for i in range(n-1)])
        FEs_vcv = np.cov(FEs.T)
        FEs_atv = np.array([FEs_vcv[i+1,i] for i in range(n-1)]) ## this is no longer needed
        
        forecast_moments_sim = {"FE":FEs_mean,
                                "Disg":forecasts_var,
                                "Var":Vars_mean}
        return forecast_moments_sim
        
    def SMM(self):
        
        γ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = np.nan
        InfVar = np.nan
        InfATV = np.nan
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim}
        return SMMMoments


# + {"code_folding": [1]}
## intialize the ar instance 
sear0 = StickyExpectationAR(exp_para = np.array([0.2]),
                            process_para = np.array([ρ0,σ0]),
                            real_time = real_time0,
                            history = history0,
                            horizon = 1)

sear0.GetRealization(realized0)

# + {"code_folding": []}
## intialize the sv instnace 
sesv = StickyExpectationSV(exp_para = np.array([0.3]),
                           process_para = np.array([0.1]),
                           real_time = xx_real_time,
                           history = xx_real_time) ## history does not matter here, 

## get the realization 

sesv.GetRealization(xx_realized)
# -

# #### Estimating SE with RE 

# + {"code_folding": [0]}
## only expectation estimation 

moments0 = ['FE',
            'FEVar',
            'FEATV',
            'Disg']

def Objsear_re(paras):
    scalor = ObjGen(sear0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_re,
                    moment_choice = moments0,
                    how = 'expectation')
    return scalor


## invoke estimation 
ParaEst(Objsear_re,
        para_guess = (0.2),
        method='Nelder-Mead',
        bounds = ((0,1),)
       )
# -

# #### Estimating SE with SE 

# + {"code_folding": [0]}
## get a fake data moment dictionary under a different parameter 

sear1 = StickyExpectationAR(exp_para = np.array([0.4]),
                            process_para = np.array([ρ0,σ0]),
                            real_time = real_time0,
                            history = history0,
                            horizon = 1)
sear1.GetRealization(realized0)
data_mom_dict_se = sear1.SMM()

moments0 = ['FE',
            'FEVar',
            'FEATV',
            'Disg']

def Objsear_se(paras):
    scalor = ObjGen(sear0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_se,
                    moment_choice = moments0,
                    how = 'expectation')
    return scalor


## invoke estimation 
ParaEst(Objsear_se,
        para_guess = (0.2),
        method='Nelder-Mead',
        bounds = ((0,1),)
       )
# -

# #### Joint Estimation 

# + {"code_folding": [0]}
## for joint estimation 

moments1 = ['InfAV',
            'InfVar',
            'InfATV',
            'FE',
            'FEVar',
            'FEATV',
            'Disg']

def Objsear_joint(paras):
    scalor = ObjGen(sear0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_se,
                    moment_choice = moments1,
                    how ='joint',
                    n_exp_paras = 1)
    return scalor

## invoke estimation 
ParaEst(Objsear_joint,
        para_guess = np.array([0.2,0.8,0.2]),
        method='Nelder-Mead'
       )


# -

# ### Noisy Information (NI) + AR1
#

# + {"code_folding": [2, 14, 18, 115]}
@jitclass(model_data)
class NoisyInformationAR:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
        
    def SimForecasts(self,
                     n_sim = 500):
        ## inputs 
        
        real_time = self.real_time
        history = self.history
        realized = self.realized
        n = len(real_time)
        n_history = len(history)
        n_burn = len(history) - n
        
        ## parameters 
        ρ,σ = self.process_para
        sigma_pb,sigma_pr = self.exp_para

        #######################
        var_init = 5    ## some initial level of uncertainty, will be washed out after long simulation
        ##################
        sigma_v = np.array([[sigma_pb**2,0.0],[0.0,sigma_pr**2]]) ## variance matrix of signal noises 
        horizon = self.horizon      
        
        ## simulate signals 
        nb_s = 2                                    ## the number of signals 
        H = np.array([[1.0],[1.0]])                 ## an multiplicative matrix summing all signals
        
        # randomly simulated signals 
        np.random.seed(12434)
        signal_pb = self.history+sigma_pb*np.random.randn(n_history)   ## one series of public signals 
        signals_pb = signal_pb.repeat(n_sim).reshape((-1,n_sim)).T     ## shared by all agents
        np.random.seed(13435)
        signals_pr = self.history + sigma_pr*np.random.randn(n_sim*n_history).reshape((n_sim,n_history))
                                                                 ### private signals are agent-specific 
    
        ## prepare matricies 
        nowcasts_to_burn = np.zeros((n_sim,n_history))  ### nowcasts matrix of which the initial simulation is to be burned 
        nowcasts_to_burn[:,0] = history[0]
        nowvars_to_burn = np.zeros((n_sim,n_history))   ### nowcasts uncertainty matrix
        nowvars_to_burn[:,0] = var_init
        Vars_to_burn = np.zeros((n_sim,n_history))      ### forecasting uncertainty matrix 
        
        
        ## fill the matricies for individual moments        
        for i in range(n_sim):
            signals_this_i = np.concatenate((signals_pb[i,:],signals_pr[i,:]),axis=0).reshape((2,-1))
            ## the histories signals specific to i: the first row is public signals and the second is private signals 
            Pkalman = np.zeros((n_history,nb_s))
            ## Kalman gains of this agent for respective signals 
            Pkalman[0,:] = 0  ## some initial values 
            
            for t in range(n_history-1):
                step1_vars_to_burn = ρ**2*nowvars_to_burn[i,t] + σ**2
                ## priror uncertainty 
                
                inv = np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v) 
                ## the inverse of the noiseness matrix  
                
                inv_sc = np.dot(np.dot(H.T,inv),H)
                ## the total noisiness as a scalar 
                
                var_reduc = step1_vars_to_burn*inv_sc*step1_vars_to_burn
                ## reduction in uncertainty from the update
                
                nowvars_this_2d = np.array([[step1_vars_to_burn]]) - var_reduc
                ## update equation of nowcasting uncertainty 
                
                nowvars_to_burn[i,t+1] = nowvars_this_2d[0,0] 
                ## nowvars_this_2d is a 2-d matrix with only one entry. We take the element and set it to the matrix
                ### this is necessary for Numba typing 
                
                Pkalman[t+1,:] = step1_vars_to_burn*np.dot(H.T,np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v))
                ## update Kalman gains recursively using the signal extraction ratios 
                
                Pkalman_all = np.dot(Pkalman[t+1,:],H)[0] 
                ## the weight to the prior forecast 
    
                nowcasts_to_burn[i,t+1] = (1-Pkalman_all)*ρ*nowcasts_to_burn[i,t]+ np.dot(Pkalman[t+1,:],signals_this_i[:,t+1])
                ## kalman filtering updating for nowcasting: weighted average of prior and signals 
                
            for t in range(n_history):
                Vars_to_burn[i,t] = ρ**(2*horizon)*nowvars_to_burn[i,t] + hstepvar(horizon,ρ,σ)
                
        nowcasts = nowcasts_to_burn[:,n_burn:]
        forecasts = ρ**horizon*nowcasts 
        Vars = Vars_to_burn[:,n_burn:]
        
        ## compuate population moments
        forecasts_mean = np_mean(forecasts,axis=0)
        forecasts_var = np_var(forecasts,axis=0)
        FEs_mean = forecasts_mean - realized
            
        Vars_mean = np_mean(Vars,axis=0) ## need to change for time-variant volatility
        
        forecast_moments_sim = {"FE":FEs_mean,
                                "Disg":forecasts_var,
                                "Var":Vars_mean}
        return forecast_moments_sim
    
    def SMM(self):
        
        ρ,σ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = 0.0
        InfVar = σ**2/(1-ρ**2)
        InfATV = ρ**2*InfVar
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim}
        return SMMMoments


# -

# ### Noisy Information (NI) + SV
#
#

# + {"code_folding": [2, 63, 124]}
@jitclass(model_sv_data)
class NoisyInformationSV:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
              
    def SimForecasts(self,
                     n_sim = 500):
        ## inputs 
        real_time = self.real_time
        history  = self.history
        n = len(real_time[0,:])
        horizon = self.horizon
        n_history = len(history[0,:]) # of course equal to len(history)
        n_burn = n_history - n
        
        ## get the information set 
        infoset = history 
        y_now, p_now, sigmas_p_now, sigmas_t_now= infoset[0,:],infoset[1,:],infoset[2,:],infoset[3,:]
        sigmas_now = np.concatenate((sigmas_p_now,sigmas_t_now),axis=0).reshape((2,-1))
        
        
        ## process parameters
        γ = self.process_para
        ## exp parameters 
        sigma_pb,sigma_pr = self.exp_para
        var_init = 1
        
        ## other parameters 
        sigma_v = np.array([[sigma_pb**2,0.0],[0.0,sigma_pr**2]]) ## variance matrix of signal noises         
        ## simulate signals 
        nb_s = 2                                    ## the number of signals 
        H = np.array([[1.0],[1.0]])                 ## an multiplicative matrix summing all signals
        
        # randomly simulated signals 
        np.random.seed(12434)
        ##########################################################
        signal_pb = p_now+sigma_pb*np.random.randn(n_history)   ## one series of public signals 
        signals_pb = signal_pb.repeat(n_sim).reshape((-1,n_sim)).T     ## shared by all agents
        np.random.seed(13435)
        signals_pr = p_now + sigma_pr*np.random.randn(n_sim*n_history).reshape((n_sim,n_history))
        #####################################################################################

        ## prepare matricies 
        nowcasts_to_burn = np.zeros((n_sim,n_history))
        nowcasts_to_burn[:,0] = p_now[0]
        nowvars_to_burn = np.zeros((n_sim,n_history))
        nowvars_to_burn[:,0] = var_init
        Vars_to_burn = np.zeros((n_sim,n_history))
        
        ## fill the matricies for individual moments        
        for i in range(n_sim):
            signals_this_i = np.concatenate((signals_pb[i,:],signals_pr[i,:]),axis=0).reshape((2,-1))
            ## the histories signals specific to i: the first row is public signals and the second is private signals 
            Pkalman = np.zeros((n_history,nb_s))
            ## Kalman gains of this agent for respective signals 
            Pkalman[0,:] = 0  ## some initial values 
            
            for t in range(n_history-1):
                step1var = hstepvarSV(1,
                                      sigmas_now[:,t],
                                      γ[0])
                step1_vars_to_burn = nowvars_to_burn[i,t] + step1var
                ## priror uncertainty 
                
                inv = np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v) 
                ## the inverse of the noiseness matrix  
                
                inv_sc = np.dot(np.dot(H.T,inv),H)
                ## the total noisiness as a scalar 
                
                var_reduc = step1_vars_to_burn*inv_sc*step1_vars_to_burn
                ## reduction in uncertainty from the update
                
                nowvars_this_2d = np.array([[step1_vars_to_burn]]) - var_reduc
                ## update equation of nowcasting uncertainty 
                
                nowvars_to_burn[i,t+1] = nowvars_this_2d[0,0] 
                ## nowvars_this_2d is a 2-d matrix with only one entry. We take the element and set it to the matrix
                ### this is necessary for Numba typing 
                
                Pkalman[t+1,:] = step1_vars_to_burn*np.dot(H.T,np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v))
                ## update Kalman gains recursively using the signal extraction ratios 
                
                Pkalman_all = np.dot(Pkalman[t+1,:],H)[0] 
                ## the weight to the prior forecast 
    
                nowcasts_to_burn[i,t+1] = (1-Pkalman_all)*nowcasts_to_burn[i,t]+ np.dot(Pkalman[t+1,:],signals_this_i[:,t+1])
                ## kalman filtering updating for nowcasting: weighted average of prior and signals 
                
            for t in range(n_history):
                stephvar = hstepvarSV(horizon,
                                      sigmas_now[:,t],
                                      γ[0])
                Vars_to_burn[i,t] = nowvars_to_burn[i,t] + stephvar
                
        nowcasts = nowcasts_to_burn[:,n_burn:]
        forecasts = nowcasts 
        Vars = Vars_to_burn[:,n_burn:]

        
        ## compuate population moments
        forecasts_mean = np_mean(forecasts,axis=0)
        forecasts_var = np_var(forecasts,axis=0)
        FEs_mean = forecasts_mean - self.realized
        Vars_mean = np_mean(Vars,axis=0) ## need to change for time-variant volatility
        
        forecast_moments_sim = {"FE":FEs_mean,
                                "Disg":forecasts_var,
                                "Var":Vars_mean}
        return forecast_moments_sim
        
    def SMM(self):
        
        γ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = np.nan
        InfVar = np.nan
        InfATV = np.nan
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim}
        return SMMMoments

# + {"code_folding": [0, 1]}
## intialize the ar instance 
niar0 = NoisyInformationAR(exp_para = np.array([0.1,0.2]),
                            process_para = np.array([ρ0,σ0]),
                            real_time = real_time0,
                            history = history0,
                            horizon = 1)

niar0.GetRealization(realized0)

# + {"code_folding": []}
## initial a sv instance
nisv = NoisyInformationSV(exp_para = np.array([0.3,0.2]),
                           process_para = np.array([0.1]),
                           real_time = xx_real_time,
                           history = xx_real_time) ## history does not matter here, 

## get the realization 

nisv.GetRealization(xx_realized)
# -

# #### Estimating NI using RE data 

# + {"code_folding": [0, 8, 9, 17]}
moments0 = ['FE',
            'FEATV',
            'FEVar',
            'Disg',
            'DisgATV',
            'DisgVar',
            'Var']

def Objniar_re(paras):
    scalor = ObjGen(niar0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_re,
                    moment_choice = moments0,
                    how = 'expectation')
    return scalor

## invoke estimation 
ParaEst(Objniar_re,
        para_guess = np.array([0.2,0.1]),
        method='Nelder-Mead')
# -

# #### Estimate NI with NI

# + {"code_folding": [0, 2]}
## get a fake data moment dictionary under a different parameter 

niar1 = NoisyInformationAR(exp_para = np.array([0.3,0.1]),
                            process_para = np.array([ρ0,σ0]),
                            real_time = real_time0,
                            history = history0,
                            horizon = 1)

niar1.GetRealization(realized0)

data_mom_dict_ni = niar1.SMM()

moments0 = ['FE',
            'FEVar',
            'FEATV',
            'Disg',
            'DisgVar',
            'Var']

def Objniar_ni(paras):
    scalor = ObjGen(niar0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_ni,
                    moment_choice = moments0,
                    how = 'expectation')
    return scalor

## invoke estimation 

ParaEst(Objniar_ni,
        para_guess = np.array([0.2,0.3]),
        method='L-BFGS-B',
        bounds = ((0,None),(0,None),)
       )
# -

# #### Joint Estimation
#

# + {"code_folding": [0]}
## for joint estimation 

moments1 = ['InfAV',
            'InfVar',
            'InfATV',
            'FE',
            'FEVar',
            'FEATV',
            'Disg']

def Objniar_joint(paras):
    scalor = ObjGen(niar0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_ni,
                    moment_choice = moments1,
                    how ='joint',
                    n_exp_paras = 2)
    return scalor

## invoke estimation 
ParaEst(Objniar_joint,
        para_guess = np.array([0.2,0.3,0.8,0.2]),
        method='Nelder-Mead'
       )


# -

# ###  Diagnostic Expectation(DE) + AR1

# + {"code_folding": [2, 18, 71]}
@jitclass(model_data)
class DiagnosticExpectationAR:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
        
    def SimForecasts(self,
                     n_sim = 500):
        ## inputs 
        real_time = self.real_time
        history  = self.history
        realized = self.realized
        n = len(real_time)
        horizon = self.horizon 
        n_burn = len(history) - n
        n_history = len(history)  # of course equal to len(history)
        
        ## parameters 
        ρ,σ = self.process_para
        theta,theta_sigma = self.exp_para
        
    
        ## simulation
        np.random.seed(12345)
        thetas = theta_sigma*np.random.randn(n_sim) + theta  ## randomly drawn representativeness parameters
        
        nowcasts_to_burn = np.empty((n_sim,n_history))
        Vars_to_burn = np.empty((n_sim,n_history))
        nowcasts_to_burn[:,0] = history[0]
        Vars_to_burn[:,:] = hstepvar(horizon,ρ,σ)
        
        ## diagnostic and extrapolative expectations 
        for i in range(n_sim):
            this_theta = thetas[i]
            for j in range(n_history-1):
                nowcasts_to_burn[i,j+1] = history[j+1]+ this_theta*(history[j+1]-ρ*history[j])  # can be nowcasting[j-1] instead
        
        ## burn initial forecasts since history is too short 
        nowcasts = nowcasts_to_burn[:,n_burn:]
        forecasts = ρ**horizon*nowcasts
        Vars = Vars_to_burn[:,n_burn:]
        
        FEs = forecasts - realized
        
        ## compuate population moments
        forecasts_mean = np_mean(forecasts,axis = 0)
        forecasts_var = np_var(forecasts,axis = 0)
        FEs_mean = forecasts_mean - realized  
        Vars_mean = np_mean(Vars,axis = 0) ## need to change 
        
        #forecasts_vcv = np.cov(forecasts.T)
        #forecasts_atv = np.array([forecasts_vcv[i+1,i] for i in range(n-1)])
        #FEs_vcv = np.cov(FEs.T)
        
        forecast_moments_sim = {"FE":FEs_mean,
                                "Disg":forecasts_var,
                                "Var":Vars_mean}
        return forecast_moments_sim
    
    def SMM(self):
        
        ρ,σ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = 0.0
        InfVar = σ**2/(1-ρ**2)
        InfATV = ρ**2*InfVar
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim}
        return SMMMoments


# -

# ###  Diagnostic Expectation(DE) + SV

# + {"code_folding": [2, 18, 55, 84]}
@jitclass(model_sv_data)
class DiagnosticExpectationSV:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
        
    def SimForecasts(self,
                     n_sim = 500):
        ## inputs 
        real_time = self.real_time
        history  = self.history
        realized = self.realized
        n = len(real_time[0,:])
        horizon = self.horizon
        n_history = len(history[0,:]) # of course equal to len(history)
        n_burn = n_history - n
        
        
        ## get the information set 
        infoset = history 
        y_now, p_now, sigmas_p_now, sigmas_t_now= infoset[0,:],infoset[1,:],infoset[2,:],infoset[3,:]
        sigmas_now = np.concatenate((sigmas_p_now,sigmas_t_now),axis=0).reshape((2,-1))
        
        ## process parameters
        γ = self.process_para
        ## exp parameters 
        theta,theta_sigma= self.exp_para
        
    
        ## simulation of representativeness parameters 
        np.random.seed(12345)
        thetas = theta_sigma*np.random.randn(n_sim) + theta  ## randomly drawn representativeness parameters
        
        
        ## simulation of individual forecasts     
        nowcasts_to_burn = np.empty((n_sim,n_history))
        nowcasts_to_burn[:,0] = p_now[0]
        
        Vars_to_burn = np.empty((n_sim,n_history))
        Vars_to_burn[:,0] = hstepvarSV(horizon,
                                       sigmas_now[:,0],
                                       γ[0])
        
        for i in range(n_sim):
            this_theta = thetas[i]
            for j in range(n_history-1):
                ###########################################################################################
                nowcasts_to_burn[i,j+1] = y_now[j+1]+ this_theta*(y_now[j+1]- p_now[j])  # can be nowcasting[j-1] instead
                Var_now_re = hstepvarSV(horizon,
                                        sigmas_now[:,j+1],
                                        γ[0])
                Vars_to_burn[i,j+1] = Var_now_re + this_theta*(Var_now_re - Vars_to_burn[i,j])
                ######### this line of codes needs to be double checked!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                ##########################################################################################
                
        ## burn initial forecasts since history is too short 
        nowcasts = nowcasts_to_burn[:,n_burn:]
        forecasts = nowcasts
        Vars = Vars_to_burn[:,n_burn:]
        
        ## compuate population moments
        forecasts_mean = np_mean(forecasts,axis=0)
        forecasts_var = np_var(forecasts,axis=0)
        FEs_mean = forecasts_mean - realized
        Vars_mean = np_mean(Vars,axis = 0) ## need to change 
        
        forecast_moments_sim = {"FE":FEs_mean,
                                "Disg":forecasts_var,
                                "Var":Vars_mean}
        return forecast_moments_sim


    def SMM(self):
        
        γ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = np.nan
        InfVar = np.nan
        InfATV = np.nan
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim}
        return SMMMoments

# + {"code_folding": [0]}
## intialize the ar instance 
dear0 = DiagnosticExpectationAR(exp_para = np.array([0.5,0.2]),
                                process_para = np.array([ρ0,σ0]),
                                real_time = real_time0,
                                history = history0,
                                horizon = 1)

dear0.GetRealization(realized0)

# + {"code_folding": []}
## initial a sv instance
desv = DiagnosticExpectationSV(exp_para = np.array([0.3,0.2]),
                               process_para = np.array([0.1]),
                               real_time = xx_real_time,
                               history = xx_real_time) ## history does not matter here, 

## get the realization 

desv.GetRealization(xx_realized)
# -

# #### Estimating DE using RE data

# + {"code_folding": [0]}
## only expectation estimation 

moments0 = ['FE',
            'FEVar',
            'FEATV',
            'Disg',
            'DisgVar',
            'Var']

def Objdear_re(paras):
    scalor = ObjGen(dear0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_re,
                    moment_choice = moments0,
                    how = 'expectation')
    return scalor


## invoke estimation 
ParaEst(Objdear_re,
        para_guess = np.array([0.2,0.3]),
        method='Nelder-Mead')
# -

# #### Estimating DE using DE data 

# + {"code_folding": [0]}
## get a fake data moment dictionary under a different parameter 

dear1 = DiagnosticExpectationAR(exp_para = np.array([0.3,0.1]),
                                process_para = np.array([ρ0,σ0]),
                                real_time = real_time0,
                                history = history0,
                                horizon = 1)

dear1.GetRealization(realized0)

data_mom_dict_de = dear1.SMM()


## only expectation estimation 

moments0 = ['FE',
            'FEVar',
            'FEATV',
            'Disg',
            'DisgVar',
            'Var']

def Objdear_de(paras):
    scalor = ObjGen(dear0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_de,
                    moment_choice = moments0,
                    how = 'expectation')
    return scalor


## invoke estimation 
ParaEst(Objdear_de,
        para_guess = np.array([0.2,0.3]),
        method='Nelder-Mead')
# -

# #### Joint Estimation 

# + {"code_folding": [0]}
## for joint estimation 

moments1 = ['InfAV',
            'InfVar',
            'InfATV',
            'FE',
            'FEVar',
            'FEATV',
            'Disg']

def Objdear_joint(paras):
    scalor = ObjGen(dear0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_de,
                    moment_choice = moments1,
                    how ='joint',
                    n_exp_paras = 2)
    return scalor

## invoke estimation 
ParaEst(Objdear_joint,
        para_guess = np.array([0.2,0.3,0.8,0.2]),
        method='Nelder-Mead')


# -

# ###  Sticky Expectation and Noisy Information Hybrid(SENI) + AR1

# + {"code_folding": [2, 14, 18, 150]}
@jitclass(model_data)
class SENIHybridAR:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
        
    def SimForecasts(self,
                     n_sim = 500):
        ## inputs 
        real_time = self.real_time
        history  = self.history
        realized = self.realized
        n = len(real_time)
        horizon = self.horizon 
        n_history =len(history)
        n_burn = len(history) - n
        horizon = self.horizon      

        ## parameters 
        ρ,σ = self.process_para
        lbd, sigma_pb,sigma_pr = self.exp_para
                
        #### The first NI part of the model 
        var_init = 5    ## some initial level of uncertainty, will be washed out after long simulation
        sigma_v = np.array([[sigma_pb**2,0.0],[0.0,sigma_pr**2]]) ## variance matrix of signal noises 
        
        ## simulate signals 
        nb_s = 2                                    ## the number of signals 
        H = np.array([[1.0],[1.0]])                 ## an multiplicative matrix summing all signals
        
        # randomly simulated signals 
        np.random.seed(12434)
        signal_pb = self.history+sigma_pb*np.random.randn(n_history)   ## one series of public signals 
        signals_pb = signal_pb.repeat(n_sim).reshape((-1,n_sim)).T     ## shared by all agents
        np.random.seed(13435)
        signals_pr = self.history + sigma_pr*np.random.randn(n_sim*n_history).reshape((n_sim,n_history))
                                                                 ### private signals are agent-specific 
            
        ## SE part of the model, which governs if updating for each agent at each point of time 
        ## simulation of updating profile
        ## simulation
        np.random.seed(12345)
        update_or_not_val = np.random.uniform(0,
                                              1,
                                              size = (n_sim,n_history))
        update_or_not_bool = update_or_not_val>=1-lbd
        update_or_not = update_or_not_bool.astype(np.int64)
        most_recent_when = np.empty((n_sim,n_history),dtype = np.int64)    
        ########################################################################
        nowsignals_pb_to_burn = np.empty((n_sim,n_history),dtype = np.float64)
        nowsignals_pr_to_burn = np.empty((n_sim,n_history),dtype=np.float64)
        ######################################################################
    
        # look back for the most recent last update for each point of time  
        for i in range(n_sim):
            for j in range(n_history):
                most_recent = j 
                for x in range(j):
                    if update_or_not[i,j-x]==1 and most_recent<=x:
                        most_recent = most_recent
                    elif update_or_not[i,j-x]==1 and most_recent>x:
                        most_recent = x
                most_recent_when[i,j] = most_recent
                ################################################################################
                nowsignals_pr_to_burn[i,j] = signal_pb[j - most_recent_when[i,j]]
                nowsignals_pr_to_burn[i,j] = signals_pr[i,j - most_recent_when[i,j]]
                ## both above are the matrices of signals available to each agent depending on if updating
                #####################################################################################
                
        
        ## The second NI part of the model 
        ## Once sticky signals are prepared, agents filter as NI
        
        ## prepare matricies 
        nowcasts_to_burn = np.zeros((n_sim,n_history))  ### nowcasts matrix of which the initial simulation is to be burned 
        nowcasts_to_burn[:,0] = history[0]
        nowvars_to_burn = np.zeros((n_sim,n_history))   ### nowcasts uncertainty matrix
        nowvars_to_burn[:,0] = var_init
        Vars_to_burn = np.zeros((n_sim,n_history))      ### forecasting uncertainty matrix 
        
        ## fill the matricies for individual moments  
        for i in range(n_sim):
            signals_this_i = np.concatenate((nowsignals_pb_to_burn[i,:],nowsignals_pr_to_burn[i,:]),axis=0).reshape((2,-1))
            ## the histories signals specific to i: the first row is public signals and the second is private signals 
            Pkalman = np.zeros((n_history,nb_s))
            ## Kalman gains of this agent for respective signals 
            Pkalman[0,:] = 0  ## some initial values 
            
            for t in range(n_history-1):
                step1_vars_to_burn = ρ**2*nowvars_to_burn[i,t] + σ**2
                ## priror uncertainty 
                
                inv = np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v) 
                ## the inverse of the noiseness matrix  
                
                inv_sc = np.dot(np.dot(H.T,inv),H)
                ## the total noisiness as a scalar 
                
                var_reduc = step1_vars_to_burn*inv_sc*step1_vars_to_burn
                ## reduction in uncertainty from the update
                
                nowvars_this_2d = np.array([[step1_vars_to_burn]]) - var_reduc
                ## update equation of nowcasting uncertainty 
                
                nowvars_to_burn[i,t+1] = nowvars_this_2d[0,0] 
                ## nowvars_this_2d is a 2-d matrix with only one entry. We take the element and set it to the matrix
                ### this is necessary for Numba typing 
                
                Pkalman[t+1,:] = step1_vars_to_burn*np.dot(H.T,np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v))
                ## update Kalman gains recursively using the signal extraction ratios 
                
                Pkalman_all = np.dot(Pkalman[t+1,:],H)[0] 
                ## the weight to the prior forecast 
                nowcasts_to_burn[i,t+1] = (1-Pkalman_all)*ρ*nowcasts_to_burn[i,t]+ np.dot(Pkalman[t+1,:],signals_this_i[:,t+1])
                ## kalman filtering updating for nowcasting: weighted average of prior and signals 
                
            for t in range(n_history):
                Vars_to_burn[i,t] = ρ**(2*horizon)*nowvars_to_burn[i,t] + hstepvar(horizon,ρ,σ)
        
        
        ## burn initial histories  
        nowcasts = nowcasts_to_burn[:,n_burn:]
        forecasts = ρ**horizon*nowcasts 
        Vars = Vars_to_burn[:,n_burn:]
        
        ## compuate population moments
        forecasts_mean = np_mean(forecasts,axis=0)
        forecasts_var = np_var(forecasts,axis=0)
        
        FEs_mean = forecasts_mean - realized
            
        Vars_mean = np_mean(Vars,axis=0) ## need to change for time-variant volatility
        
        forecast_moments_sim = {"FE":FEs_mean,
                                "Disg":forecasts_var,
                                "Var":Vars_mean}
        return forecast_moments_sim
        
    def SMM(self):
        
        ρ,σ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = 0.0
        InfVar = σ**2/(1-ρ**2)
        InfATV = ρ**2*InfVar
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim}
        return SMMMoments


# -

# ###  Sticky Expectation and Noisy Information Hybrid(SENI) + SV
#
#

# + {"code_folding": [2, 14, 73, 101, 163]}
#@jitclass(model_sv_data)
class SENIHybridSV:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
        
    def SimForecasts(self,
                     n_sim = 500):
        ## inputs 
        real_time = self.real_time
        history  = self.history
        realized = self.realized
        n = len(real_time[0,:])
        horizon = self.horizon
        n_history = len(history[0,:]) # of course equal to len(history)
        n_burn = n_history - n
        
        ## get the information set 
        infoset = history 
        y_now, p_now, sigmas_p_now, sigmas_t_now= infoset[0,:],infoset[1,:],infoset[2,:],infoset[3,:]
        sigmas_now = np.concatenate((sigmas_p_now,sigmas_t_now),axis=0).reshape((2,-1))
        
        
        ## process parameters
        γ = self.process_para
        ## exp parameters 
        lbd,sigma_pb,sigma_pr = self.exp_para
        var_init = 1
        
        ## other parameters 
        sigma_v = np.array([[sigma_pb**2,0.0],[0.0,sigma_pr**2]]) ## variance matrix of signal noises         
        ## simulate signals 
        nb_s = 2                                    ## the number of signals 
        H = np.array([[1.0],[1.0]])                 ## an multiplicative matrix summing all signals
        
        
        ## The first NI part of the model 
        # randomly simulated signals 
        np.random.seed(12434)
        ##########################################################
        signal_pb = p_now+sigma_pb*np.random.randn(n_history)   ## one series of public signals 
        signals_pb = signal_pb.repeat(n_sim).reshape((-1,n_sim)).T     ## shared by all agents
        np.random.seed(13435)
        signals_pr = p_now + sigma_pr*np.random.randn(n_sim*n_history).reshape((n_sim,n_history))
        ##################################################################################### 
            
        ## SE part of the model, which governs if updating for each agent at each point of time 
        ## simulation of updating profile
        ## simulation
        np.random.seed(12345)
        update_or_not_val = np.random.uniform(0,
                                              1,
                                              size = (n_sim,n_history))
        update_or_not_bool = update_or_not_val>=1-lbd
        update_or_not = update_or_not_bool.astype(np.int64)
        most_recent_when = np.empty((n_sim,n_history),dtype = np.int64)    
        ########################################################################
        nowsignals_pb_to_burn = np.empty((n_sim,n_history),dtype = np.float64)
        nowsignals_pr_to_burn = np.empty((n_sim,n_history),dtype=np.float64)
        ######################################################################
    
        # look back for the most recent last update for each point of time  
        for i in range(n_sim):
            for j in range(n_history):
                most_recent = j 
                for x in range(j):
                    if update_or_not[i,j-x]==1 and most_recent<=x:
                        most_recent = most_recent
                    elif update_or_not[i,j-x]==1 and most_recent>x:
                        most_recent = x
                most_recent_when[i,j] = most_recent
                ################################################################################
                nowsignals_pr_to_burn[i,j] = signal_pb[j - most_recent_when[i,j]]
                nowsignals_pr_to_burn[i,j] = signals_pr[i,j - most_recent_when[i,j]]
                ## both above are the matrices of signals available to each agent depending on if updating
                #####################################################################################
                
        
        ## The second NI part of the model 
        ## Once sticky signals are prepared, agents filter as NI
        
        ## prepare matricies 
        nowcasts_to_burn = np.zeros((n_sim,n_history))  ### nowcasts matrix of which the initial simulation is to be burned 
        nowcasts_to_burn[:,0] = p_now[0]
        nowvars_to_burn = np.zeros((n_sim,n_history))   ### nowcasts uncertainty matrix
        nowvars_to_burn[:,0] = var_init
        Vars_to_burn = np.zeros((n_sim,n_history))      ### forecasting uncertainty matrix 
        
        
        ## fill the matricies for individual moments        
        for i in range(n_sim):
            signals_this_i = np.concatenate((nowsignals_pr_to_burn[i,:],nowsignals_pr_to_burn[i,:]),axis=0).reshape((2,-1))
            ## the histories signals specific to i: the first row is public signals and the second is private signals 
            Pkalman = np.zeros((n_history,nb_s))
            ## Kalman gains of this agent for respective signals 
            Pkalman[0,:] = 0  ## some initial values 
            
            for t in range(n_history-1):
                step1var = hstepvarSV(1,
                                      sigmas_now[:,t],
                                      γ[0])
                step1_vars_to_burn = nowvars_to_burn[i,t] + step1var
                ## priror uncertainty 
                
                inv = np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v) 
                ## the inverse of the noiseness matrix  
                
                inv_sc = np.dot(np.dot(H.T,inv),H)
                ## the total noisiness as a scalar 
                
                var_reduc = step1_vars_to_burn*inv_sc*step1_vars_to_burn
                ## reduction in uncertainty from the update
                
                nowvars_this_2d = np.array([[step1_vars_to_burn]]) - var_reduc
                ## update equation of nowcasting uncertainty 
                
                nowvars_to_burn[i,t+1] = nowvars_this_2d[0,0] 
                ## nowvars_this_2d is a 2-d matrix with only one entry. We take the element and set it to the matrix
                ### this is necessary for Numba typing 
                
                Pkalman[t+1,:] = step1_vars_to_burn*np.dot(H.T,np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v))
                ## update Kalman gains recursively using the signal extraction ratios 
                
                Pkalman_all = np.dot(Pkalman[t+1,:],H)[0] 
                ## the weight to the prior forecast 
    
                nowcasts_to_burn[i,t+1] = (1-Pkalman_all)*nowcasts_to_burn[i,t]+ np.dot(Pkalman[t+1,:],signals_this_i[:,t+1])
                ## kalman filtering updating for nowcasting: weighted average of prior and signals 
                
            for t in range(n_history):
                stephvar = hstepvarSV(horizon,
                                      sigmas_now[:,t],
                                      γ[0])
                Vars_to_burn[i,t] = nowvars_to_burn[i,t] + stephvar
                
        nowcasts = nowcasts_to_burn[:,n_burn:]
        forecasts = nowcasts 
        Vars = Vars_to_burn[:,n_burn:]

        
        ## compuate population moments
        forecasts_mean = np_mean(forecasts,axis=0)
        forecasts_var = np_var(forecasts,axis=0)
        
        FEs_mean = forecasts_mean - realized            
        Vars_mean = np_mean(Vars,axis=0) ## need to change for time-variant volatility
        
        forecast_moments_sim = {"FE":FEs_mean,
                                "Disg":forecasts_var,
                                "Var":Vars_mean}
        return forecast_moments_sim
              
    def SMM(self):
        
        γ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = np.nan
        InfVar = np.nan
        InfATV = np.nan
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim}
        return SMMMoments

# + {"code_folding": [0]}
## intialize the ar instance 
seniar0 = SENIHybridAR(exp_para = np.array([0.3,0.3,0.2]),
                       process_para = np.array([ρ0,σ0]),
                       real_time = real_time0,
                       history = history0,
                       horizon = 1)

seniar0.GetRealization(realized0)

# +
## initial a sv instance
senisv = SENIHybridSV(exp_para = np.array([0.5,0.23,0.32]),
                               process_para = np.array([0.1]),
                               real_time = xx_real_time,
                               history = xx_real_time) ## history does not matter here, 


senisv.GetRealization(xx_realized)
# -

# #### Estimate Hybrid using RE data 

# + {"code_folding": [0]}
## only expectation estimation 

moments0 = ['FE',
            'FEVar',
            'FEATV',
            'Disg',
            'DisgVar',
            'DisgVar',
            'Var']

def Objseniar_re(paras):
    scalor = ObjGen(seniar0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_re,
                    moment_choice = moments0,
                    how = 'expectation')
    return scalor

## invoke estimation 
ParaEst(Objseniar_re,
        para_guess = np.array([0.5,0.5,0.5]),
        method='Nelder-Mead')
# -

# #### Estimate Hybrid using Hybrid 
#
#

# + {"code_folding": [0, 2]}
## get a fake data moment dictionary under a different parameter 

seniar1 = SENIHybridAR(exp_para = np.array([0.2,0.4,0.5]),
                     process_para = np.array([ρ0,σ0]),
                     real_time = real_time0,
                     history = history0,
                     horizon = 1)

seniar1.GetRealization(realized0)

data_mom_dict_seni= seniar1.SMM()

def Objseniar_seni(paras):
    scalor = ObjGen(seniar0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_seni,
                    moment_choice = moments0,
                    how = 'expectation')
    return scalor

## invoke estimation 
ParaEst(Objseniar_seni,
        para_guess = np.array([0.5,0.5,0.5]),
        method='Nelder-Mead')
# -

# #### Joint Estimation 

# + {"code_folding": [0]}
## for joint estimation 

moments1 = ['InfAV',
            'InfVar',
            'InfATV',
            'FE',
            'FEVar',
            'FEATV',
            'Disg',
           'DisgVar',
           'DisgATV',
           'Var']

def Objseniar_joint(paras):
    scalor = ObjGen(seniar0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_seni,
                    moment_choice = moments1,
                    how ='joint',
                    n_exp_paras = 3)
    return scalor

## invoek estimation 
ParaEst(Objseniar_joint,
        para_guess = np.array([0.4,0.2,0.3,0.8,0.2]),
        method='Nelder-Mead')
