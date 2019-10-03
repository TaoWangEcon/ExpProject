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

# ## Estimation of Model Parameters with SCE and SPF data
#

# ### 1. Estimation algorithms 

from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import arma_generate_sample
import pandas as pd


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


# + {"code_folding": []}
## auxiliary functions 
def hstepvar(h,sigma,rho):
    return sum([ rho**(2*i)*sigma**2 for i in range(h)] )

np.random.seed(12345)
def hstepfe(h,sigma,rho):
    return sum([rho**i*(np.random.randn(1)*sigma)*np.random.randn(h)[i] for i in range(h)])
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
        
        
def ForecastPlotDiag(test,data):
    plt.figure(figsize=([3,13]))
    for i,val in enumerate(test):
        plt.subplot(4,1,i+1)
        plt.plot(test[val],label='model:'+ val)
        plt.plot(np.array(data[val]),label='data:'+ val)
        plt.legend(loc=1)


# + {"code_folding": [0]}
## AR1 series for testing 
nobs = 100
rho = process_para['rho']
sigma = process_para['sigma']
xxx = AR1_simulator(rho,sigma,nobs)


# + {"code_folding": [1]}
# a function that generates population moments according to FIRE 
def FIREForecaster(real_time,horizon =10,process_para = process_para):
    n = len(real_time)
    rho = process_para['rho']
    sigma = process_para['sigma']
    Disg =np.zeros(n)
    FE = np.random.randn(n)*sigma  ## forecast errors depend on realized shocks 
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
# plot different moments for FIRE
ForecastPlot(FIREtest)

# + {"code_folding": []}
## SE parameters

SE_FE_para = {'lambda':1}
SE_para = {'lambda':0.75}
SE_para2 = {'lambda':0.25}


# + {"code_folding": [0]}
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


# + {"code_folding": []}
## check if SE collapses to FIRE when lambda=1

FE_from_SE = SEForecaster(xxx,horizon=1,exp_para = SE_FE_para)
ForecastPlot(FE_from_SE)

# + {"code_folding": []}
## test 

SEtest = SEForecaster(xxx,horizon=1)
ForecastPlot(SEtest)

# + {"code_folding": []}
## prepare inputs for the estimation

horizon = 1
real_time = xxx 
process_para = process_para

## some fake data using a different lambda
SEtest2 = SEForecaster(xxx,horizon=1,exp_para = SE_para2)
data_moms_dct = SEtest2


# + {"code_folding": [0]}
## a function estimating SE model parameter only 

def SE_EstObjfunc(lbd,moments = ['Forecast','Disg','Var']):
    """
    input
    -----
    lbd: the parameter of SE model to be estimated
    
    output
    -----
    the objective function to minmize
    """

    SE_para = {"lambda":lbd}
    SE_moms_dct = SEForecaster(real_time,horizon=1,process_para = process_para,exp_para = SE_para)
    SE_moms = np.array([SE_moms_dct[key] for key in moments] )
    data_moms = np.array([data_moms_dct[key] for key in moments] )
    obj_func = PrepMom(SE_moms,data_moms)
    return obj_func 


# + {"code_folding": [0]}
## invoke the estimation of SE 

lbd_est = Estimator(SE_EstObjfunc,para_guess =0.5,method='CG')
lbd_est
# -

SE_para_default = SE_para


# + {"code_folding": [0, 10]}
## SE class 

class StickyExpectation:
    def __init__(self,real_time,horizon=1,process_para = process_para,exp_para = SE_para_default,max_back =10):
        self.real_time = real_time
        self.horizon = horizon
        self.process_para = process_para
        self.exp_para = exp_para
        self.max_back = max_back
        
    def SEForecaster(self):
        ## parameters
        n = len(self.real_time)
        rho = self.process_para['rho']
        sigma =self.process_para['sigma']
        lbd = self.exp_para['lambda']
        max_back = self.max_back
        
        ## forecast moments 
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
    
    ## a function estimating SE model parameter only 
    def SE_EstObjfunc(self,lbd,data_moms_dct = data_moms_dct, moments = ['Forecast','Disg','Var']):
        """
        input
        -----
        lbd: the parameter of SE model to be estimated
        
        output
        -----
        the objective function to minmize
        """

        SE_para = {"lambda":lbd}
        self.exp_para = SE_para  # give the new lambda
        SE_moms_dct = self.SEForecaster()
        SE_moms = np.array([SE_moms_dct[key] for key in moments] )
        data_moms = np.array([data_moms_dct[key] for key in moments] )
        obj_func = PrepMom(SE_moms,data_moms)
        return obj_func 
    


# -

SE_test = StickyExpectation(xxx)

SE_test.SE_EstObjfunc(lbd =0.2)

# + {"code_folding": [0]}
## NI parameters

NI_para_default = {'sigma_pub':0.1,
          'sigma_prv':0.1}


# + {"code_folding": [2]}
## class of Noisy information 

def NoisyInformation(self,real_time,horizon=1,process_para = process_para, exp_para = NI_para_default):
    def __init__(self):
        self.real_time = real_time
        self.horizon = horizon
        self.process_para = process_para
        self.exp_para = exp_para
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


# -

# ### 2. Preparing real-time data 

# + {"code_folding": [0]}
## CPI Core
InfCPICMRT=pd.read_stata('../OtherData/InfCPICMRealTime.dta')  
InfCPICMRT = InfCPICMRT[-InfCPICMRT.date.isnull()]

## CPI 
InfCPIMRT=pd.read_stata('../OtherData/InfCPIMRealTime.dta')  
InfCPIMRT = InfCPIMRT[-InfCPIMRT.date.isnull()]

# + {"code_folding": [0]}
## dealing with dates 
dateM_cpic = pd.to_datetime(InfCPICMRT['date'],format='%Y%m%d')
dateM_cpi = pd.to_datetime(InfCPIMRT['date'],format='%Y%m%d')

InfCPICMRT.index = pd.DatetimeIndex(dateM_cpic,freq='infer')
InfCPIMRT.index = pd.DatetimeIndex(dateM_cpi,freq='infer')


# + {"code_folding": [0]}
## a function that turns vintage matrix to a one-dimension vector of real time data
def GetRealTimeData(matrix):
    periods = len(matrix)
    real_time = np.zeros(periods)
    for i in range(periods):
        real_time[i] = matrix.iloc[i,i+1]
    return real_time


# + {"code_folding": [0]}
## generate real-time series 
matrix_cpic = InfCPICMRT.copy().drop(columns=['date','year','month'])
matrix_cpi = InfCPIMRT.copy().drop(columns=['date','year','month'])

real_time_cpic = pd.Series(GetRealTimeData(matrix_cpic) )
real_time_cpi =  pd.Series(GetRealTimeData(matrix_cpi) ) 
real_time_cpic.index =  InfCPICMRT.index #+ pd.DateOffset(months=1) 
real_time_cpi.index = InfCPIMRT.index #+ pd.DateOffset(months=1)

# + {"code_folding": [0]}
## turn index into yearly inflation
real_time_index =pd.concat([real_time_cpic,real_time_cpi], join='inner', axis=1)
real_time_index.columns=['RTCPI','RTCPICore']
real_time_inf = real_time_index.pct_change(periods=12)*100
# -

real_time_inf.tail()

# ### 3. Estimating using real-time inflation and expectation data
#

# + {"code_folding": [0, 6]}
## exapectation data from SPF 
PopQ=pd.read_stata('../SurveyData/InfExpQ.dta')  
PopQ = PopQ[-PopQ.date.isnull()]

dateQ = pd.to_datetime(PopQ['date'],format='%Y%m%d')

dateQ_str = dateQ.dt.year.astype(int).astype(str) + \
             "Q" + dateQ.dt.quarter.astype(int).astype(str)
PopQ.index = pd.DatetimeIndex(dateQ_str)

SPFCPI = PopQ[['SPFCPI_Mean','SPFCPI_FE','SPFCPI_Disg','SPFCPI_Var']].dropna(how='any')

## Inflation data 
InfQ = pd.read_stata('../OtherData/InfShocksQClean.dta')
InfQ = InfQ[-InfQ.date.isnull()]
dateQ2 = pd.to_datetime(InfQ['date'],format='%Y%m%d')
dateQ_str2 = dateQ2 .dt.year.astype(int).astype(str) + \
             "Q" + dateQ2 .dt.quarter.astype(int).astype(str)
InfQ.index = pd.DatetimeIndex(dateQ_str2)


# + {"code_folding": [0]}
## expectation data from SCE

PopM = pd.read_stata('../SurveyData/InfExpM.dta')

PopM = PopM[-PopM.date.isnull()]

dateM = pd.to_datetime(PopM['date'],format='%Y%m%d')

dateM_str = dateM.dt.year.astype(int).astype(str) + \
             "M" + dateM.dt.month.astype(int).astype(str)
PopM.index = pd.DatetimeIndex(dateM)

SCECPI = PopM[['SCE_Mean','SCE_FE','SCE_Disg','SCE_Var']].dropna(how='any')


# + {"code_folding": [0]}
## Combine expectation data and real-time data 

SPF_est = pd.concat([SPFCPI,real_time_inf,InfQ['Inf1y_CPICore']], join='inner', axis=1)
SCE_est = pd.concat([SCECPI,real_time_inf], join='inner', axis=1)

# + {"code_folding": [0]}
# How large is the difference between current vintage and real-time data
rev = SPF_est['Inf1y_CPICore'] - SPF_est['RTCPI']
hist_rv = plt.hist(rev,bins=20,color='orange')

# + {"code_folding": [0]}
# real time inflation 
real_time = np.array(SPF_est['RTCPI'])
xx = plt.figure()
SPF_est[['RTCPI','Inf1y_CPICore']].plot()
plt.title('Current vintage and real-time Core CPI Inflation')

# + {"code_folding": [0]}
## preparing for estimation 

exp_data_SPF = SPF_est[['SPFCPI_Mean','SPFCPI_FE','SPFCPI_Disg','SPFCPI_Var']]
exp_data_SPF.columns = ['Forecast','FE','Disg','Var']
data_moms_dct_SPF = dict(exp_data_SPF)

exp_data_SCE = SCE_est[['SCE_Mean','SCE_FE','SCE_Disg','SCE_Var']]
exp_data_SCE.columns = ['Forecast','FE','Disg','Var']
data_moms_dct_SCE = dict(exp_data_SCE)

# + {"code_folding": []}
## estimation for SPF
real_time = np.array(SPF_est['RTCPI'])
data_moms_dct = data_moms_dct_SPF
lbd_est_SPF = Estimator(SE_EstObjfunc,para_guess =0.2,method='CG')

## estimation for SCE
real_time = np.array(SCE_est['RTCPI'])
data_moms_dct = data_moms_dct_SCE
lbd_est_SCE = Estimator(SE_EstObjfunc,para_guess =0.2,method='CG')

# + {"code_folding": []}
## what is the estimated lambda?
print("SPF: "+str(lbd_est_SPF))
print("SCE: "+str(lbd_est_SCE))

## rough estimation that did not take care of following issues
## quarterly survey of 1-year-ahead forecast
## real-time data is yearly 

# + {"code_folding": []}
## compare the data with estimation

SE_para_est = {"lambda":lbd_est_SCE}
SE_est = SEForecaster(real_time,horizon=1,exp_para = SE_para_est)
ForecastPlotDiag(SE_est,data_moms_dct)
