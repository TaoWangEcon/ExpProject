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

# ## Do the Estimation with SCE and SPF data
#

# ### 1. Importing estimation modules

# +
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.optimize import minimize

import statsmodels.api as sm
from statsmodels.tsa.api import AutoReg as AR
#from UCSVEst import UCSVEst as ucsv
# -

from GMMEst_Dev import RationalExpectation as re
from GMMEst import StickyExpectation as seb
from GMMEst import NoisyInformation as nib
from GMMEst import DiagnosticExpectation as de
from GMMEst import StickyNoisyHybrid as senib
from GMMEst import ParameterLearning as pl
from GMMEst_Dev import AR1_simulator, ForecastPlotDiag, ForecastPlot

# + {"code_folding": []}
## some parameters 
rho = 0.95
sigma = 0.1
process_para = {'rho':rho,
                'sigma':sigma}
# -

# ### 2. Preparing real-time data 

# + {"code_folding": []}
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

# + {"code_folding": []}
## exapectation data from SPF 
PopQ=pd.read_stata('../SurveyData/InfExpQ.dta')  
PopQ = PopQ[-PopQ.date.isnull()]

dateQ = pd.to_datetime(PopQ['date'],format='%Y%m%d')

dateQ_str = dateQ.dt.year.astype(int).astype(str) + \
             "Q" + dateQ.dt.quarter.astype(int).astype(str)
PopQ.index = pd.DatetimeIndex(dateQ_str,freq='infer')

####################################################
#### specific to including more moments ###########
####################################################
SPFCPI = PopQ[['SPFCPI_Mean','SPFCPI_FE','SPFCPI_Disg','SPFCPI_Var',
               'SPFCPI_ATV','SPFCPI_FEATV']].dropna(how='any')


# + {"code_folding": [0]}
## Inflation data quarterly 
InfQ = pd.read_stata('../OtherData/InfShocksQClean.dta')
InfQ = InfQ[-InfQ.date.isnull()]
dateQ2 = pd.to_datetime(InfQ['date'],format='%Y%m%d')
dateQ_str2 = dateQ2.dt.year.astype(int).astype(str) + \
             "Q" + dateQ2.dt.quarter.astype(int).astype(str)
InfQ.index = pd.DatetimeIndex(dateQ_str2,freq='infer')

# + {"code_folding": []}
## process parameters estimation 
# period filter 
start_t='1995-01-01'
end_t = '2019-03-30'   

### quarterly data 
CPICQ = InfQ['Inf1y_CPICore'].copy().loc[start_t:end_t]
Y = np.array(CPICQ[1:])
X = np.array(CPICQ[:-1])

ARmodel = AR(CPICQ,
             lags = 1,
             trend = 'n')
ar_rs = ARmodel.fit()
rhoQ_est = ar_rs.params[0]
sigmaQ_est = np.sqrt(sum(ar_rs.resid**2)/(len(CPICQ)-1))
print('Quarterly CPI core inflation AR1 model estimates are '+ str([rhoQ_est,sigmaQ_est]))


# + {"code_folding": [0]}
## Inflation data monthly
InfM = pd.read_stata('../OtherData/InfShocksMClean.dta')
InfM = InfM[-InfM.date.isnull()]
dateM = pd.to_datetime(InfM['date'],format='%Y%m%d')
#dateM_str = dateM .dt.year.astype(int).astype(str) + \
#             "M" + dateM .dt.month.astype(int).astype(str)
InfM.index = pd.DatetimeIndex(dateM,freq='infer')

# + {"code_folding": [0]}
### monthly data 
CPIM = InfM['Inf1y_CPIAU'].copy().loc[start_t:end_t]
Y = np.array(CPIM[1:])
X = np.array(CPIM[:-1])

ARmodel2 = AR(CPIM,
             lags = 1, 
             trend = 'n')
ar_rs2 = ARmodel2.fit()
rhoM_est = ar_rs2.params[0]
sigmaM_est = np.sqrt(sum(ar_rs2.resid**2)/(len(CPIM)-1))
print('Monthly CPI AR1 model estimates are '+ str([rhoM_est,sigmaM_est]))

# + {"code_folding": [0]}
## expectation data from SCE

PopM = pd.read_stata('../SurveyData/InfExpM.dta')

PopM = PopM[-PopM.date.isnull()]

dateM = pd.to_datetime(PopM['date'],format='%Y%m%d')

dateM_str = dateM.dt.year.astype(int).astype(str) + \
             "M" + dateM.dt.month.astype(int).astype(str)
PopM.index = pd.DatetimeIndex(dateM,freq='infer')

####################################################
#### specific to including more moments ###########
####################################################

SCECPI = PopM[['SCE_Mean','SCE_FE','SCE_Disg','SCE_Var',
              'SCE_ATV','SCE_FEATV']].dropna(how='any')


# + {"code_folding": [0]}
## Combine expectation data and real-time data 

SPF_est = pd.concat([SPFCPI,real_time_inf,InfQ['Inf1y_CPICore'],InfQ['Inf1yf_CPICore']], join='inner', axis=1)
SCE_est = pd.concat([SCECPI,real_time_inf,InfM['Inf1yf_CPIAU']], join='inner', axis=1)

# + {"code_folding": []}
## hisotries data, the series ends at the same dates with real-time data but startes earlier 

st_t_history = '2000-01-01'
ed_t_SPF = SPF_est.index[-1].strftime('%Y%m%d')
ed_t_SCE = SCE_est.index[-1].strftime('%Y-%m-%d')

historyQ = real_time_inf.copy().loc[st_t_history:ed_t_SPF]
historyM = real_time_inf.loc[st_t_history:ed_t_SCE]
# -

historyQ.index

# + {"code_folding": [0]}
# How large is the difference between current vintage and real-time data
rev = SPF_est['Inf1y_CPICore'] - SPF_est['RTCPICore']

plt.style.use('ggplot')
plt.figure(figsize=([6,4]))
hist_rv = plt.hist(rev,
                   bins=30,
                   density = True)
plt.title('Core CPI Revisions between Real-time \n \
    data and the Current Vintage')
plt.ylabel('Density(0-1)',size=12)
plt.xlabel('Current Vintage - Real-Time',size=12)
plt.savefig('figures/hist_rev_realtime.png')

# + {"code_folding": [0]}
# real time inflation 
real_time = np.array(SPF_est['RTCPI'])

ax = SPF_est[['RTCPI','Inf1y_CPICore']].plot(style=['s-','o-'],
                                             figsize=([6,4]))
#plt.style.use('ggplot')
ax.set_title('Real-time and Current-vintage Core CPI ')
ax.set_xlabel('Date')
ax.set_ylabel('Core CPI (%)')
ax.legend(['Real-time', 'Current-vintage'])
plt.savefig('figures/ts_rev_realtime.png')

# + {"code_folding": [0]}
## realized 1-year-ahead inflation
realized_CPIC = np.array(SPF_est['Inf1yf_CPICore'])
realized_CPI = np.array(SCE_est['Inf1yf_CPIAU'])
#SPF_est['Inf1yf_CPICore'].plot()
#plt.title('Realized 1-year-ahead Core CPI Inflation')

# + {"code_folding": [0]}
## preparing data moments

####################################################
#### specific to including more moments ###########
####################################################

exp_data_SPF = SPF_est[['SPFCPI_Mean','SPFCPI_FE','SPFCPI_Disg','SPFCPI_Var',
                        'SPFCPI_ATV','SPFCPI_FEATV']]
exp_data_SPF.columns = ['Forecast','FE','Disg','Var',
                        'ATV','FEATV']
data_moms_dct_SPF = dict(exp_data_SPF)

exp_data_SCE = SCE_est[['SCE_Mean','SCE_FE','SCE_Disg','SCE_Var',
                       'SCE_ATV','SCE_FEATV']]
exp_data_SCE.columns = ['Forecast','FE','Disg','Var',
                        'ATV','FEATV']
data_moms_dct_SCE = dict(exp_data_SCE)

# + {"code_folding": [0]}
## estimating parameters 

################
## quarterly ###
#################

real_time_Q = np.array(SPF_est['RTCPI'])
history_Q = historyQ['RTCPICore']

data_moms_dctQ = data_moms_dct_SPF


process_paraQ_est = {'rho':rhoQ_est,
                    'sigma':sigmaQ_est}
##############
## monthly ###
#############

real_time_M = np.array(SCE_est['RTCPI'])
history_M = historyM['RTCPI']
data_moms_dctM = data_moms_dct_SCE


process_paraM_est = {'rho':rhoM_est,
                    'sigma':sigmaM_est}


# + {"code_folding": [0]}
## function of moving average 
def mvavg(array,
          window = 3):
    array_av = np.array([ (array[i]+array[i+1]+array[i+2]/3) for i in range(len(array)-3)])
    return array_av

## a tweek using 3-month average 
data_moms_dct_SPF_mv = {}
data_moms_dct_SCE_mv = {}

for key in data_moms_dct_SPF.keys():
    data_moms_dct_SPF_mv[key] = mvavg(data_moms_dct_SPF[key],window = 2)
    
for key in data_moms_dct_SCE.keys():
    data_moms_dct_SCE_mv[key] = mvavg(data_moms_dct_SCE[key],window = 3)

## plot

spf_data_mv = ForecastPlot(data_moms_dct_SPF_mv)
sce_data_mv = ForecastPlot(data_moms_dct_SCE_mv)
# -

# ### Rational Expectation 

# + {"code_folding": [0]}
#################
## RE for SPF #
###############

#RE_model = re(real_time = real_time_Q,
#              history = history_Q,
#              process_para = process_paraQ_est)
#RE_model.moments = ['Forecast','FE','Disg','Var','ATV','FEATV']
#RE_model.GetRealization(realized_CPIC)
#RE_model.GetDataMoments(data_moms_dct_SPF)
#re_dict = RE_model.Forecaster()

## plot RE forecast moments and data moments 

#re_data_plot = ForecastPlotDiag(re_dict,
#                                data_moms_dctQ)
#plt.savefig('figures/spf_re_est_diag.png')
# -

# ### SE Estimation

# + {"code_folding": [0]}
## SE loop SMM estimation over different choieces of moments for SPF

moments_choices_short = [ ['FE','FEVar','FEATV','DisgVar','DisgATV']]
moments_choices = [['FE','FEVar','FEATV'],
                   ['Disg','DisgVar','DisgATV'],
                   ['Disg','FEVar','FEATV','DisgVar','DisgATV'],
                   ['FE','FEVar','FEATV','DisgVar','DisgATV']
                  ]

para_est_SPF_holder = []
para_est_SPF_joint_holder = []


## Initiate an instance
real_time = np.array(SPF_est['RTCPI'])
history_Q = historyQ['RTCPICore']
data_moms_dct = data_moms_dct_SPF
process_paraQ_est = {'rho':rhoQ_est,
                     'sigma':sigmaQ_est}
SE_model = seb(real_time = real_time,
              history = history_Q,
              process_para = process_paraQ_est)
    
SE_model.GetRealization(realized_CPIC)
SE_model.GetDataMoments(data_moms_dct)

# loop starts from here for SPF 
for i,moments_to_use in enumerate(moments_choices):
    print("moments used include "+ str(moments_to_use))
    SE_model.moments = moments_to_use
    
    # only expectation
    SE_model.ParaEstimateSMM(method='Nelder-Mead',
                              para_guess = (0.5),
                              options={'disp':True})
    para_est_SPF_holder.append(SE_model.para_estSMM)
    SE_model.all_moments = ['FE','Disg','Var']
    SE_model.ForecastPlotDiag(how = 'SMM',
                              all_moms = True,
                             diff_scale = True)
    plt.savefig('figures/spf_se_est_diag'+str(i)+'.png')
    
    # joint estimate
    SE_model.ParaEstimateSMMJoint(method='Nelder-Mead',
                                   para_guess =(0.5,0.6,0.1),
                                   options={'disp':True})
    para_est_SPF_joint_holder.append(SE_model.para_est_SMM_joint)
    SE_model.all_moments = ['FE','Disg','Var']
    SE_model.ForecastPlotDiag(how = 'SMMjoint',
                              all_moms = True,
                              diff_scale = True)
    plt.savefig('figures/spf_se_est_joint_diag'+str(i)+'.png')
    
print(para_est_SPF_holder)
print(para_est_SPF_joint_holder)

# + {"code_folding": []}
## tabulate the estimates 
spf_se_est_para = pd.DataFrame(para_est_SPF_holder,columns=[r'SE: $\hat\lambda_{SPF}$(Q)'])
spf_se_joint_est_para = pd.DataFrame(para_est_SPF_joint_holder,
                                     columns = [r'SE: $\hat\lambda_{SPF}$(Q)',
                                                r'SE: $\rho$',
                                                r'SE: $\sigma$'])
# -

spf_se_est_para

spf_se_joint_est_para

# + {"code_folding": [0]}
## SE loop estimation over different choieces of moments for SCE

moments_choices_short = [['FEVar','DisgATV','DisgVar']]
moments_choices = [['FEVar','FEATV'],
                   ['FEVar','DisgATV','DisgVar'],
                   ['FEVar','FEATV','DisgVar','DisgATV']
                  ]

para_est_SCE_holder = []
para_est_SCE_joint_holder =[]


## Initiate an instance
real_time = np.array(SCE_est['RTCPI'])
history_M = historyM['RTCPI']
data_moms_dct = data_moms_dct_SCE_mv
process_paraM_est = {'rho':rhoM_est,
                     'sigma':sigmaM_est}
SE_model2 = seb(real_time = realized_CPI,
                history = history_M,
                process_para = process_paraM_est)

SE_model2.GetRealization(realized_CPI)
SE_model2.GetDataMoments(data_moms_dct)
    
for i,moments_to_use in enumerate(moments_choices):
    print("moments used include "+ str(moments_to_use))
    SE_model2.moments = moments_to_use
    
    ## only expectation
    SE_model2.ParaEstimateSMM(method='Nelder-Mead',
                               para_guess =(0.2),
                               options={'disp':True})
    para_est_SCE_holder.append(SE_model2.para_estSMM)
    SE_model2.all_moments = ['FE','Disg','Var']
    SE_model2.ForecastPlotDiag(how = 'SMM',
                               all_moms = True,
                               diff_scale = True)
    plt.savefig('figures/sce_se_est_diag'+str(i)+'.png')
    
    ## joint estimation
    
    SE_model2.ParaEstimateSMMJoint(method='Nelder-Mead',
                                    para_guess =(0.5,0.8,0.1),
                                    options={'disp':True})
    para_est_SCE_joint_holder.append(SE_model2.para_est_SMM_joint)
    SE_model2.all_moments = ['FE','Disg','Var']
    SE_model2.ForecastPlotDiag(how = 'SMMjoint',
                               all_moms = True,
                               diff_scale = True)
    plt.savefig('figures/sce_se_est_joint_diag'+str(i)+'.png')

print(para_est_SCE_holder)
print(para_est_SCE_joint_holder)

# + {"code_folding": [0]}
sce_se_est_para = pd.DataFrame(para_est_SCE_holder,
                               columns=[r'SE: $\hat\lambda_{SCE}$(M)'])
sce_se_joint_est_para = pd.DataFrame(para_est_SCE_joint_holder,
                                     columns = [r'SE: $\hat\lambda_{SCE}$(M)',
                                                r'SE: $\rho$',
                                                r'SE: $\sigma$'])


# + {"code_folding": []}
est_moms = pd.DataFrame(moments_choices)

## combining SCE and SPF 
se_est_df = pd.concat([est_moms,
                       #spf_se_est_para,
                       #spf_se_joint_est_para,
                       sce_se_est_para,
                       sce_se_joint_est_para],
                      join = 'inner', axis=1)
# -

sce_se_est_para

sce_se_joint_est_para

se_est_df

# + {"code_folding": []}
se_est_df.to_excel('tables/SE_Est.xlsx',
                   float_format='%.2f',
                   index=False)
# -

# ### NI Estimation 

# + {"code_folding": [0, 33, 49]}
## NI loop estimation over different choices of moments for SPF


moments_choices_short = [['FEVar','FEATV','DisgVar','DisgATV','Var']]
moments_choices = [['FEVar','FEATV'],
                   ['FEVar','FEATV','DisgVar','DisgATV'],
                   ['FEVar','FEATV','DisgVar','DisgATV','Var']
                  ]

para_est_SPF_holder = []
para_est_SPF_joint_holder = []


## initiate an instance 

real_time = np.array(SPF_est['RTCPI'])
history_Q = historyQ['RTCPICore']

data_moms_dct = data_moms_dct_SPF_mv


process_paraQ_est = {'rho':rhoQ_est,
                     'sigma':sigmaQ_est}
NI_model = nib(real_time = real_time,
              history = history_Q,
              process_para = process_paraQ_est)
#NI_model.SimulateSignals()
NI_model.GetRealization(realized_CPIC)
NI_model.GetDataMoments(data_moms_dct)
    
    
### loop starts from here for SPF 

for i,moments_to_use in enumerate(moments_choices):
    print("moments used include "+ str(moments_to_use))
    NI_model.moments = moments_to_use
   
    # only expectation
    NI_model.ParaEstimateSMM(method='Nelder-Mead',
                             para_guess =(0.5,0.5),
                             options={'disp':True})
    para_est_SPF_holder.append(NI_model.para_estSMM)
    NI_model.all_moments = ['FE','Disg','Var']
    NI_model.ForecastPlotDiag(how = 'SMM',
                              all_moms = True,
                              diff_scale = True)
    plt.savefig('figures/spf_ni_est_diag'+str(i)+'.png')
    
    # joint estimate
    NI_model.ParaEstimateSMMJoint(method='Nelder-Mead',
                                  para_guess =(0.5,0.5,0.8,0.1),
                                  options={'disp':True})
    para_est_SPF_joint_holder.append(NI_model.para_est_SMM_joint)
    NI_model.all_moments = ['FE','Disg','Var']
    NI_model.ForecastPlotDiag(how = 'SMMjoint',
                              all_moms = True,
                              diff_scale = True)
    plt.savefig('figures/spf_ni_est_joint_diag'+str(i)+'.png')
    
print(para_est_SPF_holder)
print(para_est_SPF_joint_holder)
# -

para_est_SPF_holder

para_est_SPF_joint_holder

# + {"code_folding": []}
## tabulate the estimates 
spf_ni_est_para = pd.DataFrame(para_est_SPF_holder,
                               columns=[r'NI: $\hat\sigma_{pb,SPF}$',
                                        r'$\hat\sigma_{pr,SPF}$'])
spf_ni_joint_est_para = pd.DataFrame(para_est_SPF_joint_holder,
                                     columns=[r'NI: $\hat\sigma_{pb,SPF}$',
                                              r'$\hat\sigma_{pr,SPF}$',
                                              r'NI: $\rho$',
                                              r'NI: $\sigma$'])
# -

spf_ni_est_para

spf_ni_joint_est_para

# +
spf_ni_est_para.to_excel('tables/NI_Est_spf.xlsx',
                   float_format='%.2f',
                   index = False)

spf_ni_joint_est_para.to_excel('tables/NI_Est_spf_joint.xlsx',
                   float_format='%.3f',
                   index = False)

# + {"code_folding": [0, 31]}
## NI loop estimation over different choices of moments for SCE

moments_choices_short = [['FEVar','FEATV','DisgVar','DisgATV']]
moments_choices = [['FEVar','FEATV'],
                    ['FEVar','FEATV','DisgVar','DisgATV'],
                   ['FEVar','FEATV','DisgVar','DisgATV','Var']
                  ]

para_est_SCE_holder = []
para_est_SCE_joint_holder = []


## initiate an instance 

real_time = np.array(SCE_est['RTCPI'])
history_M = historyM['RTCPI']
##########################################################################################
data_moms_dct = data_moms_dct_SCE_mv
########################################################################################

process_paraM_est = {'rho':rhoM_est,
                     'sigma':sigmaM_est}
NI_model2 = nib(real_time = real_time,
               history = history_M,
               process_para = process_paraM_est)
#NI_model2.SimulateSignals()
NI_model2.GetRealization(realized_CPI)
NI_model2.GetDataMoments(data_moms_dct)
    
### loop starts from here for SPF 

for i,moments_to_use in enumerate(moments_choices):
    print("moments used include "+ str(moments_to_use))
    NI_model2.moments = moments_to_use
    
    # only expectation
    NI_model2.ParaEstimateSMM(method='Nelder-Mead',
                              para_guess =(0.5,0.5),
                              options={'disp':True})
    para_est_SCE_holder.append(NI_model2.para_estSMM)
    NI_model2.all_moments = ['FE','Disg','Var']
    NI_model2.ForecastPlotDiag(how ='SMM',
                               all_moms = True,
                               diff_scale = True)
    plt.savefig('figures/sce_ni_est_diag'+str(i)+'.png')
    
    # joint estimate
    NI_model2.ParaEstimateSMMJoint(method='Nelder-Mead',
                                   para_guess =(0.5,0.5,0.8,0.1),
                                   options={'disp':True})
    para_est_SCE_joint_holder.append(NI_model2.para_est_SMM_joint)
    NI_model2.all_moments = ['FE','Disg','Var']
    NI_model2.ForecastPlotDiag(how = 'SMMjoint',
                               all_moms = True,
                               diff_scale = True)
    plt.savefig('figures/sce_ni_est_joint_diag'+str(i)+'.png')
    
#print(para_est_SPF_holder)
#print(para_est_SPF_joint_holder)
# -

para_est_SCE_holder

para_est_SCE_joint_holder

# + {"code_folding": []}
## tabulate the estimates 
sce_ni_est_para = pd.DataFrame(para_est_SCE_holder,
                               columns=[r'NI: $\hat\sigma_{pb,SCE}$',
                                        r'$\hat\sigma_{pr,SCE}$'])
sce_ni_joint_est_para = pd.DataFrame(para_est_SCE_joint_holder,
                                     columns=[r'NI: $\hat\sigma_{pb,SCE}$',
                                              r'$\hat\sigma_{pr,SCE}$',
                                              r'NI: $\rho$',
                                              r'NI: $\sigma$'])
# -


sce_ni_est_para

sce_ni_est_para.to_excel('tables/NI_Est_sce.xlsx',
                   float_format='%.2f',
                   index = False)

sce_ni_joint_est_para

sce_ni_joint_est_para.to_excel('tables/NI_Est_sce_joint.xlsx',
                   float_format='%.3f',
                   index = False)

# + {"code_folding": []}
est_moms = pd.DataFrame(moments_choices)

## combining SCE and SPF 
ni_est_df = pd.concat([est_moms,
                       spf_ni_est_para,
                       spf_ni_joint_est_para,
                       sce_ni_est_para,
                       sce_ni_joint_est_para],
                      join='inner', axis=1)

ni_est_df.to_excel('tables/NI_Est.xlsx',
                   float_format='%.2f',
                   index = False)
# -

# ### DE Estimation 
#
#

# + {"code_folding": [0, 30]}
## DE loop estimation over different choices of moments for SPF


moments_choices_short = [['FEVar','FEATV','DisgVar','DisgATV','Var']]
moments_choices = [['FE','FEVar','FEATV'],
                   ['FE','FEVar','FEATV','Disg','DisgVar','DisgATV'],
                   ['FE','FEVar','FEATV','Disg','DisgVar','DisgATV','Var']
                  ]

para_est_SPF_holder = []
para_est_SPF_joint_holder = []


## initiate an instance 

real_time = np.array(SPF_est['RTCPI'])
history_Q = historyQ['RTCPICore']
data_moms_dct = data_moms_dct_SPF
process_paraQ_est = {'rho':rhoQ_est,
                     'sigma':sigmaQ_est}
DE_model = de(real_time = real_time,
               history = history_Q,
               process_para = process_paraQ_est)
#NI_model.SimulateSignals()
DE_model.GetRealization(realized_CPIC)
DE_model.GetDataMoments(data_moms_dct)
    
    
### loop starts from here for SPF 

for i,moments_to_use in enumerate(moments_choices):
    print("moments used include "+ str(moments_to_use))
    DE_model.moments = moments_to_use
   
    # only expectation
    DE_model.ParaEstimateSMM(method='Nelder-Mead',
                             para_guess =(0.5,0.1),
                             options={'disp':True})
    para_est_SPF_holder.append(DE_model.para_estSMM)
    DE_model.all_moments = ['FE','Disg','Var']
    DE_model.ForecastPlotDiag(how = 'SMM',
                              all_moms = True,
                              diff_scale = True)
    plt.savefig('figures/spf_de_est_diag'+str(i)+'.png')
    
    # joint estimate
    DE_model.ParaEstimateSMMJoint(method='Nelder-Mead',
                                  para_guess =(0.5,0.5,0.8,0.1),
                                  options={'disp':True})
    para_est_SPF_joint_holder.append(DE_model.para_est_SMM_joint)
    DE_model.all_moments = ['FE','Disg','Var']
    DE_model.ForecastPlotDiag(how = 'SMMjoint',
                              all_moms = True,
                              diff_scale = True)
    plt.savefig('figures/spf_de_est_joint_diag'+str(i)+'.png')
    
print(para_est_SPF_holder)
print(para_est_SPF_joint_holder)
# -

para_est_SPF_holder

para_est_SPF_joint_holder

# + {"code_folding": []}
## tabulate the estimates 
spf_de_est_para = pd.DataFrame(para_est_SPF_holder,
                               columns=[r'DE: $\theta_{SPF}$',
                                        r'DE: $\sigma_{\theta,SPF}$'])
spf_de_joint_est_para = pd.DataFrame(para_est_SPF_joint_holder,
                                     columns=[r'DE: $\theta_{SPF}$',
                                              r'DE: $\sigma_{\theta,SPF}$',
                                              r'DE: $\rho$',
                                              r'DE: $\sigma$'])
# -

spf_de_est_para.to_excel('tables/DE_Est_spf.xlsx',
                          float_format='%.2f',
                          index = False)
spf_de_joint_est_para.to_excel('tables/DE_Est_joint_spf.xlsx',
                          float_format='%.2f',
                          index = False)

# + {"code_folding": [0]}
## DE loop estimation over different choices of moments for SCE

moments_choices_short = [['FEVar','FEATV','DisgVar','DisgATV','Var']]
moments_choices = [['FE','FEVar','FEATV'],
                   ['FE','FEVar','FEATV','Disg','DisgVar','DisgATV'],
                   ['FE','FEVar','FEATV','DisgVar','DisgATV','Var']
                  ]

para_est_SCE_holder = []
para_est_SCE_joint_holder =[]


## Initiate an instance
real_time = np.array(SCE_est['RTCPI'])
history_M = historyM['RTCPI']
data_moms_dct = data_moms_dct_SCE
process_paraM_est = {'rho':rhoM_est,
                     'sigma':sigmaM_est}
DE_model2 = de(real_time = realized_CPI,
                history = history_M,
                process_para = process_paraM_est)

DE_model2.GetRealization(realized_CPI)
DE_model2.GetDataMoments(data_moms_dct)
    
for i,moments_to_use in enumerate(moments_choices):
    print("moments used include "+ str(moments_to_use))
    DE_model2.moments = moments_to_use
    
    ## only expectation
    DE_model2.ParaEstimateSMM(method='Nelder-Mead',
                               para_guess =(0.2,0.2),
                               options={'disp':True})
    para_est_SCE_holder.append(DE_model2.para_estSMM)
    DE_model2.all_moments = ['FE','Disg','Var']
    DE_model2.ForecastPlotDiag(how = 'SMM',
                               all_moms = True,
                               diff_scale = True)
    plt.savefig('figures/sce_de_est_diag'+str(i)+'.png')
    
    ## joint estimation
    
    DE_model2.ParaEstimateSMMJoint(method='Nelder-Mead',
                                    para_guess =(0.5,0.8,0.9,0.1),
                                    options={'disp':True})
    para_est_SCE_joint_holder.append(DE_model2.para_est_SMM_joint)
    DE_model2.all_moments = ['FE','Disg','Var']
    DE_model2.ForecastPlotDiag(how = 'SMMjoint',
                               all_moms = True,
                               diff_scale = True)
    plt.savefig('figures/sce_de_est_joint_diag'+str(i)+'.png')

print(para_est_SCE_holder)
print(para_est_SCE_joint_holder)

# + {"code_folding": [0, 3]}
sce_de_est_para = pd.DataFrame(para_est_SCE_holder,
                               columns=[r'DE: $\theta_{SCE}$',
                                        r'DE: $\sigma_{\theta,SCE}$'])
sce_de_joint_est_para = pd.DataFrame(para_est_SCE_joint_holder,
                                     columns=[r'DE: $\theta_{SCE}$',
                                              r'DE: $\sigma_{\theta,SCE}$',
                                              r'DE: $\rho$',
                                              r'DE: $\sigma$'])
# -

## tabulate the estimates 
sce_de_est_para = pd.DataFrame(para_est_SCE_holder,
                               columns=[r'DE: $\theta_{SCE}$',
                                        r'DE: $\sigma_{\theta,SCE}$'])
sce_de_joint_est_para = pd.DataFrame(para_est_SCE_joint_holder,
                                     columns=[r'DE: $\theta_{SCE}$',
                                              r'DE: $\sigma_{\theta_SCE}$',
                                              r'DE: $\rho$',
                                              r'DE: $\sigma$'])

sce_de_est_para.to_excel('tables/DE_Est_sce.xlsx',
                          float_format='%.2f',
                          index = False)
sce_de_joint_est_para.to_excel('tables/DE_Est_joint_sce.xlsx',
                          float_format='%.2f',
                          index = False)

# +
est_moms = pd.DataFrame(moments_choices)

## combining SCE and SPF 
de_est_df = pd.concat([est_moms,
                       spf_de_est_para,
                       spf_de_joint_est_para,
                       sce_de_est_para,
                       sce_de_joint_est_para],
                      join='inner', axis=1)

de_est_df.to_excel('tables/DE_Est.xlsx',
                   float_format='%.2f',
                   index = False)
# -

# ### SENI Estimation 
#
#

# + {"code_folding": [0, 33]}
## SENI loop estimation over different choices of moments for SPF


moments_choices_short = [['FEVar','FEATV','DisgVar','DisgATV','Var']]
moments_choices = [['FE','FEVar','FEATV'],
                   ['FE','FEVar','FEATV','Disg','DisgVar','DisgATV'],
                   ['FE','FEVar','FEATV','Disg','DisgVar','DisgATV','Var']
                  ]

para_est_SPF_holder = []
para_est_SPF_joint_holder = []


## initiate an instance 

real_time = np.array(SPF_est['RTCPI'])
history_Q = historyQ['RTCPICore']
#######################################################################
data_moms_dct = data_moms_dct_SPF
########################################################################

process_paraQ_est = {'rho':rhoQ_est,
                     'sigma':sigmaQ_est}
SENI_model = senib(real_time = real_time,
               history = history_Q,
               process_para = process_paraQ_est)
#NI_model.SimulateSignals()
SENI_model.GetRealization(realized_CPIC)
SENI_model.GetDataMoments(data_moms_dct)
    
    
### loop starts from here for SPF 

for i,moments_to_use in enumerate(moments_choices):
    print("moments used include "+ str(moments_to_use))
    SENI_model.moments = moments_to_use
   
    # only expectation
    try:
        SENI_model.ParaEstimateSMM(method='Nelder-Mead',
                                   para_guess =(0.5,0.1,0.1),
                                   options={'disp':True})
        para_est_SPF_holder.append(SENI_model.para_estSMM)
        SENI_model.all_moments = ['FE','Disg','Var']
        SENI_model.ForecastPlotDiag(how = 'SMM',
                                    all_moms = True,
                                    diff_scale = True)
        plt.savefig('figures/spf_seni_est_diag'+str(i)+'.png')
    except:
        pass
    # joint estimate
    try:
        SENI_model.ParaEstimateSMMJoint(method='Nelder-Mead',
                                      para_guess =(0.5,0.1,0.1,0.8,0.1),
                                      options={'disp':True})
        para_est_SPF_joint_holder.append(SENI_model.para_est_SMM_joint)
        SENI_model.all_moments = ['FE','Disg','Var']
        SENI_model.ForecastPlotDiag(how = 'SMMjoint',
                                    all_moms = True,
                                    diff_scale = True)
        plt.savefig('figures/spf_seni_est_joint_diag'+str(i)+'.png')
    except:
        pass
    
print(para_est_SPF_holder)
print(para_est_SPF_joint_holder)

# + {"code_folding": [0]}
## tabulate the estimates 
spf_seni_est_para = pd.DataFrame(para_est_SPF_holder,
                                 columns=[r'SENI: $\lambda$',
                                          r'SENI: $\sigma_{pb,SPF}$',
                                          r'SENI: $\sigma_{pr,SPF}$'])
spf_seni_joint_est_para = pd.DataFrame(para_est_SPF_joint_holder,
                                     columns=[r'SENI: $\lambda$',
                                              r'SENI: $\sigma_{pb,SPF}$',
                                              r'SENI: $\sigma_{pr,SPF}$',
                                              r'DE: $\rho$',
                                              r'DE: $\sigma$'])
# -

spf_seni_est_para

# + {"code_folding": [0, 29]}
## SENI loop estimation over different choices of moments for SCE

moments_choices_short = [['FEVar','FEATV','DisgVar','DisgATV','Var']]
moments_choices = [['FE','FEVar','FEATV'],
                   ['FE','FEVar','FEATV','Disg','DisgVar','DisgATV'],
                   ['FE','FEVar','FEATV','Disg','DisgVar','DisgATV','Var']
                  ]

para_est_SCE_holder = []
para_est_SCE_joint_holder =[]


## Initiate an instance
real_time = np.array(SCE_est['RTCPI'])
history_M = historyM['RTCPI']

########################################################################
data_moms_dct = data_moms_dct_SCE_mv
#######################################################################

process_paraM_est = {'rho':rhoM_est,
                     'sigma':sigmaM_est}
SENI_model2 = senib(real_time = realized_CPI,
                  history = history_M,
                  process_para = process_paraM_est)

SENI_model2.GetRealization(realized_CPI)
SENI_model2.GetDataMoments(data_moms_dct)
    
for i,moments_to_use in enumerate(moments_choices):
    print("moments used include "+ str(moments_to_use))
    SENI_model2.moments = moments_to_use
    
    ## only expectation
    SENI_model2.ParaEstimateSMM(method='Nelder-Mead',
                               para_guess =(0.2,0.2,0.2),
                               options={'disp':True})
    para_est_SCE_holder.append(SENI_model2.para_estSMM)
    SENI_model2.all_moments = ['FE','Disg','Var']
    SENI_model2.ForecastPlotDiag(how = 'SMM',
                               all_moms = True,
                               diff_scale = True)
    plt.savefig('figures/sce_seni_est_diag'+str(i)+'.png')
    
    ## joint estimation
    
    SENI_model2.ParaEstimateSMMJoint(method='Nelder-Mead',
                                    para_guess =(0.5,0.1,0.1,0.9,0.1),
                                    options={'disp':True})
    para_est_SCE_joint_holder.append(SENI_model2.para_est_SMM_joint)
    SENI_model2.all_moments = ['FE','Disg','Var']
    SENI_model2.ForecastPlotDiag(how = 'SMMjoint',
                                 all_moms = True,
                                 diff_scale = True)
    plt.savefig('figures/sce_seni_est_joint_diag'+str(i)+'.png')

print(para_est_SCE_holder)
print(para_est_SCE_joint_holder)

# + {"code_folding": [0]}
## tabulate the estimates 
sce_seni_est_para = pd.DataFrame(para_est_SCE_holder,
                                 columns=[r'SENI: $\lambda$',
                                          r'SENI: $\sigma_{pb,SCE}$',
                                          r'SENI: $\sigma_{pr,SCE}$'])
sce_seni_joint_est_para = pd.DataFrame(para_est_SCE_joint_holder,
                                       columns=[r'SENI: $\lambda$',
                                                r'SENI: $\sigma_{pb,SCE}$',
                                                r'SENI: $\sigma_{pr,SCE}$',
                                                r'DE: $\sigma$'])

# + {"code_folding": [3, 11]}
est_moms = pd.DataFrame(moments_choices)

## combining SCE and SPF 
seni_est_df = pd.concat([est_moms,
                         spf_seni_est_para,
                         #spf_seni_joint_est_para,
                         sce_seni_est_para,
                         #sce_seni_joint_est_para],
                        ],
                        join='inner', axis=1)

seni_est_df.to_excel('tables/SENI_Est.xlsx',
                   float_format='%.2f',
                   index = False)
# -

spf_seni_est_para

# + {"code_folding": [0, 8]}
### Estimate of Parameter Learning for SPF

real_time = np.array(SPF_est['RTCPI'])
data_moms_dct = data_moms_dct_SPF

process_paraQ_est = {'rho':rhoQ_est,
                    'sigma':sigmaQ_est}

PL_model = pl(real_time = real_time,
              history = history_Q,
              process_para = process_paraQ_est,
              moments = ['Forecast','FE','Disg','Var'])

PL_model.GetRealization(realized_CPIC)
PL_model.LearnParameters()
moms_pl_sim = PL_model.Forecaster()
PL_model.GetDataMoments(data_moms_dct)

# + {"code_folding": [0]}
pl_plot = ForecastPlotDiag(moms_pl_sim,
                           data_moms_dct,
                           legends=['model','data'])

# + {"code_folding": [0]}
### Estimate of Parameter Learning for SCE

real_time = np.array(SCE_est['RTCPI'])
data_moms_dct2 = data_moms_dct_SCE

process_paraM_est = {'rho':rhoM_est,
                    'sigma':sigmaM_est}

PL_model2 = pl(real_time = real_time,
              history = history_M,
              process_para = process_paraM_est,
              moments = ['Forecast','FE','Disg','Var'])

PL_model2.GetRealization(realized_CPI)
PL_model2.LearnParameters()
moms_pl_sim2 = PL_model2.Forecaster()
PL_model2.GetDataMoments(data_moms_dct2)

# + {"code_folding": [0]}
pl_plot = ForecastPlotDiag(moms_pl_sim2,
                           data_moms_dct2,
                           legends=['model','data'])
# -

'''
NI_model_sim_est = {'sigma_pb':sigmas_est_SPF[0][0],
                'sigma_pr':sigmas_est_SPF[0][1],
                'var_init':sigmas_est_SPF[0][2]}

NI_model.exp_para = NI_model_sim_est
NI_model.SimulateSignals()
ni_sim_moms_dct = NI_model.Forecaster()
'''

# + {"code_folding": []}
#NI_model.exp_para
# -

'''
plt.figure(figsize=([3,13]))
for i,key in enumerate(ni_sim_moms_dct):
    plt.subplot(4,1,i+1)
    print(key)
    plt.plot(ni_sim_moms_dct[key],label='Model')
    plt.plot(np.array(data_moms_dct_SPF[key]),label='Data')
    plt.legend(loc=1)
'''
