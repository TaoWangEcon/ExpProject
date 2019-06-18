clear
global mainfolder "/Users/Myworld/Dropbox/ExpProject/workingfolder"
global folder "${mainfolder}/SurveyData/"
global sum_graph_folder "${mainfolder}/graphs/ind"
global sum_table_folder "${mainfolder}/tables"

cd ${folder}
pwd
set more off 
capture log close
log using "${mainfolder}/ind_log",replace


***************************
**  Clean and Merge Data **
***************************

use "${folder}/SPF/individual/InfExpSPFPointIndQ",clear 

duplicates report year quarter ID 

merge 1:1 year quarter ID using "${folder}/SPF/individual/InfExpSPFDstIndQ.dta"

rename _merge SPFDst_merge
table year if SPFDst_merge ==3

*merge m:1 year month using "${mainfolder}/OtherData/InfM.dta",keep(match using master)
*rename _merge inflation_merge 

merge m:1 year quarter using "${mainfolder}/OtherData/InfShocksClean.dta",keep(match using master)
rename _merge infshocks_merge


*******************************
**  Set Panel Data Structure **
*******************************

xtset ID dateQ
sort ID year quarter 

drop if ID==ID[_n-1] & INDUSTRY != INDUSTRY[_n-1]


*******************************
**  Summary Statistics of SPF **
*******************************

tabstat ID,s(count) by(dateQ) column(statistics)


******************************
*** Computing some measures **
******************************

gen SPFCPI_FE = CPI1y - Inf1yf_CPIAU
label var SPFCPI_FE "1-yr-ahead forecast error(SPF CPI)"
gen SPFCCPI_FE = CORECPI1y - Inf1yf_CPICore
label var SPFCPI_FE "1-yr-ahead forecast error(SPF core CPI)"
gen SPFPCE_FE = PCE1y - Inf1yf_PCE
label var SPFPCE_FE "1-yr-ahead forecast error(SPF PCE)"


*****************************************
****  Renaming so that more consistent **
*****************************************


rename CPI1y SPFCPI_Mean
rename PCE1y SPFPCE_Mean
rename COREPCE1y SPFCPCE_Mean
rename CORECPI1y SPFCCPI_Mean

rename PRCPCEVar1 SPFPCE_Var
rename PRCCPIVar1 SPFCPI_Var

rename SPFCPI_FE SPFCPI_FE
rename SPFPCE_FE SPFPCE_FE

*******************************
**  Generate Variables       **
*******************************

gen InfExp_Mean = .
gen InfExp_Var = .
gen InfExp_FE = .
*gen InfExp_Disg = . 


gen InfExp_Mean_ch = .
gen InfExp_Var_ch = .
gen InfExp_FE_ch = .
*gen InfExp_Disg_ch = . 

************************************************
** Auto Regression of the Individual Moments  **
************************************************

eststo clear

foreach mom in FE Var{
   foreach var in SPFCPI SPFPCE{
    replace InfExp_`mom' = `var'_`mom'
	xtset ID dateQ
    replace InfExp_`mom'_ch = InfExp_`mom'-l1.InfExp_`mom'
	eststo `var'_`mom'lvl: reg InfExp_`mom' l(1/4).InfExp_`mom', vce(cluster dateQ)
    eststo `var'_`mom'diff: reg InfExp_`mom'_ch l(1/4).InfExp_`mom'_ch, vce(cluster dateQ)
  }
}
esttab using "${sum_table_folder}/ind/autoregSPFIndQ.csv", mtitles se  r2 replace
eststo clear


******************************************************
** Response  Estimates using individual moments     **
******************************************************

keep if year>=2008


eststo clear

foreach mom in FE{
   foreach var in SPFCPI SPFPCE{
       * shocks 
       capture eststo `var'_`mom': reg `var'_`mom' l(1/2).`var'_`mom' ///
	                  l(0/1).pty_shock l(0/1).op_shock ///
					 l(0/1).mp1ut_shock l(0/1).ED8ut_shock, vce(cluster dateQ)
       }
 }
 
 
foreach mom in Var{
   foreach var in SPFCPI SPFPCE{
       * abs of shocks 
       capture eststo `var'_`mom': reg `var'_`mom' l(1/2).`var'_`mom' ///
	                  l(0/1).pty_abshock l(0/1).op_abshock ///
					 l(0/1).mp1ut_abshock l(0/1).ED8ut_abshock, vce(cluster dateQ)
   }
}


esttab using "${sum_table_folder}/SPF_ind_ashocks.csv", drop(_cons) mtitles se r2 replace


** !!!! Need to find a way to run var for panel data
/*
************************************************
** IRF using individual SPF moments     **
************************************************



foreach mom in FE{
   foreach var in SPFCPI SPFPCE{
       * shocks 
       var `var'_`mom', lags(1/4) ///
                     exo(l(0/1).pty_shock l(0/1).op_shock ///
					 l(0/1).mp1ut_shock l(0/1).ED8ut_shock)
   set seed 123456
   capture irf create `var', set(`mom') step(10) bsp replace 
}
   * Non-MP shocks plots
   irf graph dm, set(`mom') impulse(pty_shock op_shock) ///
                         byopts(col(2) title("`mom'") yrescale /// 
						 xrescale note("")) legend(col(2) /// 
						 order(1 "95% CI" 2 "IRF") symx(*.5) size(vsmall))  ///
						 xtitle("Quarters") 
   capture graph export "${sum_graph_folder}/irf/moments/SPF`mom'_shocks_nmp", as(png) replace
   
   * MP shocks plots
   capture irf graph dm, set(`mom') impulse(mp1ut_shock ED8ut_shock) ///
                         byopts(col(2) title("`mom'") yrescale /// 
						 xrescale note("")) legend(col(2) /// 
						 order(1 "95% CI" 2 "IRF") symx(*.5) size(vsmall))  ///
						 xtitle("Quarters") 
   capture graph export "${sum_graph_folder}/irf/moments/SPF`mom'_mpshocks", as(png) replace
   
}



****************************************************************
** IRF of SPF moments (all shocks(abs) exl MP at one time)    **
****************************************************************


foreach mom in Var{
   foreach var in SPFCPI SPFPCE{
       * shocks 
        capture var `var'_`mom', lags(1/4) ///
                     exo(l(0/1).pty_shock l(0/1).mp1ut_shock l(0/1).ED8ut_shock ///
					 l(0/1).pty_abshock l(0/1).op_abshock l(0/1).mp1ut_abshock l(0/1).ED8ut_abshock)
   set seed 123456
   capture irf create `var', set(`mom') step(10) bsp replace 
}
   * Non-MP shocks 
   capture irf graph dm, set(`mom') impulse(pty_abshock op_abshock) ///
                         byopts(title("`mom'") yrescale /// 
						 xrescale note("")) legend(col(2) /// 
						 order(1 "95% CI" 2 "IRF") symx(*.5) size(vsmall))  ///
						 xtitle("Quarters") 
   capture graph export "${sum_graph_folder}/irf/moments/SPF`mom'_ab_ashocks_nmp", as(png) replace
   
   * Non-MP shocks 
   capture irf graph dm, set(`mom') impulse(mp1ut_abshock ED8ut_abshock) ///
                         byopts(title("`mom'") yrescale /// 
						 xrescale note("")) legend(col(2) /// 
						 order(1 "95% CI" 2 "IRF") symx(*.5) size(vsmall))  ///
						 xtitle("Quarters") 
   capture graph export "${sum_graph_folder}/irf/moments/SPF`mom'_ab_mpshocks", as(png) replace
}


log close 
