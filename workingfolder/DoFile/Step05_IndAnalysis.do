clear
global mainfolder "/Users/Myworld/Dropbox/ExpProject/workingfolder"
global folder "${mainfolder}/SurveyData/"
global sum_graph_folder "${mainfolder}/graphs/pop"
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

merge m:1 year month using "${mainfolder}/OtherData/InfM.dta",keep(match using master)
rename _merge inflation_merge 

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


log close 
