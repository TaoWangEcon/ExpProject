clear
global mainfolder "/Users/Myworld/Dropbox/ExpProject/workingfolder"
global folder "${mainfolder}/SurveyData/"
global surveyfolder "NYFEDSurvey"
global sum_graph_folder "${mainfolder}/graphs/pop"
global sum_table_folder "${mainfolder}/tables"

cd ${folder}
pwd
set more off 
capture log close
log using "${mainfolder}/pop_log",replace

****************************************
**** Population Moments Analysis ******
****************************************

***********************
**   Merge other data**
***********************

use "${folder}/NYFEDSurvey/InfExpSCEProbPopM",clear 

merge 1:1 year month using "${mainfolder}/OtherData/RecessionDateM.dta", keep(match)
rename _merge  recession_merge

merge 1:1 year month using "${mainfolder}/OtherData/InfM.dta",keep(match using master)
rename _merge inflation_merge 

merge 1:1 year month using "${folder}/MichiganSurvey/InfExpMichM.dta"
rename inf_exp InfExpMichMed 
rename _merge michigan_merge 

***********************************************
** create quarter variable to match with spf
*********************************************

gen quarter = .
replace quarter=1 if month<4 & month >=1
replace quarter=2 if month<7 & month >=4
replace quarter=3 if month<10 & month >=7
replace quarter=4 if month<=12 & month >=10

merge m:1 year quarter using "${folder}/SPF/individual/InfExpSPFPointPopQ.dta",keep(match using master)
rename _merge spf_merge 


merge m:1 year quarter using "${folder}/SPF/InfExpSPFDstPopQ.dta",keep(match using master)
rename _merge spf_dst_merge 


*************************
** Declare time series **
**************************

gen date2=string(year)+"m"+string(month)
gen date3= monthly(date2,"YM")
format date3 %tm 

drop date2 date 
rename date3 date

tsset date
sort year quarter month 




******************************
*** Computing some measures **
******************************

gen SCE_FE = Q9_mean - Inf1yf_CPICore
label var SCE_FE "1-yr-ahead forecast error"
gen SPF_FE = CORECPI1y - Inf1yf_CPICore
label var SPF_FE "1-yr-ahead forecast error"



*****************************
****    Summary Charts ******
*****************************


**************************
** Single series charts **
**************************

/*
foreach var in Q9_mean Q9_var Q9_iqr Q9_cent50 Q9_disg ///
               CPI1y PCE1y CORECPI1y COREPCE1y ///
			   CPI_disg PCE_disg CORECPI_disg COREPCE_disg ///
			   CPI_ct50 PCE_ct50 CORECPI_ct50 COREPCE_ct50{
local var_lb: var label `var'
tsline `var' if `var'!=.,title("`var_lb'") xtitle("Time") ytitle("")
graph export "${sum_graph_folder}/`var'", as(png) replace 
}

***************************
** Multiple series charts**
***************************

drop if CPI1y ==. | PCE1y==.

twoway (tsline Q9_mean)  (tsline Q9_cent50,lp("dash")) ///
                         (tsline InfExpMichMed, lp("dot-dash")) ///
						 (tsline CPI1y, lp("shortdash")), ///
						 title("1-yr-ahead Expected Inflation") ///
						 xtitle("Time") ytitle("") ///
						 legend(label(1 "Mean Expectation(SCE)") label(2 "Median Expectation(SCE)") ///
						        label(3 "Median Expectation(Michigan)") label(4 "Mean Expectation(SPF)"))
graph export "${sum_graph_folder}/mean_med", as(png) replace

twoway (tsline Q9_mean, ytitle(" ",axis(1))) ///
       (tsline Q9_var,yaxis(2) ytitle("",axis(2)) lp("dash")) ///
	   if Q9_mean!=., ///
	   title("1-yr-ahead Expected Inflation") xtitle("Time") ///
	   legend(label(1 "Average Expectation") label(2 "Average Uncertainty(RHS)"))
graph export "${sum_graph_folder}/mean_var", as(png) replace 


twoway (tsline Q9_mean, ytitle(" ",axis(1))) ///
       (tsline Q9_disg,yaxis(2) ytitle("",axis(2)) lp("dash")) ///
	   if Q9_mean!=., ///
	   title("1-yr-ahead Expected Inflation") xtitle("Time") ///
	   legend(label(1 "Average Expectation") label(2 "Disagreements(RHS)"))
graph export "${sum_graph_folder}/mean_disg", as(png) replace 


twoway (tsline Q9_disg, ytitle(" ",axis(1))) ///
       (tsline CORECPI_disg,yaxis(2) ytitle("",axis(2)) lp("dash")) ///
	   if CORECPI_disg!=., ///
	   title("Disagreements in 1-yr-ahead Inflation") xtitle("Time") ///
	   legend(label(1 "Disagreements (SCE)") label(2 "Disagreements(SPF)(RHS)"))
graph export "${sum_graph_folder}/disg_disg", as(png) replace 


twoway (tsline Q9_var, ytitle(" ",axis(1))) ///
       (tsline PRCCPIVar1mean, yaxis(2) ytitle("",axis(2)) lp("dash")) ///
	   if Q9_var!=., ///
	   title("Uncertainty in 1-yr-ahead Inflation") xtitle("Time") ///
	   legend(label(1 "Uncertainty (SCE)")  /// 
	          label(2 "Uncertainty (SPF CPI)(RHS)")) 
			  
graph export "${sum_graph_folder}/var_var", as(png) replace 


twoway (tsline Q9_disg, ytitle(" ",axis(1))) ///
       (tsline Q9_var,yaxis(2) ytitle("",axis(2)) lp("dash")) ///
	   if Q9_disg!=., ///
	   title("1-yr-ahead Expected Inflation") xtitle("Time") ///
	   legend(label(1 "Disagreements") label(2 "Average Uncertainty(RHS)")) 
graph export "${sum_graph_folder}/var_disg", as(png) replace 



twoway (tsline CORECPI_disg, ytitle(" ",axis(1))) ///
       (tsline PRCCPIVar1mean,yaxis(2) ytitle("",axis(2)) lp("dash")) ///
	   if Q9_disg!=., ///
	   title("1-yr-ahead Expected Inflation(SPF)") xtitle("Time") ///
	   legend(label(1 "Disagreements") label(2 "Average Uncertainty(RHS)")) 
graph export "${sum_graph_folder}/var_disg2", as(png) replace 


twoway (tsline Q9_mean) (tsline CORECPI1y,lp("longdash")) ///
       (tsline Inf1yf_PCEPI,lp("dash")) ///
       (tsline Inf1yf_CPIAU,lp("shortdash")) ///
	   (tsline Inf1yf_CPICore,lp("dash-dot")) ///
	   if Q9_mean!=., ///
	   title("1-yr-ahead Expected Inflation") xtitle("Time") ytitle("") ///
	   legend(label(1 "Mean Forecast(SCE)") label(2 "Mean Forecast(SPF)") label(3 "Realized Inflation(PCE)") ///
	          label(4 "Realized Inflation(Headline CPI)") label(5 "Realized Inflation(Core CPI)"))
graph export "${sum_graph_folder}/mean_true", as(png) replace


twoway (tsline Q9_mean)  (tsline Inf1y_PCEPI,lp("dash")) ///
       (tsline Inf1y_CPIAU,lp("shortdash")) ///
	   (tsline Inf1y_CPICore,lp("dash-dot")) ///
	   if Q9_mean!=., ///
	   title("1-yr-ahead Expected Inflation") xtitle("Time") ytitle("") ///
	   legend(label(1 "Mean Expectation") label(2 "Past Inflation(PCE)") ///
	          label(3 "Past Inflation(Headline CPI)") label(4 "Past Inflation(Core CPI)"))
graph export "${sum_graph_folder}/mean_past", as(png) replace


twoway (tsline SCE_FE)  (tsline SPF_FE, yaxis(2) lp("dash")) ///
                         if SPF_FE!=., ///
						 title("1-yr-ahead Forecast Errors") ///
						 xtitle("Time") ytitle("") ///
						 legend(label(1 "SCE") label(2 "SPF(RHS)"))
graph export "${sum_graph_folder}/fe_fe", as(png) replace



***************************
***  Population Moments *** 
***************************
tsset date

estpost tabstat Q9_mean Q9_var Q9_disg Q9_iqr CPI1y PCE1y CORECPI1y COREPCE1y ///
                CPI_disg PCE_disg CORECPI_disg COREPCE_disg PRCCPIVar1mean PRCPCEVar1mean, ///
			    st(mean var median) columns(statistics)
esttab . using "${sum_table_folder}/pop_sum_stats.csv", cells("mean(fmt(a3)) var(fmt(a3)) median(fmt(a3))") replace

eststo clear
foreach var in Q9_mean Q9_var Q9_disg CPI1y PCE1y CORECPI1y COREPCE1y{
gen `var'_ch = `var'-l1.`var'
label var `var'_ch "m to m+1 change of `var'"
eststo: reg `var' l(1/5).`var'
eststo: reg `var'_ch l(1/5).`var'_ch
corrgram `var', lags(5) 
*gen `var'1=`r(ac1)'
*label var `var'1 "Auto-correlation coefficient of `var'"
}
esttab using "${sum_table_folder}/autoreg.csv", se r2 replace
eststo clear
*/

****************************************
**** Quarterly Level Analysis  ******
****************************************

*******************************************
***  Collapse monthly data to quarterly  **
*******************************************

collapse (mean) Q9_mean Q9_var Q9_disg Q9_iqr CPI1y PCE1y CORECPI1y InfExpMichMed ///
                COREPCE1y CPI_disg PCE_disg CORECPI_disg COREPCE_disg ///
				PRCCPIVar1mean PRCPCEVar1mean PRCCPIVar0mean PRCPCEVar0mean, ///
				by(year quarter) 

gen date2=string(year)+"Q"+string(quarter)
gen date3= quarterly(date2,"YQ")
format date3 %tq 
drop date2 
rename date3 date

tsset date
sort year quarter  

******************************************
*** Multiple series charts Quarterly  ****
*******************************************
drop if CPI1y ==. | PCE1y==.

twoway (tsline Q9_mean) (tsline InfExpMichMed, lp("dash_dot")) ///
       (tsline CPI1y, lp("shortdash")) (tsline PCE1y, lp("dash")), ///
						 title("1-yr-ahead Expected Inflation") ///
						 xtitle("Time") ytitle("") ///
						 legend(label(1 "Mean Expectation(SCE)") ///
						        label(2 "Median Expectation(Michigan)") ///
								label(3 "Mean Expectation CPI (SPF)") ///
								label(4 "Mean Expectation PCE (SPF)") col(1))
graph export "${sum_graph_folder}/mean_medQ", as(png) replace


twoway (tsline Q9_disg, ytitle(" ",axis(1))) ///
       (tsline CORECPI_disg,yaxis(2) ytitle("",axis(2)) lp("dash")) ///
	   if CORECPI_disg!=., ///
	   title("Disagreements in 1-yr-ahead Inflation") xtitle("Time") ///
	   legend(label(1 "Disagreements (SCE)") label(2 "Disagreements(SPF)(RHS)"))
graph export "${sum_graph_folder}/disg_disgQ", as(png) replace 


twoway (tsline Q9_var, ytitle(" ",axis(1)) lp("solid") ) ///
       (tsline PRCCPIVar1mean, yaxis(2) ytitle("",axis(2)) lp("shortdash")) ///
	   (tsline PRCPCEVar1mean, yaxis(2) ytitle("",axis(2)) lp("dash_dot")) ///
	   if Q9_var!=., ///
	   title("Uncertainty in 1-yr-ahead Inflation") xtitle("Time") ///
	   legend(label(1 "Uncertainty (SCE)")  /// 
	          label(2 "Uncertainty (SPF CPI)(RHS)") ///
			  label(3 "Uncertainty (SPF PCE)(RHS)") col(1)) 
			  
graph export "${sum_graph_folder}/var_varQ", as(png) replace 


********************************
***  Autoregression Quarterly **
*******************************

tsset date 

eststo clear

foreach var in Q9_mean Q9_var Q9_disg CPI1y PCE1y CORECPI1y COREPCE1y PRCCPIVar1mean PRCPCEVar1mean{
gen `var'_ch = `var'-l1.`var'
label var `var'_ch "m to m+1 change of `var'"
eststo: reg `var' l(1/5).`var'
eststo: reg `var'_ch l(1/5).`var'_ch
corrgram `var', lags(5) 
*gen `var'1=`r(ac1)'
*label var `var'1 "Auto-correlation coefficient of `var'"
}
esttab using "${sum_table_folder}/autoregQ.csv", se r2 replace
eststo clear


log close 
