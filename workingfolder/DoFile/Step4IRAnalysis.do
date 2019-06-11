clear
global mainfolder "/Users/Myworld/Dropbox/ExpProject/workingfolder"
global folder "${mainfolder}/SurveyData/"
global sum_graph_folder "${mainfolder}/graphs/pop"
global sum_table_folder "${mainfolder}/tables"

cd ${folder}
pwd
set more off 
capture log close
log using "${mainfolder}/irf_log",replace


**********************************************
*** Clean Inflation Shock data from Python ***
**********************************************

use "${mainfolder}/OtherData/InfShocks.dta",clear 

drop index 

*** Date 
gen date_str=string(year)+"Q"+string(quarter) 

gen date = quarterly(date_str,"YQ")
format date %tq 

drop date_str

order date year quarter month

** Time series 

tsset date 

** Merge with survey and inflation  

merge 1:1 year quarter using "${folder}/InfExpQ.dta",keep(match using master)
rename _merge InfExp_merge

drop if month==. 
drop if quarter ==.

merge 1:1 year month using "${mainfolder}/OtherData/InfM.dta",keep(match master)
rename _merge InfM_merge 

** Label and rename

label var pty_shock "Technology shock(Gali 1999)"
label var hours_shock "Non-technology shock(Gali 1999)"
label var inf_shock "Shock to inflation(Gali 1999)"
label var pty_shock_max "Technology shock (Francis etal 2014)"
label var news_shock "News shock(Sims etal.2011)"
rename OPShock_nm op_shock 
label var op_shock "Oil price shock (Hamilton 1996)"
rename MP1 mp1_shock
label var mp1_shock "Unexpected change in federal funds rate(%)"
label var ED4 "1-year ahead future-implied change in federal funds rate(%)"
label var ED8 "2-year ahead future-implied change in federal funds rate(%)"

** MP-path shock 

foreach ed in ED4 ED8{
  reg `ed' mp1_shock
  predict `ed'_shock, residual
label var `ed'_shock "Unexpected shock to future federal funds rate(%)"
}



** Generated unidentified shocks. 

tsset date

eststo clear

foreach Inf in CPIAU CPICore PCEPI{ 
   reg Inf1y_`Inf' l(1/4).Inf1y_`Inf' l(0/1).pty_shock l(0/1).op_shock l(0/1).mp1_shock l(0/1).ED8_shock
   predict `Inf'_uid_shock, residual
   label var `Inf'_uid_shock "Unidentified shocks to inflation"
 }

** Plot all shocks for checking 

twoway (tsline pty_shock) (tsline op_shock) ///
        (tsline mp1_shock) (tsline ED8_shock) ///
		(tsline CPIAU_uid_shock), ///
		title("Shocks to Inflation",size(6)) ///
		xtitle("Time") ytitle("") ///
		legend(cols(1)) 
		
graph export "${sum_graph_folder}/inf_shocksQ", as(png) replace


** First-run of inflation 

eststo clear
foreach sk in pty op mp1 ED4 ED8 {
  foreach Inf in CPIAU CPICore PCEPI{ 
   eststo `Inf'_`sk': reg Inf1y_`Inf' l(0/1).`sk'_shock, robust
   eststo `Inf'_uid: reg Inf1y_`Inf' l(0/1).`Inf'_uid_shock,robust 
 }
}
esttab using "${sum_table_folder}/IRFQ.csv", mtitles se r2 replace


** IRF of inflation 

eststo clear
foreach sk in pty op mp1 ED4 ED8 {
  foreach Inf in CPIAU CPICore PCEPI{ 
   var Inf1y_`Inf', lags(1/4) exo(l(0/1).`sk'_shock)
   set seed 123456
   irf create irf1, set(`Inf'_`sk',replace) step(10) bsp
   irf graph dm, impulse(`sk'_shock)
   graph export "${sum_graph_folder}/irf/`Inf'_`sk'", as(png) replace

 }
}

log close 
