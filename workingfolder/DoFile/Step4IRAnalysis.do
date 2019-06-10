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

** MP shock 

foreach ed in ED4 ED8{
  reg `ed' mp1_shock
  predict `ed'_shock, residual
label var `ed'_shock "Unexpected shock to future federal funds rate"
}


** First-run of inflation 
tsset date

eststo clear
foreach sk in pty op mp1 ED4 ED8{
  foreach Inf in CPIAU CPICore PCEPI{ 
   eststo `Inf'_`sk': reg Inf1y_`Inf' l(1/4).`sk'_shock, robust 
 }
}
esttab using "${sum_table_folder}/IRFQ.csv", mtitles se r2 replace

/*
eststo clear
foreach sk in pty op mp1 ED4 ED8{
  foreach Inf in CPIAU CPICore PCEPI{ 
   eststo `Inf'_`sk': var Inf1y_`Inf', lags(3) ex(`sk'_shock)
   irf create `inf'_`sk', step(10) set(`inf'_`sk',replace)
   irf graph oirf, impulse(`sk'_shock) response(Inf1y_`inf')
 }
}
esttab using "${sum_table_folder}/IRFQ2.csv", mtitles se r2 replace
*/


log close 
