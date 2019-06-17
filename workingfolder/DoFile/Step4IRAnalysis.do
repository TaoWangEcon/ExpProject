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
label var pty_max_shock "Technology shock (Francis etal 2014)"
label var news_shock "News shock(Sims etal.2011)"
rename OPShock_nm op_shock 
label var op_shock "Oil price shock (Hamilton 1996)"
rename MP1 mp1_shock
label var mp1_shock "Unexpected change in federal funds rate"
label var ED4 "1-year ahead future-implied change in federal funds rate"
label var ED8 "2-year ahead future-implied change in federal funds rate"


** MP-path shock 

foreach ed in ED4 ED8{
  reg `ed' mp1_shock
  predict `ed'_shock, residual
label var `ed'_shock "Unexpected shock to future federal funds rate"
}

** Normorlize MP shcoks

foreach var in mp1 ED4 ED8{
  egen `var'_shock_sd =sd(`var'_shock)
  gen `var'ut_shock = `var'_shock/`var'_shock_sd 
  local lb : var label `var'_shock
  label var `var'ut_shock "`lb' in std unit"
}


** Generated unidentified shocks. 

tsset date

eststo clear

foreach Inf in CPIAU CPICore PCEPI{ 
   reg Inf1y_`Inf' l(1/4).Inf1y_`Inf' l(0/1).pty_shock l(0/1).op_shock l(0/1).mp1ut_shock l(0/1).ED8ut_shock
   predict `Inf'_uid_shock, residual
   label var `Inf'_uid_shock "Unidentified shocks to inflation"
 }


/*
** Plot all shocks for checking 

twoway (tsline pty_shock) (tsline op_shock) ///
        (tsline mp1ut_shock) (tsline ED8ut_shock) ///
		(tsline CPIAU_uid_shock), ///
		title("Shocks to Inflation",size(4)) ///
		xtitle("Time") ytitle("") ///
		legend(cols(1)) 
		
graph export "${sum_graph_folder}/inf_shocksQ", as(png) replace

** First-run of inflation 

eststo clear
foreach sk in pty pty_max op mp1ut ED4ut ED8ut{
  foreach Inf in CPIAU CPICore PCEPI{ 
   eststo `Inf'_`sk': reg Inf1y_`Inf' l(0/1).`sk'_shock, robust
   eststo `Inf'_uid: reg Inf1y_`Inf' l(0/1).`Inf'_uid_shock,robust 
 }
}
esttab using "${sum_table_folder}/IRFQ.csv", mtitles se r2 replace

** IRF of inflation (one shock each time) 

eststo clear
foreach sk in pty pty_max op mp1ut ED4ut ED8ut{
  foreach Inf in CPIAU CPICore PCEPI{ 
   var Inf1y_`Inf', lags(1/4) exo(l(0/1).`sk'_shock)
   set seed 123456
   irf create irf1, set(irf,replace) step(10) bsp
   irf graph dm, impulse(`sk'_shock)
   graph export "${sum_graph_folder}/irf/`Inf'_`sk'", as(png) replace
 }
}

***********************************************
** IRF of inflation (all shocks at one time) **
***********************************************

eststo clear

foreach Inf in CPIAU CPICore PCEPI{ 
   var Inf1y_`Inf', lags(1/4) ///
                     exo(l(0/1).pty_shock l(0/1).op_shock ///
					 l(0/1).mp1ut_shock l(0/1).ED8ut_shock)   
   set seed 123456
   irf create irf1, set(irf,replace) step(10) bsp replace 
   irf graph dm, impulse(pty_shock op_shock mp1ut_shock ED8ut_shock) ///
                 byopts(title("`mom'") yrescale xrescale note("")) ///
                 legend(col(2) order(1 "95% CI" 2 "IRF") symx(*.5) size(vsmall)) ///
				 xtitle("Quarters") 
   graph export "${sum_graph_folder}/irf/`Inf'_ashocks", as(png) replace

}



***********************************************
** IRF of inflation (all shocks exl MP at one time) **
***********************************************


eststo clear

foreach Inf in CPIAU CPICore PCEPI{ 
   var Inf1y_`Inf', lags(1/4) ///
                     exo(l(0/1).pty_shock l(0/1).op_shock)   
   set seed 123456
   irf create irf1, set(irf,replace) step(10) bsp replace 
   irf graph dm, impulse(pty_shock op_shock) ///
                 byopts(col(1) title("`mom'") yrescale xrescale note("")) ///
                 legend(col(2) order(1 "95% CI" 2 "IRF") symx(*.5) size(vsmall)) ///
				 xtitle("Quarters") 
   graph export "${sum_graph_folder}/irf/`Inf'_ashocks_nmp", as(png) replace

}



****************************************************
** IRF of SPF moments (all shocks at one time)    **
****************************************************


foreach mom in Mean Var Disg FE{
   foreach var in SPFCPI SPFPCE{
       capture var `var'_`mom', lags(1/4) ///
                     exo(l(0/1).pty_shock l(0/1).op_shock ///
					 l(0/1).mp1ut_shock )
   set seed 123456
   capture irf create `var', set(`mom') step(10) bsp 
}
   capture irf graph dm, impulse(pty_shock op_shock mp1ut_shock) ///
                         byopts(title("`mom'") yrescale /// 
						 xrescale note("")) legend(col(2) /// 
						 order(1 "95% CI" 2 "IRF") symx(*.5) size(vsmall))  ///
						 xtitle("Quarters") 
   capture graph export "${sum_graph_folder}/irf/moments/SPF`mom'_ashocks", as(png) replace
}
*/



***********************************************************
** IRF of SPF moments (all shocks exl MP at one time)    **
***********************************************************


foreach mom in Mean Var Disg FE{
   foreach var in SPFCPI SPFPCE{
       capture var `var'_`mom', lags(1/4) ///
                     exo(l(0/1).pty_shock l(0/1).op_shock) 
   set seed 123456
   capture irf create `var', set(`mom') step(10) bsp  
}
   capture irf graph dm, impulse(pty_shock op_shock) ///
                         byopts(title("`mom'") yrescale /// 
						 xrescale note("")) legend(col(2) /// 
						 order(1 "95% CI" 2 "IRF") symx(*.5) size(vsmall))  ///
						 xtitle("Quarters") 
   capture graph export "${sum_graph_folder}/irf/moments/SPF`mom'_ashocks_nmp", as(png) replace
}

/*
****************************************************
** IRF of SCE moments (all shocks at one time)    **
****************************************************


foreach mom in Mean Var Disg FE{
   foreach var in SCE{
       capture var `var'_`mom', lags(1/4) ///
                     exo(l(0/1).pty_shock l(0/1).op_shock ///
					 l(0/1).mp1ut_shock )
   set seed 123456
   capture irf create `var', set(`mom') step(10) bsp replace 
}
   capture irf graph dm, impulse(pty_shock op_shock mp1ut_shock) ///
                         byopts(title("`mom'") yrescale /// 
						 xrescale note("")) legend(col(2) /// 
						 order(1 "95% CI" 2 "IRF") symx(*.5) size(vsmall))  ///
						 xtitle("Quarters") 
   capture graph export "${sum_graph_folder}/irf/moments/SCE`mom'_ashocks", as(png) replace
}
*/

log close 
