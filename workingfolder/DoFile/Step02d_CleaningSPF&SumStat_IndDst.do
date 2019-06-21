*******************************************************************************
** This do file cleans the density/moment estimats of individual SPF 
** InfExpSPFDstIndQ.dta. 
** And then it generates the population moments from SPF. InfExpSPFDstPopQ.dta.
** This data is quarterly. So one may need to convert SCE to quarterly for a 
** comparison of the two. 
********************************************************************************* 

clear 
set more off 
global mainfolder "/Users/Myworld/Dropbox/ExpProject/workingfolder"
global datafolder "${mainfolder}/SurveyData/SPF/individual"
global sum_graph_folder "${mainfolder}/graphs/pop"
global sum_table_folder "${mainfolder}/tables"

cd ${datafolder}

use InfExpSPFDstIndQ,clear


local Moments PRCCPIMean0 PRCCPIMean1 PRCPCEMean0 PRCPCEMean1 /// 
              PRCCPIVar0 PRCCPIVar1 PRCPCEVar0 PRCPCEVar1

*****************************
**   Summary Stats of SPF  **
*****************************
			  
	  
* table 1
tabstat `Moments', st(N min p1 p10 p25 p50 p75 p90 p99 max) save 

return list
mat T = r(StatTotal)'
matlist T
putexcel set "${sum_table_folder}/InfExpSPFDstSum.xlsx", modify  
putexcel B2 = matrix(T), sheet("rawdata") 

foreach var in `Moments'{

      egen `var'p1=pctile(`var'),p(1)
	  egen `var'p99=pctile(`var'),p(99)
	  replace `var' = . if `var' <`var'p1 | (`var' >`var'p99 & `var'!=.)
}

* table 2
tabstat `Moments', st(N min p1 p10 p25 p50 p75 p90 p99 max) save

return list
mat T = r(StatTotal)'
matlist T
putexcel set "${sum_table_folder}/InfExpSPFDstSum.xlsx", modify 
putexcel B2 = matrix(T), sheet("data_winsored") 


*********************************
**   Time Series Graph of SPF  **
*********************************
			  

******************************
**   More Series Charts   ****
******************************

foreach mom in Mean Var{
   foreach var in PRCCPI PRCPCE{
    forvalues i=0/1{
     egen `var'`mom'`i'p75 =pctile(`var'`mom'`i'),p(75) by(year quarter)
	 egen `var'`mom'`i'p25 =pctile(`var'`mom'`i'),p(25) by(year quarter)
	 egen `var'`mom'`i'p50=pctile(`var'`mom'`i'),p(50) by(year quarter)
	 local lb: variable label `var'`mom'`i'
	 label var `var'`mom'`i'p75 "`lb': 75 pctile"
	 label var `var'`mom'`i'p25 "`lb': 25 pctile"
	 label var `var'`mom'`i'p50 "`lb': 50 pctile"
 }
 }
}

** These are moments of moments 
local MomentsMom PRCCPIMean0p25 PRCCPIMean1p25 PRCPCEMean0p25 PRCPCEMean1p25 /// 
              PRCCPIVar0p25 PRCCPIVar1p25 PRCPCEVar0p25 PRCPCEVar1p25 ///
			  PRCCPIMean0p50 PRCCPIMean1p50 PRCPCEMean0p50 PRCPCEMean1p50 /// 
              PRCCPIVar0p50 PRCCPIVar1p50 PRCPCEVar0p50 PRCPCEVar1p50 ///
			  PRCCPIMean0p75 PRCCPIMean1p75 PRCPCEMean0p75 PRCPCEMean1p75 /// 
              PRCCPIVar0p75 PRCCPIVar1p75 PRCPCEVar0p75 PRCPCEVar1p75 ///


** quarterly population data 
preserve 
collapse (mean) `Moments' `MomentsMom', by(date year quarter) 

foreach var in `Moments'{
rename `var' `var'mean
label var `var'mean "Population moments: mean of `var'"
}
save "${mainfolder}/SurveyData/SPF/InfExpSPFDstPopQ1",replace 
restore 

collapse (sd) `Moments', by(date year quarter) 
foreach var in `Moments'{
replace `var' = `var'^2
rename `var' `var'disg
label var `var'disg "Population moments: variance of `var'"
}
merge using "${mainfolder}/SurveyData/SPF/InfExpSPFDstPopQ1"
drop _merge

drop date 
 
save "${mainfolder}/SurveyData/SPF/InfExpSPFDstPopQ",replace 

rm "${mainfolder}/SurveyData/SPF/InfExpSPFDstPopQ1.dta"

