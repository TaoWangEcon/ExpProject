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

tabstat `Moments', st(N min p1 p5 p10 p50 p90 p95 p99 max) save

return list
mat T = r(StatTotal)'
matlist T
putexcel set "${sum_table_folder}/InfExpSPFDstSum.xlsx", replace
putexcel A1 = matrix(T), sheet("rawdata")


foreach var in `Moments'{

      egen `var'p1=pctile(`var'),p(1)
	  egen `var'p99=pctile(`var'),p(99)
	  replace `var' = . if `var' <`var'p1 | (`var' >`var'p99 & `var'!=.)
}

tabstat `Moments', st(N min p1 p5 p10 p50 p90 p95 p99 max) save

return list
mat T = r(StatTotal)'
matlist T
putexcel set "${sum_table_folder}/InfExpSPFDstSum.xlsx", replace
putexcel A1 = matrix(T), sheet("data_winsored")


** quarterly population data 
preserve 
collapse (mean) `Moments', by(date year quarter) 

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

