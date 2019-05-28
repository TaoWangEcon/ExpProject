*************************
**** Import SPF *********
*************************

clear
set more off
global folder "/Users/Myworld/Dropbox/ExpProject/workingfolder/SurveyData"
global datafolder "/SPF/individual"

cd ${folder}/${datafolder}
 
import excel "Individual_CPI.xlsx", sheet("CPI") firstrow clear 
save SPFCPI,replace

import excel "Individual_PCE.xlsx", sheet("PCE") firstrow clear 
save SPFPCE,replace

import excel "Individual_COREPCE.xlsx", sheet("COREPCE") firstrow clear
save SPFCorePCE,replace

import excel "Individual_CORECPI.xlsx", sheet("CORECPI") firstrow clear 
save SPFCoreCPI,replace

use SPFPCE,clear
merge 1:1 YEAR QUARTER INDUSTRY ID using SPFCPI
rename _merge SPFCPI_merge
merge 1:1 YEAR QUARTER INDUSTRY ID using SPFCorePCE
rename _merge SPFCorePCE_merge
merge 1:1 YEAR QUARTER INDUSTRY ID using SPFCoreCPI
rename _merge SPFCoreCPI_merge

rm "SPFPCE.dta"
rm "SPFCPI.dta"
rm "SPFCorePCE.dta"
rm "SPFCoreCPI.dta"



gen date_str = string(YEAR)+"Q"+string(QUARTER)
gen date = quarterly(date_str,"YQ")
format date %tq
drop date_str
rename YEAR year
rename QUARTER quarter 
rename date dateQ

gen month =.
replace month =1 if quarter==1
replace month =4 if quarter==2
replace month =7 if quarter ==3
replace month =9 if quarter==4

order dateQ year quarter month


***********************************
*********Destring and Labels ******
***********************************

foreach var in PCE CPI CORECPI COREPCE{
destring `var'1,force replace 
label var `var'1 "inflation `var' from q-2 to q-1"
destring `var'2,force replace 
label var `var'2 "inflation `var' from q-1 to q"
destring `var'3,force replace 
label var `var'3 "inflation `var' from q to q+1"
destring `var'4,force replace 
label var `var'4 "inflation `var' from q+1 to q+2"
destring `var'5,force replace 
label var `var'5 "inflation `var' from q+2 to q+3"
destring `var'6,force replace 
label var `var'6 "inflation `var' from q+3 to q+4"

destring `var'A,force replace 
label var `var'A "inflation `var' from y-1 to y"

destring `var'B,force replace 
label var `var'B "inflation `var' from y to y+1"

destring `var'C,force replace 
label var `var'C "inflation `var' from y+1 to y+2"

}



***********************************************
********* Computing annualized rate ***********
***********************************************


foreach var in PCE CPI CORECPI COREPCE{
gen `var'1y = 100*(((1+`var'3/100)*(1+`var'4/100)*(1+`var'5/100)*(1+`var'6/100))^0.25-1)
label var `var'1y "inflation from q to q+4"
}

*******************************
*******  Population moments  **
*******************************


foreach var in PCE CPI CORECPI COREPCE{
egen `var'_std = sd(`var'1y), by(year quarter)
gen `var'_disg = `var'_std^2 
label var `var'_disg "disagreements of `var'"
egen `var'_ct50 = median(`var'1y), by(year quarter) 
label var `var'_ct50 "Median of `var'"
}


save InfExpSPFPointIndQ,replace

collapse (mean) PCE1y CPI1y CORECPI1y COREPCE1y ///
                PCE_disg CPI_disg CORECPI_disg COREPCE_disg ///
				PCE_ct50 CPI_ct50 CORECPI_ct50 COREPCE_ct50, by(year quarter month)

label var PCE1y "1-yr-ahead PCE inflation"
label var CPI1y "1-yr-ahead CPI inflation"
label var COREPCE1y "1-yr-ahead Core PCE inflation"
label var CORECPI1y "1-yr-ahead Core CPI inflation"

label var PCE_disg "Disagreements in 1-yr-ahead PCE inflation"
label var CPI_disg "Disagreements in 1-yr-ahead CPI inflation"
label var COREPCE_disg "Disagreements in 1-yr-ahead Core PCE inflation"
label var CORECPI_disg "Disagreements in 1-yr-ahead Core CPI inflation"


label var PCE_ct50 "Median 1-yr-ahead PCE inflation"
label var CPI_ct50 "Median 1-yr-ahead CPI inflation"
label var COREPCE_ct50 "Median 1-yr-ahead Core PCE inflation"
label var CORECPI_ct50 "Median 1-yr-ahead Core CPI inflation"

save InfExpSPFPointPopQ,replace  
