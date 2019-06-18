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

merge m:1 year quarter ID using "${folder}/SPF/individual/InfExpSPFDstIndQ.dta"


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

*******************************
**  Generate Variables       **
*******************************




log close 
