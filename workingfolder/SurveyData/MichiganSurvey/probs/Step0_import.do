*******************************************************************
*  Stata "do-file" file with labels and missing data specifications
*  Created by ddltox on Mar 25, 2019  (Mon 08:38 PM EDT)
*  DDL source file: "/z/sca-v2/sda/public/htdocs/tmpdir/AA2hI81K.txt".
*
*  Note that the data dictionary is given at the end of this file.
*  Put the dictionary into a separate file (by editing this file).
*  Then specify below the name of the dictionary file.
*
*  DDL file gives the following dataset description:
*    Records per case: 1
*    Record length:    61
*******************************************************************
clear

label data "Surveys of Consumers"

#delimit ;
label define PINC      998 "DK" 999 "NA" ;
label define PINC2     996 "Volunteered 'No personal income'" 998 "DK" 
                       999 "NA" ;
label define PJOB      998 "DK" 999 "NA" ;
label define PSSA      998 "DK" 999 "NA" ;
label define PCRY      1 "Gone up" 3 "Same" 5 "Gone down" 8 "DK" 9 "NA" ;
label define PSTK      998 "DK" 999 "NA" ;


#delimit cr

*******************************************************************
infile using dictionary
* Replace 'X' with the name of the dictionary file. 
*
* The contents of the dictionary are given at the end of this file.
* Put the dictionary into a separate file (by editing this file).
* Then specify here the name of the dictionary file.
*******************************************************************
* The md, min and max specifications were translated 
* into the following "REPLACE...IF" statements:

replace PINC = . if (PINC >= 998.00 ) 
replace PINC2 = . if (PINC2 >= 998.00 ) 
replace PJOB = . if (PJOB >= 998.00 ) 
replace PSSA = . if (PSSA >= 998.00 ) 
replace PCRY = . if (PCRY >= 8 ) 
replace PSTK = . if (PSTK >= 998.00 ) 

save prob.dta,replace 
