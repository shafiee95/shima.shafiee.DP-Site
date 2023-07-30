import numpy as np
import pandas as pd

#in case of Test.Set True in case of Train.Set False
isTest = False
if isTest:
    filename = "Test.Set.txt"
else:#Train
    filename = "Train.Set.txt"

# Read file in a numpy array
mat = np.loadtxt(filename,dtype=str)

#Add 20 columns for 20d ss feature
mat = np.hstack((mat, np.zeros((mat.shape[0], 20),dtype=np.float64)))

#20d aminoacids calculation. This part needs python version 3.10
for i,amino in enumerate(mat[:,2]):
    match amino:
        case 'A':
            mat[i,149]=1
        case 'R':
            mat[i,150]=1
        case 'N':
            mat[i,151]=1
        case 'D':
            mat[i,152]=1
        case 'C':
            mat[i,153]=1
        case 'Q':
            mat[i,154]=1
        case 'E':
            mat[i,155]=1
        case 'G':
            mat[i,156]=1
        case 'H':
            mat[i,157]=1
        case 'I':
            mat[i,158]=1
        case 'L':
            mat[i,159]=1
        case 'K':
            mat[i,160]=1
        case 'M':
            mat[i,161]=1
        case 'F':
            mat[i,162]=1
        case 'P':
            mat[i,163]=1
        case 'S':
            mat[i,164]=1
        case 'T':
            mat[i,165]=1
        case 'W':
            mat[i,166]=1
        case 'Y':
            mat[i,167]=1
        case 'V':
            mat[i,168]=1


if isTest:
 
    output = "Test.Set.txt"
else:
   
    output = "Train.Set.txt"
with open(output,"w+") as out:
    head = 'name	no	AA	SS  P(C)	P(E)	P(H)	C_SegLen	E_SegLen	H_SegLen  C_min	C_max	E_min	E_max	H_min	H_max   Phi   Psi	 Theta(i-1=>i+1)	Tau(i-2=>i+1)	 A	R	N	D	C	Q	E	G	H	I	L	K	M	F	P	S	T	W	Y	V	P1	P2	P3	P4	P5	P6	P7	ASA  rASA	rASA_avg   CNCC-4	CNCC-3	CNCC-2	CNCC-1	CNCC-0   CNCC+1	CNCC+2	CNCC+3	CNCC+4	Entropy  AAO  Label'  
    np.savetxt(out,mat,delimiter='\t',header=head,fmt='%s')

#---------------------------------------------------
#SS  P(C)	P(E)	P(H)	C_SegLen	E_SegLen	H_SegLen  C_min	C_max	E_min	E_max	H_min	H_max are for SS group
#Phi   Psi	 Theta(i-1=>i+1)	Tau(i-2=>i+1) are for LBA
#A	R	N	D	C	Q	E	G	H	I	L	K	M	F	P	S	T	W	Y	V	are for PSSM group
#P1	P2	P3	P4	P5	P6	P7	are for PP group
#ASA  rASA	rASA_avg are for ASA group
#CNCC-4	CNCC-3	CNCC-2	CNCC-1	CNCC-0   CNCC+1	CNCC+2	CNCC+3	CNCC+4	Entropy  are for CNCC and Entropy
#AAO is a binary representation for each residue
#---------------------------------------------------

