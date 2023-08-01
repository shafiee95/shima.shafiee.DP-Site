###################################
# Feature groups.py
#DP-Site: A Dual deep learning-based method for Protein-peptide binding Site prediction
# Shima Shafiee
#shafiee.shima@razi.ac.ir
###################################
import numpy as np
import pandas as pd
import sys
import string
import math
import os
import numpy
import scipy
import scipy.io
import cPickle
import pickle


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
#---------------------------------------------------
def compute_entropy(dis_list):
    """compute shannon entropy for a distribution.
    base = len(dis_list) is the base of log function 
    to make entropy between 0 and 1."""
    
    if sum(dis_list) == 0:
        return 0.0
    prob_list = map(lambda x:(x+0.0)/sum(dis_list),dis_list)
    ent = 0.0
    for prob in prob_list:
        if prob != 0:
            ent -= prob*math.log(prob,len(dis_list))
    return ent
#---------------------------------------------------
fin = file(rsa_path+pid+'.spd2','r')
sp = fin.readlines()[1:]
fin.close()
rsa_res = ''.join([x.split()[1] for x in sp])
rsa_pre = [string.atof(x.split()[6]) for x in sp]
fin = file(pssm_path+pid+'.pssm','r')
pssm = fin.readlines()
fin.close()
if len(pssm[-6].split()) != 0 or pssm[3].split()[0] != '1': 
    print 'error on reading pssm, line -6 is not a spare line;\
     or line 3 is not the first line'
    sys.exit(1)
pssm = pssm[3:-6]
fin = file(fasta_path+pid+'.seq','r')
ann = fin.readlines()
fin.close()
if len(ann) != 2:
    print 'check sequence',pid
    sys.exit(1)
fastaseq = ann[1].split()[0]
if not fastaseq == rsa_res:
    print 'Sequence inconsistent!'
    print 'fasta: ',fastaseq
    print '  rsa: ',rsa_avg
    exit(1)
fout = file(fea_path+pid+'.info','w')
fout.write('>%s\n' %pid)
pos = 0
for i in xrange(len(fastaseq)):
    res = fastaseq[i]
    fout.write('%5d%5s%5s'%(i+1,res,res))
    if pssm[pos].split()[1] == res:# check for residue type
        for p_e in pssm[pos].split()[2:22]:
            fout.write(':%2s' %p_e)
        for p_e in pssm[pos].split()[22:42]:
            fout.write(':%3s' %p_e)
        fout.write(':%5s' %pssm[pos].split()[42])
    else:
        print 'Error reading pssm file!'
        flog = file(error_file,'a')
        flog.write(pid+': error on writing pssm, %s:%s\n' \
        %(pssm[pos].split()[1],res))
        flog.close()
        sys.exit(1)
    if rsa_res[pos] == res:
        fout.write(':%5.3f' %rsa_pre[pos])
    else:
        print 'Error reading rsa file!'
        flog = file(error_file,'a')
        flog.write(pid+': error on writing rsa, %s:%s\n' %(rsa_res[pos],res))
        flog.close()
        sys.exit(1)
    pos += 1
    fout.write('\n')
fout.close()
#---------------------------------------------------
def compute_ss_content(ss_seq_win):
    """compute ss content in a window."""
    con_C = con_H = con_E = 0
    for ss in ss_seq_win:
        if ss == 'C':
            con_C += 1
        elif ss == 'H':
            con_H += 1
        elif ss == 'E':
            con_E += 1
        else:
            print('X')
             
    act_len = con_C+con_H+con_E+0.0
    return ['%.3f'%(con_C/act_len),'%.3f'%(con_H/act_len),'%.3f'%(con_E/act_len)]

def ss_binary(ss_type):
    binary = []
    for ss in ss_type:
	if ss == 'C':
	   binary.append('1.000')
           binary.append('0.000')
           binary.append('0.000')
	elif ss == 'H':
	   binary.append('0.000')
	   binary.append('1.000')
	   binary.append('0.000')
	elif ss == 'E':
	   binary.append('0.000')
	   binary.append('0.000')
	   binary.append('1.000')
	else:
	   binary.append('0.000')
	   binary.append('0.000')
	   binary.append('0.000')
    return binary

def ss_to_num(sin_ss):
    """C->0,H->1,E->2,'$'->-1"""
    if sin_ss == 'C':
        return 0
    elif sin_ss == 'H':
        return 1
    elif sin_ss == 'E':
        return 2
    else:
        return -1

def seg_bound(s_win):
    """Two boundaries of a segment"""
    c_ss = s_win[len(s_win)/2]
    l_len = r_len = 0
    i = len(s_win)/2 - 1
    while i >= 0:
        if s_win[i] != c_ss:
            break
        l_len += 1
        i -= 1
    i = len(s_win)/2 + 1
    while i < len(s_win):
        if s_win[i] != c_ss:
            break
        r_len += 1
        i += 1
    return (l_len,r_len)
#---------------------------------------------------
pdb_residue_temp = []
pdb_residue = []
all_coord = []
all_X = []
all_Y = []
all_Z = []
b_factor = []
pdbfile = []
all_residue_number = []
residue_number = []
#Angles are {θ, τ, φ, and ψ}
with open('your_pdb_directory'+pid+'.pdb') as pdb_file:
	for line in pdb_file:
		if line[:4] == 'ATOM' or line[:6] == "HETATM" and line:
			temp_pdbfile = ('%s%5.2f'%(line[:55].rstrip(),1.00))
			pdbfile.append(temp_pdbfile)
			all_residue_number.append(line[22:26])
			if line[12:16] == ' CA ':
				residue_number.append(line[22:26])
				pdb_residue_temp.append(line[17:20])
				all_coord.append([line[30:38], line[38:46], line[46:54]])
				all_X.append(line[30:38])
				all_Y.append(line[38:46])
				all_Z.append(line[46:54])
				b_factor.append(line[57:60])
all_residue_number = [x.replace(' ', '') for x in all_residue_number]
for i in xrange(len(pdb_residue_temp)):
	pdb_residue.append(ext_pdb_residue[pdb_residue_temp[i]])
#---------------------------------------------------------
for i in xrange(win):
    wop.insert(0,'%7.5f' %(0))
    wop.append('%7.5f' %(0))
#PP = {steric parameters, polarizability, helix probability, hydrophobicity, volume, isoelectric point, sheet probability}
for i in xrange(P7_win,seq_len):
    for j in xrange(i-P7_win,i+P7):
        out_list[i-P7_win].append(PP[j])
#---------------------------------------------------------
for i in xrange(win):
    rt.insert(0,'X')
    rt.append('X') 
for i in xrange(len_win):
    rsa.insert(0,'1.000')
    rsa.append('1.000')
for i in xrange(len(out_list)):
    output.write(','.join(out_list[i])+'\n')
output.close()

#in case of Test.Set True in case of Train.Set False
isTest = False
if isTest:
    filename = "Test.Set.txt"
else:#Train
    filename = "Train.Set.txt"

# Read file in a numpy array
mat = np.loadtxt(filename,dtype=str)

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

#Bioinformatics Tools:

1.Taherzadeh, G., et al., Structure-based prediction of protein–peptide binding regions using Random Forest. Bioinformatics, (2018). 34(3): p. 477-484.

2.Taherzadeh, G., et al., Sequence‐based prediction of protein–peptide binding sites using support vector machine. Journal of computational chemistry, (2016). 37(13): p. 1223-1229.

3.Altschul, S. F., Madden, T. L., Schäffer, A. A., Zhang, J., Zhang, Z., Miller, W., &amp; Lipman, D. J. (1997).“Gapped BLAST and PSI-BLAST: a new generation of protein database search programs.” In Nucleic acidsresearch, 25, 3389-3402. https://doi.org/10.1093/nar/25.17.3389.

4.Yang.Y, Heffernan.R, Paliwal.K, Lyons.J, Dehzangi.A, et al., (2017). &quot;Spider2: A package to predict secondary structure, accessible surface area, and main-chain torsional angles by deep neural networks,&quot; In
Prediction of Protein Secondary Structure. DOI: 10.1007/978-1-4939-6406-2_6.

#---------------------------------------------------



