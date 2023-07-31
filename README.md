# shima.shafiee.DP-Site
# DP-Site
** Sequence-based prediction of protein-peptide residue-level interaction using a dual deep learning-based method. **\

Please cite the relevant publication if you will use this work.

**Citation: ** S. Shafiee, A. Fathi and G. Taherzadeh, DP-Site: A Dual deep learning-based method for Protein-peptide binding Site prediction.

## Data Files

-Protein-peptide datasets are including

- Test.Set.txt

- Train.Set.txt

- Test.Labels.txt

- Train.Labels.txt

- Subset of data

- Subset of data2

## Guide
- Three lines for each sample include the seq label, the chain sequence, and the enriched binding annotation:  0=non-binding,1=binding, respectively.
- Two files contain labels for the Test. Set and Train. Set.
- Subset of data and subset of data2 are two files that are a subset of the original dataset.
- The original dataset is relying on a residue by residue. Thus, two files contain part of the used dataset which is employed for training and testing of the proposed method (DP-Site).

## Code Files

- DP-Site_model.py 
- Feature groups.py
- Pre-processing.py

## Guide
DP-Site_model.py is the main file.
To run, all the above files should provide in one folder.
Run in Linux: python DP-Site_model.py to identify the peptide binding regions in proteins.

 ## Contact
For further details or questions, it is possible to communicate through email (shafiee.shima@razi.ac.ir)

Best regards
