###################################
# Pre-processing.py
#DP-Site: A Dual deep learning-based method for Protein-peptide binding Site prediction
# Shima Shafiee
#shafiee.shima@razi.ac.ir
###################################
"""Pre-processing for predicting Peptide-binding residues"""
#Library
# get all library version
import tensorflow as tf
import numpy as np
import imblearn as im
import pandas as pd
import sklearn as sk
import skopt as skpt
import mlxtend as mxt
import matplotlib as matplot

print("tensorflow version: " ,tf.__version__)
print("numpy      version: " ,np.__version__)
print("imblearn   version: " ,im.__version__)
print("pandas     version: " ,pd.__version__)
print("sklearn    version: " ,sk.__version__)
print("skopt      version: " ,skpt.__version__)
print("mlxtend    version: " ,mxt.__version__)
print("matplotlib version: " ,matplot.__version__)

#Read Data
import pandas as pd
# read train and test data by pandas
train_pd=pd.read_csv("train.csv",delimiter=",")
test_pd=pd.read_csv("test.csv",delimiter=",")
train_pd.columns

#Feature extraction
# Run Feature groups.py
#columns_delete=["# name","no","AA","Label"]

#Pre-processing
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Interval is [0-1]

#------------- clean---------------------
# remove some columns of train and test dataset
# train
lbl_train=np.array(train_pd["Label"],dtype=int)
train_pre=train_pd.drop(columns=columns_delete)
# test
lbl_test=np.array(test_pd["Label"],dtype=int)
test_pre=test_pd.drop(columns=columns_delete)

#------------- drop and fill nan------------
#***train
indices_to_keep = ~train_pre.isin([np.nan, np.inf, -np.inf]).any(1)
train_pre=train_pre[indices_to_keep].astype(np.float64)
lbl_train=lbl_train[indices_to_keep].astype(np.float64)
#***test
indices_to_keep = ~test_pre.isin([np.nan, np.inf, -np.inf]).any(1)
test_pre=test_pre[indices_to_keep].astype(np.float64)
lbl_test=lbl_test[indices_to_keep].astype(np.float64)
train_pre=train_pre.fillna(0)
test_pre=test_pre.fillna(0)
data_train=np.array(train_pre,dtype=float)
data_test=np.array(test_pre,dtype=float)

#------------- join train and test--------
Data=np.concatenate((data_train,data_test),axis=0)
lbl=np.concatenate((lbl_train,lbl_test),axis=0)

#---------------normalization ------------
method_norm= MinMaxScaler()
if(normal_type==1):
  method_norm=StandardScaler()
Data_normal=method_norm.fit_transform(Data)

#------------- show----------------------
print("number of sample: ",Data_normal.shape[0] )
print("number of features: ",Data_normal.shape[1] )
print("number of class one: ",np.sum(lbl==0) )
print("number of class two: ",np.sum(lbl==1)  )

x_train=Data_normal[0:data_train.shape[0],:]
x_test=Data_normal[data_train.shape[0]:,:]
lbl_train=lbl[0:data_train.shape[0]]
lbl_test=lbl[data_train.shape[0]:]

print("x_train shape  : ",x_train.shape )
print("x_test shape   :  ",x_test.shape )
print("lbl_train shape: ",lbl_train.shape)
print("lbl_test shape :  ",lbl_test.shape  )

#------------- Over Sampling|Under Sampling----------------------
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import CondensedNearestNeighbour

types_sampling=0 # 0 is under_sampling, 1 is over_sampling
neighbors_sampling=1000;

#-------------  Befor Sampling-----------------------------------
print("-"*10,"Befor Sampling","-"*10)
print("number of sample: ",x_train.shape[0] )
print("number of features: ",x_train.shape[1] )
print("number of class one: ",np.sum(lbl_train==0) )
print("number of class two: ",np.sum(lbl_train==1)  )

#---------over_sampling------------------------------------------
def over_sampling(data,lbl,k=100):
    smote = SMOTE(k_neighbors=k)
    X_sm, y_sm = smote.fit_resample(data, lbl)
    return X_sm, y_sm

#---------under_sampling-----------------------------------------
def under_sampling(data,lbl,k=100):
  undersample = NearMiss(version=1, n_neighbors=k)
  #undersample = CondensedNearestNeighbour(n_neighbors=100)
  X_sm, y_sm= undersample.fit_resample(data, lbl)
  return X_sm, y_sm

#---------sampling method-----------------------------------------
if(types_sampling==0): # over_sampling
  data_sampling,lbl_sampling=over_sampling(x_train,lbl_train,neighbors_sampling)
else: # under_sampling
  data_sampling,lbl_sampling=under_sampling(x_train,lbl_train,neighbors_sampling)

#------------- After Sampling-------------------------------------
print()
print("-"*10,"After Sampling","-"*10)
print("number of sample: ",data_sampling.shape[0] )
print("number of features: ",data_sampling.shape[1] )
print("number of class one: ",np.sum(lbl_sampling==0) )
print("number of class two: ",np.sum(lbl_sampling==1)  )

np.savetxt("Data_normal.csv",Data_normal,delimiter=",")

#------------- Image-based transformation ------------------------
win_size=7
from matplotlib import pyplot as plt
def create_window_protein(data,label,win_size):
  n=data.shape[0]
  center=win_size//2
  x,y=[],[]
  for i in range(data.shape[0]):
    if(i<center):
      fwd=list(range(0,center+i+1))
      back=[]
      for k in range(win_size-len(fwd)):
        back.insert(0,fwd[k+i+1])
      back.extend(fwd)
      item_window=data[back,:]
      item_lbl=label[i]
    elif(i>=n-center):
      back=list(range(i-center,data.shape[0]))
      fwd=[]
      for k in range(win_size-len(back)):
        fwd.append(back[center-k-1])
      back.extend(fwd)
      item_window=data[back,:]
      item_lbl=label[i]

    else:
      item_window=data[i-center:i+center+1,:]
      item_lbl=label[i]
    x.append(item_window)
    y.append(item_lbl)
  x=np.array(x)
  y=np.array(y)
  return x,y

x_train_image,lbl_new_train=create_window_protein(x_train,lbl_train,win_size)
#-------------  After image----------------------
print()
print("-"*10,"After iamge","-"*10)
print("number of sample: ",x_train_image.shape[0] )
print("data image size ",x_train_image.shape[1:] )

#-----------Input data---------------------------

print("---------one sample image data show---------")
axs = plt.figure(figsize=(44,7))
plt.imshow(x_train_image[1793], cmap='gray')
plt.yticks(range(0,win_size+1,3))
plt.xlabel("number of features")
plt.ylabel("window size")
plt.show()

x_test_image,lbl_new_test=create_window_protein(x_test,lbl_test,win_size)
#-------------  After image-----------------------
print()
print("-"*10,"After iamge","-"*10)
print("number of sample: ",x_test_image.shape[0] )
print("data image size ",x_test_image.shape[1:] )

#-----------Input data----------------------------

print("---------one sampl image data show---------")
axs = plt.figure(figsize=(44,7))
plt.imshow(x_test_image[0], cmap='gray')
plt.yticks(range(0,win_size+1,3))
plt.xlabel("number of features")
plt.ylabel("window size")
plt.show()

data_image=np.concatenate((x_train_image,x_test_image),axis=0)
lbl_new=np.concatenate((lbl_new_train,lbl_new_test),axis=0)


#/*************************************************/



