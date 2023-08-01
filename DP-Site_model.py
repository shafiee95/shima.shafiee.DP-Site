
###################################
# DP-Site_model.py
# DP-Site: A Dual deep learning-based method for Protein-peptide binding Site prediction
# Shima Shafiee
# shafiee.shima@razi.ac.ir
###################################
"""Processing for predicting Peptide-binding residues"""
# get all library version
import tensorflow as tf
import numpy as np
import imblearn as im
import pandas as pd
import sklearn as sk
import skopt as skpt
import mlxtend as mxt
import matplotlib as matplot

def get_feature_extraction_batch(layer_extract,data,dim_ext):
    N=data.shape[0]
    lst=list(range(N))
    batch_idx=[lst[i:i + batch_size_ext] for i in range(0, N, batch_size_ext)]
    feature_extract=np.array([]).reshape(0,dim_ext)
    for item_idx in batch_idx:
      feature_new=layer_extract(data[item_idx])
      feature_extract=np.concatenate((feature_extract,feature_new),axis=0)
    return feature_extract
from tensorflow import keras
from keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier

from tensorflow import keras
from tensorflow.keras.models import Model

# create CNN model 
def create_model(learning_rate,num_filters1,num_filters2,
                 num_dense1_nodes,num_dense2_nodes, activation_name,drop_value,data_kf):
  input_size=data_kf.shape[1],data_kf.shape[2],1
  model= keras.models.Sequential()
  model.add(keras.layers.Input(shape=(input_size), name='input_layer'))
  model.add(keras.layers.Conv2D(filters=num_filters1,
                                    kernel_size=(2),strides=(1),
                            padding="same",name="conv1"))
  model.add(keras.layers.BatchNormalization(name="BN1"))
  model.add(keras.layers.AveragePooling2D(pool_size=(2),name="pool1"))
  model.add(keras.layers.Conv2D(filters=num_filters2,
                                    kernel_size=(2),strides=(1),
                            padding="same",name="conv2"))
  model.add(keras.layers.BatchNormalization(name="BN2"))
  model.add(keras.layers.MaxPool2D(pool_size=(2),name="pool2"))

  model.add(keras.layers.Flatten(name="flt"))
  model.add(keras.layers.Dense(num_dense1_nodes,activation=activation_name))
  model.add(keras.layers.Dropout(drop_value))
  model.add(keras.layers.Dense(num_dense2_nodes,activation=activation_name))
  model.add(keras.layers.Dropout(drop_value))
  model.add(keras.layers.Dense(32,activation=activation_name))
 # model.add(keras.layers.Softmax())
  model.add(keras.layers.Dense(2,activation="softmax",name="outlayer"))
  optimer_new=keras.optimizers.Adam(learning_rate=learning_rate)
  model.compile(optimizer=optimer_new,loss=keras.losses.categorical_crossentropy, metrics=merics_name)
  return Pipeline1
# Pipeline1 or PDCNN
#--------------------LSTM_RF-------------------------------#
def LSTM_RF(model_lstm,x_train,y_train,x_test):
  RF = RandomForestClassifier(max_depth=10, random_state=0)
  extract_layer_LSTM = tf.keras.models.Model(inputs=model_lstm.inputs,
                                            outputs=model_lstm.get_layer(name="lstm2").output)
  dim_lstm=model_lstm.get_layer(name="lstm2").output.shape[1]
  feature_Train=get_feature_extraction_batch(extract_layer_LSTM,x_train,dim_lstm)
  feature_Test=get_feature_extraction_batch(extract_layer_LSTM,x_test,dim_lstm)


  RF.fit(feature_Train, y_train)
  Pipeline2=np.array(RF.predict_proba(feature_Test))
  return Pipeline2
# Pipeline2 or PDLSTM+RF
#--------------------CNN_LSTM_RF(Proposed)-------------------------------#

def CNN_LSTM_RF(model_cnn,model_lstm,x_train,y_train,x_test):
  RF = RandomForestClassifier(max_depth=10, random_state=0)
  extract_layer_CNN = tf.keras.models.Model(inputs=model_cnn.inputs,
                                            outputs=model_cnn.get_layer(name="outlayer").output)
  extract_layer_LSTM = tf.keras.models.Model(inputs=model_lstm.inputs,
                                             outputs=model_lstm.get_layer(name="lstm2").output)
  dim_lstm=model_lstm.get_layer(name="lstm2").output.shape[1]
  dim_cnn=model_cnn.get_layer(name="outlayer").output.shape[1]
  feature_LSTM_Train=get_feature_extraction_batch(extract_layer_LSTM,x_train,dim_lstm)
  feature_LSTM_Test=get_feature_extraction_batch(extract_layer_LSTM,x_test,dim_lstm)

  RF.fit(feature_LSTM_Train, y_train)
  PLSTMRF=np.array(RF.predict_proba(feature_LSTM_Test))

  CNN_test_extract=get_feature_extraction_batch(extract_layer_CNN,x_test,dim_cnn)
  PCNNS=np.array(CNN_test_extract)
  PENCLS_R = (a*PCNNS + b*PLSTMRF )
  #lbl_cnn_lstm=np.argmax(PENCLS_R,axis=1)
  return PENCLS_R

def CNN_LSTM(model_cnn,model_lstm,x_train,y_train,x_test):
  RF = RandomForestClassifier(max_depth=10, random_state=0)
  extract_layer_CNN = tf.keras.models.Model(inputs=model_cnn.inputs,
                                            outputs=model_cnn.get_layer(name="outlayer").output)
  extract_layer_LSTM = tf.keras.models.Model(inputs=model_lstm.inputs,
                                             outputs=model_lstm.get_layer(name="outlayer").output)
  dim_lstm=model_lstm.get_layer(name="outlayer").output.shape[1]
  dim_cnn=model_cnn.get_layer(name="outlayer").output.shape[1]

  feature_LSTM_Test=get_feature_extraction_batch(extract_layer_LSTM,x_test,dim_lstm)
  CNN_test_extract=get_feature_extraction_batch(extract_layer_CNN,x_test,dim_cnn)

  PLSTM=np.array(CNN_test_extract)
  PCNNS=np.array(feature_LSTM_Test)
  PENCL = (0.481*PCNNS + 0.519*PLSTM )
  #lbl_cnn_lstm=np.argmax(PENCLS_R,axis=1)
  return PENCL
# PCOM-DCDLR or PENCL
#--------------------Hyper-parameters-------------------------------#
from tensorflow import keras
lstm_bt=500
lstm_epocs=100
cnn_bt=500
cnn_epocs=100
Resnet_bt=100
Resnet_epocs=100
batch_size_ext=1000
lr_lstm=0.01 
lr_cnn=0.01  
loss_lstm=keras.losses.categorical_crossentropy
loss_cnn=keras.losses.categorical_crossentropy
a=0.481
b=0.519
from tensorflow.keras.callbacks import EarlyStopping

#---------patience_early_stopping----------
patience_early_stopping = EarlyStopping(
    monitor='val_kappa',
    patience=20,
    min_delta=0.0001)

#---------custom_early_stopping----------
class stop_early(keras.callbacks.Callback):
  def __init__(self):
    super(stop_early,self).__init__()
  def on_epoch_end(self,epoch,logs=None):
    if(epoch>=40  and logs["val_kappa"]<0.10):
      self.model.stop_training = True
custom_early_stopping=stop_early()

#---------KFold-----------------------
#KFold used to avoid overfitting
from sklearn.model_selection import KFold,StratifiedKFold
import tensorflow as tf
import os

if( not os.path.isdir(path) ):
  os.mkdir(path)
Row=-1
c=0
kf = StratifiedKFold(n_splits=cv,shuffle=True,random_state=0)
for train_idx, test_idx in kf.split(data_image,lbl_new):
  c+=1
  train_data, test_data = data_image[train_idx], data_image[test_idx]
  train_lbl, test_lbl = lbl_new[train_idx], lbl_new[test_idx]

  if(train_data.shape[0]+1==count_train_kf):
    train_data=np.concatenate((train_data,train_data[-1].reshape(1,win_size,-1)),axis=0)
    train_lbl=np.append(train_lbl,train_lbl[-1])

  if(test_data.shape[0]+1==count_test_kf):
    test_data=np.concatenate((test_data,test_data[-1].reshape(1,win_size,-1)),axis=0)
    test_lbl=np.append(test_lbl,test_lbl[-1])

  print("_"*100)
  print("TRAIN:", train_data.shape[0], "TEST:", test_data.shape[0])
  y_train_bin = tf.keras.utils.to_categorical(train_lbl, 2)
  y_test_bin = tf.keras.utils.to_categorical(test_lbl, 2)

#We trained and optimized different learning-based methods. The best method is CNN-LSTM-RF.
cv=10;
count_train_kf= int(np.ceil((1-(1/cv))*data_image.shape[0]))
count_test_kf= int(np.ceil(((1/cv))*data_image.shape[0]))
algorithm_name=["LSTM","CNN","CNN_LSTM_RF","LSTM_RF","CNN_LSTM"]
zeros_data=np.zeros((len(algorithm_name)*cv,16));
res_metrics_train=pd.DataFrame(zeros_data,columns=["algorithm_name","accuracy","precision","auc",
                                                   "sensitivity","f1_score","Mathews Correlation Coefficient",
                                                   "specificity","TN","FP","FN","TP","FPR","FNR","k"])
                                                   

res_metrics_train["algorithm_name"]=algorithm_name*cv
res_metrics_test=res_metrics_train.copy()

res_metrics_test


