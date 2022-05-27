#!/usr/bin/env python
# coding: utf-8

# # Quant Modeling
# 5/16/2022  
# Lei G Renmin Univ. of China

# ### Import packages and setting up 

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeClassifier
from dtreeviz.trees import dtreeviz
from keras_visualizer import visualizer 
from sklearn import tree
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ### 1. Data Processsing

# In[2]:


df = pd.read_csv('car_prices.csv',on_bad_lines='skip')
df=df.dropna(subset=['sellingprice'])

df_Train,df_Test=train_test_split(df,test_size=0.3,random_state=123)

def fun_ann_prepare_train(Train_a,var_sale_knn,target):
    xtrain_knn=Train_a[var_sale_knn]
    print('Sample Size train: {}'.format(xtrain_knn.shape))
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    imp_mean_knn=SimpleImputer(missing_values=np.nan,strategy='mean')
    scaler_knn=StandardScaler()
    imp_mean_knn.fit(xtrain_knn)
    xtrain_knn=pd.DataFrame(imp_mean_knn.transform(xtrain_knn)
                            ,index=xtrain_knn.index,
                            columns=xtrain_knn.columns)
    scaler_knn.fit(xtrain_knn)
    xtrain_knn=pd.DataFrame(scaler_knn.transform(xtrain_knn)
                            ,index=xtrain_knn.index,
                            columns=xtrain_knn.columns)
    return xtrain_knn,imp_mean_knn,scaler_knn

def fun_ann_prepare_score(Test_a,var_sale_knn,
                        target,imp_mean_knn,scaler_knn):
    xtest_knn=Test_a[var_sale_knn]
    print('Sample Size score: {}'.format(xtest_knn.shape))
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    xtest_knn=pd.DataFrame(imp_mean_knn.transform(xtest_knn)
                            ,index=xtest_knn.index,
                           columns=xtest_knn.columns)
    xtest_knn=pd.DataFrame(scaler_knn.transform(xtest_knn)
                            ,index=xtest_knn.index,
                           columns=xtest_knn.columns)
    return xtest_knn


target='sellingprice'
ytrain_ann=df_Train[target].copy()
ytest_ann=df_Test[target].copy()
print(ytrain_ann.shape,ytest_ann.shape)
var_ann=['year', 'condition', 'odometer', 'mmr' ]

xtrain_ann,imp_mean_ann,scaler_ann=fun_ann_prepare_train(
    df_Train,var_ann,target)

xtest_ann=fun_ann_prepare_score(df_Test,var_ann,
                          target,imp_mean_ann,scaler_ann)


# ### 2. Model: ANN

# In[3]:


get_ipython().run_cell_magic('time', '', 'K.clear_session()\nepochs=30\nbatch_size=128\nmodel_ann = Sequential()\noptimizer = keras.optimizers.Adam(lr=0.1)\ncallback = tf.keras.callbacks.EarlyStopping(monitor=\'val_loss\', \n                                            patience=5,verbose=1,\n                                            restore_best_weights=True)\nmodel_ann.add(Dense(8,activation = \'relu\',\n                    input_dim = len(xtrain_ann.columns)))\nmodel_ann.add(BatchNormalization())\nmodel_ann.add(Dense(4,activation = \'relu\'))\nmodel_ann.add(BatchNormalization())\nmodel_ann.add(Dense(1,activation =\'linear\'))\nmodel_ann.compile(optimizer =optimizer,loss = \'mse\',metrics = [\'mae\'])\nmodel_ann.summary()\n\nkeras.utils.plot_model(model_ann, show_shapes=True, rankdir="TB",dpi=200)\n\nnp.random.seed(666)\ntf.random.set_seed(666)\nwith tf.device(\'/cpu:0\'):\n    model_ann.fit(xtrain_ann,ytrain_ann,\n             batch_size=batch_size,epochs=epochs,\n                  validation_split=0.1,validation_freq=1,\n              verbose=2,callbacks=[callback])\nypred_ann = model_ann.predict(xtest_ann,batch_size = 32)\n')


# ### 2. Lasso Model

# In[4]:


reg_Lasso=Lasso(alpha=0.5)
reg_Lasso.fit(xtrain_ann,ytrain_ann)
ypred_lasso = reg_Lasso.predict(xtest_ann)


# ### 3. XgBoost Model

# In[5]:


get_ipython().run_cell_magic('time', '', "hyperparameters_xgb={'n_estimators':200,'learning_rate':0.1,\n                     'max_depth':5,'n_jobs':-1}\nreg_xgb=XGBRegressor(**hyperparameters_xgb)\nevals_result = {}\n")


# In[10]:


reg_xgb.fit(xtrain_ann,ytrain_ann,    
            early_stopping_rounds=5,
            verbose=10,
            eval_metric='rmse',
            callbacks=[xgb.callback.record_evaluation(evals_result)])
ypred_xgb = reg_xgb.predict(xtest_ann)


# ### 4. GradientBoosting from the Sklearn

# In[11]:


get_ipython().run_cell_magic('time', '', "hyperparameters_gb={'n_estimators':200,'learning_rate':0.1,'max_depth':5}\nreg_gb=GradientBoostingRegressor(**hyperparameters_gb)\nreg_gb.fit(xtrain_ann,ytrain_ann)\nypred_gb = reg_gb.predict(xtest_ann)\n\nplt.figure(figsize=(20,10)) \ntree.plot_tree(reg_gb.estimators_[5][0],max_depth=9,\n               filled=True,feature_names=xtrain_ann.columns,\n               rounded=True,precision=1)\nplt.show()\n")


# ### 5. Performance Comparison Between Difference Models

# In[12]:


df_Test['pred_price_ann']=ypred_ann
df_Test['pred_price_lasso']=ypred_lasso
df_Test['pred_price_xgb']=ypred_xgb
df_Test['pred_price_gb']=ypred_gb


# In[13]:


df_Test['error_ann']=(df_Test['pred_price_ann']-df_Test['sellingprice'])/df_Test['sellingprice']
df_Test['error_lasso']=(df_Test['pred_price_lasso']-df_Test['sellingprice'])/df_Test['sellingprice']
df_Test['error_xgb']=(df_Test['pred_price_xgb']-df_Test['sellingprice'])/df_Test['sellingprice']
df_Test['error_gb']=(df_Test['pred_price_gb']-df_Test['sellingprice'])/df_Test['sellingprice']


# In[14]:


df_Test['error_ann'].abs().mean()


# In[15]:


df_Test['error_xgb'].abs().mean()


# In[16]:


df_Test['error_gb'].abs().mean()


# In[17]:


df_Test['error_lasso'].abs().mean()


# In[ ]:




