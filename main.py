#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import cvxopt as opt

# df_project=pd.read_csv("C:/Users/12498/Quantify/project/project.csv")
# df_riskfactor=pd.read_csv("C:/Users/12498/Quantify/project/riskfactor.csv")
df_project=pd.read_csv("project.csv")
df_riskfactor=pd.read_csv("riskfactor.csv")
df_project.tail(1)


# **firm-level attributes (columns 4-11) are firm-level information (used as predictors) lagged by one month relative to column3, the excess return.**

# In[2]:


df_riskfactor.head(1)


# **these riskfactor do no lag!!**

# ## Step 1

# In[3]:


#insorize the monthly outliers
# 1.First, if any characteristic of a firm is missing, delete that firm-month observation.
df_project.isnull().any()
# So there are NaN


# In[4]:


def deletena(tmp_df):
    Series=tmp_df.isnull().any()

    for col in list(tmp_df):
        if Series[col]:
            indexs=list(tmp_df[tmp_df[col].isna()].index)
            tmp_df=tmp_df.drop(indexs)
            tmp_df.reset_index(drop=True,inplace=True)
    return tmp_df
df_project=deletena(df_project)

#2.Winsorize and standardize z-score
def winsorize(df):
    scaler = StandardScaler()
    date_list=list(set(df["date"]))
    date_list.sort()

    name_lst=list(df)
    del name_lst[0:3]
    stand_lst=list(df)
    del stand_lst[0:4]
    # remove date,permno,exret
    df_lst=[]
    for date in date_list:
        tmp_df=df[df["date"] == date]
        for name in name_lst:
            maxium_95=np.percentile(np.array(tmp_df[name]),99) # 95分位数
            miniun_5=np.percentile(np.array(tmp_df[name]),1) # 5分位数
            tmp_df[name]=tmp_df[name].apply(lambda x: x if x>=miniun_5 else miniun_5)
            tmp_df[name]=tmp_df[name].apply(lambda x: x if x<=maxium_95 else maxium_95)
        # standized
        tmp_df[stand_lst]=scaler.fit_transform(tmp_df[stand_lst])
        df_lst.append(tmp_df)
    df=pd.concat(df_lst)
    return df
df_project=winsorize(df_project)


# In[5]:


df_project.tail(1)


# In[6]:


# regression
date_list=list(set(df_project["date"]))
date_list.sort()
T = len(date_list)
coef = np.zeros((T, 8))

# create a dictionary for N stock so that we know what stock refer to index
permno_set= list(set(df_project["PERMNO"]))
permno_set.sort()
permno_dict={}
for index,item in enumerate(permno_set):
    permno_dict[item]=index

ita  = np.full((T, len(permno_set)),np.nan)

for t,date in enumerate(date_list):
    tmp_df = df_project[df_project["date"] == date]
    tmp_df.reset_index(drop=True,inplace=True)
    X = tmp_df.iloc[:,3:].astype(float) # regression without intercept
    y = tmp_df.iloc[:, 2]

    # Perform linear regression using statsmodels
    model = sm.OLS(y, X)
    results = model.fit()
    coef[t, :] = results.params[0:]
    
    residual=results.resid
    permno=list(tmp_df["PERMNO"])
    for name_index,name in enumerate(permno):
        index=permno_dict[name]
        ita[t,index]=residual[name_index]


# ## investment

# In[7]:


gmv_Exret=[]
star_Exret=[]

for Mt in range(120,T): # the reason why to T-1 is T is the last prediction date
    #################################### step 2 ##########################################
    tmp_coef_df=coef[0:Mt]
    tmp_coef_array=np.array(tmp_coef_df)
    tmp_theta_mean=tmp_coef_df.mean(axis=0) # 1*8 vector
    # step 2 result: tmp_theta_mean
    
    ################################### step 3 #############################################
    tmp_theta_cov=np.cov(tmp_coef_array,rowvar=False,bias=False)*Mt/(Mt-8) # 8*8 vector
    # step 3 result: tmp_theta_cov
    
    ################################### step 4 #############################################
    # filter the stock permno
    invest_stock_lst=set(df_project[df_project["date"]==date_list[Mt-1]]["PERMNO"])
    invest_stock_lst2=set(df_project[df_project["date"]==date_list[Mt]]["PERMNO"]) # next month the stock exist
    invest_stock_lst=invest_stock_lst.intersection(invest_stock_lst2)
#     print(len(invest_stock_lst))
    
    invest_stock_lst_res=[] # larger than 100 records
    tmp_df_project=df_project[df_project["date"]<=date_list[Mt-1]]
    count=tmp_df_project.groupby("PERMNO")["date"].count()
    for permno in invest_stock_lst:
        if count[permno]>=100:
            invest_stock_lst_res.append(permno)
    invest_stock_lst_res.sort()

#     print(len(invest_stock_lst_res))
    tmp_ita=ita[0:Mt]
    tmp_ita_cov=np.zeros((len(invest_stock_lst_res),len(invest_stock_lst_res))) # step 4 results
    for i,permno in enumerate(invest_stock_lst_res):
        index=permno_dict[permno]
        tmp_individual_ita=tmp_ita[:,index]
        tmp_individual_ita_var=np.nanstd(tmp_individual_ita)**2
        tmp_ita_cov[i,i]=tmp_individual_ita_var 
    # step 4 results tmp_ita_cov
    
    ############################### step 5 ##########################################
    tmp_prediction=[] # step 5 result it is a ndarray
    tmp_df_project=df_project[df_project["date"]==date_list[Mt]]
    for permno in invest_stock_lst_res:
        tmp_X=np.array(tmp_df_project[tmp_df_project["PERMNO"]==permno])[0,3:]
        tmp_pred=np.dot(tmp_X,tmp_theta_mean)
        tmp_prediction.append(tmp_pred)
    tmp_prediction=np.array(tmp_prediction)
   # step 5 result: tmp_prediction

   ################################ step 6 ##########################################   
    tmp_Big_X=np.array(tmp_df_project[tmp_df_project["PERMNO"].isin(invest_stock_lst_res)])[:,3:]
    tmp_conditional_var=tmp_Big_X @ tmp_theta_cov @ tmp_Big_X.T+tmp_ita_cov

    # step 6 :tmp_conditional_var
    ############################## step 7a #############################################
    n=tmp_conditional_var.shape[0]
    e = np.ones(n)
    w_gmv = np.linalg.inv(tmp_conditional_var).dot(e)
    w_gmv = w_gmv / np.sum(w_gmv)

    real_R=df_project[df_project["date"]==date_list[Mt]][["PERMNO","ExRet","Beta","logme"]]
    real_R=real_R[real_R["PERMNO"].isin(invest_stock_lst_res)]
    real_R.reset_index(drop=True,inplace=True)
    real_R.sort_values(by='PERMNO',inplace=True)
    
    ################## use this result in step 7b##############################
    beta=np.array(real_R["Beta"] )
    logme=np.array(real_R["logme"] )
    ##########################################################################
    
    real_R=real_R["ExRet"]
    
    R_gmv = w_gmv.dot(real_R)
    gmv_Exret.append(R_gmv)
    
    V_gmv = w_gmv.dot(tmp_conditional_var).dot(w_gmv)
    # result: R_gmv
    
    ############################ step 7b ###############################################
    # Solve the quadratic programming problem using cvxopt

    P = opt.matrix(tmp_conditional_var)
    q = opt.matrix(-tmp_prediction/7)
    G=np.zeros((2*n,n))
    for row in range(2*n):
        if row<n:
            G[row,row]=1
        else:
            G[row,row-n]=-1
    G=opt.matrix(G)
    h=opt.matrix(np.array([0.01 for i in range(2*n)]))
    A=np.zeros((3,n))
    b=opt.matrix(np.zeros(3))
    A[0,:]=1 # sum of weight equal to zero
    A[1,:]=beta
    A[2,:]=logme
    A=opt.matrix(A)
    
    
    result = opt.solvers.qp(P, q, G, h, A, b)
    w_star = np.array(result['x']).flatten()    
    R_star = w_star.dot(real_R)
    star_Exret.append(R_star)
    
    
    
    
    ############################## 运行时删掉break 只是用来 debug##########################
#     break
    #######################################################################################

portfolio_ret=pd.DataFrame()
portfolio_ret["date"]=date_list[120:]
portfolio_ret["gmv"]=gmv_Exret
portfolio_ret["star"]=star_Exret
portfolio_ret.head()


# In[11]:


portfolio_ret.to_csv("portfolio_ret2.csv",index=False)


# Solves a quadratic program
# 
#     minimize    (1/2)*x'*P*x + q'*x
#     subject to  G*x <= h
#                 A*x = b.
# 

# In[12]:


# adf=pd.read_csv("portfolio_ret.csv")
# adf.head()

