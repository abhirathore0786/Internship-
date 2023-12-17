#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[32]:


df=pd.read_csv(r'https://raw.githubusercontent.com/dsrscientist/Dataset2/main/temperature.csv')


# In[33]:


df


# In[ ]:


#There are 7752 rows and 25 columns in dataset.Two columns are our target next day minimum temperature 
#and next day maximum temperature
#null values are seen on dataset.


# In[34]:


df.drop(['Date','station'],axis=1,inplace=True)


# In[35]:


for i in df:
    print(df[i].value_counts())


# In[36]:


df.isnull().sum()


# In[167]:


""""there are lot of null values in each columns.We have enough data and there are a few nan's in columns, so we can directly drop nan values."""""


# In[37]:


df.dropna(inplace=True)


# In[38]:


plt.figure(figsize=(10,20))
p=1

for column in df:
    if p<=24:
        ax=plt.subplot(6,4,p)
        sns.distplot(df[column])
        
    p+=1
plt.show()


# In[39]:


plt.figure(figsize=(10,20))
p=1

for column in df:
    if p<=24:
        ax=plt.subplot(6,4,p)
        sns.boxplot(df[column])
        
    p+=1
plt.show()


# In[41]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr().abs(),annot=True)


# In[166]:


""""" data distribution is quite well shape. But some columns have maximum values as outliers.After looking into heatmap we found a very less relationship with target data and no multi-colinearity in these columns :LDAPS_PPT1','LDAPS_PPT2','LDAPS_PPT3','LDAPS_PPT4. So droping these values as most of them values are 0."""""


# In[40]:


df.drop(['LDAPS_PPT1','LDAPS_PPT2','LDAPS_PPT3','LDAPS_PPT4'],axis=1, inplace=True)


# In[42]:


df.columns


# In[43]:


df.reset_index()


# In[168]:


""""" making a copy of dataset for 2nd prediction."""""


# In[44]:


df2=df.copy()


# In[171]:


""""" it is found that LDAPS_CC2','LDAPS_CC3 is having muli-colinearity in dataset.So dropping them"""""


# In[45]:


df.drop(['LDAPS_CC2','LDAPS_CC3'],axis=1, inplace=True)


# In[46]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr().abs(),annot=True)


# In[47]:


df.drop(['LDAPS_CC1','Slope'],axis=1, inplace=True)


# In[172]:


""""" 'LDAPS_CC1','Slope' is alos having high-colinearity in dataset.So dropping them"""""


# In[60]:


plt.figure(figsize=(10,20))
p=1

for column in df:
    if p<=24:
        ax=plt.subplot(6,4,p)
        sns.boxplot(df[column])
        
    p+=1
plt.show()


# In[173]:


""""" now treating outliers firsty trying with quantile method.and after this doing with zscoe"""""


# In[69]:


q1=df.quantile(0.25)
q3=df.quantile(0.75)

iqr=q3-q1


# In[70]:


outlier=df[~((df<(q1-1.5*iqr))|df>((q3+1.5*iqr)))]


# In[71]:


outlier.shape


# In[175]:


""""" no value remove by this technique"""""


# In[61]:


from scipy.stats import zscore
data=df[['Present_Tmax','Present_Tmin','LDAPS_RHmax','LDAPS_Tmax_lapse','LDAPS_Tmin_lapse','LDAPS_WS','LDAPS_LH','Next_Tmax']]

z=np.abs(zscore(data))
z


# In[62]:


np.where(z>3)


# In[66]:


out=df[(z<2.8).all(axis=1)]

out.shape


# In[177]:


"""""now shape left is 7221 and 15 columns.checking outliers now"""""


# In[67]:


plt.figure(figsize=(10,20))
p=1

for column in out:
    if p<=24:
        ax=plt.subplot(6,4,p)
        sns.distplot(out[column])
        
    p+=1
plt.show()


# In[68]:


plt.figure(figsize=(10,20))
p=1

for column in out:
    if p<=24:
        ax=plt.subplot(6,4,p)
        sns.boxplot(out[column])
        
    p+=1
plt.show()


# In[178]:


"""" outliers are still left.Checking skewness"""""


# In[72]:


out.skew()


# In[73]:


out['LDAPS_RHmax'] = np.cbrt(out['LDAPS_RHmax'])
out['LDAPS_WS'] = np.cbrt(out['LDAPS_WS'])
out['LDAPS_LH'] = np.cbrt(out['LDAPS_LH'])
out['Solar radiation'] = np.cbrt(out['Solar radiation'])
out['LDAPS_CC4'] = np.cbrt(out['LDAPS_CC4'])


# In[74]:


out.skew()


# In[180]:


""""" skewness is negative in RHMAX, solar radiation.Leaving them as usual"""""


# In[116]:


from sklearn.model_selection import cross_val_score


# In[75]:


from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[76]:


X=out.drop('Next_Tmin',axis=1)
y=out['Next_Tmin']


# In[181]:


"""" first predicting Nextday minimum temperature"""""


# In[77]:


std=StandardScaler()

scld=std.fit_transform(X)


# In[78]:


VIF=pd.DataFrame()

VIF['vif_score']=[variance_inflation_factor(scld,i)for i in range(scld.shape[1])]
VIF['NAME']=X.columns


# In[79]:


VIF


# In[84]:


X.drop('LDAPS_Tmax_lapse',axis=1,inplace=True)


# In[182]:


"""" from X droping LDAPS_Tmax_lapse because of high mulitcolinearity"""""


# In[85]:


scld=std.fit_transform(X)


# In[86]:


VIF=pd.DataFrame()

VIF['vif_score']=[variance_inflation_factor(scld,i)for i in range(scld.shape[1])]
VIF['NAME']=X.columns


# In[87]:


VIF


# In[88]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
ln=LinearRegression()
best_acc=0
best_random_state=0

for i in range(1,800):
    x_train,x_test,y_train,y_test=train_test_split(scld,y, test_size=0.25, random_state=i)
    

    ln.fit(x_train,y_train)

    pred=ln.predict(x_train)
    acc=r2_score(pred,y_train)

    if acc > best_acc:
        best_acc=acc
        best_random_state=i

print(best_acc)
print(best_random_state)


# In[89]:


x_train,x_test,y_train,y_test=train_test_split(scld,y, test_size=0.25, random_state=321)
ln.fit(x_train,y_train)

ypred=ln.predict(x_train)
acc_train=r2_score(ypred,y_train)
print('Best training accuracy is',acc_train)

pred=ln.predict(x_test)
acc_test=r2_score(pred,y_test)
print('Best test accuracy is',acc_test)


# In[90]:


from sklearn.linear_model import LassoCV, Lasso


# In[92]:


lasso=LassoCV(alphas=None, cv=15,max_iter=15,random_state=444)


# In[93]:


lasso.fit(x_train,y_train)


# In[94]:


alpha=lasso.alpha_
alpha


# In[95]:


lasso_reg=Lasso(alpha)
lasso_reg


# In[96]:


lasso_reg.fit(x_train,y_train)


# In[97]:


print(lasso_reg.score(x_train,y_train))
print(lasso_reg.score(x_test,y_test))


# In[ ]:


#after cross validating trainig score is 83% and test score is 80% with linear regression


# In[98]:


from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()

best_acc=0
best_random_state=0

for i in range(1,1000):
    x_train,x_test,y_train,y_test=train_test_split(scld,y, test_size=0.25, random_state=i)
    

    dt.fit(x_train,y_train)

    pred=dt.predict(x_train)
    acc=r2_score(pred,y_train)
    
    if acc > best_acc:
        best_acc=acc
        best_random_state=i

print(best_acc)
print(best_random_state)


# In[99]:


x_train,x_test,y_train,y_test=train_test_split(scld,y, test_size=0.25, random_state=1)
dt.fit(x_train,y_train)

ypred=dt.predict(x_train)
acc_train=r2_score(ypred,y_train)
print('Best training accuracy is',acc_train)

pred=dt.predict(x_test)
acc_test=r2_score(pred,y_test)
print('Best test accuracy is',acc_test)


# In[100]:


from sklearn.model_selection import GridSearchCV


# In[101]:


param={'criterion':["squared_error","friedman_mse", "absolute_error","poisson"],
      'max_leaf_nodes':[1,2,3,4,5,6],
      'min_samples_split':[1,2,3,4,5,6],
      'min_samples_leaf':[1,2,3,4,5,6],
      }


# In[102]:


gd=GridSearchCV(dt,param_grid=param)


# In[103]:


gd.fit(x_train,y_train)


# In[104]:


gd.best_params_


# In[105]:


dtt=DecisionTreeRegressor(criterion= 'absolute_error',
 max_leaf_nodes= 6,
 min_samples_leaf= 1,
 min_samples_split= 2,
 random_state= 200)


# In[106]:


dtt.fit(x_train,y_train)


# In[107]:


ypred=dtt.predict(x_train)
acc_train=r2_score(ypred,y_train)
print('Best training accuracy is',acc_train)

pred=dtt.predict(x_test)
acc_test=r2_score(pred,y_test)
print('Best test accuracy is',acc_test)


# In[ ]:


#accuracy in decision tree after tunnig is 60% both 


# In[108]:


from sklearn.ensemble import AdaBoostRegressor


# In[109]:


ada=AdaBoostRegressor()


# In[110]:


ada.fit(x_train,y_train)


# In[111]:


ypred=ada.predict(x_train)
acc_train=r2_score(ypred,y_train)
print('Best training accuracy is',acc_train)

pred=ada.predict(x_test)
acc_test=r2_score(pred,y_test)
print('Best test accuracy is',acc_test)


# In[ ]:


#in adaboost training score is 74 and test 72% without tunning


# In[112]:


from sklearn.neighbors import KNeighborsRegressor


# In[113]:


knn=KNeighborsRegressor()


# In[114]:


best_acc=0
best_random_state=0

for i in range(1,1000):
    x_train,x_test,y_train,y_test=train_test_split(scld,y, test_size=0.25, random_state=i)
    

    knn.fit(x_train,y_train)

    pred=dt.predict(x_train)
    acc=r2_score(pred,y_train)
    
    if acc > best_acc:
        best_acc=acc
        best_random_state=i

print(best_acc)
print(best_random_state)


# In[115]:


x_train,x_test,y_train,y_test=train_test_split(scld,y, test_size=0.25, random_state=1)
knn.fit(x_train,y_train)
ypred=knn.predict(x_train)
acc_train=r2_score(ypred,y_train)
print('Best training accuracy is',acc_train)

pred=knn.predict(x_test)
acc_test=r2_score(pred,y_test)
print('Best test accuracy is',acc_test)


# In[ ]:


#knn giving 89% and 82% accuracy.


# In[120]:


cross_val_score(knn, scld,y, cv=5).mean()


# In[119]:


cross_val_score(knn, x_train,y_train, cv=5).mean()


# In[121]:


cross_val_score(dt, x_train,y_train, cv=5).mean()


# In[ ]:


#cross validation scores of decision tree and knn are 75 % and 85%


# In[122]:


from sklearn.ensemble import BaggingRegressor


# In[ ]:


#using bagging for good accuracy score.


# In[183]:


bag=BaggingRegressor(KNeighborsRegressor(),
                      n_estimators=20,bootstrap=True,oob_score=True
                     )


# In[184]:


bag.fit(x_train,y_train)
bag.fit(x_test,y_test)


# In[185]:


bag.score(x_train,y_train)


# In[127]:


df2


# In[187]:


"""" df2 is 2nd model dataset"""""


# In[128]:


plt.figure(figsize=(20,20))
sns.heatmap(df2.corr().abs(),annot=True)


# In[ ]:


X=out.drop('Next_Tmax',axis=1)
y=out['Next_Tmax']


# In[129]:


df2.drop(['LDAPS_CC2','LDAPS_CC3','LDAPS_CC1','Slope'],axis=1, inplace=True)


# In[131]:


from scipy.stats import zscore
data2=df2[['Present_Tmax','Present_Tmin','LDAPS_RHmax','LDAPS_Tmax_lapse','LDAPS_Tmin_lapse','LDAPS_WS','LDAPS_LH','Next_Tmin']]

z=np.abs(zscore(data2))
z


# In[132]:


np.where(z>3)


# In[133]:


dd=df[(z<2.8).all(axis=1)]

dd.shape


# In[134]:


dd.skew()


# In[ ]:


#mostly treating all as last dataset.Just change target for 2nd model


# In[136]:


dd['LDAPS_RHmax'] = np.cbrt(dd['LDAPS_RHmax'])
dd['LDAPS_WS'] = np.cbrt(dd['LDAPS_WS'])
dd['LDAPS_LH'] = np.cbrt(dd['LDAPS_LH'])
dd['Solar radiation'] = np.cbrt(dd['Solar radiation'])
dd['LDAPS_CC4'] = np.cbrt(dd['LDAPS_CC4'])


# In[137]:


dd.skew()


# In[138]:


from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[139]:


X=dd.drop('Next_Tmax',axis=1)
y=dd['Next_Tmax']


# In[140]:


std=StandardScaler()

scld=std.fit_transform(X)


# In[141]:


VIF=pd.DataFrame()

VIF['vif_score']=[variance_inflation_factor(scld,i)for i in range(scld.shape[1])]
VIF['NAME']=X.columns


# In[142]:


VIF


# In[143]:


X=X.drop('LDAPS_Tmin_lapse',axis=1)


# In[189]:


"""""LDAPS_Tmin_lapse is highly co_related so droping"""""


# In[144]:


scld=std.fit_transform(X)


# In[145]:


VIF=pd.DataFrame()

VIF['vif_score']=[variance_inflation_factor(scld,i)for i in range(scld.shape[1])]
VIF['NAME']=X.columns


# In[146]:


VIF


# In[147]:


from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()

best_acc=0
best_random_state=0

for i in range(1,1000):
    x_train,x_test,y_train,y_test=train_test_split(scld,y, test_size=0.25, random_state=i)
    

    dt.fit(x_train,y_train)

    pred=dt.predict(x_train)
    acc=r2_score(pred,y_train)
    
    if acc > best_acc:
        best_acc=acc
        best_random_state=i

print(best_acc)
print(best_random_state)


# In[148]:


x_train,x_test,y_train,y_test=train_test_split(scld,y, test_size=0.25, random_state=1)
dt.fit(x_train,y_train)

ypred=dt.predict(x_train)
acc_train=r2_score(ypred,y_train)
print('Best training accuracy is',acc_train)

pred=dt.predict(x_test)
acc_test=r2_score(pred,y_test)
print('Best test accuracy is',acc_test)


# In[ ]:


#decisiontree accuracy is 100 and 68%


# In[149]:


from sklearn.model_selection import GridSearchCV


# In[159]:


param={'criterion':["squared_error","friedman_mse", "absolute_error","poisson"],
      'max_leaf_nodes':[1,2,3,4,5,6],
      'min_samples_split':[1,2,3,4,5],
      'min_samples_leaf':[1,2,3,4,5],
      }


# In[160]:


gd=GridSearchCV(dt,param_grid=param)


# In[161]:


gd.fit(x_train,y_train)


# In[192]:


gd.best_params_


# In[193]:


dtt=DecisionTreeRegressor(criterion= 'squared_error',
 max_leaf_nodes= 6,
 min_samples_leaf= 1,
 min_samples_split= 5,
 random_state= 2)


# In[194]:


dtt.fit(x_train,y_train)


# In[195]:


ypred=dtt.predict(x_train)
acc_train=r2_score(ypred,y_train)
print('Best training accuracy is',acc_train)

pred=dtt.predict(x_test)
acc_test=r2_score(pred,y_test)
print('Best test accuracy is',acc_test)


# In[196]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
ln=LinearRegression()
best_acc=0
best_random_state=0

for i in range(1,800):
    x_train,x_test,y_train,y_test=train_test_split(scld,y, test_size=0.25, random_state=i)
    

    ln.fit(x_train,y_train)

    pred=ln.predict(x_train)
    acc=r2_score(pred,y_train)

    if acc > best_acc:
        best_acc=acc
        best_random_state=i

print(best_acc)
print(best_random_state)


# In[197]:


x_train,x_test,y_train,y_test=train_test_split(scld,y, test_size=0.25, random_state=394)
ln.fit(x_train,y_train)

ypred=ln.predict(x_train)
acc_train=r2_score(ypred,y_train)
print('Best training accuracy is',acc_train)

pred=ln.predict(x_test)
acc_test=r2_score(pred,y_test)
print('Best test accuracy is',acc_test)


# In[198]:


from sklearn.linear_model import LassoCV, Lasso


# In[199]:


lasso=LassoCV(alphas=None, cv=15,max_iter=15,random_state=444)


# In[200]:


lasso.fit(x_train,y_train)


# In[201]:


alpha=lasso.alpha_
alpha


# In[202]:


lasso_reg=Lasso(alpha)
lasso_reg


# In[203]:


lasso_reg.fit(x_train,y_train)


# In[204]:


print(lasso_reg.score(x_train,y_train))
print(lasso_reg.score(x_test,y_test))


# In[205]:


from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor()


# In[206]:


best_acc=0
best_random_state=0

for i in range(1,1000):
    x_train,x_test,y_train,y_test=train_test_split(scld,y, test_size=0.25, random_state=i)
    

    knn.fit(x_train,y_train)

    pred=dt.predict(x_train)
    acc=r2_score(pred,y_train)
    
    if acc > best_acc:
        best_acc=acc
        best_random_state=i

print(best_acc)
print(best_random_state)


# In[207]:


x_train,x_test,y_train,y_test=train_test_split(scld,y, test_size=0.25, random_state=1)
knn.fit(x_train,y_train)

ypred=knn.predict(x_train)
acc_train=r2_score(ypred,y_train)
print('Best training accuracy is',acc_train)

pred=knn.predict(x_test)
acc_test=r2_score(pred,y_test)
print('Best test accuracy is',acc_test)


# In[ ]:


"""" for 2nd model linear regression is working good.and 1st model is knn is working 89% and 82%


# In[208]:


import pickle


# In[209]:


pickle.dump(df,open('temperature forecast','wb'))


# In[ ]:




