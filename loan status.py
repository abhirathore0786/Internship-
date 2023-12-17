#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[20]:


df=pd.read_csv(r'https://raw.githubusercontent.com/dsrscientist/DSData/master/loan_prediction.csv')
df


# In[21]:


for i in df:
    print(df[i].value_counts())


# In[22]:


df.dtypes


# In[23]:


df.isnull().sum()


# In[24]:


df.describe()


# In[ ]:


""""there are 13 columns in dataset in which 13 columns and 614 rows. Below observations are from upper view:"""""
""""there are lot of null values in dataset.Which can be treat with mode and mean method differently."""""
"""" datatype is found ok as dataset is.But after encoding will check again."""""
"""""there are outliers in applicant income applicantincome, coapplicantincome, Loan amount and loan term."""""
"""""further observation will check after encoding and imputing"""""


# In[25]:


from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


# In[29]:


sim=SimpleImputer(strategy='most_frequent')


# In[27]:


new=df[['Gender','Married','Dependents','Self_Employed','Credit_History','Loan_Amount_Term','LoanAmount']]


# In[31]:


for i in new:
    df[i]=sim.fit_transform(df[[i]])


# In[32]:


df


# In[33]:


df.isnull().sum()


# In[34]:


le=LabelEncoder()


# In[35]:


new2=df[['Gender','Married','Dependents','Self_Employed','Education','Property_Area','Loan_Status']]


# In[39]:


for i in new2:
    df[i]=le.fit_transform(df[[i]])


# In[40]:


df


# In[41]:


df.drop('Loan_ID',axis=1,inplace=True)


# In[42]:


df.dtypes


# In[43]:


df['Loan_Status'].value_counts()


# In[45]:


""""value count is found imbalance"""""


# In[49]:


plt.figure(figsize=(15,15))
p=1

for column in df:
    if p<=11:
        ax=plt.subplot(4,3,p)
        sns.distplot(df[column])
       
    p+=1
plt.show()


# In[ ]:


#most of columns are categorical data so we don't check their distribution.
#But we look and treat applicant, coapplicant income, loan amount.In these columns data is skewed to left.Outlier presence is found.


# In[50]:


plt.figure(figsize=(10,15))
p=1

for column in df:
    if p<=11:
        ax=plt.subplot(4,3,p)
        sns.boxplot(df[column])
        
    p+=1
plt.show()


# In[51]:


from scipy.stats import zscore
dlist=df[['ApplicantIncome','CoapplicantIncome','LoanAmount']]

z=np.abs(zscore(dlist))
z


# In[52]:


np.where(z>3)


# In[53]:


dff=df[(z<3).all(axis=1)]

dff.shape


# In[54]:


q1=df.quantile(0.25)
q3=df.quantile(0.75)

iqr=q3-q1


# In[56]:


outlier=df[~((df<(q1-1.5*iqr))|df>((q3+1.5*iqr)))]


# In[57]:


outlier.shape


# In[58]:


plt.figure(figsize=(10,15))
p=1

for column in outlier:
    if p<=11:
        ax=plt.subplot(4,3,p)
        sns.distplot(outlier[column])
        
    p+=1
plt.show()


# In[ ]:


plt.figure(figsize=(10,15))
p=1

for column in df:
    if p<=11:
        ax=plt.subplot(4,3,p)
        sns.distplot(df[column])
        
    p+=1
plt.show()


# In[59]:


outlier.skew()


# In[60]:


outlier['ApplicantIncome']=np.cbrt(outlier['ApplicantIncome'])
outlier['CoapplicantIncome']=np.cbrt(outlier['CoapplicantIncome'])
outlier['LoanAmount']=np.cbrt(outlier['LoanAmount'])


# In[61]:


outlier.skew()


# In[62]:


outlier['ApplicantIncome']=np.cbrt(outlier['ApplicantIncome'])


# In[63]:


outlier.skew()


# In[65]:


plt.figure(figsize=(10,10))
sns.heatmap(outlier.corr().abs(),annot=True)


# In[66]:


outlier.drop_duplicates()


# In[67]:


plt.figure(figsize=(10,10))
p=1

for column in outlier:
    if p<=12:
        ax=plt.subplot(4,3,p)
        sns.regplot(outlier[column],outlier['Loan_Status'])
        p+=1
plt.tight_layout()


# In[ ]:


""""" Gender,married,dependents,applicantincome, coaaplicantincome,property area is having direct relation.
credit history is highly related to target.
loan amount and loan term is having incerse relation to target.
self_employed and education is much realted.


# In[72]:


sns.scatterplot(outlier['LoanAmount'],outlier['Loan_Status'],data=outlier)


# In[73]:


outlier.drop(['Education','Self_Employed'],axis=1,inplace=True)


# In[74]:


X=df.drop('Loan_Status',axis=1)
y=df['Loan_Status']


# In[78]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix 
ln=LogisticRegression()
best_acc=0
best_random_state=0

for i in range(1,800):
    x_train,x_test,y_train,y_test=train_test_split(X,y, test_size=0.25, random_state=i)
    

    ln.fit(x_train,y_train)

    pred=ln.predict(x_train)
    acc=accuracy_score(pred,y_train)

    if acc > best_acc:
        best_acc=acc
        best_random_state=i

print(best_acc)
print(best_random_state)


# In[79]:


x_train,x_test,y_train,y_test=train_test_split(X,y, test_size=0.25, random_state=540)
ln.fit(x_train,y_train)

ypred=ln.predict(x_train)
acc_train=accuracy_score(ypred,y_train)
print('Best training accuracy is',acc_train)

pred=ln.predict(x_test)
acc_test=accuracy_score(pred,y_test)
print('Best test accuracy is',acc_test)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[80]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix 
dt=DecisionTreeClassifier()
best_acc=0
best_random_state=0

for i in range(1,800):
    x_train,x_test,y_train,y_test=train_test_split(X,y, test_size=0.25,random_state=i)
    

    dt.fit(x_train,y_train)

    pred=dt.predict(x_train)
    acc=accuracy_score(pred,y_train)
    
    if acc > best_acc:
        best_acc=acc
        best_random_state=i

print(best_acc)
print(best_random_state)


# In[81]:


x_train,x_test,y_train,y_test=train_test_split(X,y, test_size=0.25, random_state=1)
ln.fit(x_train,y_train)

ypred=dt.predict(x_train)
acc_train=accuracy_score(ypred,y_train)
print('Best training accuracy is',acc_train)

pred=dt.predict(x_test)
acc_test=accuracy_score(pred,y_test)
print('Best test accuracy is',acc_test)


# In[83]:


confusion_matrix(ypred,y_train)


# In[84]:


confusion_matrix(pred,y_test)


# In[85]:


from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


# In[86]:


parm={'criterion':['ginni','entropy'],
     'max_depth':range(1,20),
     
     
     'max_leaf_nodes':range(3,10)
     
     }


# In[87]:


grid=GridSearchCV(dt,param_grid=parm)


# In[88]:


grid.fit(x_train,y_train)


# In[89]:


grid.best_params_


# In[90]:


dt=DecisionTreeClassifier(criterion='entropy',max_depth= 2, max_leaf_nodes= 4)


# In[91]:


dt.fit(x_train,y_train)


# In[92]:


dt.score(x_train,y_train)


# In[93]:


dt.score(x_test,y_test)


# In[94]:


confusion_matrix(ypred,y_train)


# In[95]:


confusion_matrix(pred,y_test)


# In[96]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix 
knn=KNeighborsClassifier()
best_acc=0
best_random_state=0

for i in range(1,800):
    x_train,x_test,y_train,y_test=train_test_split(X,y, test_size=0.25,random_state=i)
    

    knn.fit(x_train,y_train)

    pred=knn.predict(x_train)
    acc=accuracy_score(pred,y_train)
    
    if acc > best_acc:
        best_acc=acc
        best_random_state=i

print(best_acc)
print(best_random_state)


# In[97]:


x_train,x_test,y_train,y_test=train_test_split(X,y, test_size=0.25, random_state=1)
knn.fit(x_train,y_train)

ypred=knn.predict(x_train)
acc_train=accuracy_score(ypred,y_train)
print('Best training accuracy is',acc_train)

pred=knn.predict(x_test)
acc_test=accuracy_score(pred,y_test)
print('Best test accuracy is',acc_test)


# In[98]:


confusion_matrix(ypred,y_train)


# In[99]:


confusion_matrix(pred,y_test)


# In[100]:


parm={'algorithm':['kd_tree','brute'],
     'leaf_size':[1,2,3,4,5,6,7,8,9],
     'n_neighbors':[1,2,3,4,5,6,7,8,9],
     
     }


# In[101]:


grid=GridSearchCV(knn,param_grid=parm)


# In[102]:


grid.fit(x_train,y_train)


# In[103]:


grid.best_params_


# In[104]:


knn=KNeighborsClassifier(algorithm= 'kd_tree', leaf_size= 1, n_neighbors= 9)


# In[105]:


knn.fit(x_train,y_train)


# In[106]:


knn.score(x_train,y_train)


# In[107]:


knn.score(x_test,y_test)


# In[108]:


parm={'algorithm':['kd_tree','brute'],
     'leaf_size':[1,2,3,4,5,6,7,8,9],
     'n_neighbors':[1,2,3,4,5,6,7,8,9],
     }


# In[109]:


random=RandomizedSearchCV(KNeighborsClassifier(),cv=5,param_distributions=parm)


# In[110]:


random.fit(x_train,y_train)


# In[111]:


random.best_params_


# In[112]:


knn=KNeighborsClassifier(algorithm= 'brute', leaf_size= 9, n_neighbors= 9)


# In[113]:


knn.fit(x_train,y_train)


# In[114]:


knn.score(x_train,y_train)


# In[115]:


knn.score(x_test,y_test)


# In[116]:


y_pred=knn.predict(x_train)
confusion_matrix(y_pred,y_train)


# In[117]:


pred=knn.predict(x_test)
confusion_matrix(pred,y_test)


# In[119]:


from sklearn.ensemble import BaggingClassifier


# In[121]:


bag=BaggingClassifier(DecisionTreeClassifier(criterion='entropy',max_depth= 2, max_leaf_nodes= 4),
                      n_estimators=20,bootstrap=True,oob_score=True
                     )


# In[122]:


bag.fit(x_train,y_train)
bag.fit(x_test,y_test)


# In[123]:


bag.score(x_train,y_train)


# In[124]:


bag.score(x_test,y_test)


# In[ ]:


#plotting roc auc curve for curve


# In[140]:


from sklearn.metrics import roc_curve


# In[141]:


fpr,tpr,thresholds=roc_curve(y_test,pred)


# In[142]:


print(fpr)
print(tpr)
print(thresholds)


# In[143]:


plt.plot(fpr,tpr, color='Blue',label='ROC')
plt.plot([0,1],[0,1],color='green',linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend()
plt.show()


# In[144]:


import pickle


# In[145]:


pickle.dump(df,open('loan status','wb'))


# In[ ]:




