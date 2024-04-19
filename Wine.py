import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import mean_absolute_error,confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

dataset=pd.read_csv('winequality-red.csv',sep=';')
temp_mean=dataset.mean(0)

#replace the nan data with mean
for col in range(0,len(dataset.columns)):
    dataset.iloc[:, col] = dataset.iloc[:, col].fillna(temp_mean.iloc[col])

#replace outlier with mean
temp_mean=dataset.mean(0)
Q1 = dataset.quantile(0.25)
Q3 = dataset.quantile(0.75)
IQR=Q3-Q1
outlier_min=Q1-1.5*IQR
outlier_max=Q3+1.5*IQR
for col in range(0,len(dataset.columns)-1): #tag 'quality' doesn't need to do this
   for row in range(0,len(dataset)):
       if (dataset.iloc[row,col]<outlier_min.iloc[col]) | (dataset.iloc[row,col]>outlier_max.iloc[col]):
           dataset.iloc[row, col] = temp_mean.iloc[col]

#normalize dataset : new_value= value/max-min
min_value=dataset.min()
max_value=dataset.max()
for col in range(0,len(dataset.columns)-1):
    for row in range(0, len(dataset)):
       dataset.iloc[row,col]=(dataset.iloc[row,col]-min_value.iloc[col])/(max_value.iloc[col]-min_value.iloc[col])

#using sklearn to do regression
var_x=dataset.iloc[:,0:len(dataset.columns)-1]
var_y=dataset.iloc[:,len(dataset.columns)-1]

#split the dataset into half train and half test
x_train, x_test, y_train, y_test = train_test_split(var_x, var_y, test_size=0.25, random_state=1)

#initiate the model
lr = LinearRegression()
lr.fit(x_train,y_train)

#evalue the model
y_pred=(lr.predict(x_test)).round()
print(lr.score(x_test,y_test))
accuracy1 = accuracy_score(y_test, y_pred)
print("Accuracy1:", accuracy1)


#initiate the model logistic
clf =LogisticRegression(penalty='l2',dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                   class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='ovr',
                   verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

#evalue the model : solution 1
clf.fit(x_train,y_train)
pre_prob_y= clf.predict_proba(x_train)
cur_auc= roc_auc_score(y_train, pre_prob_y, multi_class='ovr')
print("auc",cur_auc)
y_pred=clf.predict(x_test)
accuracy2 = accuracy_score(y_test, y_pred)
print("Accuracy2:", accuracy2)

#pick the value which is significant: solution 2
select_var = []
score = 0
auc_rec=[]
var_pool=np.arange(len(x_train.columns))
while(len(var_pool)>0):
     max_auc=0
     best_var=None
     for i in var_pool:
         cur_x=x_train.iloc[:,select_var+[i]]
         clf.fit(cur_x,y_train)
         pre_prob_y=clf.predict_proba(cur_x)
         cur_auc=metrics.roc_auc_score(y_train, pre_prob_y, multi_class='ovr')
         if(cur_auc > max_auc):
             max_auc = cur_auc
             best_var=i
     last_auc = auc_rec[-1] if len(auc_rec)>0 else 0.0001
     valid = True if ((max_auc-last_auc)/last_auc>0.005) else False
     if(valid):
         auc_rec.append(max_auc)
         select_var.append(best_var)
         var_pool=var_pool[var_pool != best_var]
     else:
         break

clf.fit(x_train.iloc[:,select_var],y_train)
y_pred=clf.predict(x_test.iloc[:,select_var])
accuracy3 = accuracy_score(y_test, y_pred)
print("Accuracy3:", accuracy3)

lr.fit(x_train.iloc[:,select_var],y_train)
y_pred=(lr.predict(x_test.iloc[:,select_var])).round()
print(lr.score(x_test.iloc[:,select_var],y_test))
accuracy4 = accuracy_score(y_test, y_pred)
print("Accuracy4:", accuracy4)

reg2=SGDRegressor()
reg2.fit(x_train.iloc[:,select_var],y_train)
y_pred=(reg2.predict(x_test.iloc[:,select_var])).round()
print(reg2.score(x_test.iloc[:,select_var],y_test))
accuracy5 = accuracy_score(y_test, y_pred)
print("Accuracy5:", accuracy5)

# test result

