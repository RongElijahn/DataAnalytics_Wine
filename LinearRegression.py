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
# min_value=dataset.min()
# max_value=dataset.max()
# for col in range(0,len(dataset.columns)-1):
#     for row in range(0, len(dataset)):
#        dataset.iloc[row,col]=(dataset.iloc[row,col]-min_value.iloc[col])/(max_value.iloc[col]-min_value.iloc[col])

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


#draw the scatter plot
fig, ax= plt.subplots()
ax.scatter(y_test,y_pred)
ax.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--',lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)




