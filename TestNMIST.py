# Initialization
#   1.1 import package and load data
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import svm

traindata = pd.read_csv("FashionMNIST/fashion-mnist_train.csv")
testdata = pd.read_csv("FashionMNIST/fashion-mnist_test.csv")

#   1.2 Seperate data and label
data_train = traindata.iloc[: , 1 : 785] / 255.0
label_train = pd.DataFrame([traindata.iloc[: , 0]]).T
data_test = testdata.iloc[: , 0 : 784] / 255

#   1.3 View image data
# label_train.value_counts()
categoryMap = {0 : 'T-shirt/Top',
               1 : 'Trouser',
               2 : 'Pullover',
               3 : 'Dress',
               4 : 'Coat',
               5 : 'Sandal',
               6 : 'shirt',
               7 : 'Sneaker',
               8 : 'Bag',
               9 : 'Ankle boot'}
label_train['category'] = label_train['label'].map(categoryMap)

L = 5
W = 6
fig , axes = plt.subplots(L , W , figsize=(12 , 12))
axes = axes.ravel()

for i in range(30):
    axes[i].imshow(data_train.values.reshape((data_train.shape[0], 28, 28))[i], cmap=plt.get_cmap('gray'))
    axes[i].set_title('class' + str(label_train['label'][i]) + ": " + label_train['category'][i])
    axes[i].axis('off')
plt.show()

# 2. Feature Engineer

#   2.1 Check for null and missing values
# print("check for data_train:\n", data_train.isnull().any().describe() , "\n\ncheck for label_train:\n" ,
#       label_train.isnull().any().describe() , "\n\ncheck for data_test:\n",data_test.isnull().any().describe())

#   2.2 Split training and validation set
l_train = pd.DataFrame([traindata.iloc[:,0]]).T
X_train , X_val , Y_train , Y_val = train_test_split(data_train , l_train , test_size=0.25 , random_state=255)

#   2.3 Stardarding
# print(np.mean(X_train.values) , np.std(X_train.values) , np.mean(X_val.values) , np.std(X_val.values))
X_train = StandardScaler().fit_transform(X_train)
X_val = StandardScaler().fit_transform(X_val)
# print(np.mean(X_train),np.std(X_train),np.mean(X_val),np.std(X_val))
column_name = ['pixel' + str(i) for i in range(1 , 785)]
X_train = pd.DataFrame(X_train , columns = column_name)
X_val = pd.DataFrame(X_val , columns = column_name)

#   2.4 Dimensionality Reduction
pca = PCA(n_components=0.9 , copy=True , whiten=False)
X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)
# print(pca.explained_variance_ratio_)
var = np.cumsum(np.round(pca.explained_variance_ratio_ , decimals=3) * 100)
fig = go.Figure(data=go.Scatter(x = list(range(1 , len(var) + 1)) , y=var))
fig.update_layout(title='PCA variance Explained',
                  xaxis_title='# Of Features',
                  yaxis_title='% Variance Explained')
# fig.show()
pcn = X_train.shape[1]
X_train = pd.DataFrame(X_train , columns=column_name[0:pcn])
X_val = pd.DataFrame(X_val , columns=column_name[0:pcn])

# 3. Evaluate the model
#   3.1 KNN
start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train , Y_train.values.ravel())
y_train_prd = knn.predict(X_train)
y_val_prd = knn.predict(X_val)
acc_train_knn = accuracy_score(Y_train , y_train_prd)
acc_val_knn = accuracy_score(Y_val , y_val_prd)
print('KNN: ')
print("accuracy on the train set:{:0.4f}\naccuracy on validation set:{:.4f}".format(acc_train_knn , acc_val_knn))
print("--- %s seconds ---\n" % (time.time() - start_time))

# con_matrix = pd.crosstab(pd.Series(Y_val.values.flatten() , name='Actual') , pd.Series(y_val_prd , name='predict'))
# plt.figure(figsize= (9 , 6))
# plt.title("Confusion Matrix on KNN")
# sns.heatmap(con_matrix , cmap="Greys" , annot=True , fmt='g')
# plt.show()

#   3.2 Gaussian Naive Bayes
start_time = time.time()
NB = GaussianNB()
NB.fit(X_train , Y_train.values.ravel())
y_train_prd = NB.predict(X_train)
y_val_prd = NB.predict(X_val)
acc_train_nb = accuracy_score(Y_train , y_train_prd)
acc_val_nb = accuracy_score(Y_val , y_val_prd)
print('Gaussian Naive Bayes: ')
print("accuracy on train set:{:.4f}\naccuracy on validation set:{:.4f}".format(acc_train_nb , acc_val_nb))
print("--- %s seconds ---\n" % (time.time() - start_time))

# con_matrix = pd.crosstab(pd.Series(Y_val.values.flatten() , name='Actual') , pd.Series(y_val_prd , name='predict'))
# plt.figure(figsize= (9 , 6))
# plt.title("Confusion Matrix on Gaussian Naiye Bayes")
# sns.heatmap(con_matrix , cmap="Greys" , annot=True , fmt='g')
# plt.show()

#   3.3 Logistic Regression
start_time = time.time()
lg = LogisticRegression(solver='liblinear')
lg.fit(X_train , Y_train.values.ravel())
y_train_prd = lg.predict(X_train)
y_val_prd = lg.predict(X_val)
acc_train_lg = accuracy_score(Y_train , y_train_prd)
acc_val_lg = accuracy_score(Y_val  ,y_val_prd)
print("Logistic Regression:")
print("accuracy on train set:{:.4f}\naccuracy on validation set:{:.4f}".format(acc_train_lg , acc_val_lg))
print("--- %s seconds ---\n" % (time.time() - start_time))

# con_matrix = pd.crosstab(pd.Series(Y_val.values.flatten() , name='Actual') , pd.Series(y_val_prd , name='predict'))
# plt.figure(figsize= (9 , 6))
# plt.title("Confusion Matrix on Logistic Regression")
# sns.heatmap(con_matrix , cmap="Greys" , annot=True , fmt='g')
# plt.show()

#   3.4 Random Forest Classfier
start_time = time.time()
rf = RandomForestClassifier(random_state=8)
rf.fit(X_train , Y_train.values.ravel())
y_train_prd = rf.predict(X_train)
y_val_prd = rf.predict(X_val)
acc_train_rf = accuracy_score(Y_train , y_train_prd)
acc_val_rf = accuracy_score(Y_val , y_val_prd)
print("Random Forest")
print("accuracy on train set:{:.4f}\naccuracy on validation set:{:.4f}".format(acc_train_rf , acc_val_rf))
print("--- %s seconds ---\n" % (time.time() - start_time))

# con_matrix = pd.crosstab(pd.Series(Y_val.values.flatten(), name='Actual'),pd.Series(y_val_prd, name='Predicted'))
# plt.figure(figsize = (9,6))
# plt.title("Confusion Matrix on Random Forest Classifier")
# sns.heatmap(con_matrix, cmap="Greys", annot=True, fmt='g')
# plt.show()

#   3.5 SVM Classfier
start_time = time.time()
sv = svm.SVC(decision_function_shape='ovo')
sv.fit(X_train , Y_train.values.ravel())
y_train_prd = sv.predict(X_train)
y_val_prd = sv.predict(X_val)
acc_train_sv = accuracy_score(Y_train  , y_train_prd)
acc_val_sv = accuracy_score(Y_val , y_val_prd)
print("SVM")
print("accuracy on train set:{:.4f}\naccuracy on validation set:{:.4f}".format(acc_train_sv, acc_val_sv))
print("--- %s seconds ---\n" % (time.time() - start_time))

# con_matrix = pd.crosstab(pd.Series(Y_val.values.flatten(), name='Actual'),pd.Series(y_val_prd, name='Predicted'))
# plt.figure(figsize = (9,6))
# plt.title("Confusion Matrix on SVM Classifier")
# sns.heatmap(con_matrix, cmap="Greys", annot=True, fmt='g')
# plt.show()

#   3.6 XGBoost
# start_time = time.time()
# xgb = XGBClassifier(use_label_encoder=False)
# xgb.fit(X_train, Y_train.values.ravel())
# y_train_prd = xgb.predict(X_train)
# y_val_prd = xgb.predict(X_val)
# acc_train_xgb = accuracy_score(Y_train,y_train_prd)
# acc_val_xgb = accuracy_score(Y_val,y_val_prd)
# print('XGBoost:')
# print("accuracy on train set:{:.4f}\naccuracy on validation set:{:.4f}".format(acc_train_xgb , acc_val_xgb))
# print("--- %s seconds ---\n" % (time.time() - start_time))
# con_matrix = pd.crosstab(pd.Series(Y_val.values.flatten(), name='Actual'),pd.Series(y_val_prd, name='Predicted'))
# plt.figure(figsize = (9,6))
# plt.title("Confusion Matrix on XGBoost Classifier")
# sns.heatmap(con_matrix, cmap="Greys", annot=True, fmt='g')
# plt.show()


#   3.7 Model Comparison
acc_combine = {'Model' : ['KNN' , 'Gaussian Naive Bayes' , 'Logistic Regression' , 'Random Forest Classfier',
                          'SVM Classfier' , 'XGBoost'],
               'Accuracy_Tra':[acc_train_knn , acc_train_nb , acc_train_lg , acc_train_rf , acc_train_sv],
               'Accuracy_Val':[acc_val_knn , acc_val_nb , acc_val_lg , acc_val_rf , acc_val_sv]}
fig = go.Figure(data=[
    go.Bar(name = 'train set' , x=acc_combine['Model'] , y=acc_combine['Accuracy_Tra'] , text=np.round(acc_combine['Accuracy_Tra'] , 2) , textposition='outside') ,
    go.Bar(name= 'validation set' , x=acc_combine['Model'] , y=acc_combine['Accuracy_Val'], text=np.round(acc_combine['Accuracy_Val'] , 2) , textposition='outside') ,
])
fig.update_layout(barmode='group' , title_text='Accuracy Comparison On Different Models' , yaxis = dict(title='Accuracy'))
fig.show()