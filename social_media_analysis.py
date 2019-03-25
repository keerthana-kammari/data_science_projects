import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Loading the dataset
df = pd.read_csv('OnlineNewsPopularity.csv', header=0)
df.head()
def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn
df.columns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from time import time
X = df[0:58]
Y = df['shares']

# Get the statistics of original target attribute
data = df[df.keys()[-1]]
data.describe()
# Encode the label by threshold 1400
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
popular_label = pd.Series(label_encoder.fit_transform(data>=1400))

popular = df.shares >= 1400
unpopular = df.shares < 1400
df.loc[popular,'shares'] = 1
df.loc[unpopular,'shares'] = 0

features=list(df.columns[2:60])

# split dataset to 60-40 training and testing resp.
X_train, X_test, y_train, y_test = train_test_split(df[features], df['shares'], test_size=0.4, random_state=0)

#Shape of  training and test datasets
print(X_train.shape)
print(y_train.shape)

#Decision Tree model and accuracy
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.metrics import accuracy_score, log_loss
from sklearn.naive_bayes import GaussianNB
model1 = DecisionTreeClassifier(min_samples_split = 12, random_state = 52)
model1_dt = model1.fit(X_train,y_train)
print("The accuracy of the model is {:.2f}".format(model1_dt.score(X_test,y_test)))

#Accuracy of Decision Tree model
scores = cross_val_score(model1, df[features], df['shares'], cv=5)
print(scores)
print("The mean of the model is {:.2f}".format(scores.mean()))

# Random Forest model and accuracy
model2 = RandomForestClassifier(n_estimators=100,n_jobs=-1)
model2_rf = model2.fit(X_train,y_train)
print("The accuracy of the model is {:.2f}".format(model2_rf.score(X_test,y_test)))

#Accuracy of Random Forest model
scores = cross_val_score(model2, df[features], df['shares'], cv=5)
print(scores)
print("The mean of the model is {:.2f}".format(scores.mean()))

# KNN model and accuracy
model3 = KNeighborsClassifier()
model3_knn=model3.fit(X_train, y_train)
print("The accuracy of the model is {:.2f}".format(model3_knn.score(X_test,y_test)))

#Accuracy of KNN model
scores = cross_val_score(model3, df[features], df['shares'], cv=5)
print(scores)
print("The mean of the model is {:.2f}".format(scores.mean()))

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GaussianNB()]
    
# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__  
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()

sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")

plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')
plt.show()
