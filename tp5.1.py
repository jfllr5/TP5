# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:15:39 2024

@author: julia
"""

#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib 
matplotlib.use('QtAgg')  #Using the 'Agg' backend for matplotlib (non-interactive)

#Load data from an XML file
path = 'C:\\Users\\julia\\Downloads\\labeled_flows_xml\\TestbedMonJun14Flows.xml'
df = pd.read_xml(path)

#Display information about the DataFrame
print(df.info())

#Data analysis
AppCount = pd.value_counts(df['appName'])
AttackCount = pd.value_counts(df['Tag'])
AttackDataframe = pd.DataFrame(df.loc[df['Tag']=='Attack'])
AttackCount2 = pd.value_counts(AttackDataframe['appName'])
NormalDataframe = pd.DataFrame(df.loc[df["Tag"]=="Normal"])
NormalDataframeY = NormalDataframe[["Tag"]]
AttackDataframeY = AttackDataframe[["Tag"]]

#Select relevant columns for attack data
AttackDataframe = AttackDataframe[["totalSourceBytes", "totalDestinationBytes", "totalDestinationPackets", "totalSourcePackets", "sourcePort", "destinationPort"]]

#Select relevant columns for normal data
NormalDataframe = NormalDataframe[["totalSourceBytes", "totalDestinationBytes", "totalDestinationPackets", "totalSourcePackets", "sourcePort", "destinationPort"]]

#Convert normal labels to numeric
NormalDataframeY = NormalDataframeY.to_numpy()
NormalDataframeY = NormalDataframeY.ravel()
labels, uniques = pd.factorize(NormalDataframeY)
NormalDataframeY = labels
NormalDataframeY = NormalDataframeY.ravel()

#Convert attack labels to numeric
AttackDataframeY = AttackDataframeY.to_numpy()
AttackDataframeY = AttackDataframeY.ravel()
labelsS, uniquesS = pd.factorize(AttackDataframeY)
AttackDataframeY = labelsS
AttackDataframeY = AttackDataframeY.ravel()

#Correct labels for the attack class
indices_zero = AttackDataframeY == 0
AttackDataframeY[indices_zero] = 1

#Split data into training and testing sets
from sklearn.model_selection import train_test_split

X_train_N, X_test_N, Y_train_N, Y_test_N = train_test_split(NormalDataframe, NormalDataframeY, random_state=0, test_size=8000)
X_train_A, X_test_A, Y_train_A, Y_test_A = train_test_split(AttackDataframe, AttackDataframeY, random_state=0, test_size=500)

#Concatenate training and testing sets
X_train = pd.concat([X_train_N, X_train_A])
X_train = X_train.sample(frac=1, random_state=42)

X_test = pd.concat([X_test_N, X_test_A])
X_test = X_test.sample(frac=1, random_state=42)

#Convert labels to series and concatenate
Y_train_N = pd.Series(Y_train_N, name='Tag') 
Y_train_A = pd.Series(Y_train_A, name='Tag')

Y_train = pd.concat([Y_train_N, Y_train_A])
Y_train = pd.DataFrame(Y_train)
Y_train = Y_train.sample(frac=1, random_state=42)

Y_test_N = pd.Series(Y_test_N, name='Tag') 
Y_test_A = pd.Series(Y_test_A, name='Tag')

Y_test = pd.concat([Y_test_N, Y_test_A])
Y_test = pd.DataFrame(Y_test)
Y_test = Y_test.sample(frac=1, random_state=42)


#Visualize data with t-SNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

transform = TSNE
X = X_test
trans = transform(n_components=2)
X_reduced = trans.fit_transform(X)
Y = pd.DataFrame(Y_test)
fig, ax = plt.subplots(figsize=(7, 7))


scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=Y[Y.columns[-1]].astype('category').cat.codes, cmap="cool", alpha=0.7)

legend_labels = ['Normal', 'Attacks']
legend = ax.legend(legend_labels)

ax.set(xlabel="$X_1$", ylabel="$X_2$", title=f"{transform.__name__} visualization of IDS testing dataset",)

plt.show()

#Train decision tree classification model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score, KFold

#Initialize and train the model
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, Y_train)

#Cross-validation to evaluate model performance
cv = KFold(n_splits=10, random_state=0, shuffle=True)
accuracy = clf.score(X_test, Y_test)
KFold10_accuracy = cross_val_score(clf, X_train, Y_train, scoring="accuracy", cv=cv, n_jobs=-1)
print(KFold10_accuracy.mean())

#Prediction on the test set
predict = clf.predict(X_test)

#Calculate and display confusion matrix and other metrics
cm=confusion_matrix(Y_test, predict)
precision = precision_score(Y_test, predict, average = "weighted", labels=np.unique(predict))
recall = recall_score(Y_test, predict, average = "weighted", labels=np.unique(predict))
f1scoreMacro = f1_score(Y_test, predict, average = "macro", labels=np.unique(predict))
print(classification_report(Y_test, predict, target_names=["Normal", "Attacks"]))

#Identify false alerts
predict = np.reshape(predict, (8500, 1))
fp_rows = []
for i in range(len(predict)):
    if predict[i] == 1 and Y_test.iloc[i, -1] == 0:
        fp_rows.append(i)
 #%%       
#Confusion Matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix

#Compute the confusion matrix
cm = confusion_matrix(Y_test, predict)

#Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attacks"], yticklabels=["Normal", "Attacks"])
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
#%%
#ROC Curve (Receiver Operating Characteristic)
from sklearn.metrics import roc_curve, auc

#Compute ROC curve and area under the curve
fpr, tpr, thresholds = roc_curve(Y_test, predict)
roc_auc = auc(fpr, tpr)

#Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('ROC curve.png')
#%%
#Precision-Recall Curve
from sklearn.metrics import precision_recall_curve

#Compute precision-recall curve
precision, recall, _ = precision_recall_curve(Y_test, predict)

#Plot precision-recall curve
plt.figure()
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.savefig('PR curve.png')

#%%
#%%

import pandas as pd
import numpy as np
import matplotlib 
matplotlib.use('QtAgg')


path = 'C:\\Users\\julia\\Downloads\\labeled_flows_xml\\TestbedSatJun12Flows.xml'
df = pd.read_xml(path)

print(df.info())

AppCount = pd.value_counts(df['appName'])
AttackCount = pd.value_counts(df['Tag'])
AttackDataframe = pd.DataFrame(df.loc[df['Tag']=='Attack'])
AttackCount2 = pd.value_counts(AttackDataframe['appName'])
NormalDataframe = pd.DataFrame(df.loc[df["Tag"]=="Normal"])
NormalDataframeY = NormalDataframe[["Tag"]]
AttackDataframeY = AttackDataframe[["Tag"]]

AttackDataframe = AttackDataframe[["totalSourceBytes", "totalDestinationBytes", "totalDestinationPackets", "totalSourcePackets", "sourcePort", "destinationPort"]]

NormalDataframe = NormalDataframe[["totalSourceBytes", "totalDestinationBytes", "totalDestinationPackets", "totalSourcePackets", "sourcePort", "destinationPort"]]


NormalDataframeY = NormalDataframeY.to_numpy()
NormalDataframeY = NormalDataframeY.ravel()
labels, uniques = pd.factorize(NormalDataframeY)
NormalDataframeY = labels
NormalDataframeY = NormalDataframeY.ravel()


AttackDataframeY = AttackDataframeY.to_numpy()
AttackDataframeY = AttackDataframeY.ravel()
labelsS, uniquesS = pd.factorize(AttackDataframeY)
AttackDataframeY = labelsS
AttackDataframeY = AttackDataframeY.ravel()

indices_zero = AttackDataframeY == 0
AttackDataframeY[indices_zero] = 1

from sklearn.model_selection import train_test_split

# La division des données a été corrigée ici
X_train_N, X_test_N, Y_train_N, Y_test_N = train_test_split(NormalDataframe, NormalDataframeY, random_state=0, test_size=8000)
X_train_A, X_test_A, Y_train_A, Y_test_A = train_test_split(AttackDataframe, AttackDataframeY, random_state=0, test_size=500)

X_train = pd.concat([X_train_N, X_train_A])
X_train = X_train.sample(frac=1, random_state=42)

X_test = pd.concat([X_test_N, X_test_A])
X_test = X_test.sample(frac=1, random_state=42)

Y_train_N = pd.Series(Y_train_N, name='Tag') 
Y_train_A = pd.Series(Y_train_A, name='Tag')

Y_train = pd.concat([Y_train_N, Y_train_A])
Y_train = pd.DataFrame(Y_train)
Y_train = Y_train.sample(frac=1, random_state=42)

Y_test_N = pd.Series(Y_test_N, name='Tag') 
Y_test_A = pd.Series(Y_test_A, name='Tag')

Y_test = pd.concat([Y_test_N, Y_test_A])
Y_test = pd.DataFrame(Y_test)
Y_test = Y_test.sample(frac=1, random_state=42)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

transform = TSNE
X = X_test
trans = transform(n_components=2)
X_reduced = trans.fit_transform(X)
Y = pd.DataFrame(Y_test)
fig, ax = plt.subplots(figsize=(7, 7))


scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=Y[Y.columns[-1]].astype('category').cat.codes, cmap="cool", alpha=0.7)

legend_labels = ['Normal', 'Attacks']
legend = ax.legend(legend_labels)

ax.set(xlabel="$X_1$", ylabel="$X_2$", title=f"{transform.__name__} visualization of IDS testing dataset",)

plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score, KFold

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, Y_train)
cv = KFold(n_splits=10, random_state=0, shuffle=True)
accuracy = clf.score(X_test, Y_test)
KFold10_accuracy = cross_val_score(clf, X_train, Y_train, scoring="accuracy", cv=cv, n_jobs=-1)
print(KFold10_accuracy.mean())
predict = clf.predict(X_test)
cm=confusion_matrix(Y_test, predict)
precision = precision_score(Y_test, predict, average = "weighted", labels=np.unique(predict))
recall = recall_score(Y_test, predict, average = "weighted", labels=np.unique(predict))
f1scoreMacro = f1_score(Y_test, predict, average = "macro", labels=np.unique(predict))
print(classification_report(Y_test, predict, target_names=["Normal", "Attacks"]))


predict = np.reshape(predict, (8500, 1))
fp_rows = []
for i in range(len(predict)):
    if predict[i] == 1 and Y_test.iloc[i, -1] == 0:
        fp_rows.append(i)
        
#%%


import pandas as pd
import numpy as np
import matplotlib 
matplotlib.use('QtAgg')


path = 'C:\\Users\\julia\\Downloads\\labeled_flows_xml\\TestbedSunJun13Flows.xml'
df = pd.read_xml(path)

print(df.info())

AppCount = pd.value_counts(df['appName'])
AttackCount = pd.value_counts(df['Tag'])
AttackDataframe = pd.DataFrame(df.loc[df['Tag']=='Attack'])
AttackCount2 = pd.value_counts(AttackDataframe['appName'])
NormalDataframe = pd.DataFrame(df.loc[df["Tag"]=="Normal"])
NormalDataframeY = NormalDataframe[["Tag"]]
AttackDataframeY = AttackDataframe[["Tag"]]

AttackDataframe = AttackDataframe[["totalSourceBytes", "totalDestinationBytes", "totalDestinationPackets", "totalSourcePackets", "sourcePort", "destinationPort"]]

NormalDataframe = NormalDataframe[["totalSourceBytes", "totalDestinationBytes", "totalDestinationPackets", "totalSourcePackets", "sourcePort", "destinationPort"]]


NormalDataframeY = NormalDataframeY.to_numpy()
NormalDataframeY = NormalDataframeY.ravel()
labels, uniques = pd.factorize(NormalDataframeY)
NormalDataframeY = labels
NormalDataframeY = NormalDataframeY.ravel()


AttackDataframeY = AttackDataframeY.to_numpy()
AttackDataframeY = AttackDataframeY.ravel()
labelsS, uniquesS = pd.factorize(AttackDataframeY)
AttackDataframeY = labelsS
AttackDataframeY = AttackDataframeY.ravel()

indices_zero = AttackDataframeY == 0
AttackDataframeY[indices_zero] = 1

from sklearn.model_selection import train_test_split

# La division des données a été corrigée ici
X_train_N, X_test_N, Y_train_N, Y_test_N = train_test_split(NormalDataframe, NormalDataframeY, random_state=0, test_size=8000)
X_train_A, X_test_A, Y_train_A, Y_test_A = train_test_split(AttackDataframe, AttackDataframeY, random_state=0, test_size=500)

X_train = pd.concat([X_train_N, X_train_A])
X_train = X_train.sample(frac=1, random_state=42)

X_test = pd.concat([X_test_N, X_test_A])
X_test = X_test.sample(frac=1, random_state=42)

Y_train_N = pd.Series(Y_train_N, name='Tag') 
Y_train_A = pd.Series(Y_train_A, name='Tag')

Y_train = pd.concat([Y_train_N, Y_train_A])
Y_train = pd.DataFrame(Y_train)
Y_train = Y_train.sample(frac=1, random_state=42)

Y_test_N = pd.Series(Y_test_N, name='Tag') 
Y_test_A = pd.Series(Y_test_A, name='Tag')

Y_test = pd.concat([Y_test_N, Y_test_A])
Y_test = pd.DataFrame(Y_test)
Y_test = Y_test.sample(frac=1, random_state=42)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

transform = TSNE
X = X_test
trans = transform(n_components=2)
X_reduced = trans.fit_transform(X)
Y = pd.DataFrame(Y_test)
fig, ax = plt.subplots(figsize=(7, 7))


scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=Y[Y.columns[-1]].astype('category').cat.codes, cmap="cool", alpha=0.7)

legend_labels = ['Normal', 'Attacks']
legend = ax.legend(legend_labels)

ax.set(xlabel="$X_1$", ylabel="$X_2$", title=f"{transform.__name__} visualization of IDS testing dataset",)

plt.show()

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score, KFold

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, Y_train)
cv = KFold(n_splits=10, random_state=0, shuffle=True)
accuracy = clf.score(X_test, Y_test)
KFold10_accuracy = cross_val_score(clf, X_train, Y_train, scoring="accuracy", cv=cv, n_jobs=-1)
print(KFold10_accuracy.mean())
predict = clf.predict(X_test)
cm=confusion_matrix(Y_test, predict)
precision = precision_score(Y_test, predict, average = "weighted", labels=np.unique(predict))
recall = recall_score(Y_test, predict, average = "weighted", labels=np.unique(predict))
f1scoreMacro = f1_score(Y_test, predict, average = "macro", labels=np.unique(predict))
print(classification_report(Y_test, predict, target_names=["Normal", "Attacks"]))


predict = np.reshape(predict, (8500, 1))
fp_rows = []
for i in range(len(predict)):
    if predict[i] == 1 and Y_test.iloc[i, -1] == 0:
        fp_rows.append(i)
        
