# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 15:38:35 2024

@author: julia
"""

#TASK3
#%%
#Importation of necessary libraries
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

#Loading data from the CSV file
dataset = pd.read_csv('SDN_traffic.csv')

#Displaying the first rows of the dataset
print('\n\nhead :\r')
print(dataset.head())

#Displaying information about the dataset
print('\n\ninfo : \r')
print(dataset.info())

#Displaying descriptive statistics of the dataset
print('\n\ndescribe : \r')
print(dataset.describe())

#Checking for duplicates in the dataset
print('\n\nduplicated : \r')
print(dataset.duplicated())

#Selecting necessary features
X = dataset[["forward_bps_var", "tp_src", "tp_dst", "nw_proto", "forward_pc", "forward_bc", "forward_pl",
             "forward_piat", "forward_pps", "forward_bps", "forward_pl_mean", "forward_piat_mean", 
             "forward_pps_mean", "forward_bps_mean", "forward_pl_var", "forward_piat_var", "forward_pps_var",
             "forward_pl_q1", "forward_pl_q3", "forward_piat_q1", "forward_piat_q3", "forward_pl_max", 
             "forward_pl_min", "forward_piat_max", "forward_piat_min", "forward_pps_max", "forward_pps_min",
             "forward_bps_max", "forward_bps_min", "forward_duration", "forward_size_packets", 
             "forward_size_bytes", "reverse_pc", "reverse_bc", "reverse_pl", "reverse_piat", "reverse_pps", 
             "reverse_bps", "reverse_pl_mean", "reverse_piat_mean", "reverse_pps_mean", "reverse_bps_mean", 
             "reverse_pl_var", "reverse_piat_var", "reverse_pps_var", "reverse_pl_q1", "reverse_pl_q3", 
             "reverse_piat_q1", "reverse_piat_q3", "reverse_pl_max", "reverse_pl_min", "reverse_piat_max", 
             "reverse_piat_min", "reverse_pps_max", "reverse_pps_min", "reverse_bps_max", "reverse_bps_min", 
             "reverse_duration", "reverse_size_packets", "reverse_size_bytes"]]

#Correction of incorrect values
X.loc[1877, 'forward_bps_var'] = float(11968065203349)
X.loc[1931, 'forward_bps_var'] = float(12880593804833)
X.loc[2070, 'forward_bps_var'] = float(9022747730895)
X.loc[2381, 'forward_bps_var'] = float(39987497172945)
X.loc[2562, 'forward_bps_var'] = float(663300742992)
X.loc[2567, 'forward_bps_var'] = float(37770223877794)
X.loc[2586, 'forward_bps_var'] = float(97227875083751)
X.loc[2754, 'forward_bps_var'] = float(18709751403737)
X.loc[2765, 'forward_bps_var'] = float(33969277035759)
X.loc[2904, 'forward_bps_var'] = float(39204786962856)
X.loc[3044, 'forward_bps_var'] = float(9169996063653)
X.loc[3349, 'forward_bps_var'] = float(37123283690575)
X.loc[3507, 'forward_bps_var'] = float(61019064590464)
X.loc[3610, 'forward_bps_var'] = float(46849620984072)
X.loc[3717, 'forward_bps_var'] = float(97158873841506)
X.loc[3845, 'forward_bps_var'] = float(11968065203349)
X.loc[3868, 'forward_bps_var'] = float(85874278395372)


#Conversion of the 'forward_bps_var' column to numeric type
X['forward_bps_var'] = pd.to_numeric(X['forward_bps_var'])

# Displaying information about X
print(X.info())

#Selecting the target variable (Y) and converting it to a NumPy array
Y = dataset[['category']].to_numpy().ravel()
#Converting labels to numeric values
labels, uniques = pd.factorize(Y)
Y = labels.ravel()

print('New value code of the Category column (but reshaped in a horizontal vector)')
print(Y)

#Normalizing features
X = stats.zscore(X)
X = np.nan_to_num(X)

#Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.7)

#Initializing and training the model
clf = DecisionTreeClassifier(random_state=0, max_depth=6)
clf.fit(X_train, Y_train)
print(clf)

#Evaluating the model
cv = KFold(n_splits=10, random_state=0, shuffle=True)
accuracy = clf.score(X_test, Y_test)
Kfold10_accuracy = cross_val_score(clf, X_train, Y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print("Kfold10 accuracy mean : ")
print(Kfold10_accuracy.mean())
predict = clf.predict(X_test)
cm = confusion_matrix(Y_test, predict)
precision = precision_score(Y_test, predict, average='weighted', labels=np.unique(predict))
recall = recall_score(Y_test, predict, average='weighted', labels=np.unique(predict))
f1scoreMacro = f1_score(Y_test, predict, average='macro', labels=np.unique(predict))

print("Classification report : ")
print(classification_report(Y_test, predict, target_names=uniques))

#Calculating values for the confusion matrix
tn = np.sum(np.diag(cm)) - np.sum(cm[0, :])
fp = np.sum(cm[0, :]) - tn
fn = np.sum(cm[:, 0]) - tn
tp = np.sum(np.diag(cm))

# Calculating additional metrics
false_alarm_rate = fp / (fp + tn)
true_negative_rate = tn / (tn + fp)

#Displaying metrics
print("Precision: ", precision)
print("Recall: ", recall)
print("False Alarm Rate: ", false_alarm_rate)
print("True Negative Rate: ", true_negative_rate)
print("Accuracy: ", accuracy)

#Feature importance
importance = clf.feature_importances_
important_feature_dict = {}
for idx, val in enumerate(importance):
    important_feature_dict[idx] = val
important_feature_list = sorted(important_feature_dict, key=important_feature_dict.get, reverse=True)

print(f'The 10 most important features :  {important_feature_list[:10]}')

#Displaying the decision tree
fn = ["forward_bps_var",
             "tp_src", "tp_dst", "nw_proto",
             "forward_pc", "forward_bc", "forward_pl",
             "forward_piat", "forward_pps", "forward_bps", "forward_pl_mean",
             "forward_piat_mean", "forward_pps_mean", "forward_bps_mean", "forward_pl_var", "forward_piat_var",
             "forward_pps_var",     "forward_pl_q1",    "forward_pl_q3",
             "forward_piat_q1",     "forward_piat_q3",  "forward_pl_max", "forward_pl_min",
             "forward_piat_max",    "forward_piat_min", "forward_pps_max", "forward_pps_min",
             "forward_bps_max",     "forward_bps_min", "forward_duration", "forward_size_packets",
             "forward_size_bytes", "reverse_pc", "reverse_bc", "reverse_pl", "reverse_piat",
             "reverse_pps", "reverse_bps", "reverse_pl_mean", "reverse_piat_mean", "reverse_pps_mean",
             "reverse_bps_mean", "reverse_pl_var", "reverse_piat_var", "reverse_pps_var",
             "reverse_pl_q1",    "reverse_pl_q3", "reverse_piat_q1",     "reverse_piat_q3",  "reverse_pl_max", 
             "reverse_pl_min", "reverse_piat_max",    "reverse_piat_min", "reverse_pps_max", "reverse_pps_min",
             "reverse_bps_max",     "reverse_bps_min", "reverse_duration", "reverse_size_packets","reverse_size_bytes" ]
la = ['WWW', 'DNS', 'FTP', 'ICMP', 'P2P', 'VOIP']
plt.figure(2, dpi=300)
fig = plot_tree(clf, filled=True, feature_names=fn, class_names=la)
plt.title("Decision Tree trained on all the features")
plt.show()

#%% #%% ID3
#Importing necessary libraries
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import export_text, plot_tree, DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

#Loading data from the CSV file
dataset = pd.read_csv('SDN_traffic.csv')

#Displaying the first rows of the dataset
print('\n\nhead:\r')
print(dataset.head())

#Displaying information about the dataset
print('\n\ninfo:\r')
print(dataset.info())

#Displaying descriptive statistics of the dataset
print('\n\ndescribe:\r')
print(dataset.describe())

#Checking for duplicates in the dataset
print('\n\nduplicated:\r')
print(dataset.duplicated())

#Selecting the necessary features
X = dataset[["forward_bps_var", "tp_src", "tp_dst", "nw_proto", "forward_pc", "forward_bc", "forward_pl",
             "forward_piat", "forward_pps", "forward_bps", "forward_pl_mean", "forward_piat_mean", 
             "forward_pps_mean", "forward_bps_mean", "forward_pl_var", "forward_piat_var", "forward_pps_var",
             "forward_pl_q1", "forward_pl_q3", "forward_piat_q1", "forward_piat_q3", "forward_pl_max", 
             "forward_pl_min", "forward_piat_max", "forward_piat_min", "forward_pps_max", "forward_pps_min",
             "forward_bps_max", "forward_bps_min", "forward_duration", "forward_size_packets", 
             "forward_size_bytes", "reverse_pc", "reverse_bc", "reverse_pl", "reverse_piat", "reverse_pps", 
             "reverse_bps", "reverse_pl_mean", "reverse_piat_mean", "reverse_pps_mean", "reverse_bps_mean", 
             "reverse_pl_var", "reverse_piat_var", "reverse_pps_var", "reverse_pl_q1", "reverse_pl_q3", 
             "reverse_piat_q1", "reverse_piat_q3", "reverse_pl_max", "reverse_pl_min", "reverse_piat_max", 
             "reverse_piat_min", "reverse_pps_max", "reverse_pps_min", "reverse_bps_max", "reverse_bps_min", 
             "reverse_duration", "reverse_size_packets", "reverse_size_bytes"]]

#Correcting incorrect values
X.loc[1877, 'forward_bps_var'] = float(11968065203349)
X.loc[1931, 'forward_bps_var'] = float(12880593804833)
X.loc[2070, 'forward_bps_var'] = float(9022747730895)
X.loc[2381, 'forward_bps_var'] = float(39987497172945)
X.loc[2562, 'forward_bps_var'] = float(663300742992)
X.loc[2567, 'forward_bps_var'] = float(37770223877794)
X.loc[2586, 'forward_bps_var'] = float(97227875083751)
X.loc[2754, 'forward_bps_var'] = float(18709751403737)
X.loc[2765, 'forward_bps_var'] = float(33969277035759)
X.loc[2904, 'forward_bps_var'] = float(39204786962856)
X.loc[3044, 'forward_bps_var'] = float(9169996063653)
X.loc[3349, 'forward_bps_var'] = float(37123283690575)
X.loc[3507, 'forward_bps_var'] = float(61019064590464)
X.loc[3610, 'forward_bps_var'] = float(46849620984072)
X.loc[3717, 'forward_bps_var'] = float(97158873841506)
X.loc[3845, 'forward_bps_var'] = float(11968065203349)
X.loc[3868, 'forward_bps_var'] = float(85874278395372)



#Converting the 'forward_bps_var' column to numeric type
X['forward_bps_var'] = pd.to_numeric(X['forward_bps_var'])

#Displaying information about X
print(X.info())

#Selecting the target variable (Y) and converting it to a NumPy array
Y = dataset[['category']].to_numpy().ravel()
# Converting labels to numeric values
labels, uniques = pd.factorize(Y)
Y = labels.ravel()

print('New value code of the Category column (but reshaped in a horizontal vector)')
print(Y)

#Normalizing features
X = stats.zscore(X)
X = np.nan_to_num(X)

#Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.7)

#Initializing and training the model
clf_id3 = DecisionTreeClassifier(criterion="entropy", random_state=0, max_depth=6)

clf_id3.fit(X_train, Y_train)
print(clf_id3)

#Evaluating the ID3 model
cv = KFold(n_splits=10, random_state=0, shuffle=True)
accuracy_id3 = clf_id3.score(X_test, Y_test)
Kfold10_accuracy_id3 = cross_val_score(clf_id3, X_train, Y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print("Kfold10 accuracy mean for ID3: ")
print(Kfold10_accuracy_id3.mean())
predict_id3 = clf_id3.predict(X_test)
cm_id3 = confusion_matrix(Y_test, predict_id3)
precision_id3 = precision_score(Y_test, predict_id3, average='weighted', labels=np.unique(predict_id3))
recall_id3 = recall_score(Y_test, predict_id3, average='weighted', labels=np.unique(predict_id3))
f1scoreMacro_id3 = f1_score(Y_test, predict_id3, average='macro', labels=np.unique(predict_id3))
print("Classification report for ID3: ")
print(classification_report(Y_test, predict_id3, target_names=uniques))

#Calculating confusion matrix values for ID3
tn_id3 = np.sum(np.diag(cm_id3)) - np.sum(cm_id3[0, :])
fp_id3 = np.sum(cm_id3[0, :]) - tn_id3
fn_id3 = np.sum(cm_id3[:, 0]) - tn_id3
tp_id3 = np.sum(np.diag(cm_id3))

#Calculating additional metrics for ID3
false_alarm_rate_id3 = fp_id3 / (fp_id3 + tn_id3)
true_negative_rate_id3 = tn_id3 / (tn_id3 + fp_id3)

#Displaying metrics for ID3
print("Precision for ID3: ", precision_id3)
print("Recall for ID3: ", recall_id3)
print("False Alarm Rate for ID3: ", false_alarm_rate_id3)
print("True Negative Rate for ID3: ", true_negative_rate_id3)
print("Accuracy for ID3: ", accuracy_id3)

#Feature Importance
importance = clf_id3.feature_importances_
important_feature_dict = {}
for idx, val in enumerate(importance):
    important_feature_dict[idx] = val
important_feature_list = sorted(important_feature_dict, key=important_feature_dict.get, reverse=True)

print(f'The 10 most important features :  {important_feature_list[:10]}')

#Displaying the decision tree
fn = ["forward_bps_var",
             "tp_src", "tp_dst", "nw_proto",
             "forward_pc", "forward_bc", "forward_pl",
             "forward_piat", "forward_pps", "forward_bps", "forward_pl_mean",
             "forward_piat_mean", "forward_pps_mean", "forward_bps_mean", "forward_pl_var", "forward_piat_var",
             "forward_pps_var",     "forward_pl_q1",    "forward_pl_q3",
             "forward_piat_q1",     "forward_piat_q3",  "forward_pl_max", "forward_pl_min",
             "forward_piat_max",    "forward_piat_min", "forward_pps_max", "forward_pps_min",
             "forward_bps_max",     "forward_bps_min", "forward_duration", "forward_size_packets",
             "forward_size_bytes", "reverse_pc", "reverse_bc", "reverse_pl", "reverse_piat",
             "reverse_pps", "reverse_bps", "reverse_pl_mean", "reverse_piat_mean", "reverse_pps_mean",
             "reverse_bps_mean", "reverse_pl_var", "reverse_piat_var", "reverse_pps_var",
             "reverse_pl_q1",    "reverse_pl_q3", "reverse_piat_q1",     "reverse_piat_q3",  "reverse_pl_max", 
             "reverse_pl_min", "reverse_piat_max",    "reverse_piat_min", "reverse_pps_max", "reverse_pps_min",
             "reverse_bps_max",     "reverse_bps_min", "reverse_duration", "reverse_size_packets","reverse_size_bytes" ]
la = ['WWW', 'DNS', 'FTP', 'ICMP', 'P2P', 'VOIP']
plt.figure(2, dpi=300)
fig = plot_tree(clf_id3, filled=True, feature_names=fn, class_names=la)
plt.title("Decision Tree trained on all the features")
plt.show()

# %% Metrics Comparison:

import matplotlib.pyplot as plt

#Comparison of precision, recall, false alarm rate, and true negative rate
metrics_names = ['Precision', 'Recall', 'False Alarm Rate', 'True Negative Rate']
cart_metrics = [precision, recall, false_alarm_rate, true_negative_rate]
id3_metrics = [precision_id3, recall_id3, false_alarm_rate_id3, true_negative_rate_id3]

#Creating a bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
bar_positions_cart = range(len(metrics_names))
bar_positions_id3 = [pos + bar_width for pos in bar_positions_cart]

#Adding bars for CART and ID3 with colors, width, and grid
plt.bar(bar_positions_cart, cart_metrics, width=bar_width, label='CART', color='pink', edgecolor='black', linewidth=1.5)
plt.bar(bar_positions_id3, id3_metrics, width=bar_width, label='ID3', color='green', edgecolor='black', linewidth=1.5)

#Adding labels and title
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Comparison of Metrics between CART and ID3')
plt.xticks([pos + bar_width/2 for pos in bar_positions_cart], metrics_names)
plt.legend()

#Adding grid
plt.grid(axis='y', linestyle='--', alpha=0.7)

#Setting the vertical axis scale with values spaced by 0.02
plt.yticks(range(0, 2.2, 0.02))

#Displaying the chart
plt.show()

#%%

import matplotlib.pyplot as plt
import numpy as np

#Metrics for each algorithm
algorithms = ['CART', 'ID3']
precision_values = [0.8059956574452123, 0.8053431163665464]
recall_values = [0.8181511470985156, 0.8157894736842105]
far_values = [0.5760422783323547, 0.5801526717557252]
tnr_values = [0.42395772166764534, 0.4198473282442748]
accuracy_values = [0.8181511470985156, 0.8157894736842105]

#Side-by-side bars
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.15
index = np.arange(len(algorithms))

bar1 = ax.bar(index, precision_values, bar_width, label='Precision', color='#FF69B4')  # Pink
bar2 = ax.bar(index + bar_width, recall_values, bar_width, label='Recall', color='#87CEEB')  # Light Blue
bar3 = ax.bar(index + 2 * bar_width, far_values, bar_width, label='False Alarm Rate', color='#98FB98')  # Pastel Green
bar4 = ax.bar(index + 3 * bar_width, tnr_values, bar_width, label='True Negative Rate', color='#FFD700')  # Yellow
bar5 = ax.bar(index + 4 * bar_width, accuracy_values, bar_width, label='Accuracy', color='#DA70D6')  # Light Purple

ax.set_xlabel('Algorithms')
ax.set_ylabel('Metric Values')
ax.set_title('Comparison of Metrics between CART and ID3')
ax.set_xticks(index + 2 * bar_width)
ax.set_xticklabels(algorithms)

#Hide values on the left y-axis
ax.set_yticklabels([])

#Adding a grid with a scale from 0 to 1 in steps of 0.02
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, which='major', color='black', alpha=0.2)
ax.set_yticks(np.arange(0, 1.002, 0.02))

#Adding the legend outside
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

#Adding values above the bars
for i, values in enumerate(zip(precision_values, recall_values, far_values, tnr_values, accuracy_values)):
    for j, value in enumerate(values):
        ax.text(i + j * bar_width, value + 0.02, f'{value:.3f}', ha='center')

plt.show()



