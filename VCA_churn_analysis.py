# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 10:29:19 2018

@author: hp
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from sklearn.preprocessing import scale
from numpy import sort
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
#Load data
churn_df = pd.read_csv("E:/VCA Business Case Challenge/hackathon_final.csv")
col_names = churn_df.columns.tolist()
#Remove unnecessary columns
to_drop = ["issuer", "acct_card_no"]
churn_feature_space = churn_df.drop(to_drop, axis=1)
churn_feature_space.dtypes
#Check for non-numeric columns
categorical_columns = [x for x in churn_feature_space.dtypes.index if churn_feature_space.dtypes[x] == 'object']
print(categorical_columns)
#Binning vintage_months field
churn_feature_space["vintage_months_bin"] = pd.cut(churn_feature_space["vintage_months"],3, labels=[1,2,3])
churn_feature_space.drop("vintage_months", inplace = True, axis = 1)
#Redundant features
domestic_features = ["d_active_months",
                "B_D_LST_1MNTH_F2F_TXNS",
                "B_D_LST_1MNTH_MG_RT_AMT",
                "B_D_LST_1MNTH_TOTAL_AMT",
                "B_D_LST_1MNTH_TOTAL_TXNS",
                "B_D_LST_3MNTHS_TOTAL_TXNS",
                "B_D_LST_6MNTHS_TOTAL_TXNS",
                "B_D_LST_9MNTHS_TOTAL_TXNS",
                "B_D_PRV_6MNTHS_TOTAL_AMT",
                "B_D_PRV_6MNTHS_TOTAL_TXNS",
                "D_MCC_Lst_1Mnth_cnt",
                "d_MG_Emrgin_active_months",
                "RATIO_D_TOTAL_AMT_L1_L6",
                "RATIO_D_TOTAL_AMT_L6_P6",
                "RATIO_D_TOTAL_TXNS_L1_L6",
                "RATIO_D_TOTAL_AMT_L3_P3"]
redundant_features = ["B_BP_LST_1MNTH_TOTAL_AMT",
                      "B_BP_LST_1MNTH_TOTAL_TXNS",
                      "B_LAST_1MNTH_F2F_AMT",
                      "B_LAST_1MNTH_MG_EDAY_AMT",
                      "B_LAST_1MNTH_MG_EMRGIN_AMT",
                      "B_LAST_1MNTH_F2F_TXNS",
                      "B_LAST_3MNTHS_F2F_TXNS",
                      "B_LAST_3MNTHS_MG_EDAY_TXNS",
                      "B_LAST_3MNTHS_MG_EMRGIN_TXNS",
                      "B_DECL_COUNT",
                      "RATIO_DI_MG_EMRGIN_AMT_L1_L3",
                      "RATIO_DI_MS_TRASPRT_AMT_L1_L3",
                      "RATIO_D_MG_EMRGIN_TXNS_L1_L6",
                      "RATIO_D_MG_RT_AMT_L3_P3",
                      "RATIO_DI_F2F_AMT_L3_P3",
                      "RATIO_DI_F2F_TXNS_L1_L3",
                      "RATIO_DI_MS_INSU_AMT_L3_P3"]
merged_features = domestic_features + redundant_features
churn_feature_space = churn_feature_space.drop(merged_features, axis=1)
#Train and Test split
churn_train = churn_feature_space[churn_feature_space['Data_Set'] == "train"]
churn_test = churn_feature_space[churn_feature_space['Data_Set'] == "test"]
#Target value
y_train = churn_train["Dormant_Flag"]
y_test = churn_test["Dormant_Flag"]
y = y_train + y_test
#Remove features: Dormant_Flag and Data_Set
X_train = churn_train.drop(["Dormant_Flag", "Data_Set"], axis=1)
X_test = churn_test.drop(["Dormant_Flag", "Data_Set"], axis=1)
X = X_train + X_test
#Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
#Implementing KFold Cross Validation on Train set
#kf = KFold(len(X_train), n_folds=2, shuffle = True, random_state = None)

#Initialising a Classifier
###RANDOM FOREST###
clf = RandomForestClassifier(n_estimators = 30)
clf.fit(X_train_scaled, y_train)
#Predictions on test data
predictions = clf.predict(X_test_scaled)
probs = clf.predict_proba(X_test_scaled)
display(predictions)
#Evaluation of the model
score_rf = clf.score(X_test_scaled, y_test)
print("Accuracy for Random Forest Classifier: ", score_rf)
#Confusion matrix
get_ipython().magic('matplotlib inline')
confusion_matrix = pd.DataFrame(
        confusion_matrix(y_test, predictions),
        columns = ["Predicted False", "Predicted True"],
        index = ["Actual False", "Actual True"]
        )
display(confusion_matrix)
#Calculation of FPR and TPR
fpr, tpr, threshold = roc_curve(y_test, probs[:,1])
plt.title("Receiver Operating Characteristics")
plt.plot(fpr, tpr, 'b')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True positive rate")
plt.xlabel("False positive rate")
plt.show()
#Feature importance
fig = plt.figure(figsize=(25,15))
ax = fig.add_subplot(111)

df_f = pd.DataFrame(clf.feature_importances_, columns = ["importance"])
features = churn_feature_space.drop(["Dormant_Flag", "Data_Set"], axis=1)
features = features.columns.tolist()
df_f["labels"] = features
df_f.sort_values("importance", inplace = True, ascending = False)
display(df_f.head(5))

index = np.arange(len(clf.feature_importances_))
bar_width = 0.5
rects = plt.barh(index, df_f["importance"], bar_width, alpha=0.4, color='b', label="Main")
plt.yticks(index, df_f["labels"])
plt.show()

#Churn propensity
churn_test["prob_true"] = probs[:, 1]
df_risky = churn_test[churn_test["prob_true"] > 0.9]
display(df_risky.head(5)[["prob_true"]])
df_risky["acct_card_no"] = churn_df["acct_card_no"]
df = df_risky[["acct_card_no", "prob_true"]]
df = df.sort_values(df.columns[1], ascending = False)
df.to_csv("E:/Python Exercises/VCA Churn prediction/Customers_highest_probability.csv", index = False)

###K-NEAREST NEIGHBORS
knn = KNeighborsClassifier(n_neighbors=20 , n_jobs=2 , weights='distance')
knn.fit(X_train_scaled, y_train)
#Predictions on test data
predictions = knn.predict(X_test_scaled)
probs = knn.predict_proba(X_test_scaled)
display(predictions)
#Evaluation of the model
score_knn = knn.score(X_test_scaled, y_test)
print("Model score for K-Nearest Neighbors:{:.3f}".format(score_knn))

###GRADIENT BOOSTED DECISION TREE
gb = GradientBoostingClassifier(learning_rate = 0.01, max_depth=2, random_state=0)
gb.fit(X_train_scaled, y_train)
#Predictions on test data
predictions = gb.predict(X_test_scaled)
probs = gb.predict_proba(X_test_scaled)
display(predictions)
#Evaluation of the model
score_gbdt = gb.score(X_test_scaled, y_test)
print('Accuracy of GBDT classifier on test set: {:.3f}'.format(score_gbdt))

###LOGISTIC REGRESSION
logistic = LogisticRegression(C=1)
logistic.fit(X_train_scaled, y_train)
#Predictions on test data
predictions = logistic.predict(X_test_scaled)
probs = logistic.predict_proba(X_test_scaled)
display(predictions)
#Evaluation of the model
score_lr = logistic.score(X_test_scaled, y_test)
print('Accuracy of Logistic Regression on test set: {:.3f}'.format(score_lr))

###GAUSSIAN NAIVE BAYES CLASSIFIER
gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train_scaled, y_train)
#Predictions on test data
predictions = gaussian_nb.predict(X_test_scaled)
probs = gaussian_nb.predict_proba(X_test_scaled)
display(predictions)
#Evaluation of the model
score_gaussian_nb = gaussian_nb.score(X_test_scaled, y_test)
print('Accuracy of Gaussian Naive Bayes on test set: {:.3f}'.format(score_gaussian_nb))

###SINGLE LAYER PERCEPTRON
score_single_MLP = []
for units in ([1,10,25,100]):
        single_MLP = MLPClassifier(hidden_layer_sizes=[units], solver='lbfgs', random_state=0)
        single_MLP.fit(X_train_scaled, y_train)
        #Predictions on test data
        predictions = single_MLP.predict(X_test_scaled)
        probs = single_MLP.predict_proba(X_test_scaled)
        display(predictions)
        score_mlp = single_MLP.score(X_test_scaled, y_test)
        score_single_MLP.append(score_mlp)
maxscore = max(score_single_MLP)
print('Maximum accuracy of Single Layer Perceptron on test set: {:.3f}'.format(maxscore))
        
###MULTILAYER PERCEPTRON
MLP = MLPClassifier(hidden_layer_sizes = [10, 100], solver='lbfgs',random_state = 0)
MLP.fit(X_train_scaled, y_train)
predictions = MLP.predict(X_test_scaled)
probs = MLP.predict_proba(X_test_scaled)
display(predictions)       
score_MLP = MLP.score(X_test_scaled, y_test) 
print('Accuracy of Multiple Layer Perceptron on test set: {:.3f}'.format(score_MLP))

###KERAS ANN###
model = Sequential()
model.add(Dense(64, activation = 'relu', input_dim = 23))
model.add(Dropout(0.9))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.9))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.9))
model.add(Dense(1, activation = 'sigmoid'))
sgd = SGD(lr = 0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer =sgd, metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs= 35, batch_size = 30, validation_split=0.3)
score_NN = model.evaluate(X_test_scaled, y_test, batch_size=30)
model.summary()

#Consolidated score
consolidated_score = [{'ALGORITHM': 'Keras Neural Network', 'SCORE': score_NN[1]},
         {'ALGORITHM': 'Multilayer Perceptron', 'SCORE': score_MLP},
         {'ALGORITHM': 'Gaussian Naive Bayes', 'SCORE': score_gaussian_nb },
         {'ALGORITHM': 'Gradient Boosted Decision tree', 'SCORE': score_gbdt },
         {'ALGORITHM': 'K Nearest Neighbor', 'SCORE': score_knn },
         {'ALGORITHM': 'Logistic Regression', 'SCORE': score_lr },
         {'ALGORITHM': 'Random Forests', 'SCORE': score_rf },
         {'ALGORITHM': 'Single MLP', 'SCORE': maxscore },
         ]
df = pd.DataFrame(consolidated_score)
print(df)
df['SCORE'].idxmax()
print("Maximum Accuracy model")
print(df.loc[df['SCORE'].idxmax()])