import numpy as np
import pandas as pd
# ------- Prepairing DATA -----------

pattern = str(5)

X_train_temp = np.load('Features and Models/'+pattern+'/X_train.npy')
Y_train_temp = np.load('Features and Models/'+pattern+'/Y_train.npy')
X_test = np.load('Features and Models/'+pattern+'/X_test.npy')
Y_test = np.load('Features and Models/'+pattern+'/Y_test.npy')

X_max = np.load('Features and Models/'+pattern+'/X_max.npy')
X_min = np.load('Features and Models/'+pattern+'/X_min.npy')
Y_max = np.load('Features and Models/'+pattern+'/Y_max.npy')
Y_min = np.load('Features and Models/'+pattern+'/Y_min.npy')

lst_random = np.random.permutation(len(Y_train_temp))
X_train = np.zeros_like(X_train_temp)
Y_train = np.zeros_like(Y_train_temp)
for i in range(lst_random.shape[0]):
    X_train[i,:] = X_train_temp[lst_random[i] , :]
    Y_train[i] = Y_train_temp[lst_random[i]]
np.save('Features and Models/'+pattern+'/X_train_permute' , X_train)
np.save('Features and Models/'+pattern+'/Y_train_permute', Y_train)


X_train = (X_train - X_min)/(X_max - X_min)
Y_train = (Y_train - Y_min)/(Y_max - Y_min)
X_test = (X_test - X_min)/(X_max - X_min)
Y_test = (Y_test - Y_min)/(Y_max - Y_min)
Ali


# ------- Fitting and Evaluating AI Model -----------
from sklearn.svm import SVR,NuSVR,LinearSVR
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error as mse
from sklearn.externals import joblib
from LevenBerg1 import NN_LM_RG
from Plot_output import plot_curve
import Compute_Result as CR

# M1 = NN_LM_RG(layers=(4,22,1),lr=0.5,activaton_func='sigmoid',max_epoch=100, batch_size=100,Best_weights=True)
# loss , val_loss = M1.fit(X_train, Y_train, X_test, Y_test)
"""
M1 = MLPRegressor(hidden_layer_sizes=(10, 200), activation='logistic', solver='adam', batch_size=10, learning_rate='adaptive',
                  learning_rate_init=0.005, max_iter=500, shuffle=True, tol=0.0000001, verbose=True,
                  momentum=0.9).fit(X_train, Y_train)
"""

#M1 = NuSVR(nu=1, C=0.9, kernel='poly', gamma='scale', tol=0.0000001).fit(X_train, Y_train)
M1 = KNN(n_neighbors=2, weights='uniform', algorithm='auto', p=1, metric='minkowski').fit(X_train,Y_train)


CR.Get_result(X_train, Y_train, (Y_max,Y_min), X_test, Y_test, M1)
joblib.dump(M1, 'Features and Models/'+pattern+'/Models/M1/KNN.joblib')