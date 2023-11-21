from sklearn.linear_model import LinearRegression ,ridge_regression , Ridge ,LogisticRegression
from sklearn.linear_model import Lasso ,BayesianRidge

from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsRegressor , RadiusNeighborsRegressor
from sklearn.metrics import mean_squared_error as sk_mse , r2_score
from sklearn.svm import LinearSVR,NuSVR , SVR
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as mse
from sklearn.svm import LinearSVR


def Get_result(X_train,Y_train,maxmin_vector,X_test , Y_test, Model):
    Y_max = maxmin_vector[0]
    Y_min = maxmin_vector[1]
    M1 = Model
    M1_predict_train = M1.predict(X_train) * (Y_max - Y_min) + Y_min
    M1_predict_test = M1.predict(X_test) * (Y_max - Y_min) + Y_min
    Y_train_pure = Y_train * (Y_max - Y_min) + Y_min
    Y_test_pure = Y_test * (Y_max - Y_min) + Y_min
    M1_mseR_train = round(np.sqrt(sk_mse(Y_train_pure, M1_predict_train)),2)
    M1_mseR_test = round(np.sqrt(sk_mse(Y_test_pure, M1_predict_test)),2)
    r2_train = r2_score(Y_train_pure, M1_predict_train)
    r2_test = r2_score(Y_test_pure, M1_predict_test)
    print(r2_train , '    ',r2_test)

    M1_R_train = round(np.sqrt(r2_train),4)
    M1_R_test = round(np.sqrt(r2_test),4)

    return np.array([M1_mseR_train , M1_R_train , M1_mseR_test,M1_R_test])
