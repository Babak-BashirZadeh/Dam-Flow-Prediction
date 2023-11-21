import numpy as np
import pandas as pd


# ------- Prepairing DATA -----------

X_train_temp = np.load('Features and Models/5/X_train.npy')
Y_train_temp = np.load('Features and Models/5/Y_train.npy')
X_max = np.load('Features and Models/5/X_max.npy')
X_min = np.load('Features and Models/5/X_min.npy')
Y_max = np.load('Features and Models/5/Y_max.npy')
Y_min = np.load('Features and Models/5/Y_min.npy')

lst_random = np.random.permutation(len(Y_train_temp))
X_train = np.zeros_like(X_train_temp)
Y_train = np.zeros_like(Y_train_temp)
for i in range(lst_random.shape[0]):
    X_train[i,:] = X_train_temp[lst_random[i] , :]
    Y_train[i] = Y_train_temp[lst_random[i]]
np.save('Features and Models/5/X_train_permute' , X_train)
np.save('Features and Models/5/Y_train_permute', Y_train)


X_test = np.load('Features and Models/5/X_test.npy')
Y_test = np.load('Features and Models/5/Y_test.npy')
X_train = np.load('Features and Models/5/X_train_permute.npy')
Y_train = np.load('Features and Models/5/Y_train_permute.npy' )

X_train = (X_train - X_min)/(X_max - X_min)
Y_train = (Y_train - Y_min)/(Y_max - Y_min)
X_test = (X_test - X_min)/(X_max - X_min)
Y_test = (Y_test - Y_min)/(Y_max - Y_min)
ali

# ------- Predictive Models ---------
from sklearn.linear_model import LinearRegression ,ridge_regression , Ridge ,LogisticRegression
from sklearn.linear_model import Lasso ,BayesianRidge

from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsRegressor , RadiusNeighborsRegressor
from sklearn.metrics import mean_squared_error as sk_mse , r2_score
from sklearn.svm import LinearSVR,NuSVR , SVR
from sklearn.neural_network import MLPRegressor
import Compute_Result as CR
from keras.models import Sequential
import keras
from keras import layers , models , Model
from keras.layers import Dense , Dropout , BatchNormalization , Conv2D , Flatten , LSTM , Bidirectional
from keras.activations import relu , linear , tanh , sigmoid , elu , softmax
from keras.optimizers import SGD , Adam , Adadelta
from keras.losses import categorical_crossentropy, mean_squared_error as mse
from keras.models import save_model
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)


M1 = MLPRegressor(hidden_layer_sizes=(20,100) , activation='tanh' ,
                  solver = 'adam' , learning_rate='adaptive' ,
                  verbose=True , tol = 1e-5 , max_iter= 1000,
                  shuffle=True , momentum=0.9 , batch_size=10).fit(X_train,Y_train)

#M1 = SVR(kernel='rbf' , gamma='auto' , C=10).fit(X_train,Y_train)
#M1 = NuSVR(nu=0.5 , C=1,kernel='linear',gamma='auto',degree=4).fit(X_train,Y_train)
#M1=LinearSVR(C=10,dual=True).fit(X_train,Y_train)
#M1 = Ridge(alpha=0.5).fit(X_train,Y_train)
#M1 = LinearRegression().fit(X_train , Y_train)
#M1 = BayesianRidge(n_iter=1000 , tol=1e-5,alpha_1=1e-3,alpha_2=1e-3).fit(X_train,Y_train)
#M1 = RadiusNeighborsRegressor(radius=5,weights='uniform',algorithm='auto',p=1).fit(X_train,Y_train)
#M1 = KNeighborsRegressor(n_neighbors=3 , weights='distance' , p=2 , algorithm='auto').fit(X_train,Y_train)
result = CR.Get_result(X_train , Y_train,(Y_max,Y_min) , X_test , Y_test , M1)
# joblib.dump(M1,'Features and Models/5/Models/M1/Nu_SVR.joblib')
#

# X_train1 = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
# X_test1 = X_test.reshape(X_test.shape[0] , X_test.shape[1],1)
# from keras.models import Sequential
# from keras.activations import selu , exponential
# M1 = Sequential()
# M1.add(Dense(15 , activation=exponential,input_shape=(X_train.shape[1],)))
# M1.add(Dropout(0.1))
# M1.add(Dense(7, activation=exponential))
# #M1.add(Dropout(0.1))
# #M1.add(Dense(25,activation=sigmoid))
# #M1.add(Dropout(0.1))
# M1.add(Dense(1,activation=relu))
# M1.compile(optimizer=Adam(learning_rate=0.001) , loss=mse)
# train_detail = M1.fit(x=X_train,y=Y_train,batch_size=8 , epochs=300, validation_data=(X_test , Y_test))
# #
# loss = train_detail.history['loss']
# loss_test = train_detail.history['val_loss']
# # #M1.save('Features and Models/5/Models/M1/M2.h5')
# result = CR.Get_result(X_train , Y_train,(Y_max,Y_min) , X_test , Y_test , M1)




from LevenBerg1 import NN_LM_RG
M1 = NN_LM_RG(layers=(5,16,1), lr=1 ,activaton_func='sigmoid',max_epoch=500 , batch_size=50 , Best_weights=True)
loss , val_loss = M1.fit(X_train,Y_train,X_test,Y_test)
#joblib.dump(M1 , 'Features and Models/5/Models/M1/M1_LVM.joblib')
#M1 = joblib.load('Features and Models/5/Models/M1/M1_LVM.joblib')
#np.save('Features and Models/5/Models/M1/W_LVM',W)
result = CR.Get_result(X_train , Y_train,(Y_max,Y_min) , X_test , Y_test , M1)


import matplotlib.pyplot as plt
M1 = joblib.load('Features and Models/5/Models/M1/Nu_SVR.joblib')
X_train = (np.load('Features and Models/5/X_train.npy') - X_min)/(X_max - X_min)
X_test = (np.load('Features and Models/5/X_test.npy') - X_min)/(X_max - X_min)
Y_train = np.load('Features and Models/5/Y_train.npy')
Y_test  = np.load('Features and Models/5/Y_test.npy')
X = (np.load('Features and Models/5/X.npy') - X_min)/(X_max - X_min)
Y = np.load('Features and Models/5/Y.npy')
Y_predict = M1.predict(X_train) * (Y_max - Y_min) + Y_min

plot_curve((Y_train , Y_predict) , ('observed' , 'Levenbrg Marquadrt Model') , ('-b','--r')
           , 'Time period (month)' , 'inflow (MCM)')
#
# plt.plot(np.arange(len(Y_temp)) , Y_temp , 'b-', label='')
import matplotlib.pyplot as plt
def plot_curve(data , labels , color , x_label , y_label):
    plt.figure(figsize=(10,5))

    for i in range(len(data)) :
        curve = data[i]
        lbl = labels[i]
        clr = color[i]
        plt.plot(np.arange(curve.shape[0]),curve , clr , label=lbl)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
    plt.legend()
    plt.show()