import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as mse
from sklearn.svm import LinearSVR
from keras.models import Sequential
import keras
from keras import layers , models , Model
from keras.layers import Dense , Dropout , BatchNormalization , Conv2D , Flatten
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


# ------- Prepairing DATA -----------
data = pd.read_excel("Dataset/inputs.xlsx")
data = data.to_numpy()
Y_temp = data[: , 1]
X_temp = data[: , 2:4] # halate 2

'''
X_temp = data[: , 2] # halate 1
X_temp = data[: , 2:5] # halate 3
X_temp = data[: , 2:6] # halate 4
X_temp = data[: , 2:7] # halate 5
X_temp = data[: , 2:8] # halate 6
X_temp = data[: , 2:9] # halate 7
X_temp = data[: , 2:]  # halate 8
'''

X = []
Y = []
permute = np.random.permutation(len(Y_temp))
for i in range(len(permute)):
    #print(i)
    idx = permute[i]
    X.append(X_temp[idx,:])
    Y.append(Y_temp[idx])

np.save('Features/2/X',X)
np.save('Features/2/Y',Y)

tr_idx = int(len(Y_temp)*0.7)
val_idx = int(len(Y_temp)* 0.85)
te_idx = len(X)
X = np.array(X)
Y = np.array(Y)

X_train = X[:tr_idx ,:]
#X_trian = X_train.reshape((len(X_train) , 1))
Y_train = Y[:tr_idx ]

X_val = X[tr_idx:val_idx,: ]
#X_val = X_val.reshape((len(X_val) , 1))
Y_val = Y[tr_idx:val_idx]

X_test = X[val_idx : ,:]
#X_test = X_test.reshape((len(X_test) , 1))
Y_test = Y[val_idx : ]

X_max = np.max(X_train , 0)
X_min = np.min(X_train , 0)
Y_max = np.max(Y_train)
Y_min = np.min(Y_train)
np.save('Features/2/X_max',X_max)
np.save('Features/2/X_min',X_min)
np.save('Features/2/Y_max',Y_max)
np.save('Features/2/Y_min',Y_min)


X_train = (X_train - X_max) / (X_max - X_min)
Y_train = (Y_train - Y_min)/(Y_max - Y_min)

X_val = (X_val - X_max) / (X_max - X_min)
Y_val = (Y_val - Y_min)/(Y_max - Y_min)

X_test = (X_test - X_max) / (X_max - X_min)
Y_test = (Y_test - Y_min)/(Y_max - Y_min)


# ------- Predictive Models ---------
from sklearn.linear_model import LinearRegression ,ridge_regression
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsRegressor , RadiusNeighborsRegressor
from sklearn.metrics import mean_squared_error as sk_mse , r2_score
from sklearn.svm import LinearSVR,NuSVR , SVR
from keras.applications import
from matplotlib.pyplot import

# M1 = MLPRegressor(hidden_layer_sizes=(100,20) , activation='tanh' ,
#                   solver = 'adam' , learning_rate='adaptive' ,
#                   verbose=True , tol = 1e-10 , max_iter= 1000,
#                   shuffle=True , momentum=0.95 , batch_size=5)

#M1 = SVR(kernel='rbf' , gamma='auto' , C=10)
M1 = KNeighborsRegressor(n_neighbors=10 , weights='distance' , p=1 , algorithm='ball_tree')


M1.fit(X_train , Y_train)

M1_predict_train = M1.predict(X_train) * (Y_max - Y_min)  + Y_min
M1_predict_val = M1.predict(X_val) * (Y_max - Y_min)  + Y_min
M1_predict_test = M1.predict(X_test) * (Y_max - Y_min)  + Y_min
Y_train = Y_train *(Y_max - Y_min) + Y_min
Y_val = Y_val *(Y_max - Y_min) + Y_min
Y_test = Y_test *(Y_max - Y_min) + Y_min

M1_R_train = r2_score(Y_train , M1_predict_train)
M1_R_val = r2_score(Y_val , M1_predict_val)
M1_R_test = r2_score(Y_test , M1_predict_test)

M1_mseR_train = np.sqrt(sk_mse(Y_train, M1_predict_train))
M1_mseR_val = np.sqrt(sk_mse(Y_val , M1_predict_val))
M1_mseR_test = np.sqrt(sk_mse(Y_test, M1_predict_test))


'''
import matplotlib.pyplot as plt

plt.figure(1)
# plot actual output and output of network for train data
plt.subplot(2 ,1 ,1)
plt.plot(np.arange(len(Y_train)) , Y_train , 'b-' , marker = 'o')
plt.plot(np.arange(len(Y_train)) , train_pred , 'r-' , marker = '*')
plt.xlabel('train samples')
plt.ylabel('output')
plt.title('comparision of MLP and Real Target')

# plot actual output and output of network for tese data
plt.subplot(2 ,1 ,2)
plt.plot(np.arange(len(Y_test)) , Y_test , 'b-' , marker = 'o' )
plt.plot(np.arange(len(Y_test)) , test_pred , 'r-' , marker = '*' )
plt.xlabel('test samples')
plt.ylabel('output')
plt.show()
# plot loss function
plt.figure(2)
plt.subplot(1, 1, 1)
plt.plot(np.arange(len(loss_train)) , loss_train , 'b-')
plt.xlabel('epochs')
plt.ylabel('error MSE')
plt.title('Train Error Mse')
plt.show()
'''

