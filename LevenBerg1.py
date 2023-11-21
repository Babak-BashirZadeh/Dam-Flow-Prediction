from sklearn.svm import SVC , LinearSVC , NuSVC , OneClassSVM
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix as cm
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error as mse
import numpy as np

class NN_LM_RG:
    def __init__(self,layers=(10,5,1),lr=1,activaton_func='tanh'
                 , max_epoch=50 , batch_size=10 , Best_weights = False):
        self.layers = layers
        self.lr = lr
        self.activation = activaton_func
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.Best_weights = Best_weights
        self.list_weights = []

    def activ_fun(self, net):

        if self.activation == 'relu':
            if len(net.shape)==2:
                for i in range(net.shape[-1]):
                    if net[0,i] < 0:
                        net[0,i] = 0
            else:
                for i in range(net.shape[-1]):
                    if net[i] < 0:
                        net[i] = 0

            return net

        elif self.activation == 'sigmoid':
            # logsig
            return 1 / (1 + np.exp(-1 * net))

        else:
            # tanh
            return (2 / (1 + np.exp(-2 * net))) - 1

    def d_activation(self,output):
        # derivation of relu
        if self.activation == 'relu':
            return np.ones_like(output)

        # derivation of logsig
        elif self.activation == 'sigmoid':
            return output * (1 - output)

        # derivation of tanh
        elif self.activation == 'tanh':
            return 1 - output ** 2
        else:
            print('ERROR: you can use "tanh" , "logsig" and "relu" ---' , self.activation , ' is undefined')

    def feedforward(self,input):
        global net1
        #input = input.reshape(-1,1)
        net1 = np.dot(input, self.w1.T)
        o1 = self.activ_fun(net1)
        net2 = np.dot(o1, self.w2.T)
        o2 = self.activ_fun(net2)
        net3 = np.dot(o2, self.w3.T)
        o3 = net3
        return o3, o2, o1

    def fit(self,X_train,Y_train,X_val , Y_val):
        X_test = X_val
        Y_test = Y_val


        n0 = X_train.shape[1]
        n1 = self.layers[0]
        n2 = self.layers[1]
        n3 = self.layers[2]

        w1 = np.random.uniform(-1, 1, size=(n1, n0))
        w2 = np.random.uniform(-1, 1, size=(n2, n1))
        w3 = np.random.uniform(-100, 100, size=(n3, n2))
        # w1 = np.load('Features and Models/3/Models/M1/W_LVM.npy')[0]
        # w2 = np.load('Features and Models/3/Models/M1/W_LVM.npy')[1]
        # w3 = np.load('Features and Models/3/Models/M1/W_LVM.npy')[2]
        self.w1 , self.w2 , self.w3 = w1 , w2 , w3
        self.WEIGHTS = np.array([w1, w2, w3])

        max_epoch = self.max_epoch
        mse_train = []
        mse_test = []
        # train_output = []
        # test_output = []
        activation = self.activation
        eta_old = self.lr
        eta = eta_old

        w1_t = w1.reshape((1, n0 * n1))
        w2_t = w2.reshape((1, n1 * n2))
        w3_t = w3.reshape((1, n2 * n3))

        w_all = []
        w_all.extend(w1_t[0])
        w_all.extend(w2_t[0])
        w_all.extend(w3_t[0])
        w_all = np.array(w_all)
        w_all = w_all.reshape((1, len(w_all)))
        Jacobian = np.zeros(shape=(len(X_train), n0 * n1 + n1 * n2 + n2 * n3))
        I = np.eye(n0 * n1 + n1 * n2 + n2 * n3)

        for epoch in range(max_epoch):

            X_train_temp = np.zeros_like(X_train)
            Y_train_temp = np.zeros_like(Y_train)
            lst_permute = np.random.permutation(X_train.shape[0])
            for i in range(lst_permute.shape[0]):
                X_train_temp[i, :] = X_train[lst_permute[i], :]
                Y_train_temp[i] = Y_train[lst_permute[i]]
            X_train = X_train_temp
            Y_train = Y_train_temp

            error_train = []
            error_train_=[]
            error_test = []
            train_output = []
            test_output = []

            # Learning Step
            batch_counter = 0
            for i in range(len(X_train)):
                input = X_train[i]
                input = np.reshape(input, newshape=(1, n0))
                target = Y_train[i]

                # feedforward step
                o3, o2, o1 = self.feedforward(input=input)
                # print(i)

                # backforwar step
                error = target - o3
                error_train_.append(error)

                c = self.d_activation(o2)
                A = np.diagflat(c)

                c = self.d_activation(o1)
                B = np.diagflat(c)

                # w3 * A
                w3A = np.dot(w3, A)

                # w3 * A * w2
                w3Aw2 = np.dot(w3A, w2)

                # w3 * A * w2 * B
                w3Aw2B = np.dot(w3Aw2, B)

                J_temp = []

                temp = w3Aw2B.T
                gw1 = -1 * 1 * np.dot(temp, input)
                gw1 = gw1.reshape((1, n0 * n1))
                J_temp.extend(gw1[0])

                temp = w3A.T
                gw2 = -1 * 1 * np.dot(temp, o1)
                gw2 = gw2.reshape((1, n1 * n2))
                J_temp.extend(gw2[0])

                gw3 = -1 * 1 * o2
                gw3 = gw3.reshape((1, n2 * n3))
                J_temp.extend(gw3[0])

                J_temp = np.array(J_temp)
                J_temp = J_temp.reshape((1, n0 * n1 + n1 * n2 + n2 * n3))

                Jacobian[i] = J_temp[0]
                batch_counter +=1
            #if batch_counter == self.batch_size or i == len(X_train):
            batch_counter = 0
            error_train_ = np.array(error_train_)
            error_train_ = error_train_.reshape((len(X_train), 1))
            #error_train_ = error_train_.reshape(self.batch_size, 1)
            mu = eta_old * error_train_.T.dot(error_train_)
            # updating weights
            hessian = np.linalg.inv(Jacobian.T.dot(Jacobian) + mu * I)
            temp = hessian.dot(Jacobian.T)
            temp = temp.dot(error_train_)
            w_all = w_all - temp.T

            border = 0
            w1 = w_all[0, :n1 * n0].reshape((n1, n0))
            border += n1 * n0
            w2 = w_all[0, border: border + n2 * n1].reshape((n2, n1))
            border += n2 * n1
            w3 = w_all[0, border: border + n3 * n2].reshape((n3, n2))
            self.w1 , self.w2 , self.w3 = w1 , w2 , w3

            # calculating train set
            for i in range(len(X_train)):
                input = X_train[i]
                # input = np.reshape(input, newshape=(1, col))
                target = Y_train[i]
                # print('j :' , i)
                o3, _, _ = self.feedforward(input=input)

                train_output.append(o3)
                error = target - o3
                error_train.append(error)

            # calculating output of test set
            for i in range(len(X_test)):
                input = X_test[i]
                # input = np.reshape(input, newshape=(1, col))
                target = Y_test[i]

                o3, _, _ = self.feedforward(input=input)
                test_output.append(o3)
                error = target - o3
                error_test.append(error)

            # calculating mse for test and train
            #print(np.shape(Y_train) , '--' , np.shape(train_output))
            trn_mse = mse(Y_train, train_output)
            tst_mse = mse(Y_test, test_output)
            mse_train.append(trn_mse)
            mse_test.append(tst_mse)
            print('epoch', epoch, ':', 'loss : ', trn_mse ,'   val_loss : ' , tst_mse)
            self.list_weights.append([self.w1,self.w2,self.w3])

        self.loss = mse_train
        self.val_loss = mse_test
        if self.Best_weights :
            Best_idx = np.argmin(self.val_loss)
            self.w1 = self.list_weights[Best_idx][0]
            self.w2 = self.list_weights[Best_idx][1]
            self.w3 = self.list_weights[Best_idx][2]

        return self.loss , self.val_loss

    def predict(self,X):
        out = []
        for i in range(X.shape[0]):
            input = np.reshape(X[i,:],(1,len(X[i,:])))
            output,_,_ = self.feedforward(input)
            out.append(output[0])
        out = np.array(out)
        #print(out.shape)
        return out



