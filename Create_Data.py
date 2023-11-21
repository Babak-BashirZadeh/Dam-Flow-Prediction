import numpy as np
import pandas as pd

pattern = str(6)
raw_data2 = np.load('Dataset/raw_data2.npy')
ali

def prepairing():
    data = pd.read_excel("Dataset/inputs.xlsx")
    date = data['Date'].to_numpy()
    Qt = data['Q(t)'].to_numpy()
    Pt = data['P(t)'].to_numpy()
    raw_data = np.zeros(shape=(396,3))
    raw_data[:Qt.shape[0],0] = date.copy()
    raw_data[:Qt.shape[0],1] = Qt.copy()
    raw_data[:Qt.shape[0],2] = Pt.copy()
    Qt ,Pt= raw_data[:,1] , raw_data[:,2]
    Qt_anual = np.reshape(Qt,newshape=(33,12))
    Pt_anual = np.reshape(Pt , newshape=(33,12))

    Qt_mean = np.mean(Qt_anual,axis=0)
    Qt_std = np.std(Qt_anual, axis=0)

    Pt_mean = np.mean(Pt_anual,axis=0)
    Pt_std = np.std(Pt_anual, axis=0)

    month_idxs = Qt_anual.copy()
    Qt_Means= Qt_anual.copy()
    Qt_Stds = Qt_anual.copy()
    Pt_Means= Qt_anual.copy()
    Pt_Stds = Qt_anual.copy()
    for i in range(month_idxs.shape[0]):
        month_idxs[i,:] = np.arange(1,13)/12
        Qt_Means[i,:] = Qt_mean
        Qt_Stds[i,:] = Qt_std
        Pt_Means[i,:]=Pt_mean
        Pt_Stds[i,:] = Pt_std

    raw_data2 = np.zeros(shape=(396,8))
    raw_data2[:,0] = raw_data[:,0]
    raw_data2[:,1] = Qt.copy()
    raw_data2[:,2] = np.reshape(Qt_Means,newshape=(396,))
    raw_data2[:,3] = np.reshape(Qt_Stds,newshape=(396,))
    raw_data2[:,4] = Pt.copy()
    raw_data2[:,5] = np.reshape(Pt_Means , newshape=(396,))
    raw_data2[:,6] = np.reshape(Pt_Stds, newshape=(396,))
    raw_data2[:,7] = np.reshape(month_idxs , newshape=(396,))
    raw_data2= raw_data2[:391,:]
    #np.save('Dataset/raw_data2',raw_data2)
"""
>>> State 1 :   X0      X1     ||     y
>>>            Q(t-1)   T      ||    Q(t)
"""
def create_data1():
    prev = 1
    data1 = np.zeros(shape=(raw_data2.shape[0] - prev , 3))
    for i in range(raw_data2.shape[0] - prev):
        data1[i,0] = raw_data2[i,1]
        data1[i,1] = raw_data2[i+prev , -1]
        data1[i ,-1] = raw_data2[i+prev , 1]
    X = data1[:,:-1]
    Y = data1[:,-1]
    trn_tst_split = int(X.shape[0] * 0.75)
    X_train = X[:trn_tst_split]
    Y_train = Y[:trn_tst_split]
    X_test = X[trn_tst_split:]
    Y_test = Y[trn_tst_split:]
    np.save('Features and Models/1/X' , X)
    np.save('Features and Models/1/Y',Y)
    np.save('Features and Models/1/X_train', X_train)
    np.save('Features and Models/1/Y_train', Y_train)
    np.save('Features and Models/1/X_test', X_test)
    np.save('Features and Models/1/Y_test', Y_test)
    np.save('Features and Models/1/X_max', X_train.max(0))
    np.save('Features and Models/1/X_min', X_train.min(0))
    np.save('Features and Models/1/Y_max', Y_train.max(0))
    np.save('Features and Models/1/Y_min', Y_train.min(0))


"""
>>> State 2 :   X0      X1      X2      X3     ||     y
>>>            MQ(t)  SQ(t)     T      Q(t-1)  ||    Q(t)
"""
def create_data2():
    prev = 1
    data1 = np.zeros(shape=(raw_data2.shape[0] - prev , 5))
    for i in range(raw_data2.shape[0] - prev):
        data1[i,0] = raw_data2[i+prev,2] # X0 = MQ(t)
        data1[i,1] = raw_data2[i+prev,3] # X1 = SQ(t)
        data1[i,2] = raw_data2[i+prev , -1]  # X2 = T
        data1[i,3] = raw_data2[i,1]  # X3 = Q(t-1)
        data1[i ,-1] = raw_data2[i+prev , 1]  # Y = Q(t)
    X = data1[:,:-1]
    Y = data1[:,-1]
    trn_tst_split = int(X.shape[0] * 0.75)
    X_train = X[:trn_tst_split]
    Y_train = Y[:trn_tst_split]
    X_test = X[trn_tst_split:]
    Y_test = Y[trn_tst_split:]
    np.save('Features and Models/2/X' , X)
    np.save('Features and Models/2/Y',Y)
    np.save('Features and Models/2/X_train', X_train)
    np.save('Features and Models/2/Y_train', Y_train)
    np.save('Features and Models/2/X_test', X_test)
    np.save('Features and Models/2/Y_test', Y_test)
    np.save('Features and Models/2/X_max', X_train.max(0))
    np.save('Features and Models/2/X_min', X_train.min(0))
    np.save('Features and Models/2/Y_max', Y_train.max(0))
    np.save('Features and Models/2/Y_min', Y_train.min(0))



"""
>>> State 3 :   X0      X1      X2      X3    X4    X5     ||     y
>>>            MQ(t)  SQ(t)    MP(t)  SP(t)   T    Q(t-1)  ||    Q(t)
"""
def create_data3():
    prev = 1
    data1 = np.zeros(shape=(raw_data2.shape[0] - prev , 7))
    for i in range(raw_data2.shape[0] - prev):
        data1[i,0] = raw_data2[i+prev,2] # X0 = MQ(t)
        data1[i,1] = raw_data2[i+prev,3] # X1 = SQ(t)
        data1[i,2] = raw_data2[i+prev,5]  # X2 = MP(t)
        data1[i,3] = raw_data2[i + prev, 6]  # X3 = SP(t)
        data1[i,4] = raw_data2[i+prev , -1]  # X4 = T
        data1[i,5] = raw_data2[i,1]  # X5 = Q(t-1)
        data1[i ,-1] = raw_data2[i+prev , 1]  # Y = Q(t)
    X = data1[:,:-1]
    Y = data1[:,-1]
    trn_tst_split = int(X.shape[0] * 0.75)
    X_train = X[:trn_tst_split]
    Y_train = Y[:trn_tst_split]
    X_test = X[trn_tst_split:]
    Y_test = Y[trn_tst_split:]
    np.save('Features and Models/3/X' , X)
    np.save('Features and Models/3/Y',Y)
    np.save('Features and Models/3/X_train', X_train)
    np.save('Features and Models/3/Y_train', Y_train)
    np.save('Features and Models/3/X_test', X_test)
    np.save('Features and Models/3/Y_test', Y_test)
    np.save('Features and Models/3/X_max', X_train.max(0))
    np.save('Features and Models/3/X_min', X_train.min(0))
    np.save('Features and Models/3/Y_max', Y_train.max(0))
    np.save('Features and Models/3/Y_min', Y_train.min(0))




"""
>>> State 4 :   X0      X1        X2       X3       X4                    ||     y
>>>            MQ(t)   SQ(t)     MP(t)    SP(t)     T                     ||    Q(t)
>>>             --------------------------------------------------------
>>>             X5      X6        X7       X8       X9      X10     X11   ||     y
>>>            Q(t-1)  MQ(t-1)  SQ(t-1)   P(t-1)  MP(t-1)  SP(t-1)  T-1   ||    Q(t)     

"""
def create_data4():
    prev = 1
    cols = 13
    data1 = np.zeros(shape=(raw_data2.shape[0] - prev , cols))
    for i in range(raw_data2.shape[0] - prev):
        data1[i,0] = raw_data2[i+prev-0 , 2]  # X0 = MQ(t)
        data1[i,1] = raw_data2[i+prev-0 , 3]  # X1 = SQ(t)
        data1[i,2] = raw_data2[i+prev-0 , 5]  # X2 = MP(t)
        data1[i,3] = raw_data2[i+prev-0 , 6]  # X3 = SP(t)
        data1[i,4] = raw_data2[i+prev-0 ,-1]  # X4 = T
        data1[i,5] = raw_data2[i+prev-1 , 1]  # X5 =  Q(t-1)
        data1[i,6] = raw_data2[i+prev-1 , 2]  # X6 = MQ(t-1)
        data1[i,7] = raw_data2[i+prev-1 , 3]  # X7 = SQ(t-1)
        data1[i,8] = raw_data2[i+prev-1 , 4]  # X8 = P(t-1)
        data1[i,9] = raw_data2[i+prev-1 , 5]  # X9 = MP(t-1)
        data1[i,10]= raw_data2[i+prev-1 , 6]  # X10= SP(t-1)
        data1[i,11]= raw_data2[i+prev-1 ,-1]  # X11= T-1

        data1[i ,-1] = raw_data2[i+prev , 1]  # Y = Q(t)

    X = data1[:,:-1]
    Y = data1[:,-1]
    trn_tst_split = int(X.shape[0] * 0.75)
    X_train = X[:trn_tst_split]
    Y_train = Y[:trn_tst_split]
    X_test = X[trn_tst_split:]
    Y_test = Y[trn_tst_split:]
    np.save('Features and Models/4/X' , X)
    np.save('Features and Models/4/Y',Y)
    np.save('Features and Models/4/X_train', X_train)
    np.save('Features and Models/4/Y_train', Y_train)
    np.save('Features and Models/4/X_test', X_test)
    np.save('Features and Models/4/Y_test', Y_test)
    np.save('Features and Models/4/X_max', X_train.max(0))
    np.save('Features and Models/4/X_min', X_train.min(0))
    np.save('Features and Models/4/Y_max', Y_train.max(0))
    np.save('Features and Models/4/Y_min', Y_train.min(0))





"""
>>> State 5 :   X0      X1        X2       X3       X4                    ||     y
>>>            MQ(t)   SQ(t)     MP(t)    SP(t)     T                     ||    Q(t)
>>>             --------------------------------------------------------
>>>             X5      X6        X7       X8       X9      X10     X11   ||     y
>>>            Q(t-1)  MQ(t-1)  SQ(t-1)   P(t-1)  MP(t-1)  SP(t-1)  T-1   ||    Q(t)
>>>            -------------------------------------------------------- 
>>>             X12      X13      X14     X15      X16      X17     X18   ||     y
>>>            Q(t-2)  MQ(t-2)  SQ(t-2)   P(t-2)  MP(t-2)  SP(t-2)  T-2   ||    Q(t)

"""
def create_data5():
    prev = 2
    cols = 20
    data1 = np.zeros(shape=(raw_data2.shape[0] - prev , cols))
    for i in range(raw_data2.shape[0] - prev):
        data1[i,0] = raw_data2[i+prev-0 , 2]  # X0 = MQ(t)
        data1[i,1] = raw_data2[i+prev-0 , 3]  # X1 = SQ(t)
        data1[i,2] = raw_data2[i+prev-0 , 5]  # X2 = MP(t)
        data1[i,3] = raw_data2[i+prev-0 , 6]  # X3 = SP(t)
        data1[i,4] = raw_data2[i+prev-0 ,-1]  # X4 = T
        temp = np.concatenate((raw_data2[i+prev-1,1:] , raw_data2[i+prev-2,1:])) # X5 - X18
        data1[i,5:-1] = temp  # X5 - X18

        data1[i ,-1] = raw_data2[i+prev , 1]  # Y = Q(t)

    X = data1[:,:-1]
    Y = data1[:,-1]
    trn_tst_split = int(X.shape[0] * 0.75)
    X_train = X[:trn_tst_split]
    Y_train = Y[:trn_tst_split]
    X_test = X[trn_tst_split:]
    Y_test = Y[trn_tst_split:]
    np.save('Features and Models/5/X' , X)
    np.save('Features and Models/5/Y',Y)
    np.save('Features and Models/5/X_train', X_train)
    np.save('Features and Models/5/Y_train', Y_train)
    np.save('Features and Models/5/X_test', X_test)
    np.save('Features and Models/5/Y_test', Y_test)
    np.save('Features and Models/5/X_max', X_train.max(0))
    np.save('Features and Models/5/X_min', X_train.min(0))
    np.save('Features and Models/5/Y_max', Y_train.max(0))
    np.save('Features and Models/5/Y_min', Y_train.min(0))

"""
>>> State 6 :   X0      X1        X2       X3       X4                    ||     y
>>>            Q(t-1)   Q(t-2)   Q(t-3)   Q(t-4)    Q(t-5)                ||    Q(t)
>>>             --------------------------------------------------------
>>>             X5      X6        X7       X8       X9      X10     X11   ||     y
>>>            Q(t-6)  Q(t-12)  P(t)      P(t-1)  P(t-2)  P(t-3)   P(t-4) ||    Q(t)
>>>            -------------------------------------------------------- 
>>>             X12      X13      X14     X15                             ||     y
>>>            P(t-5)  P(t-6)   P(t-12)   T                               ||    Q(t)

"""

prev = 12
cols = 17
data1 = np.zeros(shape=(raw_data2.shape[0] - prev, cols))
for i in range(raw_data2.shape[0] - prev):
    data1[i, 0] = raw_data2[i + prev - 0, 2]  # X0 = MQ(t)
    data1[i, 1] = raw_data2[i + prev - 0, 3]  # X1 = SQ(t)
    data1[i, 2] = raw_data2[i + prev - 0, 5]  # X2 = MP(t)
    data1[i, 3] = raw_data2[i + prev - 0, 6]  # X3 = SP(t)
    data1[i, 4] = raw_data2[i + prev - 0, -1]  # X4 = T
    temp = np.concatenate((raw_data2[i + prev - 1, 1:], raw_data2[i + prev - 2, 1:]))  # X5 - X18
    data1[i, 5:-1] = temp  # X5 - X18

    data1[i, -1] = raw_data2[i + prev, 1]  # Y = Q(t)

X = data1[:, :-1]
Y = data1[:, -1]
trn_tst_split = int(X.shape[0] * 0.75)
X_train = X[:trn_tst_split]
Y_train = Y[:trn_tst_split]
X_test = X[trn_tst_split:]
Y_test = Y[trn_tst_split:]
np.save('Features and Models/'+pattern+'/X', X)
np.save('Features and Models/'+pattern+'/Y', Y)
np.save('Features and Models/'+pattern+'/X_train', X_train)
np.save('Features and Models/'+pattern+'/Y_train', Y_train)
np.save('Features and Models/'+pattern+'/X_test', X_test)
np.save('Features and Models/'+pattern+'/Y_test', Y_test)
np.save('Features and Models/'+pattern+'/X_max', X_train.max(0))
np.save('Features and Models/'+pattern+'/X_min', X_train.min(0))
np.save('Features and Models/'+pattern+'/Y_max', Y_train.max(0))
np.save('Features and Models/'+pattern+'/Y_min', Y_train.min(0))
