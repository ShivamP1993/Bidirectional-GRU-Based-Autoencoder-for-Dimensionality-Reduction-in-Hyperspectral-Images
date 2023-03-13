# Function to perform one hot encoding of the class labels 

def my_ohc(lab_arr):
    lab_arr_unique =  np.unique(lab_arr)
    r,c = lab_arr.shape
    r_u  = lab_arr_unique.shape
    
    
    one_hot_enc = np.zeros((r,r_u[0]), dtype = 'float')
    
    for i in range(r):
        for j in range(r_u[0]):
            if lab_arr[i,0] == lab_arr_unique[j]:
                one_hot_enc[i,j] = 1
    
    return one_hot_enc

# Function that takes the confusion matrix as input and
# calculates the overall accuracy, producer's accuracy, user's accuracy,
# Cohen's kappa coefficient and standard deviation of 
# Cohen's kappa coefficient

def accuracies(cm):
  import numpy as np
  num_class = np.shape(cm)[0]
  n = np.sum(cm)

  P = cm/n
  ovr_acc = np.trace(P)

  p_plus_j = np.sum(P, axis = 0)
  p_i_plus = np.sum(P, axis = 1)

  usr_acc = np.diagonal(P)/p_i_plus
  prod_acc = np.diagonal(P)/p_plus_j

  theta1 = np.trace(P)
  theta2 = np.sum(p_plus_j*p_i_plus)
  theta3 = np.sum(np.diagonal(P)*(p_plus_j + p_i_plus))
  theta4 = 0
  for i in range(num_class):
    for j in range(num_class):
      theta4 = theta4+P[i,j]*(p_plus_j[i]+p_i_plus[j])**2

  kappa = (theta1-theta2)/(1-theta2)

  t1 = theta1*(1-theta1)/(1-theta2)**2
  t2 = 2*(1-theta1)*(2*theta1*theta2-theta3)/(1-theta2)**3
  t3 = ((1-theta1)**2)*(theta4 - 4*theta2**2)/(1-theta2)**4

  s_sqr = (t1+t2+t3)/n

  return ovr_acc, usr_acc, prod_acc, kappa, s_sqr

# Import Relevant libraries and classes
import scipy.io as sio
import numpy as np
import tqdm
from sklearn.decomposition import PCA
import tensorflow as tf
keras = tf.keras
from keras import backend as K
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout
from keras.layers import Conv2D, Flatten, Lambda, Conv3D, Conv3DTranspose,BatchNormalization,Conv1D, Activation, Layer, MaxPooling1D, GRU, Bidirectional
from keras.layers import Reshape, Conv2DTranspose, Concatenate, Multiply, Add, MaxPooling2D, MaxPooling3D, GlobalAveragePooling2D,  Conv1DTranspose
from keras import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
#from keras.utils import plot_model
from keras import backend as K
from sklearn.metrics import confusion_matrix

def get_gt_index(y):

  import tqdm

  shapeY = np.shape(y)

  pp,qq = np.unique(y, return_counts=True)
  sum1 = np.sum(qq)-qq[0]

  index = np.empty([sum1,3], dtype = 'int')

  cou = 0
  for k in tqdm.tqdm(range(1,np.size(np.unique(y)))):
    for i in range(shapeY[0]):
      for j in range(shapeY[1]):
        if y[i,j] == k:
          index[cou,:] = np.expand_dims(np.array([k,i,j]),0)
          #print(cou)
          cou = cou+1
  return index

# The code takes the entire hsi/lidar image as input for 'X' and grounttruth file as input for 'y'
# and the patchsize as for 'windowSize'.
# The output are the patches centered around the groundtruth pixel, the corresponding groundtruth label and the
# pixel location of the patch.

def make_patches(X, y, windowSize):

  shapeX = np.shape(X)

  margin = int((windowSize-1)/2)
  newX = np.zeros([shapeX[0]+2*margin,shapeX[1]+2*margin,shapeX[2]])

  newX[margin:shapeX[0]+margin:,margin:shapeX[1]+margin,:] = X

  index = get_gt_index(y)
  
  patchesX = np.empty([index.shape[0],2*margin+1,2*margin+1,shapeX[2]], dtype = 'float32')
  patchesY = np.empty([index.shape[0]],dtype = 'uint8')

  for i in range(index.shape[0]):
    p = index[i,1]
    q = index[i,2]
    patchesX[i,:,:,:] = newX[p:p+windowSize,q:q+windowSize,:]
    patchesY[i] = index[i,0]

  return patchesX, patchesY, index

train_vec = np.reshape(np.load('.data/train_vec.npy'), [-1,1,204,1])
train_labels = np.load('.data/train_labels.npy')

test_vec = np.reshape(np.load('.data/test_vec.npy'), [-1,1,204,1])
test_labels = np.load('.data/test_labels.npy')

def rae(x):

  conv1 = Bidirectional(GRU(204))(x)
  conv1 = Reshape([1,408])(conv1)
  conv2 = Bidirectional(GRU(68))(conv1)

  f1 = Flatten()(conv2)
  d1 = Dense(10, activation="relu")(f1)
  
  xA = Dense(1 * 1 * 204, activation="relu")(d1)
  r1 = Reshape([1,204])(xA)

  conv6 =Bidirectional(GRU(68))(r1)
  conv6 = Reshape([1,136])(conv6)
  conv7 = Bidirectional(GRU(102))(conv6)

  rs = Reshape([1,204])(conv7)

  decoder = Model(x, rs, name="decoder")

  return decoder


xA = Input(shape = (1,204))
aeR = rae(xA)

aeR.summary()

aeR.compile(loss = 'mean_squared_error', optimizer=keras.optimizers.Adam(0.001), metrics = ['mse'])

# Random FOrest Classifier to check the performance of encoder.
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0, n_estimators=200)

acc_temp=0
import gc
for epoch in range(500): 
  gc.collect()
  aeR.fit(x = train_vec, y = train_vec,
                  epochs=1, batch_size = 128, verbose = 1)
  

  new_model = Model(aeR.input, aeR.layers[5].output, name = 'new_model') 
  code_feat_train = np.reshape(new_model.predict(train_vec),[-1,10])
  code_feat_test = np.reshape(new_model.predict(test_vec),[-1,10])

  clf.fit(code_feat_train, train_labels)
  preds = clf.predict(code_feat_test)

  conf = confusion_matrix(test_labels, preds)
  ovr_acc, _, _, _, _ = accuracies(conf)

  print(epoch)
  print(np.round(100*ovr_acc,2))
  if ovr_acc>=k:
    aeR.save('.models/model')
  print('acc_max = ', np.round(100*acc_temp,2), '% at epoch', ep)
