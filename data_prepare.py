import scipy.io as sio
import numpy as np
import tqdm
import tensorflow as tf
import random

def createPatches2(X, y, windowSize):
  
  from sklearn.utils import shuffle
  import tqdm
  shapeX = np.shape(X)

  margin = int((windowSize-1)/2)
  newX = np.empty([shapeX[0]+2*margin,shapeX[1]+2*margin,shapeX[2]])

  newX[margin:shapeX[0]+margin:,margin:shapeX[1]+margin,:] = X

  index = np.empty([0,3], dtype = 'int')

  for k in tqdm.tqdm(range(1,np.size(np.unique(y)))):
    for i in range(shapeX[0]):
      for j in range(shapeX[1]):
        if y[i,j] == k:
          index = np.append(index,np.expand_dims(np.array([k,i,j]),0),0)

  patchesX = np.empty([index.shape[0],2*margin+1,2*margin+1,shapeX[2]])
  patchesY = np.empty([index.shape[0]])

  for i in range(index.shape[0]):
    p = index[i,1]
    q = index[i,2]
    patchesX[i,:,:,:] = newX[p:p+windowSize,q:q+windowSize,:]
    patchesY[i] = index[i,0]

  patchesX,  patchesY = shuffle(patchesX,  patchesY)
  return patchesX, patchesY-1, index

# Read the data
data = sio.loadmat('.data/Salinas_corrected.mat')['salinas_corrected']
label = sio.loadmat('.data/Salinas_gt.mat')['salinas_gt']

# Normalize the data
feats_norm = np.empty([512,217,204], dtype = 'float32')
for i in tqdm.tqdm(range(204)):
  feats_norm[:,:,i] = data[:,:,i] - np.min(data[:,:,i])
  feats_norm[:,:,i] = feats_norm[:,:,i]/np.max(feats_norm[:,:,i])

train_test_patches, train_test_labels, index = createPatches2(feats_norm, label, 3)

# Divide data into train test splits
from sklearn.model_selection import StratifiedShuffleSplit
s3 = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=0)
s3.get_n_splits(train_test_patches, train_test_labels, index)

for train_index, test_index in s3.split(train_test_patches, train_test_labels, index):
   train_patches, test_patches, = train_test_patches[train_index], train_test_patches[test_index]
   train_labels, test_labels = train_test_labels[train_index], train_test_labels[test_index]
   tr_indx, ts_indx = index[train_index], index[test_index]

np.save('.data/train_vec',train_patches[:,1,1,:])
np.save('.data/test_vec',test_patches[:,1,1,:])
np.save('.data/train_labels',train_labels)
np.save('.data/test_labels',test_labels)
