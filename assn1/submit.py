import numpy as np
import time as tm
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( Z_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# The first 64 columns contain the config bits
	# The next 4 columns contain the select bits for the first mux
	# The next 4 columns contain the select bits for the second mux
	# The first 64 + 4 + 4 = 72 columns constitute the challenge
	# The last column contains the response
  new_data=slip_data_for_different_models(train_data)
  classifiers=[]
  for d2 in new_data:
    d=d2[1:]
    if(len(d)):
      x = np.array(d)
      clf = LinearSVC( loss = "squared_hinge",max_iter=1e7 )
      clf.fit(x[:,:-1] ,x[:,-1] )
      classifiers.append(clf)
    else: classifiers.append(None)
  return classifiers	
def slip_data_for_different_models(data):
  for row in data:
    a = row[64]*8+row[65]*4+row[66]*2+row[67]
    b = row[68]*8+row[69]*4+row[70]*2+row[71]
    if a>b:
      for i in range(4):
        row[68+i],row[64+i]=row[64+i],row[68+i]
      row[72]=1-row[72]

  arr=[]

  for i in range(256):
    # for initialising  
    arr.append([np.zeros(65)])
  for row in data:
    a = row[64]*8+row[65]*4+row[66]*2+row[67]
    b = row[68]*8+row[69]*4+row[70]*2+row[71]
    a = int(a*16+b)
    arr[a].append(np.append(row[:64],row[72]))

  return arr					# Return the trained model


################################
# Non Editable Region Starting #
################################
def my_predict( test_data,classifiers ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to make predictions on test challenges
  i=0
  pred=np.zeros(test_data.shape[0])
  for row in test_data:
    a = row[64]*8+row[65]*4+row[66]*2+row[67]
    b = row[68]*8+row[69]*4+row[70]*2+row[71]
    f=1
    if a>b:
      a,b=b,a
      f=-1
    x=classifiers[int(16*a+b)].predict([row[:64]])
    if f==-1:
      x[0]=1-x[0]
    pred[i]=x[0]
    i=i+1
  return pred