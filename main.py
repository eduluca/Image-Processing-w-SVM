# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:02:19 2021

@author: jsalm
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from scipy.ndimage import convolve
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score
from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.ensemble import GradientBoostingClassifier

import SVM
import Filters
import ML_interface_SVM_V3
import DataManager


####PARAMS####
# param_range = [0.0001,0.001,0.01,0.1,1,10,100,1000]
# param_range= np.arange(0.01,1,0.001)
param_range2_C = [40,50,60,70,80,90,100,110,120,130,140]
param_range2_ga = [0.0005,0.0006,0.0007,0.001,0.002,0.003,0.004]
deg_range = [2,3,4,5,6,7]
deg_range2 = [2,3,4,5,6,10]
poly_range = np.arange(2,10,1)
poly_range_C = np.arange(1e-15,1e-7,6e-10)
poly_range_ga = np.arange(1e8,1e15,6e12)
# param_grid = [{'svc__C':param_range,
#                 'svc__kernel':['linear']},
#               {'svc__C': param_range,
#                 'svc__gamma':param_range,
#                 'svc__kernel':['rbf']},
#               {'svc__C': param_range,
#                 'svc__gamma':param_range,
#                 'svc__kernel':['poly'],
#                 'svc__degree':deg_range}]
param_grid2 = [{'svc__C': param_range2_C,
                'svc__gamma':param_range2_ga,
                'svc__decision_function_shape':['ovo','ovr'],
                'svc__kernel':['rbf']},
                {'svc__C': param_range2_C,
                 'svc__gamma':param_range2_ga,
                 'svc__kernel':['poly'],
                 'svc__decision_function_shape':['ovo','ovr'],
                 'svc__degree':deg_range2}]
# param_grid3 = [{'svc__C': poly_range_C,
#                 'svc__gamma':poly_range_ga,
#                 'svc__kernel':['poly'],
#                 'svc__degree':poly_range}]
###END###

### DATA PROCESSING IMAGE 1 ###

#initialize test data set
dirname = os.path.dirname(__file__)
foldername = os.path.join(dirname,"images_5HT")
im_dir = DataManager.DataMang(foldername)

### PARAMS ###
channel = 2
ff_width = 121
wiener_size = (5,5)
med_size = 10
start = 0
count = 42
###

#load image folder for training data
dirname = os.path.dirname(__file__)
foldername = os.path.join(dirname,"images_5HT")
im_dir = DataManager.DataMang(foldername)
# change the 'start' in PARAMS to choose which file you want to start with.
im_list = [i for i in range(start,im_dir.dir_len)]
hog_features = []
for gen in im_dir.open_dir(im_list):
    #load image and its information
    image,nW,nH,chan,name = gen
    print('procesing image : {}'.format(name))
    #only want the red channel (fyi: cv2 is BGR (0,1,2 respectively) while most image processing considers 
    #the notation RGB (0,1,2 respectively))
    image = image[:,:,channel]
    #Import train data (if training your model)
    train_bool = ML_interface_SVM_V3.import_train_data(name,(nW,nH),'train_71420')
    #extract features from image using method(SVM.filter_pipeline) then watershed data useing thresholding algorithm (work to be done here...) to segment image.
    #Additionally, extract filtered image data and hog_Features from segmented image. (will also segment train image if training model) 
    im_segs, bool_segs, domains, paded_im_seg, paded_bool_seg, hog_features = SVM.feature_extract(image, ff_width, wiener_size, med_size,True,train_bool)
    #choose which data you want to merge together to train SVM. Been using my own filter, but could also use hog_features.
    X,y = SVM.create_data(hog_features,True,bool_segs)
    break

print('done')
#adding in some refence numbers for later
y = np.vstack([y,np.arange(0,len(y),1)]).T
#split dataset

print('Splitting dataset...')
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=count)
ind_train = y_train[:,1]
ind_test = y_test[:,1]

y_train = y_train[:,0]
y_test = y_test[:,0]



print("y_train: " + str(np.unique(y_train)))
print("y_test: " + str(np.unique(y_test)))



#create SVM pipline
#try using a GBC
pipe_svc = make_pipeline(RobustScaler(),SVC())

#SVM MODEL FITTING
#we create an instance of SVM and fit out data.
# print("starting modeling career...")

# gs = GridSearchCV(estimator = pipe_svc,
#                   param_grid = param_grid2,
#                   scoring = 'roc_auc',
#                   cv = 5,
#                   n_jobs = -1,
#                   verbose = 10)


# print("Fitting...")
# gs = gs.fit(X_train,y_train)
# print('best score: ' + str(gs.best_score_))
# print(gs.best_params_)
# pipe_svc = gs.best_estimator_
### END Gridsearch ####

### Setting Parameters ###
print('fitting...')
#{'svc__C': 100, 'svc__gamma': 0.001, 'svc__kernel': 'rbf'} (~0.72% f1_score)
#{svc__C=130, svc__decision_function_shape=ovr, svc__gamma=0.0005, svc__kernel=rbf}

pipe_svc.set_params(svc__C =  130, 
                    svc__gamma = 0.0005, 
                    svc__kernel =  'rbf',
                    svc__probability = True,
                    svc__shrinking = False,
                    svc__decision_function_shape = 'ovr')

### Cross Validate ###
scores = cross_val_score(estimator = pipe_svc,
                          X = X_train,
                          y = y_train,
                          cv = 10,
                          verbose = True,
                          n_jobs=-1)

print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))) 

### Fitting Model ###
fitted = pipe_svc.fit(X_train,y_train)
y_score = pipe_svc.fit(X_train,y_train).decision_function(X_test)
print(pipe_svc.score(X_test,y_test))

### DATA PROCESSING IMAGE 2 ###
#pick a test image
# os.chdir(r'C:\Users\jsalm\Documents\Python Scripts\SVM_7232020')
Test_im = np.array(cv2.imread("images_5HT/injured 60s_sectioned_CH2.tif")[:,:,2]/255).astype(np.float32)

#extract features
# im_segs_test, _, domains_test, paded_im_seg_test, _, hog_features_test = SVM.feature_extract(Test_im, ff_width, wiener_size, med_size,False)
# X_test = SVM.create_data(im_segs_test,False)

### SVM MODEL PREDICTION ###
predictions = fitted.predict(X_test)   

# predict_im = data_to_img(boolim2_2,predictfions)
SVM.overlay_predictions(image, train_bool, predictions, y_test, ind_test,domains)

### Confusion Matrix: Save fig if interesting ###
confmat = confusion_matrix(y_true = y_test, y_pred=predictions)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
#plt.savefig('images/06_09.png', dpi=300)
plt.show()

### ROC Curve ###
fpr, tpr,_ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# #PLOT IMAGES
# # Filters.imshow_overlay(Test_im,predict_im,'predictions2',True)

# name_list = ["image","denoised_im","median_im","thresh_im","dir_im","gau_im","di_im","t_im"]
# for i in range(0,len(image_tuple)):
#     plt.figure(name_list[i]);plt.imshow(image_tuple[i])