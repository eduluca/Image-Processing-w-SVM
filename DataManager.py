# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:31:36 2020

@author: jsalm
"""

import os
import numpy as np
import cv2
import pickle

class DataMang:
    def __init__(self,directory):
        # dir_n = os.path.join(os.path.dirname(__file__),dirname)
        self.directory = directory
        self.dir_len = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory,name))])
    'end def'
    
    def save_obj(obj):
        with open('mat.pkl', 'wb') as outfile:
            pickle.dump(obj, outfile, pickle.HIGHEST_PROTOCOL)
        'end with'
    'end def'
    
    def load_obj(obj):
        with open('mat.pkl', 'rb') as infile:
            result = pickle.load(infile)
        'end with'
        return result
    'end def'
    
    def _load_image(self,rootdir):
        im = np.array(cv2.imread(rootdir)[:,:,:]/255).astype(np.float32)
        # im[im==0] = "nan"
        # im[im==1] = np.nanmin(im)
        # im[np.isnan(im)] = np.nanmin(im)
        return im
    
    def open_dir(self,im_list):
        """
        This is a chunky directory manager. 

        Parameters
        ----------
        *args : int or list of int
            

        Yields
        ------
        im : TYPE
            DESCRIPTION.
        nW : TYPE
            DESCRIPTION.
        nH : TYPE
            DESCRIPTION.

        """
        count = 0
        for root, dirs, files in os.walk(self.directory):
            for f in files:
                if isinstance(im_list,list):
                    if count in im_list:
                        impath = os.path.join(root,f)
                        im = self._load_image(impath)
                        name = [x for x in map(str.strip, f.split('.')) if x]
                        nW,nH,chan = im.shape
                        yield (im,nW,nH,chan,name[0])
                    'end if'
                count += 1
                'end if'
            'end for'
        'end for'
    'end def'
'end class'
'''
####Learning Curve####
train_sizes, train_scores, test_scores =\
                learning_curve(estimator=pipe_svc,
                               X=X_train,
                               y=y_train,
                               train_sizes=np.linspace(0.1, 1.0, 20),
                               cv=10,
                               n_jobs=-1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure("Learning Curve")
plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.tight_layout()
plt.ylim([0,1])
#plt.savefig('images/06_05.png', dpi=300)
plt.show()
'''