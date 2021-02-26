# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:04:42 2020

@author: jsalm
"""
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt
import cv2 
# from PIL import Image
from IPython import get_ipython
import os
import csv
from scipy.ndimage import convolve
from skimage.morphology import remove_small_objects,binary_closing
from skimage.measure import label
import pickle



os.getcwd()

plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (10,10)
get_ipython().run_line_magic('matplotlib','qt5')

class DataMang:
    def __init__(self,dirname):
        dir_n = os.path.join(os.path.dirname(__file__),dirname)
        self.directory = dir_n
        self.dir_len = len([name for name in os.listdir(dir_n) if os.path.isfile(os.path.join(dir_n,name))])
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
        im[im==0] = "nan"
        im[im==1] = np.nanmin(im)
        im[np.isnan(im)] = np.nanmin(im)
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
        count = 1
        for root, dirs, files in os.walk(self.directory):
            for f in files:
                if isinstance(im_list,list):
                    if count in im_list:
                        impath = os.path.join(root,f)
                        im = self._load_image(impath)
                        nW,nH,chan = im.shape
                        yield (im,nW,nH,chan)
                    'end if'
                count += 1
                'end if'
            'end for'
        'end for'
    'end def'
'end class'

class Direction:
    def  __init__(self,image,boolimage):
        self.image = image
        self.boolimage = boolimage
    'end def'
    
    def _create_circle(dia,fill=False):
        """
        creates a circle of radius
        to be used in curve detection
        dia = diameter; int (must be odd)

        """
        if dia%2 == 0:
            raise ValueError("Radius must be odd")
        'end if'
        circle = np.zeros((dia,dia))
        N = int(dia/2)
        x_val = list(range(0,dia))
        y_val = list(range(0,dia))
        for x in x_val:
            for y in y_val:
                circle[int(x),int(y)] = np.sqrt((x-N)**2+(y-N)**2)
            'end for'
        'end for'
        circle_bool = np.logical_not(np.add(circle>(dia/2),circle<(dia-2)/2))
        if fill == True:
            circle_bool = circle<(dia/2)
        'end if'
        return circle_bool
    'end def'
 
    def get_index_in_array(array,val):
        boolar = array == val
        pointsout = list()
        for i,row in enumerate(boolar):
            for j,col in enumerate(row):
                if col == True:
                    pointsout.append((i,j))
                else:
                    pass
        return pointsout
    
    def get_xy_in_array(array,dia,inpoint):
        #zero axis is on the center of the array
        point = list(inpoint)
        for i in range(len(array.shape)):
            if array.shape[i]!= dia:
                point[i] = inpoint[i] + dia-array.shape[i]
            'end if'
        'end for'
        x = point[1]-int(dia/2)
        if point[0] < int(dia/2):
            if point[0] == 0:
                y = int(dia/2)
            else:
                y = -(point[0]%int(dia/2)-int(dia/2))
        elif point[0] > int(dia/2):
            
            if point[0] == dia-1:
                y = -int(dia/2)
            else:
                y = -point[0]%int(dia/2)
        else:
            y = 0
        return (x,y)
        
    
    def get_angle_frm_cntr(point,dia):
        if point[0] > 0:
            angle = np.arctan(point[1]/point[0])
        elif point[0] < 0 and point[1] > 0:
            angle = np.arctan(abs(point[1])/abs(point[0]))+np.pi/2
        elif point[0] < 0 and point[1] < 0:
            angle = np.arctan(abs(point[1])/abs(point[0]))+np.pi
        elif point[0] == 0 and point[1] > 0:
            angle = np.pi/2
        elif point[0] == 0 and point[1] < 0:
            angle = np.pi*3/2
        else:
            angle = 0
        return angle
    
    def _boundary_satisfier(array,vector,point):
        i,j = point
        r,_ = vector.shape
        added = np.array([0])
        if r%2 != 0:
            ex = 1
        else:
            ex = 0
        #case where 
        if i >= r and array.shape[0]-i > int(r/2)+ex:
            if j >= r and array.shape[1]-j > r:
                #takes care of internal cases
                added = np.multiply(vector,array[i-(int(r/2)+ex):i+(int(r/2)),j-(int(r/2)+ex):j+int(r/2)])
            elif j >= r and array.shape[1]-j < int(r/2)+ex:
                added = np.multiply(vector[:,0:(array.shape[1]-j)+(int(r/2)+ex)],array[i-(int(r/2)+ex):i+(int(r/2)),j-(int(r/2)+ex):array.shape[1]])
            elif j < int(r/2)+ex:
                added = np.multiply(vector[:,int(r/2)-j:r],array[i-(int(r/2)+ex):i+(int(r/2)),0:j+int(r/2)+ex])
        elif i >= r and array.shape[0]-i < int(r/2)+ex:
            if j >= r and array.shape[1]-j > r:
                added = np.multiply(vector[0:(array.shape[0]-i)+int(r/2)+ex,:],array[i-(int(r/2)+ex):array.shape[0],j-(int(r/2)+ex):j+int(r/2)])
            elif j >= r and array.shape[1]-j < int(r/2)+ex:
                added = np.multiply(vector[0:(array.shape[0]-i)+int(r/2)+ex,0:(array.shape[1]-j)+(int(r/2)+ex)],array[i-(int(r/2)+ex):array.shape[0],j-(int(r/2)+ex):array.shape[1]])
            elif j < int(r/2)+ex:
                added = np.multiply(vector[0:(array.shape[0]-i)+int(r/2)+ex,int(r/2)-j:r],array[i-(int(r/2)+ex):array.shape[0],0:j+int(r/2)+ex])
        elif i < int(r/2)+ex:
            if j >= r and array.shape[1]-j > r:
                added = np.multiply(vector[int(r/2)-i:r,:],array[0:i+int((r/2)+ex),j-(int(r/2)+ex):j+int(r/2)])
            elif j >= r and array.shape[1]-j < int(r/2)+ex:
                added = np.multiply(vector[int(r/2)-i:r,0:(array.shape[1]-j)+(int(r/2)+ex)],array[0:i+int((r/2)+ex),j-(int(r/2)+ex):array.shape[1]])
            elif j < int(r/2)+ex:
                added = np.multiply(vector[int(r/2)-i:r,int(r/2)-j:r],array[0:i+int((r/2)+ex),0:j+(int(r/2)+ex)])
        return added
    
    def get_angle(self,point,dia):
        if self.boolimage[point] == False:
            ValueError('point must be contained within boolimage')
        'end if'
        array = self.image
        circle = Direction._create_circle(dia)
        added = Direction._boundary_satisfier(array,circle,point)
        maxVal = np.max(added)
        if maxVal == 0:
            return 0
        else:
            point_option = Direction.get_index_in_array(added,maxVal)
            xy = Direction.get_xy_in_array(added,dia,point_option[0])
            angle = Direction.get_angle_frm_cntr(xy,dia)
            return angle/np.pi*180
    'end def'
'end class'

class Gen_Training(Direction):
    def __init__(self,image):
        self.image = image
        self.updateImage = np.array([])
        self.layer = True
        self.boolimage = np.array([])
        self.points = []
    'end def'
    
    
    def _range_cont(self,array,layers,maxlayers,shift = 0,reverse = True):
        """
        generates a list of "thresholds" to be used for _contours.
        uses the max and min of the array as bounds and the number of layers as
        segmentation.
        array = np.array()
        layers = number of differentials (int)
        maxlayers = number of layers kept (int)
        shfit = a variable that shifts the sampling up and down based on maxlayers (int)
        reverse = start at the min or end at max of the array (True == end at max) (bool)
        """
        arraymax = np.max(array)+(np.max(array)/layers)
        arraymin = np.min(array)
        incr = arraymax/layers
        maxincr = arraymin+(incr*(maxlayers))
        if reverse == True:
            thresh = [arraymax-maxincr-shift*(incr*(maxlayers+1))]
            arraystart = arraymax-maxincr - shift*(incr*(maxlayers+1))
            maxincr = arraymax - shift*(incr*(maxlayers+1))
            
        elif reverse == False:
            thresh = [arraymin + shift*(incr*(maxlayers))]
            arraystart = arraymin + shift*(incr*(maxlayers))
            if shift > 0:
                maxincr = arraystart + shift*(incr*(maxlayers-1))
            'end if'
        try:
            while thresh[-1] < maxincr:
                thresh.append(np.float32(arraystart+incr))
                incr = incr + arraymax/layers
        except:
            raise ValueError("Max and Min == 0")
        'end try'
        return thresh
    'end def'
    
    def _contours(self,array,thresh):
        """takes an array and creates a topographical map based on layers and maxlayers. Useful in creating
        differential sections of images"""
        contours = [None]*(len(thresh))
        sums = [None]*(len(thresh))
        avgweight= [None]*(len(thresh))
        avgnan= [None]*(len(thresh)) 
        count = 0
        maxlayers = len(thresh)
        #############
        contours[count] = np.logical_not(np.add(array<thresh[count],array>thresh[count+1]))
        sums[count] = np.sum(contours[count])
        overim = contours[count]*array
        avgweight[count] = np.average(overim)
        overim[overim == 0] = 'nan'
        avgnan[count] = np.nanmean(overim)
        while count < maxlayers-2:
            count += 1
            contours[count] = np.logical_not(np.add(array<thresh[count],array>thresh[count+1]))
            sums[count] = np.sum(contours[count])
            overim = contours[count]*array
            avgweight[count] = np.average(overim)
            overim[overim == 0] = 'nan'
            avgnan[count] = np.nanmean(overim)
        'end while'
        sums = sums[0:count]
        contours = contours[0:count]
        avgnan = avgnan[0:count]
        avgweight = avgweight[0:count]
        return [contours,sums,avgnan,avgweight]
    'end def'
    
    def _remove_other(self,im_array):
        labeled = label(im_array)
        layers = np.unique(labeled)
        boolesum = [None]*(len(layers))
        count = 0
        for layer in layers:
            boolesum[count] = np.sum(labeled == layer)
            count+=1
        'end for'
        boolesum = boolesum[1:]
        for layer in layers:
            if np.sum(labeled == layer) < np.max(boolesum):
                labeled[labeled == layer] = 0
            'end if'
        'end for'
        return labeled
    'end def'
    
    def back_process(self):
        #find tissue Body
        circle = np.array([[0,0,1,1,0,0],[0,1,1,1,1,0],[1,1,1,1,1,1],[1,1,1,1,1,1],[0,1,1,1,1,0],[0,0,1,1,0,0]])
        filtim = self.FFfilt(self.image,0,500)
        flatimi = self.diffmat(filtim,np.arange(0,np.pi,(np.pi)/12),(10,10))
        convolved = convolve(flatimi,Filter._onediv_d3gaussian(100,1,1))
        thresh = self._range_cont(convolved,100_000,10,0,False)
        [contours,sums,avgnan,avgweight] = self._contours(convolved,thresh)

        count = 0
        for count in range(0,len(sums)):
            if np.sum(sums[count:]) < 800_000: #and abs(avgnan[count+1]-avgnan[count])/avgnan[count+1] > 0.5:
                threshfin = thresh[count]
                break
            else: 
                continue
            'end if'
        'end while'
        backim = flatimi>threshfin
        
        #Fill image
        backim = binary_closing(backim,circle)
        backadd = np.divide(np.add(backim,self.image),self.image)
        backadd[backadd == 1] = np.max(backadd)
        thresh = self._range_cont(backadd,100,20,0,False)
        [contours,sums,avgnan,avgweight] = self._contours(backadd,thresh)
        
        count = 0
        for i in range(0,len(sums)):
            if np.sum(sums[:i]) > (self.image.shape[0]*self.image.shape[1])/2:
                break
            else:
                continue
        'end for'
        backadd = backadd < thresh[i]
        backadd = remove_small_objects(backadd,min_size=40)
        backadd = binary_closing(backadd,circle)
#        self.badd = backadd
        backaddfin = self._remove_other(backadd)
        backaddfin = backaddfin >= 1

        return backaddfin
    'end def'
    
    def check_overlap(self,check):
        """
        Identifies the degree of off kilteredness in a estimated domain on a section
        """
        for i,row in enumerate(check):
            row = np.diff(row)
            
    
    def get_backdom(self,backim,Hn,Wn,diffo=0):
        if Hn%2 != 0 or Wn%2 != 0:
            ValueError("Hn and Wn need to be divisible by 2")
        'end if'
#        print('Hn: '+str(Hn)+' '+'Wn: '+str(Wn))
#        print('diff: '+str(diffo))
        H,W = backim.shape
        dom = np.zeros((H,W))
        
        m1H = int(H/2)
        m1W = int(W/2)
        
        m2H = int(Hn/2)
        m2W = int(Wn/2)
        
        dom[m1H-m2H:m1H+m2H,m1W-m2W:m1W+m2W] = np.ones((Hn,Wn))*2
        dom.reshape((H,W))
        check = np.add(backim,dom)
        diff = np.sum(check == 2)        
        
        if (check == 2).any() and abs(diff-diffo) > 150:
            diffo = diff
            Hn = Hn-int(m1H/4)
            Wn = Wn-int(m1W/4)
            return self.get_backdom(backim,Hn,Wn,diffo)
            
        return dom
        
    
    def find_canal(self,backim):
        W,H = self.image.shape
        domain = self.get_backdom(backim,1200,1200)
        imnew = convolve(self.image,self._d3gaussian(10,1,1))*domain
        imnew[imnew==0] = 'nan'
        canal = imnew == np.nanmin(imnew)       
        i,j = np.where(canal)
        ui = np.unique(i)
        ni = ui[int(len(ui)/2)]
        uj = np.unique(j)
        nj = uj[int(len(uj)/2)]
        return (i,j)
    'end def'
    
    def load_points(self,filename):
        os.chdir(r"C:\Users\jsalm\Documents\Python Scripts\3DRecon\saved_training")
        with open(filename) as csvfile:
            csvread = csv.reader(csvfile,delimiter=',')
            for row in csvread:
                self.points.append(row)
            'end for'
        'end with'
    'end def'
    
    def gen_training_img(self,savefile=False):
        image = np.zeros(self.img.shape,dtype=np.uint8)
        for i in np.arange(0,len(self.points),2):
            cv2.line(image,self.points[i],self.points[i+1],[255,0,0],1)
        'end for'
        cv2.imshow("image boolean",image)
        if savefile == True:
            dirname = r"C:\Users\jsalm\Documents\Python Scripts\3DRecon\saved_training"
            cv2.imwrite(os.path.join(dirname,"image_"+str(self.im_num)+".tif"),image)
        'end if'
        return image
    'end def'
        
'end class'

class Aff_Trans(Gen_Training):
    
    modes = ["Identity","Scaling","Rotation","Translation","Horizontal Shear","Vertical Shear"]
    identity = np.array([[1,0,0],[0,1,0],[0,0,1]])
    
    def __init__(self,image):
        self.image = None
        self.nW = None
        self.nH = None
    'end def'
    
    def _scaling(cx,cy):
        matrix = np.array([[cx,0,0],[0,cy,0],[0,0,1]])
        return matrix
    
    def _rotation(self,theta,scale=1):
        (h,w,_) = self.image.shape
        (cx,cy) = (w//2,h//2)
        M = cv2.getRotationMatrix2D((cx,cy),-theta, 1.0)
        cos = np.abs(M[0,0])
        sin = np.abs(M[0,1])
        nW = int((h*sin)+(w*cos))
        nH = int((h*cos)+(w*sin))
        self.nW = nW
        self.nH = nH
        
        M[0,2] += (nW/2)-cx
        M[1,2] += (nH/2)-cy
        M = np.concatenate((M,np.array([[0,0,1]])),axis=0)
        
        return M
    
    
    def _translation(tx,ty):
        matrix = np.array([[1,0,tx],
                  [0,1,ty],
                  [0,0,1]])
        return matrix
    
    def _horiz_shear(sh):
        matrix = np.array([[1,sh,0],
                           [0,1,0],
                           [0,0,1]])
        return matrix
    
    def _verti_shear(sv):
        matrix = np.array([[1,0,0],
                           [sv,1,0],
                           [0,0,1]])
        return matrix
    
    def affine_trans(self,**kwargs):
        """
        takes any number of transformations (e.g. trans1 = ("Rotation",30))
        **kwargs:
            Scaling (2 vars) - scales an image var1 = x-axis, var2 = y-axis
            Rotation (1 vars) - rotates images based on angle, in degrees
            Translation (2 vars) - translates image var1 = x-axis, var2 = y-axis, in pixels
            Horizontal Shear (1 vars) -
            Vetical Shear (1 vars) - 
        """
        mattrans = Aff_Trans.identity
        for key,value in kwargs.items():
            if type(value) != tuple:
                return TypeError("**kwargs needs to be type: tuple")
            else:
                try:
                    [i for i,x in enumerate(Aff_Trans.modes) if x == value[0]][0]
                except IndexError:
                    print("wrong mode type or name")
                else:
                    if value[0] == "Scaling":
                        matn = Aff_Trans._scaling(value[1],value[2])
                    elif value[0] == "Rotation":
                        matn = self._rotation(value[1])
                    elif value[0] == "Translation":
                        matn = Aff_Trans._translation(value[1],value[2])
                    elif value[0] == "Horizontal Shear":
                        matn = Aff_Trans._horiz_shear(value[1])
                    elif value[0] == "Vertical Shear":
                        matn = Aff_Trans._verti_shear(value[1])
                    'end if'
                    mattrans = np.dot(mattrans,matn)
                'end try'
            'end if'
        'end for'
        rows,cols,ch = self.image.shape
        T_opencv = np.float32(mattrans.flatten()[:6].reshape(2,3))
        img_transformed = cv2.warpAffine(self.image, T_opencv, (self.nW,self.nH))
        return img_transformed
    'end def'
    def perc_match(self,nimage):
        overlap = np.add(self.image,nimage)
        overlap[overlap==2]
        added = np.sum(overlap)/(np.sum(self.image))
    'end def'
    
    def match_match(self,nimage):
        pass
    'end def'
'end class'

if __name__ == "__main__":
    pass
'end if'




















