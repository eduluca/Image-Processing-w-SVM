# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:24:10 2020

@author: jsalm
"""
import numpy as np
import math
from scipy.ndimage import convolve,median_filter
from scipy.signal import wiener
from skimage.filters import gabor_kernel,threshold_otsu
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os


import cv2

def wiener_filter(image,filter_width=(5,5),noise=None):
    filtered_img = wiener(image,filter_width,noise)
    return filtered_img

def median_filt(image,size_v):
    filtered_img = median_filter(image,size=size_v,mode='wrap')
    return filtered_img

def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats,filtered


def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i

def normalize_img(image):
    im_new = np.sqrt((image-np.nanmean(image))**2)/np.nanstd(image)
    return im_new


def gabor_filter(image,frq_range,theta_den,sigma_range=(0,4)):
    """
    

    Parameters
    ----------
    image : np.float32[:,:]
        DESCRIPTION.
    frq_range : tuple
        DESCRIPTION.
    theta_den : int
        DESCRIPTION.
    sigma_range : tuple, optional
        DESCRIPTION. The default is (0,4).

    Returns
    -------
    kernels : TYPE
        DESCRIPTION.

    """
    kernels = []
    for theta in range(theta_den):
        theta = theta/theta_den*np.pi
        for sigma in sigma_range:
            for frequency in frq_range:
                kernel = np.real(gabor_kernel(frequency,theta=theta,
                                              sigma_x=sigma,sigma_y=sigma))
                kernels.append(kernel)
            'end for'
        'end for'
    'end for'
    
    feats,filtered = compute_feats(image,kernels)
    return filtered,kernels,feats
    
def range_cont(set_min,set_max,incr,shift = 0,reverse = False):
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
    maxsteps = math.floor(set_max/incr)
    incr_i = incr
    maxincr = set_min+(incr*(maxsteps))
    if set_min+(incr*maxsteps) > set_max:
        maxincr = set_max    
    if reverse == True:
        thresh = [int(set_max-maxincr-shift*(incr*(maxsteps+1)))]
        arraystart = set_max-maxincr - shift*(incr*(maxsteps+1))
        maxincr = set_max - shift*(incr*(maxsteps+1))
        
    elif reverse == False:
        thresh = [int(set_min + shift*(incr*(maxsteps)))]
        arraystart = set_min + shift*(incr*(maxsteps))
        if shift > 0:
            maxincr = arraystart + shift*(incr*(maxsteps-1))
        'end if'
    try:
        while thresh[-1] < maxincr:
            thresh.append(np.int32(arraystart+incr_i))
            incr_i = incr_i + incr
        'end while'
        thresh.append(np.int32(set_max))
    except:
        raise ValueError("Max and Min == 0")
    'end try'
    return thresh
'end def'

def adaptive_threshold(image,sW,bins,sub_sampling=True):      
#this helps significantly, but les try adding sub sample
    nH,nW = image.shape
    new_image = np.zeros((nH,nW))
    sbxx = range_cont(0,nW,sW)
    sbxy = range_cont(0,nH,sW)
    if sub_sampling:
        for i in range(0,len(sbxy)-1):
            for j in range(0,len(sbxx)-1):
                try:
                    thresh = threshold_otsu(image[sbxy[i]:sbxy[i+1],sbxx[j]:sbxx[j+1]],nbins=256)
                except ValueError:
                    thresh = np.max(image[sbxy[i]:sbxy[i+1],sbxx[j]:sbxx[j+1]])
                threshed = (image[sbxy[i]:sbxy[i+1],sbxx[j]:sbxx[j+1]] > thresh)*image[sbxy[i]:sbxy[i+1],sbxx[j]:sbxx[j+1]]
                new_image[sbxy[i]:sbxy[i+1],sbxx[j]:sbxx[j+1]] = threshed
    else:
        thresh = threshold_otsu(image,nbins=bins)
        new_image = image*(image>thresh)
    return new_image

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
 
def Hi_pass_filter(image,width):
     dft = cv2.dft(np.float32(image),flags=cv2.DFT_COMPLEX_OUTPUT)
     dft_shift = np.fft.fftshift(dft)
     
     rows,cols = image.shape
     crow,ccol = rows//2,cols//2
     
     # mask = np.zeros((rows,cols,2),np.uint8)
     # mask[crow-width:crow+width, ccol-width:ccol+width] = 1
     # mask = np.uint8(mask == 0)
     
     circle = _create_circle(width,True)
     mask = np.zeros((rows,cols,2),np.uint8)
     adjust = width//2
     mask[crow-adjust:crow+adjust+1, ccol-adjust:ccol+adjust+1,0] = circle
     mask[crow-adjust:crow+adjust+1, ccol-adjust:ccol+adjust+1,1] = circle
     mask = np.uint8(mask == 0)
     
     
     
     fshift = dft_shift*mask
     f_ishift = np.fft.ifftshift(fshift)
     img_back = cv2.idft(f_ishift)
     img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
     return img_back
"end def"    

def average_filter(image,width,C):
    kernel = np.ones((width,width),dtype=np.uint8)
    kernel = kernel*C
    image_out = convolve(image,kernel)
    return image_out
# def FFfilt(image,reduction_factor,width,Lo_pass=True):
#     if Lo_pass:
#         began = int(image.shape[0]/2)-int(width/2)
#         end = int(image.shape[0]/2)+int(width/2)
#         gfilt = np.zeros((image.shape[0],image.shape[1]))
#         gfft = np.fft.fftshift(np.fft.fft(image))           
#         gfft[:,:began] = reduction_factor*gfft[:,:began]
#         gfft[:,end:] = reduction_factor*gfft[:,end:]
#         gfilt = abs(np.fft.ifft(np.fft.fftshift(gfft)))
#         # greturn = gfilt
#         greturn = np.multiply(gfilt,image)
#     else:
#         began = int(image.shape[0]/2)-int(width/2)
#         end = int(image.shape[0]/2)+int(width/2)
#         gfilt = np.zeros((image.shape[0],image.shape[1]))
#         gfft = np.fft.fftshift(np.fft.fft(image))           
#         gfft[:,began:end] = reduction_factor*gfft[:,began:end]
#         gfilt = abs(np.fft.ifft(np.fft.fftshift(gfft)))
#         # greturn = gfilt
#         greturn = np.multiply(gfilt,image)
#     'end if'
#     return greturn
# 'end def'

def _d3gaussian(vector_width,multiplier_a,multiplier_d):
    """
    creates a 3D gaussian inlayed into matrix form.
    of the form: G(x,y) = a*e^(-((x-N)**2+(y-N)**2)/(2*d**2))
    """
    x = np.arange(0,vector_width,1)
    y = np.arange(0,vector_width,1)
    d2gau = np.zeros((vector_width,vector_width)).astype(float)
    N = int(vector_width/2)
    for i in x:
        d2gau[:,i] = multiplier_a*np.exp(-((i-N)**2+(y-N)**2)/(2*multiplier_d**2))
    'end for'
    return d2gau     
'end def'

def _onediv_d3gaussian(vector_width,multiplier_a,multiplier_d):
    """
    creates a 3D gaussian inlayed into matrix form.
    of the form: G(x,y) = a*e^(-((x-N)**2+(y-N)**2)/(2*d**2))
    """
    x = np.arange(0,vector_width,1)
    y = np.arange(0,vector_width,1)
    d2gau_x = np.zeros((vector_width,vector_width)).astype(float)
    d2gau_y = np.zeros((vector_width,vector_width)).astype(float)
    N = int(vector_width/2)
    for i in x:
        d2gau_x[:,i] = -((i-N)/multiplier_d**2)*multiplier_a*np.exp(-((i-N)**2+(y-N)**2)/(2*multiplier_d**2))
        d2gau_y[:,i] = -((y-N)/multiplier_d**2)*multiplier_a*np.exp(-((i-N)**2+(y-N)**2)/(2*multiplier_d**2))
    'end for'
    d2gau = np.add(d2gau_x**2,d2gau_y**2)
    return d2gau     
'end def'

def diffmat(image,theta,dim=(10,2)):
    if type(dim) != tuple:
        raise ValueError('dim must be tuple')
    if dim[1]%2 != 0:
        raise ValueError('n must be even')
    'end if'
    outarray = np.zeros((image.shape[0],image.shape[1]))
    dfmat = np.zeros((max(dim),max(dim)))
    dfmat[:,0:int(dim[1]/2)] = -1
    dfmat[:,int(dim[1]/2):dim[1]] = 1
    
    dmatx = dfmat
    dmaty = np.transpose(dfmat)
    for angle in theta:
        dmat = dmatx*np.cos(angle)+dmaty*np.sin(angle)
        dm = np.divide(convolve(image,dmat)**2,math.factorial(max(dim)))
        outarray = np.add(dm,outarray)
    'end for'
    return outarray
'end def'

def diagonal_map(image):
    #variable filter that convolves different 
    pass
'end def'

def imshow_overlay(im, mask, title="default",savefig=False,  genfig=True, alpha=0.3, color='red', **kwargs):
    """Show semi-transparent red mask over an image"""
    mask = mask > 0
    mask = np.ma.masked_where(~mask, mask)
    if genfig:
        plt.figure(title)  
        plt.imshow(im, **kwargs)
        plt.imshow(mask, alpha=alpha, cmap=ListedColormap([color]))
    'end if'
    
    if savefig:
        foldername = "saved_im"
        savedir = os.path.join(os.path.dirname(__file__),foldername)
        plt.savefig(os.path.join(savedir,"overlayed_"+str(title)+".tif"),dpi=600,quality=95,pad_inches=0)
    'end if'
'end def'

def simplest_countless(data):
  """
  Vectorized implementation of downsampling a 2D 
  image by 2 on each side using the COUNTLESS algorithm.
  
  data is a 2D numpy array with even dimensions.
  """
  sections = []
  
  # This loop splits the 2D array apart into four arrays that are
  # all the result of striding by 2 and offset by (0,0), (0,1), (1,0), 
  # and (1,1) representing the A, B, C, and D positions from Figure 1.
  factor = (2,2)
  for offset in np.ndindex(factor):
    part = data[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
    sections.append(part)

  a, b, c, d = sections

  ab = a * (a == b) # PICK(A,B)
  ac = a * (a == c) # PICK(A,C)
  bc = b * (b == c) # PICK(B,C)

  a = ab | ac | bc # Bitwise OR, safe b/c non-matches are zeroed
  
  return a + (a == 0) * d # AB || AC || BC || D

def zero_corrected_countless(data):
  """
  Vectorized implementation of downsampling a 2D 
  image by 2 on each side using the COUNTLESS algorithm.
  
  data is a 2D numpy array with even dimensions.
  """
  # allows us to prevent losing 1/2 a bit of information 
  # at the top end by using a bigger type. Without this 255 is handled incorrectly.
  data, upgraded = upgrade_type(data) 

  data = data + 1 # don't use +=, it will affect the original data.

  sections = []
  
  # This loop splits the 2D array apart into four arrays that are
  # all the result of striding by 2 and offset by (0,0), (0,1), (1,0), 
  # and (1,1) representing the A, B, C, and D positions from Figure 1.
  factor = (2,2)
  for offset in np.ndindex(factor):
    part = data[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
    sections.append(part)

  a, b, c, d = sections

  ab = a * (a == b) # PICK(A,B)
  ac = a * (a == c) # PICK(A,C)
  bc = b * (b == c) # PICK(B,C)

  a = ab | ac | bc # Bitwise OR, safe b/c non-matches are zeroed
  
  result = a + (a == 0) * d - 1 # a or d - 1

  if upgraded:
    return downgrade_type(result)

  return result

def upgrade_type(arr):
  dtype = arr.dtype

  if dtype == np.uint8:
    return arr.astype(np.uint16), True
  elif dtype == np.uint16:
    return arr.astype(np.uint32), True
  elif dtype == np.uint32:
    return arr.astype(np.uint64), True

  return arr, False
  
def downgrade_type(arr):
  dtype = arr.dtype

  if dtype == np.uint64:
    return arr.astype(np.uint32)
  elif dtype == np.uint32:
    return arr.astype(np.uint16)
  elif dtype == np.uint16:
    return arr.astype(np.uint8)
  
  return arr

def countless(data):
  """
  Vectorized implementation of downsampling a 2D 
  image by 2 on each side using the COUNTLESS algorithm.
  
  data is a 2D numpy array with even dimensions.
  """
  # allows us to prevent losing 1/2 a bit of information 
  # at the top end by using a bigger type. Without this 255 is handled incorrectly.
  data, upgraded = upgrade_type(data) 

  data = data + 1 # don't use +=, it will affect the original data.

  sections = []
  
  # This loop splits the 2D array apart into four arrays that are
  # all the result of striding by 2 and offset by (0,0), (0,1), (1,0), 
  # and (1,1) representing the A, B, C, and D positions from Figure 1.
  factor = (2,2)
  for offset in np.ndindex(factor):
    part = data[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
    sections.append(part)

  a, b, c, d = sections
  
  
  ab_ac = a * ((a == b) | (a == c)) # PICK(A,B) || PICK(A,C) w/ optimization
  bc = b * (b == c) # PICK(B,C)

  a = ab_ac | bc # (PICK(A,B) || PICK(A,C)) or PICK(B,C)
  result = a + (a == 0) * d - 1 # (matches or d) - 1

  if upgraded:
    return downgrade_type(result)

  return result


if __name__=="__main__":
    dirname = os.path.dirname(__file__)
    image = np.array(cv2.imread(os.path.join(dirname,"images_5HT/dAIH_20x_sectioning_2_CH2.tif"))[:,:,2]/255).astype(np.float32)
    kernels = gabor_filter(image,(0.05,0.25),8,(0,4))
    normalized = normalize_img(image)
    jj = wiener_filter(normalized)
    filtfilt = median_filt(jj,5)
    filt_cont = abs(normalized-filtfilt)
    threshed = adaptive_threshold(filt_cont,50,2)
    # out = countless(np.uint8(image*255))
    # plt.figure('old');plt.imshow(image)
    # plt.figure('new');plt.imshow(out)
    plt.figure('1');plt.imshow(filtfilt)
    plt.figure('2');plt.imshow(filt_cont)
    plt.figure('3');plt.imshow(normalized)
    plt.figure('4');plt.imshow(threshed)