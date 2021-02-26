# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:04:48 2020

@author: jsalm
"""

# import the necessary packages
import cv2
import os
import numpy as np
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
import DataManager



refPt = []

cropping = False
# global DEFAULT 
# DEFAULT = np.array([])

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)
'end def'

def mark_positive_line(event, x, y, flags, param):
    global refPt, tracing
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x,y)]
        tracing = True
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x,y))
        tracing = False
        cv2.line(image,refPt[0],refPt[1], [255,0,0], 2)
        cv2.imshow("image",image)
    'end if'
'end def'

class PanAndZoomState(object):
    """ Tracks the currently-shown rectangle of the image.
    Does the math to adjust this rectangle to pan and zoom."""
    MIN_SHAPE = np.array([50,50])
    def __init__(self, imShape, parentWindow):
        self.ul = np.array([0,0]) #upper left of the zoomed rectangle (expressed as y,x)
        self.imShape = np.array(imShape[0:2])
        self.shape = self.imShape #current dimensions of rectangle
        self.parentWindow = parentWindow
    def zoom(self,relativeCy,relativeCx,zoomInFactor):
        self.shape = (self.shape.astype(np.float)/zoomInFactor).astype(np.int)
        #expands the view to a square shape if possible. (I don't know how to get the actual window aspect ratio)
        self.shape[:] = np.max(self.shape) 
        self.shape = np.maximum(PanAndZoomState.MIN_SHAPE,self.shape) #prevent zooming in too far
        c = self.ul+np.array([relativeCy,relativeCx])
        self.ul = c-self.shape/2
        self._fixBoundsAndDraw()
    def _fixBoundsAndDraw(self):
        """ Ensures we didn't scroll/zoom outside the image. 
        Then draws the currently-shown rectangle of the image."""
#        print "in self.ul:",self.ul, "shape:",self.shape
        self.ul = np.maximum(0,np.minimum(self.ul, self.imShape-self.shape))
        self.shape = np.minimum(np.maximum(PanAndZoomState.MIN_SHAPE,self.shape), self.imShape-self.ul)
#        print "out self.ul:",self.ul, "shape:",self.shape
        yFraction = float(self.ul[0])/max(1,self.imShape[0]-self.shape[0])
        xFraction = float(self.ul[1])/max(1,self.imShape[1]-self.shape[1])
        cv2.setTrackbarPos(self.parentWindow.H_TRACKBAR_NAME, self.parentWindow.WINDOW_NAME,int(xFraction*self.parentWindow.TRACKBAR_TICKS))
        cv2.setTrackbarPos(self.parentWindow.V_TRACKBAR_NAME, self.parentWindow.WINDOW_NAME,int(yFraction*self.parentWindow.TRACKBAR_TICKS))
        self.parentWindow.redrawImage()
    def setYAbsoluteOffset(self,yPixel):
        self.ul[0] = min(max(0,yPixel), self.imShape[0]-self.shape[0])
        self._fixBoundsAndDraw()
    def setXAbsoluteOffset(self,xPixel):
        self.ul[1] = min(max(0,xPixel), self.imShape[1]-self.shape[1])
        self._fixBoundsAndDraw()
    def setYFractionOffset(self,fraction):
        """ pans so the upper-left zoomed rectange is "fraction" of the way down the image."""
        self.ul[0] = int(round((self.imShape[0]-self.shape[0])*fraction))
        self._fixBoundsAndDraw()
    def setXFractionOffset(self,fraction):
        """ pans so the upper-left zoomed rectange is "fraction" of the way right on the image."""
        self.ul[1] = int(round((self.imShape[1]-self.shape[1])*fraction))
        self._fixBoundsAndDraw()
'end class'

class PanZoomWindow(object):
    """ Controls an OpenCV window. Registers a mouse listener so that:
        1. right-dragging up/down zooms in/out
        2. right-clicking re-centers
        3. trackbars scroll vertically and horizontally 
        4. pressing and dragging left button appends points to form trace
    You can open multiple windows at once if you specify different window names.
    You can pass in an onLeftClickFunction, and when the user left-clicks, this 
    will call onLeftClickFunction(y,x), with y,x in original image coordinates."""
    def __init__(self,channel,img,img_name,key = -1, windowName = 'PanZoomWindow', onLeftClickFunction = None):
        self.WINDOW_NAME = windowName
        self.IMG_NAME = img_name
        self.H_TRACKBAR_NAME = 'x'
        self.V_TRACKBAR_NAME = 'y'
        self.DEFAULT_c = img[:,:,channel].copy()
        self.img = img
        self.img_prev = img.copy()
        self.onLeftClickFunction = onLeftClickFunction
        self.TRACKBAR_TICKS = 1000
        self.panAndZoomState = PanAndZoomState(img.shape, self)
        self.lButtonDownLoc = None
        self.mButtonDownLoc = None
        self.rButtonDownLoc = None
        self.tool_feature = "l"
        self.incMode = True
        self.a = None
        self.b = None
        self.points = [[]]
        self.points_display = [[]]
        self.poly_counter = -1
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        self.redrawImage()
        cv2.setMouseCallback(self.WINDOW_NAME, self.onMouse)
        cv2.createTrackbar(self.H_TRACKBAR_NAME, self.WINDOW_NAME, 0, self.TRACKBAR_TICKS, self.onHTrackbarMove)
        cv2.createTrackbar(self.V_TRACKBAR_NAME, self.WINDOW_NAME, 0, self.TRACKBAR_TICKS, self.onVTrackbarMove)
    def onMouse(self, event, xc, yc, _Ignore1, _Ignore2):
        """ Responds to mouse events within the window. 
        The x and y are pixel coordinates in the image currently being displayed.
        If the user has zoomed in, the image being displayed is a sub-region, so you'll need to
        add self.panAndZoomState.ul to get the coordinates in the full image."""
        global tracing, key
        if event == cv2.EVENT_RBUTTONDOWN:
            #record where the user started to right-drag
            self.mButtonDownLoc = np.array([yc,xc])
        elif event == cv2.EVENT_RBUTTONUP and self.mButtonDownLoc is not None:
            #the user just finished right-dragging
            dy = yc - self.mButtonDownLoc[0]
            pixelsPerDoubling = 0.2*self.panAndZoomState.shape[0] #lower = zoom more
            changeFactor = (1.0+abs(dy)/pixelsPerDoubling)
            changeFactor = min(max(1.0,changeFactor),5.0)
            if changeFactor < 1.05:
                dy = 0 #this was a click, not a draw. So don't zoom, just re-center.
            if dy > 0: #moved down, so zoom out.
                zoomInFactor = 1.0/changeFactor
            else:
                zoomInFactor = changeFactor
            self.panAndZoomState.zoom(self.mButtonDownLoc[0], self.mButtonDownLoc[1], zoomInFactor)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.img_prev = self.img.copy()
            
            
            coordsInDisplayedImage = np.array([xc,yc])
            if np.any(coordsInDisplayedImage < 0):
                print("you clicked outside the image area")
            elif coordsInDisplayedImage[0] > self.panAndZoomState.shape[1] or coordsInDisplayedImage[1] > self.panAndZoomState.shape[0]:
                print("you are an idiot")
                print(coordsInDisplayedImage)
                print(self.panAndZoomState.shape[:2])
            else:
                tracing = True
                #initiate new line
                self.poly_counter += 1
                
                self.a, self.b = xc, yc
                print('LBD: ',str(tracing))
                print("appending point: ", coordsInDisplayedImage.astype(int))
                coordsInFullImage = self.panAndZoomState.ul + coordsInDisplayedImage
                if self.tool_feature == "l":
                    self.points[self.poly_counter].append(self.incMode)
                    self.points[self.poly_counter].append('l')
                elif self.tool_feature == "f":
                    self.points[self.poly_counter].append(self.incMode)
                    self.points[self.poly_counter].append('f')
                self.points[self.poly_counter].append([int(coordsInFullImage[0]),int(coordsInFullImage[1])])
                self.points_display[self.poly_counter].append([int(coordsInDisplayedImage[0]),int(coordsInDisplayedImage[1])])
                
                if self.onLeftClickFunction is not None:
                    self.onLeftClickFunction(int(coordsInFullImage[0]),int(coordsInFullImage[1]))
                'end if'
            'end if'
        elif event == cv2.EVENT_MOUSEMOVE:
            try:
                if tracing == True:
                    # dist = np.sqrt((a-xc)**2+(b-yc)**2)
                    if self.a != xc and self.b != yc:
                        coordsInDisplayedImage = np.array([xc,yc])
                        coordsInFullImage = self.panAndZoomState.ul+coordsInDisplayedImage
                        self.points[self.poly_counter].append([int(coordsInFullImage[0]),int(coordsInFullImage[1])])
                        self.points_display[self.poly_counter].append([int(coordsInDisplayedImage[0]),int(coordsInDisplayedImage[1])])
                    'end if'
            except NameError:
                pass
            'end if'
        elif event == cv2.EVENT_LBUTTONUP:
            tracing = False
            print('LBU: ',str(tracing))
            coordsInDisplayedImage = np.array([xc,yc])
            coordsInFullImage = self.panAndZoomState.ul+coordsInDisplayedImage
            pzs = self.panAndZoomState
            self.points[self.poly_counter].append([int(coordsInFullImage[0]),int(coordsInFullImage[1])])
            self.points_display[self.poly_counter].append([int(coordsInDisplayedImage[0]),int(coordsInDisplayedImage[1])])            
            print("appending point: ",coordsInFullImage.astype(int))
            pointstate = np.array(self.points_display[self.poly_counter]).reshape((-1,1,2))
            bool_state = np.array(self.points[self.poly_counter][2:]).reshape((-1,1,2))
            if self.points[self.poly_counter][1] == "l":
                if self.points[self.poly_counter][0]:
                    cv2.polylines(self.img[int(pzs.ul[0]):int(pzs.ul[0]+pzs.shape[0]), int(pzs.ul[1]):int(pzs.ul[1]+pzs.shape[1])] , [pointstate],False ,[255,0,0], 1)
                else:
                    cv2.polylines(self.img[int(pzs.ul[0]):int(pzs.ul[0]+pzs.shape[0]), int(pzs.ul[1]):int(pzs.ul[1]+pzs.shape[1])] , [pointstate],False ,[1,0,0], 1)
            elif self.points[self.poly_counter][1] == "f":
                if self.points[self.poly_counter][0]:
                    cv2.fillPoly(self.img[int(pzs.ul[0]):int(pzs.ul[0]+pzs.shape[0]), int(pzs.ul[1]):int(pzs.ul[1]+pzs.shape[1])], [pointstate],[255,0,0])
                else:
                    cv2.fillPoly(self.img[int(pzs.ul[0]):int(pzs.ul[0]+pzs.shape[0]), int(pzs.ul[1]):int(pzs.ul[1]+pzs.shape[1])], [pointstate],[1,0,0])
            cv2.imshow(self.WINDOW_NAME,self.img[int(pzs.ul[0]):int(pzs.ul[0]+pzs.shape[0]), int(pzs.ul[1]):int(pzs.ul[1]+pzs.shape[1])])
            
            self.points.append([])
            self.points_display.append([])
        'end if'
    
    
    def retract_lines(self):
        self.img = self.img_prev.copy() 
        pzs = self.panAndZoomState
        cv2.imshow(self.WINDOW_NAME,self.img[int(pzs.ul[0]):int(pzs.ul[0]+pzs.shape[0]), int(pzs.ul[1]):int(pzs.ul[1]+pzs.shape[1])])
    'end def'
    
    def menuChange(self):
        if cv2.waitKey(0) == ord('m'):
            print("""
                  2 tools: polyline or polyfill
                  press 't' to change tool
                  press 'c' to change mode
                  press 'q' to quit 
                  press 'r' to reset line
                  press 'p' to predict
                  """)
            print("press 't' to change from "+str(self.tool_feature))
            if cv2.waitKey(0) == ord('t'):
                print('changing tool')
                if self.tool_feature == "l":
                    self.tool_feature = "f"
                elif self.tool_feature == "f":
                    self.tool_feature = "l"
            if cv2.waitKey(0) == ord('c'):
                print('changing mode')
                if self.incMode:
                    self.incMode = False
                else:
                    self.incMode = True
                return 0
            if cv2.waitKey(0) == ord('r'):
                print('redacting line')
                del self.points[-2:]
                del self.points_display[-2:]
                self.poly_counter -= 1

                self.retract_lines()
                
                self.points.append([])
                self.points_display.append([])
            if cv2.waitKey(0) == ord('p'):
                print("predicting...")
                img = self.inter_Polate(3,'poly')
                cv2.imshow(self.WINDOW_NAME,img)
        else:
            return 0
    'end def'
    def onVTrackbarMove(self,tickPosition):
        self.panAndZoomState.setYFractionOffset(float(tickPosition)/self.TRACKBAR_TICKS)
    def onHTrackbarMove(self,tickPosition):
        self.panAndZoomState.setXFractionOffset(float(tickPosition)/self.TRACKBAR_TICKS)
    def redrawImage(self):
        pzs = self.panAndZoomState
        cv2.imshow(self.WINDOW_NAME, self.img[int(pzs.ul[0]):int(pzs.ul[0]+pzs.shape[0]), int(pzs.ul[1]):int(pzs.ul[1]+pzs.shape[1])])
    def export_point_data(self,im_num,save_file):
        os.chdir(os.path.join(os.path.dirname(__file__),save_file))
        array = self.img[:,:,0] == 255
        array.astype('int16').tofile(self.IMG_NAME+"_traind.bin")
    'end def'
    def predict_rest(self,save=True):
        pass
    'end def'
'end class'

def import_train_data(im_name,imshape,filename):
    """
    

    Parameters
    ----------
    im_name : string
        name of file containing training data (.tif file)
    imshape : tuple
        Width and Length of the image
    filename : string
        directory where images are being stored

    Returns
    -------
    bitimage : int16
        binary/integer image of true false labelings for training data

    """
    filedir = os.path.join(os.path.dirname(__file__),filename,im_name+"_traind.bin")
    bitimage = np.fromfile(filedir,dtype='int16')
    bitimage = bitimage.reshape(imshape)
    return bitimage
'end def'

def main(image,im_num,name):
    window = PanZoomWindow(2,image,name,name)
    key = -1
    print("press 'm' for Menu")
    # keep looping until the 'q' key is pressed
    while key != ord('q') and key != 27 and cv2.getWindowProperty(window.WINDOW_NAME,0) >=0:
        cv2.waitKey(0)
        window.menuChange()
    'end while'
    # close all open windows
    cv2.destroyAllWindows()
    window.export_point_data(im_num,"train_71420")
    return window
'end def'
'''
if __name__ == "__main__":
    foldername = "images_5HT"
    #dealing with the Channel situation: display RGB but edit gray scale
    im_dir = ThreeD_Recon_V3.DataMang(foldername)
    im_list = [i for i in range(0,im_dir.dir_len)]
    count = 0
    for gen in im_dir.open_dir(im_list):
        image,nW,nH,chan = gen
        wind = PanZoomWindow(2,image)
        wind.points = import_point_data(count,"train_71420")
        count +=1
        break
    X, y, Xp, fitted, predictions = wind.inter_Polate(3,'poly')
    wind.print_model(X, y, Xp,fitted,predictions)
    predict_img = wind.data_to_img(predictions)
    plt.figure('prediction image');plt.imshow(predict_img)
    Filters.imshow_overlay(wind.DEFAULT_c,)

'''
if __name__ == "__main__":
    foldername = r"C:\Users\jsalm\Documents\Python Scripts\SVM_7232020\images_5HT"
    #dealing with the Channel situation: display RGB but edit gray scale
    im_dir = DataManager.DataMang(foldername)
    im_list = [i for i in range(0,im_dir.dir_len)]
    count = 0
    for gen in im_dir.open_dir(im_list):
        image,nW,nH,chan,name = gen
        print("1")
        wind = main(image,im_list[count],name)
        bool_im = import_train_data(name,(nW,nH),'train_71420')
        count += 1
        break
    'end for'
"end if"

boolim = import_train_data('dAIH_20x_sectioning_2_CH2',(720,960),"train_71420")
# jj = import_point_data(0,"train_71420")
#6/29/2020: fillPoly() works, storage works, reconstruction works, just could use some user friendliness
