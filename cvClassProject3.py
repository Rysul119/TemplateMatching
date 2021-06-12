#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 19:54:51 2021

@author: rysul
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def showImage(imgMatrix, name):
    '''
    displays an image
    '''
    cv.imshow(name, imgMatrix)
    cv.waitKey(0)
    cv.destroyAllWindows()
    

def createPyramid(image, levels):
    '''
    creates an image pyramid for given number of levels
    '''
    gaussPyr = [image]
    
    for i in range(levels-1):
        image = cv.pyrDown(image)
        gaussPyr.append(image)
        
    return gaussPyr


def getMetric(image1, image2, mode):
    '''
    get value of correlating coefficient for the given inputwindow and reference window for a given mode
    '''
    if(mode=='ncc'):
        image1Normalized = (image1)/np.linalg.norm(image1)
        
        image2Normalized = (image2)/np.linalg.norm(image2)
        
        coefficient = np.sum(image1Normalized * image2Normalized)
        
    elif(mode =='nssd'):
        
        image1Normalized = (image1)/np.linalg.norm(image1)
        
        image2Normalized = (image2)/np.linalg.norm(image2)
        
        coefficient = np.sum(np.square(image1Normalized - image2Normalized))
        
        
    return coefficient


def fullRegionSearch(inputWindow, refWindow, mode):
    '''
    it returns the best point from correlation coefficient matrix for the given input and reference window
    '''
    
    # at first create zero paddings around the input window
    
    ht, wd= inputWindow.shape # get the shape of the input window

    # create new image of desired size and color (blue) for zero padding
    ww = inputWindow.shape[0]+refWindow.shape[0]-1
    hh = inputWindow.shape[1]+refWindow.shape[1]-1
    
    result = np.full((hh,ww), 0, dtype=np.uint8)
    
    # compute center offset
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2
    
    # copy img image into center of result image
    result[yy:yy+ht, xx:xx+wd] = inputWindow
    
    corrCoeff = np.zeros_like(inputWindow, dtype = np.float)
    
    h = (refWindow.shape[0]-1)//2
    
    for i in range(inputWindow.shape[0]):
        for j in range(inputWindow.shape[1]):
            corrCoeff[i,j] = getMetric(refWindow[:refWindow.shape[0]-1,:refWindow.shape[1]-1], result[i:i+2*h+1, j:j+2*h+1], mode)
    
    if(mode == 'ncc'):
        y, x = np.unravel_index(np.argmax(corrCoeff), corrCoeff.shape)  # find the match
    elif(mode == 'nssd'):
        y, x = np.unravel_index(np.argmin(corrCoeff), corrCoeff.shape)  # find the match

    return x, y, corrCoeff

def regionSearch(image, template, xbest, ybest, mode):
    '''
    it returns the best point from correlation coefficient matrix for the given input and reference window
    '''
    m = template.shape[0]
    
    inputWindow = image[(2 * xbest)-m-2: (2 * xbest)+m, (2 * ybest)-m:(2 * ybest)+m+1]

    
    refWindow = template
    # at first create zero paddings around the input window
    
    ht, wd= inputWindow.shape # get the shape of the input window

    # create new image of desired size and color (blue) for zero padding
    ww = inputWindow.shape[0]+refWindow.shape[0]-1
    hh = inputWindow.shape[1]+refWindow.shape[1]-1
    
    result = np.full((hh,ww), 0, dtype=np.uint8)
    
    # compute center offset
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2
    
    # copy img image into center of result image
    result[yy:yy+ht, xx:xx+wd] = inputWindow
    
    corrCoeffr = np.zeros_like(inputWindow, dtype = np.float)
    corrCoeff = np.ones_like(image, dtype = np.float)
    h = (refWindow.shape[0]-1)//2
    
    for i in range(inputWindow.shape[0]):
        for j in range(inputWindow.shape[1]):
            corrCoeffr[i,j] = getMetric(refWindow[:refWindow.shape[0]-1,:refWindow.shape[1]-1], result[i:i+2*h+1, j:j+2*h+1], mode)
    
    
    

    if(mode == 'ncc'):
        corrCoeff = corrCoeff * 0
        corrCoeff[(2 * xbest)-m-2: (2 * xbest)+m, (2 * ybest)-m:(2 * ybest)+m+1] = corrCoeffr
        y, x = np.unravel_index(np.argmax(corrCoeff), corrCoeff.shape)  # find the match
    elif(mode == 'nssd'):
        corrCoeff = corrCoeff * 255
        corrCoeff[(2 * xbest)-m-2: (2 * xbest)+m, (2 * ybest)-m:(2 * ybest)+m+1] = corrCoeffr
        y, x = np.unravel_index(np.argmin(corrCoeff), corrCoeff.shape)  # find the match
    return x, y, corrCoeff


# now pyramid search 
def pyrTemplateMatch(image, template, no_of_levels, mode):
    
    level=no_of_levels
    
    gaussPyr = createPyramid(image, levels = level) # creates a pyramid of the input image
    gaussPyrT = createPyramid(template, levels = level) # creates a pyramid of the input image

    bestPoints = {'x':[],
                  'y':[]}
    coeffMat = []
    refWindow = gaussPyrT[level-1]
    inputWindow = gaussPyr[level-1]
    
    # full region search
    ybest, xbest, coeff = fullRegionSearch(inputWindow, refWindow,mode)
    bestPoints['x'].append(xbest)
    bestPoints['y'].append(ybest)
    coeffMat.append(coeff)
    
    for i in reversed(range(2)):
      
        refWindow = gaussPyrT[i]
        
        ybest, xbest, coeff = regionSearch(gaussPyr[i], gaussPyrT[i], xbest, ybest, mode)
        
        bestPoints['x'].append(xbest)
        bestPoints['y'].append(ybest)
        
        coeffMat.append(coeff)
        
    for i in range(level):
        pyrImage = gaussPyr[i].copy()
        h = (gaussPyrT[i].shape[0]-1)//2
        y = bestPoints['x'][level-1-i]
        x = bestPoints['y'][level-1-i]
        pyrImage[y-h:y+h+1, x-h:x+h+1] = 0 # for 
        #print("Correlated image size{}".format(coeffMat[level-1-i].shape))
        fig, (ax_orig, ax_template, ax_corr) = plt.subplots(3, 1, figsize=(6, 15))
        ax_orig.imshow(pyrImage, cmap='gray')
        ax_orig.set_title('Original')
        ax_orig.set_axis_off()
        ax_template.imshow(gaussPyrT[i], cmap='gray')
        ax_template.set_title('Template')
        ax_template.set_axis_off()
        ax_corr.imshow(coeffMat[level-1-i], cmap='gray')
        ax_corr.set_title('Correlation Coefficient')
        ax_corr.set_axis_off()
        fig.show()
        plt.savefig('outputs/pyrlevel'+mode+str(i+1)+'.jpg')
    
if __name__ == '__main__':
    # reading the image
    img = cv.imread('messi.jpg')
    # converting the image to grayscale
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    
    #print("Shape of the original imgae: {}".format(imgGray.shape))
    # resize the image to 256x256
    imgGrayResized = cv.resize(imgGray, (256,256), interpolation = cv.INTER_AREA)
    #print("Shape of the resized image: {}".format(imgGrayResized.shape))
    #showImage(imgGrayResized,'messiGrayResized')
    
    # create a template (messi's face)
    template = imgGray[80:132,210:265]
    templateResized = cv.resize(template, (32,32), interpolation = cv.INTER_AREA)
    #showImage(template,'template')
    #print("Shape of the template: {}".format(template.shape)) 
    
    
    pyrTemplateMatch(imgGrayResized, templateResized, 3, 'ncc')
    pyrTemplateMatch(imgGrayResized, templateResized, 3, 'nssd')


