#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 14:58:03 2021

@author: rysul
"""
import numpy as np
import cv2 as cv
from scipy import signal
import matplotlib.pyplot as plt

def showImage(imgMatrix, name):
    cv.imshow(name, imgMatrix)
    cv.waitKey(0)
    cv.destroyAllWindows()

# reading the image
img = cv.imread('messi.jpg')
# converting the image to grayscale
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#showImage(img, 'messiColor')
#showImage(imgGray,'messiGray')
#row, column = imgGray.shape

print("Shape of the original imgae: {}".format(imgGray.shape))

imgGrayResized = cv.resize(imgGray, (256,256), interpolation = cv.INTER_AREA)
print("Shape of the resized image: {}".format(imgGrayResized.shape))
#showImage(imgGrayResized,'messiGrayResized')

template = imgGray[80:132,210:265]
templateResized = cv.resize(template, (32,32), interpolation = cv.INTER_AREA)
showImage(templateResized,'template')
print("Shape of the template: {}".format(templateResized.shape))


def search(inputWindow, refWindow):
    '''
    it returns the best point from correlation coefficient matrix
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
            corrCoeff[i,j] = getNCC(refWindow[:refWindow.shape[0]-1,:refWindow.shape[1]-1], result[i:i+2*h+1, j:j+2*h+1])
    
    y, x = np.unravel_index(np.argmax(corrCoeff), corrCoeff.shape)  # find the match

    return y, x, corrCoeff

# now pyramid search 
level=3
bestPoints = {'x':[],
              'y':[]}
coeffMat = []
refWindow = gaussPyrT[level-1]
inputWindow = gaussPyr[level-1]

ybest, xbest, coeff = search( inputWindow, refWindow,)
bestPoints['x'].append(xbest)
bestPoints['y'].append(ybest)
coeffMat.append(coeff)

for i in reversed(range(2)):
    # obtain the refWindow template at the highest level 
    refWindow = gaussPyrT[i]
    m = refWindow.shape[0]
    #inputWindow = gaussPyr[i][xbest-m:xbest+m+1, ybest-m:ybest+m+1]
    
    #print(inputWindow.shape)
    inputWindow = gaussPyr[i]
    ybest, xbest, coeff = search(inputWindow, refWindow)
    bestPoints['x'].append(xbest)
    bestPoints['y'].append(ybest)
    coeffMat.append(coeff)
    
for i in range(level):
    pyrImage = gaussPyr[i]
    h = (gaussPyrT[i].shape[0]-1)//2
    y = bestPoints['x'][level-1-i]
    x = bestPoints['y'][level-1-i]
    pyrImage[y-h:y+h+1, x-h:x+h+1] = 0
    print("Correlated image size{}".format(coeffMat[level-1-i].shape))
    fig, (ax_orig, ax_template, ax_corr) = plt.subplots(3, 1, figsize=(6, 15))
    ax_orig.imshow(pyrImage, cmap='gray')
    ax_orig.set_title('Original')
    ax_orig.set_axis_off()
    ax_template.imshow(gaussPyrT[i], cmap='gray')
    ax_template.set_title('Template')
    ax_template.set_axis_off()
    ax_corr.imshow(coeffMat[level-1-i], cmap='gray')
    ax_corr.set_title('Cross-correlation')
    ax_corr.set_axis_off()
    ax_orig.plot(x, y, 'ro')
    fig.show()
    
