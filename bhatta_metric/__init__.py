import cv2
import collections
import numpy as np
import time
import os


def gethist(img):
    nopixel=img.shape[0]*img.shape[1]
    img=np.ravel(img)
    count=collections.Counter(img)
    list1=list(zip(count.keys(),count.values()))
    hist=[]
    for a in list1:
        b=[a[0],(a[1]/nopixel)**0.5]
        hist.append(b)
    #print (hist)
    return hist

def bacoe(img1,img2):
    hist1=gethist(img1)
    hist2=gethist(img2)
    bc=0
    for pixel1 in hist1:
        for pixel2 in hist2:
            if pixel1[0]==pixel2[0]:
                bc=bc+pixel1[1]*pixel2[1]
    return bc

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                              (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    if not img1.shape == img2.shape:
      raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def measurement(path1,path2):
    t0=time.clock()
    img1=cv2.imread(path1,0)
    img2 = cv2.imread(path2, 0)
    ssim=calculate_ssim(img1,img2)
    bc=bacoe(img1,img2)
    #print (ssim)
    #print(bc)
    measure=0.95*ssim+0.05*bc
#     print (measure)
    t1 = time.clock() - t0
    print ('measuring runtime is '+str(t1)+'s')
    
    return measure


if __name__=="__main__":
    path1 = r'E:\DATA\somewhere_17full\selecttestsample\gt\38662_49263.png'
    path2 = r'E:\DATA\somewhere_17full\selecttestsample\result\38662_49263.png'
    measurement(path1,path2)