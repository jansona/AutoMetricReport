import argparse, os, glob, time

import cv2
import numpy as np
from scipy.signal import convolve2d
import torch
from scipy.misc import imread, imsave

from canny_metric import CannyMetric
from bhatta_metric import measurement
from iou_metric import cal_ious


def check_shape(realB_name):
    
    file_name = "_".join(realB_name.split("/")[-1].split("_")[:first_num])
    fakeB_name = "{}/{}{}".format(root_path, file_name, fakeB_suffix)
    
    im_A = cv2.imread(realB_name, cv2.IMREAD_COLOR)
    im_B = cv2.imread(fakeB_name, cv2.IMREAD_COLOR)
    
    if im_A.shape != im_B.shape:
        print("{}: {}".format(realB_name, im_A.shape))
        print("{}: {}".format(fakeB_name, im_B.shape))
    
    return im_A.shape == im_B.shape

def MSE(pic1, pic2):
    return np.sum(np.square(pic1 - pic2)) / (pic1.shape[0] * pic1.shape[1])

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def img2tensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    img = torch.stack([img]).float()
    img = torch.tensor(img)

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):

    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))


parser = argparse.ArgumentParser()
parser.add_argument("--root_path", help="the path of the result file", default=".")
parser.add_argument("--image_type", help="the type of the image", default=".jpg")
parser.add_argument("--suffix_name_fake", default="_A_fake_B.png")
parser.add_argument("--suffix_name_real", default="_A_real_B.png")
parser.add_argument("--first_num", default='1')

if '__main__' == __name__:
    
    # 脚本所需的初始化工作
    args = parser.parse_args()
    root_path = args.root_path

    first_num = int(args.first_num)

    fakeB_suffix = args.suffix_name_fake
    realB_suffix = args.suffix_name_real

    realB_files = glob.glob("{}/*{}".format(root_path, realB_suffix))
       
    fout = open(root_path + "/report.txt", "a")
    fout.write("report time: " + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + "\n")
    
    # 各指标初始化工作
    bhatta_metric_gray = 0
    
    MSE_result = np.array([0., 0., 0.])
    PSNR_result = np.array([0., 0., 0.])
    L1Less16cnt= 0
    L1Less16sum= 0
    MSE_GRAY = 0
    
    ssim_result1 = np.array([0., 0., 0.])
    ssim_gray = np.array([0.])
    
    cms_gray = 0
    cms_rgb = 0
    cm = CannyMetric()
    
    road_iou = 0
    water_iou = 0
    
    road_len_sub = 0
    water_len_sub = 0
    
    for cnt, realB_file in enumerate(realB_files):
        
        if cnt % 20 == 0: print("No.", cnt)
        
        file_name = "_".join(realB_file.split("/")[-1].split(".")[0].split("_")[:first_num])
        fakeB_file = "{}/{}{}".format(root_path, file_name, fakeB_suffix)
        
        # bhatta metric
        if not os.path.exists(fakeB_file):
            print(fakeB_file, "not exists")
            continue
        
        im_A = cv2.imread(realB_file, cv2.IMREAD_COLOR)
        im_B = cv2.imread(fakeB_file, cv2.IMREAD_COLOR)
        try:
            gim_A = cv2.cvtColor(im_A,cv2.COLOR_BGR2GRAY).astype("int32")
            gim_B = cv2.cvtColor(im_B,cv2.COLOR_BGR2GRAY).astype("int32")
        except Exception as e:
            print("realB_file: ", realB_file)
            print("fakeB_file: ", fakeB_file)
            raise e
        im_A = im_A.astype("int32")
        im_B = im_B.astype("int32")
        
        bhatta_metric_gray += measurement(realB_file, fakeB_file)
        
        # MSE L1Less16
        im_A = cv2.imread(realB_file, cv2.IMREAD_COLOR)
        im_B = cv2.imread(fakeB_file, cv2.IMREAD_COLOR)
        try:
            gim_A = cv2.cvtColor(im_A,cv2.COLOR_BGR2GRAY).astype("int32")
            gim_B = cv2.cvtColor(im_B,cv2.COLOR_BGR2GRAY).astype("int32")
        except Exception as e:
            print("realB_file: ", realB_file)
            print("fakeB_file: ", fakeB_file)
            raise e
        im_A = im_A.astype("int32")
        im_B = im_B.astype("int32")

        for _ in range(3):
            MSE_ = MSE(im_A[:, :, _], im_B[:, :, _])
            MSE_result[_] += MSE_
            PSNR_result[_]+= 10. * np.log( 49. / MSE_ ) / np.log(10.)
            L1Less16cnt   += (np.abs(im_A[:, :, _] - im_B[:, :, _]) < 16).astype('int').sum()
            L1Less16sum   += im_A.shape[0] * im_A.shape[1]

        temp_MSE = MSE(gim_A, gim_B)
            
        MSE_GRAY += temp_MSE
        
        # SSIM
        im_A = cv2.imread(realB_file, cv2.IMREAD_COLOR)
        im_B = cv2.imread(fakeB_file, cv2.IMREAD_COLOR)
        gim_A = cv2.cvtColor(im_A,cv2.COLOR_BGR2GRAY).astype("int32")
        gim_B = cv2.cvtColor(im_B,cv2.COLOR_BGR2GRAY).astype("int32")
        im_A = im_A.astype("int32")
        im_B = im_B.astype("int32")

        for _ in range(3):
            ssim_result1[_] += 1.*compute_ssim(im_A[:, :, _], im_B[:, :, _])

        temp_ssim_gray = 1.*compute_ssim(gim_A, gim_B)
            
        ssim_gray += temp_ssim_gray
        
        # ESSI        
        im_A = cv2.imread(realB_file)
        im_B = cv2.imread(fakeB_file)
        
        im_A = torch.from_numpy(im_A.transpose((2, 0, 1)))
        im_A = torch.stack([im_A]).float()
        im_A = im_A.clone().detach()
        
        im_B = torch.from_numpy(im_B.transpose((2, 0, 1)))
        im_B = torch.stack([im_B]).float()
        im_B = im_B.clone().detach()
        
        for c in range(3):
            cms_rgb += cm(im_A[:, c:c+1, :, :], im_B[:, c:c+1, :, :], single_channel=True)
        
        temp_cms_gray = cm(im_A, im_B, single_channel=False)
        
        cms_gray += cm(im_A, im_B, single_channel=False)
        
        # IOU        
        ious = cal_ious(realB_file, fakeB_file)
        
        if np.isnan(ious['road']):
            road_len_sub += 1
        else:
            road_iou += ious['road']
        
        if np.isnan(ious['water']):
            water_len_sub += 1
        else:
            water_iou += ious['water']
        
    # print all the results
    fout.write("Bhatta Metric: " + str(bhatta_metric_gray / len(realB_files)) + "\n")
    
    fout.write("LESS16PERCENT: " + str(L1Less16cnt / L1Less16sum) + "\n")
    fout.write("MSE: " + str(sum(MSE_result / len(realB_files)) / len(MSE_result)) + "\n")
    fout.write("PSNR: " + str(sum(PSNR_result / len(realB_files)) / len(PSNR_result)) + "\n")
    fout.write("MSE GRAY: " + str(MSE_GRAY / len(realB_files)) + "\n")
    
    fout.write("SSIM: " + str(sum(ssim_result1 / len(realB_files)) / len(ssim_result1)) + "\n")
    fout.write("SSIM GRAY: " + str(ssim_gray / len(realB_files)) + "\n")
    fout.write("CANNY METRIC RGBMEAN: " + str(cms_rgb / len(realB_files)) + "\n")
    fout.write("CANNY METRIC GRAY: " + str(cms_gray / len(realB_files)) + "\n")
    
    fout.write("ROAD IOU: " + str(road_iou / (len(realB_files) - road_len_sub)) + "\n")
    fout.write("WATER IOU: " + str(water_iou / (len(realB_files) - water_len_sub)) + "\n")