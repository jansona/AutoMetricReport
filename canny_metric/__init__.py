import torch
import torch.nn as nn
import numpy as np
from scipy.signal import gaussian
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

import cv2
import traceback


def set_nan_to_nonzero(t):
#     return t
    return torch.where(t==0, torch.zeros_like(t), t)

def set_nan_to_zero(t):
#     return t
    return torch.where(t==0, torch.zeros_like(t), t)

def count_nan(t):
    return torch.sum(t != t)

def count_zero(t):
    return torch.sum(t == 0.0)

class CannyNet(nn.Module):
    def __init__(self, use_cuda=False):
        super(CannyNet, self).__init__()

#         self.threshold = threshold
        self.threshold = None
        self.use_cuda = use_cuda

        filter_size = 5
        generated_filters = gaussian(filter_size,std=1.0).reshape([1,filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 0, 1, -1],
                                [ 0, 0, 0]])

        filter_45 = np.array([  [0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, -1]])

        filter_90 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]])

        filter_135 = np.array([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]])

        filter_180 = np.array([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]])

        filter_225 = np.array([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_270 = np.array([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_315 = np.array([ [ 0, 0, -1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

    def forward(self, img, threshold):
        self.threshold = threshold
        
        img_r = img[:,0:1]
        img_g = img[:,1:2]
        img_b = img[:,2:3]

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        blurred_img = torch.stack([blurred_img_r,blurred_img_g,blurred_img_b],dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # COMPUTE THICK EDGES

        grad_mag = torch.pow(grad_x_r**2 + grad_y_r**2, 0.5)    # 惊奇发现：torch.sqrt是inplace operation，torch.pow不是
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/3.14159))
        grad_orientation += 180.0
        grad_orientation =  torch.round( grad_orientation / 45.0 ) * 45.0

        # THIN EDGES (NON-MAX SUPPRESSION)

        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]
        pixel_count = height * width
        pixel_range = torch.FloatTensor([range(pixel_count)])
        if self.use_cuda:
            pixel_range = torch.cuda.FloatTensor([range(pixel_count)])

        indices = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(1,height,width)

        indices = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(1,height,width)

        channel_select_filtered = torch.stack([channel_select_filtered_positive,channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        is_max = torch.unsqueeze(is_max, dim=0)

        thin_edges = grad_mag.clone()
        thin_edges[is_max==0] = 0.0

        # THRESHOLD

        thresholded = thin_edges.clone()
        thresholded[thin_edges<self.threshold] = 0.0

        early_threshold = grad_mag.clone()
        early_threshold[grad_mag<self.threshold] = 0.0

        assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() == early_threshold.size()

        return blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold


def my_gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = my_gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class CannyMetric(object):

    def __init__(self, use_cuda=False):

        self.net = CannyNet(use_cuda=use_cuda)
        self.use_cuda = use_cuda

        if use_cuda:
            self.net.cuda()
        self.net.eval()    

    def img2data(self, raw_img):
        img = torch.from_numpy(raw_img.transpose((2, 0, 1)))    
        batch = torch.stack([img]).float()

        data = Variable(batch)
        if self.use_cuda:
            data = Variable(batch).cuda()

        return data
    
    def get_threshold(self, image, sigma=0.33):
        
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        
        return lower
    
    def test_call(self, img0, img1):
                
        # 自适应阈值计算
        a0 = img0.cpu().detach().numpy()[0].transpose(1, 2, 0)
        a1 = img1.cpu().detach().numpy()[0].transpose(1, 2, 0)
        
        gray0 = cv2.cvtColor(a0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(a1, cv2.COLOR_BGR2GRAY)
        
        th0 = int(self.get_threshold(gray0))
        th1 = int(self.get_threshold(gray1))
        
#         print(th0, th1)

        # 利用canny求边缘
        data0 = img0
        data1 = img1

        blurred_img0, grad_mag0, grad_orientation0, thin_edges0, thresholded0, early_threshold0 = self.net(data0, th0)
        blurred_img1, grad_mag1, grad_orientation1, thin_edges1, thresholded1, early_threshold1 = self.net(data1, th1)
        
        final0 = (thresholded0.data[0, 0] > 0.0).float()
        final1 = (thresholded1.data[0, 0] > 0.0).float()     

        # 求基于canny的loss
        (w, h) = final0.size()
        N = w * h
        canny0_plain = final0.view(N)
        canny1_plain = final1.view(N)
        
        D0 = torch.var(canny0_plain)
        D1 = torch.var(canny1_plain)
        
        cov_N = torch.mean(canny0_plain * canny1_plain) - torch.mean(canny0_plain) * torch.mean(canny1_plain)
        cov = cov_N * (N/(N-1))
        
        loss = 1 - cov / (D0 * D1)**0.5
        
        return loss

    def __call__(self, img0, img1, window_size=11, single_channel=False):
        
        # 自适应阈值计算
        if not single_channel:
            a0 = img0.cpu().detach().numpy()[0].transpose(1, 2, 0)
            a1 = img1.cpu().detach().numpy()[0].transpose(1, 2, 0)
        
            gray0 = cv2.cvtColor(a0, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(a1, cv2.COLOR_BGR2GRAY)
        else:
            img0 = img0.repeat(1, 3, 1, 1)
            img1 = img1.repeat(1, 3, 1, 1)
            
            gray0 = img0
            gray1 = img1
        
        th0 = int(self.get_threshold(gray0))
        th1 = int(self.get_threshold(gray1))
        
#         print(th0, th1)

        # 利用canny求边缘
        data0 = img0
        data1 = img1

        blurred_img0, grad_mag0, grad_orientation0, thin_edges0, thresholded0, early_threshold0 = self.net(data0, th0)
        blurred_img1, grad_mag1, grad_orientation1, thin_edges1, thresholded1, early_threshold1 = self.net(data1, th1)
        
        final0 = (thresholded0.data[0, 0] > 0.0).float()
        final1 = (thresholded1.data[0, 0] > 0.0).float()     
        
#         from PIL import Image
#         import numpy as np
#         img0 = Image.fromarray(np.array(final0)*255)
#         img1 = Image.fromarray(np.array(final1)*255)
#         import matplotlib.pyplot as plt
#         plt.imshow(img0)
#         plt.show()
#         plt.imshow(img1)
#         plt.show()

        # 求基于canny的loss
        (w, h) = final0.size()
        canny0 = final0.expand(1, 1, w, h)
        canny1 = final1.expand(1, 1, w, h)
        
        (_, channel, _, _) = canny0.size()
        window = create_window(window_size, channel)
        
        if canny0.is_cuda:
            window = window.cuda(canny0.get_device())
        window = window.type_as(canny0)

        mu0 = F.conv2d(canny0, window, padding = window_size//2, groups = channel)
        mu1 = F.conv2d(canny1, window, padding = window_size//2, groups = channel)
        
        mu0_mean = torch.mean(mu0)
        mu1_mean = torch.mean(mu1)

        mu0_sq = mu0.pow(2)
        mu1_sq = mu1.pow(2)
        mu0_mu1 = mu0 * mu1

        sigma0_sq = F.conv2d(canny0*canny0, window, padding = window_size//2, groups = channel) - mu0_sq
        sigma1_sq = F.conv2d(canny1*canny1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma01 = F.conv2d(canny0*canny1, window, padding = window_size//2, groups = channel) - mu0_mu1
        
        sigma01 = set_nan_to_zero(sigma01)
        sigma0_sq_sigma1_sq = set_nan_to_nonzero(sigma0_sq * sigma1_sq)
        
        c1 = 1e-12
        c2 = 1e-12

        rho = (sigma01.mean() + c1) / (sigma0_sq_sigma1_sq.pow(0.5).mean() + c1)
        
        rho = ((2 * mu0_mean * mu1_mean + c2) / (mu0_mean ** 2 + mu1_mean ** 2 + c2)) * set_nan_to_zero(rho)
        
        if single_channel:
            rho = rho / 3

        return rho
