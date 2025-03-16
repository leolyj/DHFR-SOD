import torch
import torch.nn.functional as F
# from torch.autograd import Variable
import numpy as np
from PIL import Image

def laplacian_edge(img):
    laplacian_filter = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filter = torch.reshape(laplacian_filter, [1, 1, 3, 3])
    filter = filter.cuda()
    lap_edge = F.conv2d(img, filter, stride=1, padding=1)
    return lap_edge

def gradient_x(img):
    sobel = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter = torch.reshape(sobel,[1,1,3,3])
    filter = filter.cuda()
    gx = F.conv2d(img, filter, stride=1, padding=1)
    return gx


def gradient_y(img):
    sobel = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filter = torch.reshape(sobel, [1, 1,3,3])
    filter = filter.cuda()
    gy = F.conv2d(img, filter, stride=1, padding=1)
    return gy

def charbonnier_penalty(s):
    cp_s = torch.pow(torch.pow(s, 2) + 0.001**2, 0.5)
    return cp_s

def get_saliency_smoothness(pred, gt, size_average=True):
    gt_nonorm = gt * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    gt_nonorm = gt_nonorm + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    gt_gray = (gt_nonorm[:, 0]*0.299 + gt_nonorm[:, 1]*0.587 + gt_nonorm[:, 2]*0.114)
    #wr_img=Image.fromarray((gt_gray*255).detach().cpu()[0].numpy().astype('uint8'))
    #wr_img.save('gray_write.png')
    gt_gray = gt_gray.unsqueeze(dim=1)
    alpha = 10
    s1 = 10
    s2 = 1
    ## first oder derivative: sobel
    sal_x = torch.abs(gradient_x(pred))
    sal_y = torch.abs(gradient_y(pred))
    gt_x = gradient_x(gt_gray)
    gt_y = gradient_y(gt_gray)
    w_x = torch.exp(torch.abs(gt_x) * (-alpha))
    w_y = torch.exp(torch.abs(gt_y) * (-alpha))
    cps_x = charbonnier_penalty(sal_x * w_x)
    cps_y = charbonnier_penalty(sal_y * w_y)
    cps_xy = cps_x + cps_y

    ## second order derivative: laplacian
    lap_sal = torch.abs(laplacian_edge(pred))
    lap_gt = torch.abs(laplacian_edge(gt_gray))
    weight_lap = torch.exp(lap_gt * (-alpha))
    weighted_lap = charbonnier_penalty(lap_sal*weight_lap)

    smooth_loss = s1*torch.mean(cps_xy) + s2*torch.mean(weighted_lap)

    return smooth_loss

class smoothness_loss(torch.nn.Module):
    def __init__(self, size_average = True):
        super(smoothness_loss, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return get_saliency_smoothness(pred, target, self.size_average)
