from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import matplotlib.pyplot as plt
from util import count_parameters as count
from util import convert2cpu as cpu
from PIL import Image, ImageDraw


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''

    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image

    return canvas



def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    '''
    image = cv2.imread(img)
    image_gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # filter out small lines between counties
    kernel = np.ones((5,5),np.float32)/25
    image_gray= cv2.filter2D(image_gray,-1,kernel)

    # threshold the imageDSC_6194.JPGDSC_6194.JPG and extract contours
    ret,thresh = cv2.threshold(image_gray,50,80,cv2.THRESH_BINARY_INV)
    im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    # find the main island (biggest area)
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)

    #create a  bounding rectangle
    x,y,w,h = cv2.boundingRect(cnt)
    #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    orig_im = image[y-20:y+h, x:x+w]
    '''
    orig_im = cv2.imread(img)
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def prep_image_pil(img, network_dim):
    orig_im = Image.open(img)
    img = orig_im.convert('RGB')
    dim = img.size
    img = img.resize(network_dim)
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    img = img.view(*network_dim, 3).transpose(0,1).transpose(0,2).contiguous()
    img = img.view(1, 3,*network_dim)
    img = img.float().div(255.0)
    return (img, orig_im, dim)

def inp_to_image(inp):
    inp = inp.cpu().squeeze()
    inp = inp*255
    try:
        inp = inp.data.numpy()
    except RuntimeError:
        inp = inp.numpy()
    inp = inp.transpose(1,2,0)

    inp = inp[:,:,::-1]
    return inp
