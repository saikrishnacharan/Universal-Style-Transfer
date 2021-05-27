from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
from os import listdir
from os.path import join
import numpy as np
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class im_patches(object):
    '''
    Convert Image tensor of shape NxCxHxW
    to Patch tensor of shape (N*p_num)xCx(K_x)x(K_y) where p_num is number of patches, (K_x,K_y) is the shape of every patch
    
    Disclaimer: Select kernel_size and stride such that there is no loss of data after patch creation
    '''
    def __init__(self,kernel_size = 256, stride = 256):
        
        if(type(kernel_size) == int):
            self.kernel_size_x = kernel_size
            self.kernel_size_y = kernel_size
        elif(type(kernel_size) == tuple and len(kernel_size) == 2):
            self.kernel_size_x = kernel_size[0]
            self.kernel_size_y = kernel_size[1]
        
        if(type(stride) == int):
            self.stride_x = kernel_size
            self.stride_y = kernel_size
        elif(type(stride) == tuple and len(stride) == 2):
            self.stride_x = kernel_size[0]
            self.stride_y = kernel_size[1]
            
    def im_to_patches(self,im_tensor): # Expects a 4D Batch Tensor NxCxHxW
        
        self.N = im_tensor.shape[0] # Batch Size
        self.c = im_tensor.shape[1] # number of channels in images tensor
        self.im_h, self.im_w = im_tensor.shape[2:] # Image Height and Width
        
        im_tensor = im_tensor.unfold( 2, self.kernel_size_x, self.stride_x).unfold( 3, self.kernel_size_y, self.stride_y)
        self.num_patches_x, self.num_patches_y = im_tensor.shape[2:4]
        patches = im_tensor.permute(0,2,3,1,4,5).reshape( -1, self.c, self.kernel_size_x, self.kernel_size_y)

        return patches
    
    def patches_to_im(self,patch_tensor):
        
        im = patch_tensor.contiguous().view( self.N, self.num_patches_x, self.num_patches_y, self.c, self.kernel_size_x, self.kernel_size_y)
        im = im.permute(0, 3, 1, 4, 2, 5)
        im = im.contiguous().view(self.N, self.c, self.im_h, self.im_w)
        
        return im

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')

class Dataset(data.Dataset):
    def __init__(self,contentPath,stylePath,fineSize,do_patches = False, patch_kernel_size = None):
        super(Dataset,self).__init__()
        self.contentPath = contentPath
        self.image_list = [x for x in sorted(listdir(contentPath)) if is_image_file(x)]
        self.stylePath = stylePath
        self.fineSize = fineSize
        self.do_patches = do_patches
        self.kernel_size = patch_kernel_size
        # self.stride = stride
        if(self.do_patches and self.kernel_size != None):
            # self.image_patches = im_patches()
            self.fineSize = self.kernel_size
        #self.normalize = transforms.Normalize(mean=[103.939,116.779,123.68],std=[1, 1, 1])
        #normalize = transforms.Normalize(mean=[123.68,103.939,116.779],std=[1, 1, 1])
        self.prep = transforms.Compose([
                    transforms.Resize(fineSize),
                    transforms.ToTensor(),
                    #transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                    ])

    def __getitem__(self,index):
        contentImgPath = os.path.join(self.contentPath,self.image_list[index])
        styleImgPath = os.path.join(self.stylePath,self.image_list[index])
        contentImg = default_loader(contentImgPath)
        styleImg = default_loader(styleImgPath)

        # resize
        if(self.fineSize != 0 and not self.do_patches):
            w,h = contentImg.size
            if(w > h):
                if(w != self.fineSize):
                    neww = self.fineSize
                    newh = int(h*neww/w)
                    contentImg = contentImg.resize((neww,newh))
                    styleImg = styleImg.resize((neww,newh))
            else:
                if(h != self.fineSize):
                    newh = self.fineSize
                    neww = int(w*newh/h)
                    contentImg = contentImg.resize((neww,newh))
                    styleImg = styleImg.resize((neww,newh))
        elif(self.fineSize != 0  and self.do_patches):
            w,h = styleImg.size
            if(w > h):
                if(w != self.fineSize):
                    neww = self.fineSize
                    newh = int(h*neww/w)
                    # contentImg = contentImg.resize((neww,newh))
                    styleImg = styleImg.resize((neww,newh))
            else:
                if(h != self.fineSize):
                    newh = self.fineSize
                    neww = int(w*newh/h)
                    # contentImg = contentImg.resize((neww,newh))
                    styleImg = styleImg.resize((neww,newh))

        # Preprocess Images
        contentImg = transforms.ToTensor()(contentImg)
        styleImg = transforms.ToTensor()(styleImg)
        return contentImg.squeeze(0),styleImg.squeeze(0),self.image_list[index]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)
