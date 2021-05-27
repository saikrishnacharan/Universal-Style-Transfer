import os
import shutil

import torch
import argparse
from PIL import Image
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
from Loader import Dataset,im_patches
from util import *
import scipy.misc
from torch.utils.serialization import load_lua
# from torchfile import load as load_lua
import warnings
import time

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='WCT Pytorch')
parser.add_argument('--contentPath',default='images/content',help='path to train')
parser.add_argument('--stylePath',default='images/style',help='path to train')
parser.add_argument('--workers', default=2, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--vgg1', default='models/vgg_normalised_conv1_1.t7', help='Path to the VGG conv1_1')
parser.add_argument('--vgg2', default='models/vgg_normalised_conv2_1.t7', help='Path to the VGG conv2_1')
parser.add_argument('--vgg3', default='models/vgg_normalised_conv3_1.t7', help='Path to the VGG conv3_1')
parser.add_argument('--vgg4', default='models/vgg_normalised_conv4_1.t7', help='Path to the VGG conv4_1')
parser.add_argument('--vgg5', default='models/vgg_normalised_conv5_1.t7', help='Path to the VGG conv5_1')
parser.add_argument('--decoder5', default='models/feature_invertor_conv5_1.t7', help='Path to the decoder5')
parser.add_argument('--decoder4', default='models/feature_invertor_conv4_1.t7', help='Path to the decoder4')
parser.add_argument('--decoder3', default='models/feature_invertor_conv3_1.t7', help='Path to the decoder3')
parser.add_argument('--decoder2', default='models/feature_invertor_conv2_1.t7', help='Path to the decoder2')
parser.add_argument('--decoder1', default='models/feature_invertor_conv1_1.t7', help='Path to the decoder1')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--fineSize', type=int, default=512, help='resize image to fineSize x fineSize,leave it to 0 if not resize')
parser.add_argument('--outf', default='samples/', help='folder to output images')
parser.add_argument('--alpha', type=float,default=1, help='hyperparameter to blend wct feature and content feature')
parser.add_argument('--gpu', type=int, default=0, help="which gpu to run on.  default is 0")
parser.add_argument('--level', type = int, default = -1, help="Multi = -1, else for specific level use level = {1,2,3,4,5}")
parser.add_argument('--do_patches', type = bool, default = False)
parser.add_argument('--kernel_size', type = int, default = 512)
parser.add_argument('--stride', type = int, default = 512)

args = parser.parse_args()

try:
    os.makedirs(args.outf)
except OSError:
    pass

# Data loading code
dataset = Dataset(args.contentPath,args.stylePath,args.fineSize, args.do_patches, args.kernel_size)
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=1,
                                     shuffle=False)

wct = WCT(args)
def styleTransfer(contentImg,styleImg,imname,csF, level = -1, save_path = args.outf):
	# print("level= ",level)
    if(level == -1):
    	# print("Multi Level Style Transfer")
        sF5 = wct.e5(styleImg)
        cF5 = wct.e5(contentImg)
        sF5 = sF5.data.cpu().squeeze(0)
        # if(not args.do_patches):
        cF5 = cF5.data.cpu().squeeze(0)
        csF5 = wct.transform(cF5,sF5,csF,args.alpha)
        Im5 = wct.d5(csF5)

        sF4 = wct.e4(styleImg)
        cF4 = wct.e4(Im5)
        sF4 = sF4.data.cpu().squeeze(0)
        cF4 = cF4.data.cpu().squeeze(0)
        csF4 = wct.transform(cF4,sF4,csF,args.alpha)
        Im4 = wct.d4(csF4)

        sF3 = wct.e3(styleImg)
        cF3 = wct.e3(Im4)
        sF3 = sF3.data.cpu().squeeze(0)
        cF3 = cF3.data.cpu().squeeze(0)
        csF3 = wct.transform(cF3,sF3,csF,args.alpha)
        Im3 = wct.d3(csF3)

        sF2 = wct.e2(styleImg)
        cF2 = wct.e2(Im3)
        sF2 = sF2.data.cpu().squeeze(0)
        cF2 = cF2.data.cpu().squeeze(0)
        csF2 = wct.transform(cF2,sF2,csF,args.alpha)
        Im2 = wct.d2(csF2)

        sF1 = wct.e1(styleImg)
        cF1 = wct.e1(Im2)
        sF1 = sF1.data.cpu().squeeze(0)
        cF1 = cF1.data.cpu().squeeze(0)
        csF1 = wct.transform(cF1,sF1,csF,args.alpha)
        Im1 = wct.d1(csF1)

        if(not args.do_patches):
            vutils.save_image(Im5.data.cpu().float(),os.path.join(save_path,"Level5_"+imname))
            vutils.save_image(Im4.data.cpu().float(),os.path.join(save_path,"Level4_"+imname))
            vutils.save_image(Im3.data.cpu().float(),os.path.join(save_path,"Level3_"+imname))
            vutils.save_image(Im2.data.cpu().float(),os.path.join(save_path,"Level2_"+imname))
            vutils.save_image(Im1.data.cpu().float(),os.path.join(save_path,"Level1_"+imname))
            print("MultiLevel")
        # save_image has this wired design to pad images with 4 pixels at default.
    elif (level == 1):
    	# print(level)
        sF1 = wct.e1(styleImg)
        cF1 = wct.e1(contentImg)
        # print(cF1.shape)
        sF1 = sF1.data.cpu().squeeze(0)
        # if(not args.do_patches):
        cF1 = cF1.data.cpu().squeeze(0)
        csF1 = wct.transform(cF1,sF1,csF,args.alpha)
        Im1 = wct.d1(csF1)
        # print(level)
    elif (level == 2):
    	# print("Computation in Level 2")
        sF2 = wct.e2(styleImg)
        cF2 = wct.e2(contentImg)
        sF2 = sF2.data.cpu().squeeze(0)
        cF2 = cF2.data.cpu().squeeze(0)
        csF2 = wct.transform(cF2,sF2,csF,args.alpha)
        Im1 = wct.d2(csF2)
    elif (level == 3):
    	# print("Computation in Level 3")
    	sF3 = wct.e3(styleImg)
    	cF3 = wct.e3(contentImg)
    	sF3 = sF3.data.cpu().squeeze(0)
    	cF3 = cF3.data.cpu().squeeze(0)
    	csF3 = wct.transform(cF3,sF3,csF,args.alpha)
    	Im1 = wct.d3(csF3)
    elif (level == 4):    	
    	# print("Computation in Level 4")
        sF4 = wct.e4(styleImg)
        cF4 = wct.e4(contentImg)
        sF4 = sF4.data.cpu().squeeze(0)
        cF4 = cF4.data.cpu().squeeze(0)
        csF4 = wct.transform(cF4,sF4,csF,args.alpha)
        Im1 = wct.d4(csF4)    
        # whiten_Decode = wct.d4(whiten_cF)
        # vutils.save_image(Im1.data.cpu().float(),os.path.join(args.outf,"Level4_whiten_Decoder"imname))    

    elif (level == 5):
    	# print("Computation in Level 5")
        sF5 = wct.e5(styleImg)
        cF5 = wct.e5(contentImg)
        sF5 = sF5.data.cpu().squeeze(0)
        cF5 = cF5.data.cpu().squeeze(0)
        csF5 = wct.transform(cF5,sF5,csF,args.alpha)
        Im1 = wct.d5(csF5)

    vutils.save_image(Im1.data.cpu().float(),os.path.join(save_path,imname))
    # return Im1

avgTime = 0
cImg = torch.Tensor()
sImg = torch.Tensor()
csF = torch.Tensor()
csF = Variable(csF)
if(args.cuda):
    cImg = cImg.cuda(args.gpu)
    sImg = sImg.cuda(args.gpu)
    csF = csF.cuda(args.gpu)
    wct.cuda(args.gpu)
for i,(contentImg,styleImg,imname) in enumerate(loader):
    imname = imname[0]
    if(args.do_patches):
        image_patches = im_patches(kernel_size = args.kernel_size, stride = args.stride)
        contentImg_patches = image_patches.im_to_patches(contentImg)
        # print(contentImg_patches.shape)
        os.makedirs('./patches_temp',exist_ok=True)
        os.makedirs('./patches_out_temp',exist_ok=True)
        os.makedirs('./style_temp',exist_ok=True)
        start_time = time.time()

        for index in range(len(contentImg_patches)):
            vutils.save_image(styleImg.cpu().float(),os.path.join('./style_temp',str(index).zfill(2)+'.png'))

        for idx,im in enumerate(contentImg_patches):
            # print(im.dtype)
            vutils.save_image(im.cpu().float(),os.path.join('./patches_temp',str(idx).zfill(2)+'.png'))

        dataset_patches = Dataset('./patches_temp','./style_temp',args.fineSize, False, args.kernel_size)
        loader_patches = torch.utils.data.DataLoader(dataset=dataset_patches,
                                            batch_size=1,
                                            shuffle=False)
        for j,(contentImg_small, styleImg_small, imname_small) in enumerate(loader_patches):
            imname_small = imname_small[0]
            print('Processing Patch num',j+1,'of',len(contentImg_patches))
            if (args.cuda):
                contentImg_small = contentImg_small.cuda(args.gpu)
                styleImg_small = styleImg_small.cuda(args.gpu)
            cImg = Variable(contentImg_small,volatile=True)
            sImg = Variable(styleImg_small,volatile=True)

            styleTransfer(cImg,sImg,imname_small,csF,-1,'./patches_out_temp')

        dataset_patches_out = Dataset('./patches_out_temp','./style_temp',args.fineSize, False, args.kernel_size)
        loader_patches_out = torch.utils.data.DataLoader(dataset=dataset_patches_out,
                                            batch_size=len(contentImg_patches),
                                            shuffle=False)

        for k,(contentImg_batch, styleImg_small, imname_small) in enumerate(loader_patches_out):
            # print(contentImg_batch.shape)
            im_out = image_patches.patches_to_im(contentImg_batch)
            vutils.save_image(im_out.cpu().float(),os.path.join(args.outf,'patches_out_'+ imname))
        
        shutil.rmtree('./patches_temp')
        shutil.rmtree('./style_temp')
        shutil.rmtree('./patches_out_temp')
        end_time = time.time()

        print('Elapsed time for the current image is: %f' % (end_time - start_time))
        avgTime += (end_time - start_time)

    else:
        print("Here level is =",args.level)
        print('Computing the Image [PROCESSING]:  ' + imname)
        if (args.cuda):
            contentImg = contentImg.cuda(args.gpu)
            styleImg = styleImg.cuda(args.gpu)
        cImg = Variable(contentImg,volatile=True)
        sImg = Variable(styleImg,volatile=True)
        start_time = time.time()
        # WCT Style Transfer

        styleTransfer(cImg,sImg,imname,csF,args.level)
        # if(args.do_patches):
        #     Im1 = image_patches.patches_to_im(Im1)
        # vutils.save_image(Im1.data.cpu().float(),os.path.join(args.outf,imname))

        end_time = time.time()
        print('Elapsed time for the current image is: %f' % (end_time - start_time))
        avgTime += (end_time - start_time)

    print('Done!!! ->Processed %d images. Averaged time is %f' % ((i+1),avgTime/(i+1)))
