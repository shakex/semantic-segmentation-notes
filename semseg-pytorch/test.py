import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from models import get_model
from data import get_loader, get_data_path
from utils.utils import convert_state_dict
from config import testConfig

from skimage import io


def test():
    args = testConfig()
    print('Testing')
    print('-------------------------')

    # print("Read Input Image from : {}".format(args.img_path))
    # img = io.imread(args.img_path)
    # img = np.expand_dims(img, axis=0)
    # img = img.astype(float) / 255.0
    # img = torch.from_numpy(img).float()

    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, split='val', is_transform=True, augmentations=None)

    n_classes = loader.n_classes

    # single image test - transfrom
    img = io.imread(args.img_path)
    img = np.array(img, dtype=np.uint8)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    img = img.astype(float) / 255.0
    img = torch.from_numpy(img).float()

    # Setup model
    model = get_model(args.arch, n_classes)
    state = convert_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage)['model_state'])
    model.load_state_dict(state)

    # model.eval()
    # if torch.cuda.is_available():
    #     model.cuda(args.gpu_idx[0])
    #
    # for i, (images, labels) in tqdm(enumerate(loader)):
    #     images = Variable(images.cuda(args.gpu_idx[0]), volatile=True)
    #     labels = Variable(labels.cuda(args.gpu_idx[0]), volatile=True)
    #
    #     outputs = model(images)
    #     pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy())
    #     gt = np.squeeze(labels.data.cpu().numpy())
    #
    #     h_offset = int(np.abs(gt.shape[0] - pred.shape[0]) / 2)
    #     w_offset = int(np.abs(gt.shape[1] - pred.shape[1]) / 2)
    #
    #     gt_resize = gt[h_offset:gt.shape[0] - 1 - h_offset, w_offset:gt.shape[1] - 1 - w_offset]
    #
    #     decoded = loader.decode_segmap(pred)
    #     outpath = args.out_path + 'pred_' + loader.files['val'][i] + '.bmp'
    #     io.imsave(outpath, decoded)
    #     print("Segmentation Mask Saved at: {}".format(args.out_path))
    #
    # print("done.")


    # # single image test - model
    if torch.cuda.is_available():
        model.cuda(args.gpu_idx[0])
        images = Variable(img.cuda(args.gpu_idx[0]), volatile=True)
    else:
        images = Variable(img, volatile=True)

    conv1,conv2,conv3,conv4,center,outputs = model(images)

    fm_conv1 = []
    fm_conv2 = []
    fm_conv3 = []
    fm_conv4 = []
    fm_center = []
    fm_pred = []
    for i in range(16):
        fm_conv1.append(np.squeeze(conv1.data[:, i, :, :].cpu().numpy()))
    for i in range(32):
        fm_conv2.append(np.squeeze(conv2.data[:, i, :, :].cpu().numpy()))
    for i in range(64):
        fm_conv3.append(np.squeeze(conv3.data[:, i, :, :].cpu().numpy()))
    for i in range(128):
        fm_conv4.append(np.squeeze(conv4.data[:, i, :, :].cpu().numpy()))
    for i in range(256):
        fm_center.append(np.squeeze(center.data[:, i, :, :].cpu().numpy()))
    for i in range(4):
        fm_pred.append(np.squeeze(outputs.data[:, i, :, :].cpu().numpy()))


    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy())
    pred_conv1 = np.squeeze(conv1.data.max(1)[1].cpu().numpy())
    pred_conv2 = np.squeeze(conv2.data.max(1)[1].cpu().numpy())
    pred_conv3 = np.squeeze(conv3.data.max(1)[1].cpu().numpy())
    pred_conv4 = np.squeeze(conv4.data.max(1)[1].cpu().numpy())
    pred_center = np.squeeze(center.data.max(1)[1].cpu().numpy())
    decoded = loader.decode_segmap(pred)
    print('Classes found: ', np.unique(pred))

    # fig, axes = plt.subplots(2, 3)
    # axes[0, 0].imshow(pred_conv1)
    # axes[0, 0].set_title('conv1')
    # axes[0, 1].imshow(pred_conv2)
    # axes[0, 1].set_title('conv2')
    # axes[0, 2].imshow(pred_conv3)
    # axes[0, 2].set_title('conv3')
    # axes[1, 0].imshow(pred_conv4)
    # axes[1, 0].set_title('conv4')
    # axes[1, 1].imshow(pred_center)
    # axes[1, 1].set_title('center')
    # axes[1, 2].imshow(decoded)
    # axes[1, 2].set_title('final')
    # plt.show()

    from matplotlib.pyplot import cm
    from matplotlib.pyplot import imsave
    io.imsave(args.out_path, decoded)

    for i in range(16):
        imsave('/home/jwliu/disk/kxie/semseg-pytorch/img/s04_t093/conv1_' + str(i+1) + '.png', fm_conv1[i], cmap=cm.jet)
    for i in range(32):
        imsave('/home/jwliu/disk/kxie/semseg-pytorch/img/s04_t093/conv2_' + str(i+1) + '.png', fm_conv2[i], cmap=cm.jet)
    for i in range(64):
        imsave('/home/jwliu/disk/kxie/semseg-pytorch/img/s04_t093/conv3_' + str(i+1) + '.png', fm_conv3[i], cmap=cm.jet)
    for i in range(128):
        imsave('/home/jwliu/disk/kxie/semseg-pytorch/img/s04_t093/conv4_' + str(i+1) + '.png', fm_conv4[i], cmap=cm.jet)
    for i in range(256):
        imsave('/home/jwliu/disk/kxie/semseg-pytorch/img/s04_t093/center_' + str(i+1) + '.png', fm_center[i], cmap=cm.jet)
    for i in range(4):
        imsave('/home/jwliu/disk/kxie/semseg-pytorch/img/s04_t093/pred_' + str(i + 1) + '.png', fm_pred[i], cmap=cm.jet)

    # print("Segmentation Mask Saved at: {}".format(args.out_path))


    # for i, (images, labels) in enumerate(loader):
    #     images = Variable(images, volatile=True)
    #
    #     outputs = model(images)
    #     pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)


test()


"""
# try:
#     import pydensecrf.densecrf as dcrf
# except:
#     print("Failed to import pydensecrf,\
#            CRF post-processing will not work")

args = testConfig()

# Setup image
print("Read Input Image from : {}".format(args.img_path))
img = misc.imread(args.img_path)

data_loader = get_loader(args.dataset)
data_path = get_data_path(args.dataset)
loader = data_loader(data_path, is_transform=True)
n_classes = loader.n_classes

resized_img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]), interp='bicubic')

orig_size = img.shape[:-1]
if args.arch in ['pspnet', 'icnet', 'icnetBN']:
    img = misc.imresize(img, (orig_size[0]//2*2+1, orig_size[1]//2*2+1)) # uint8 with RGB mode, resize width and height which are odd numbers
else:
    img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))

img = img[:, :, ::-1]
img = img.astype(np.float64)
img -= loader.mean
img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))
if args.img_norm:
    img = img.astype(float) / 255.0
# NHWC -> NCWH
img = img.transpose(2, 0, 1)
img = np.expand_dims(img, 0)
img = torch.from_numpy(img).float()

# outimg = img.numpy()
# print(outimg)


# Setup Model
model = get_model(args.arch, n_classes)
state = convert_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage)['model_state'])
model.load_state_dict(state)

model.eval()

if torch.cuda.is_available():
    model.cuda(args.gpu_idx[0])
    images = Variable(img.cuda(args.gpu_idx[0]), volatile=True)
else:
    images = Variable(img, volatile=True)


outputs = model(images)
# outputs = F.softmax(outputs, dim=1)

if args.dcrf == "True":
    unary = outputs.data.cpu().numpy()
    unary = np.squeeze(unary, 0)
    unary = -np.log(unary)
    unary = unary.transpose(2, 1, 0)
    w, h, c = unary.shape
    unary = unary.transpose(2, 0, 1).reshape(loader.n_classes, -1)
    unary = np.ascontiguousarray(unary)

    resized_img = np.ascontiguousarray(resized_img)

    d = dcrf.DenseCRF2D(w, h, loader.n_classes)
    d.setUnaryEnergy(unary)
    d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)

    q = d.inference(50)
    mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
    decoded_crf = loader.decode_segmap(np.array(mask, dtype=np.uint8))
    dcrf_path = args.out_path[:-4] + '_drf.png'
    misc.imsave(dcrf_path, decoded_crf)
    print("Dense CRF Processed Mask Saved at: {}".format(dcrf_path))

if torch.cuda.is_available():
    model.cuda(args.gpu_idx[0])
    images = Variable(img.cuda(args.gpu_idx[0]), volatile=True)
else:
    images = Variable(img, volatile=True)

pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
if args.arch in ['pspnet', 'icnet', 'icnetBN']:
    pred = pred.astype(np.float32)
    pred = misc.imresize(pred, orig_size, 'nearest', mode='F') # float32 with F mode, resize back to orig_size
decoded = loader.decode_segmap(pred)
print('Classes found: ', np.unique(pred))
# io.imsave(args.out_path, decoded)
misc.imsave(args.out_path, decoded)
print("Segmentation Mask Saved at: {}".format(args.out_path))

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default='fcn8s_pascal_1_26.pkl',
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal',
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--dcrf', nargs='?', type=str, default="False",
                        help='Enable DenseCRF based post-processing')
    parser.add_argument('--img_path', nargs='?', type=str, default=None,
                        help='Path of the input image')
    parser.add_argument('--out_path', nargs='?', type=str, default=None,
                        help='Path of the output segmap')
    args = parser.parse_args()
    test(args)
'''
"""