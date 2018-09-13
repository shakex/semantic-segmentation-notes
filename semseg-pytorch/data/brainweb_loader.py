import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils import data
import glob

from tqdm import tqdm
from torch.utils import data
from skimage import io,color
from torchvision import transforms

def get_data_path(name):
    """Extract path to data from config file.

    Args:
        name (str): The name of the dataset.

    Returns:
        (str): The path to the root directory containing the dataset.
    """
    js = open('../config.json').read()
    data = json.loads(js)
    return os.path.expanduser(data[name]['data_path'])

class brainWebLoader(data.Dataset):
    """
    descriptions: todo

    A total of three data splits are provided for working with the BrainWeb data:
        train: 1262 images
        val: 631 images
        trainval: The combination of `train` and `val` - 1893 images

    """

    def __init__(self, root, split="train", is_transform=False, augmentations=None):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 4
        self.files = collections.defaultdict(list)

        for split in ["train", "val", "trainval"]:
            path = pjoin(self.root, split + '.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list
        # self.setup_annotations()

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + 'imgs/' + img_name + '.bmp'
        lbl_path = self.root + 'class4/pre_encoded/c4_' + img_name + '.bmp'

        img = io.imread(img_path)
        img = np.array(img, dtype=np.uint8)
        lbl = io.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.uint8)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def getitem(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + 'imgs/' + img_name + '.bmp'
        lbl_path = self.root + 'class4/pre_encoded/c4_' + img_name + '.bmp'

        # img = m.imread(img_path)
        # img = np.array(img, dtype=np.uint8)
        #
        # lbl = m.imread(lbl_path)
        # lbl = np.array(lbl, dtype=np.int8)

        img = io.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = io.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.uint8)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        # img = img[:, :, ::-1]
        # img = color.gray2rgb(img)
        # img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)
        img = img.astype(float) / 255.0
        # NHWC -> NCHW
        # img = img.transpose(2, 0, 1)
        #
        # img = torch.from_numpy(img).float()
        # # lbl = lbl.astype(int)
        # lbl = torch.from_numpy(lbl).long()

        # mytransform = transforms.Compose([
        #     transforms.ToTensor()
        # ])
        # print(img.shape)
        # print(img.dtype)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl


    def get_brainweb_labels(self):
        """Load the mapping that associates brainweb classes

        Returns:
            np.ndarray (4)

        0: Background(BG)
        1: Cerebro-Spinal Fluid(CSF)
        37: Grey Matter(GM)
        73: White Matter(WM)
        """
        return np.asarray([0,1,37,73])

    def get_brainweb_colormap(self):
        return np.asarray([[0,0,0],[114,26,137],[0,25,204],[67,156,202]])

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """

        mask = mask.astype(np.uint8)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for ii, label in enumerate(self.get_brainweb_labels()):
            label_mask[np.where((mask == (np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8) + label)))] = ii
        label_mask = label_mask.astype(np.uint8)
        return label_mask

    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image

        Args:

        Returns:

        """
        label_colors = self.get_brainweb_colormap()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colors[ll, 0]
            g[label_mask == ll] = label_colors[ll, 1]
            b[label_mask == ll] = label_colors[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    def setup_annotations(self):
        """Pre-encode all segmentation labels into the common label_mask format
        (if this has not already been done).
        """

        target_path = pjoin(self.root, 'class4')
        if not os.path.exists(target_path): os.makedirs(target_path)

        print("Pre-encoding segmentaion masks...")
        for ii in tqdm(self.files['trainval']):
            fname = ii + '.bmp'
            lbl_path = pjoin(self.root, 'class10', 'crisp_' + fname)
            lbl = self.encode_segmap(io.imread(lbl_path))
            io.imsave(pjoin(target_path, 'pre_encoded', 'c4_' + fname), lbl)
            rgb = self.decode_segmap(lbl)
            io.imsave(pjoin(target_path, 'rgb_decoded', 'c4_rgb_' + fname), rgb)

def max4(a,b,c,d):
    max = a
    max1 = a
    max2 = c
    if(a < b):
        max1 = b
    if(c < d):
        max2 = d
    if(max1 > max2):
        max = max1
    else:
        max = max2
    return max

def min4(a,b,c,d):
    min = a
    min1 = a
    min2 = c
    if(a > b):
        min1 = b
    if(c > d):
        min2 = d
    if(min1 > min2):
        min = min2
    else:
        min = min1
    return min

def avg4(a,b,c,d):
    return np.average((a,b,c,d))


def pool(img, type='max'):
    n_row = int(img.size()[2] / 2)
    n_col = int(img.size()[3] / 2)
    downsample = torch.zeros(n_row, n_col)
    for i in range(n_row):
        for j in range(n_col):
            if type == 'max':
                downsample[i][j] = max4(img[0,0,2*i,2*j],img[0,0,2*i,2*j+1],img[0,0,2*i+1,2*j],img[0,0,2*i+1,2*j+1])
            if type == 'min':
                downsample[i][j] = min4(img[0,0,2*i,2*j],img[0,0,2*i,2*j+1],img[0,0,2*i+1,2*j],img[0,0,2*i+1,2*j+1])
            if type == 'avg':
                downsample[i][j] = avg4(img[0,0,2*i,2*j],img[0,0,2*i,2*j+1],img[0,0,2*i+1,2*j],img[0,0,2*i+1,2*j+1])
            if type == 'mode':
                neighbor4 = np.array([img[0,0,2*i,2*j],img[0,0,2*i,2*j+1],img[0,0,2*i+1,2*j],img[0,0,2*i+1,2*j+1]])
                neighbor4_cnt = np.bincount(neighbor4)

                logic = (neighbor4_cnt == neighbor4_cnt.max())

                # if 四个数都不同：
                if np.size(neighbor4_cnt[logic] == 4):
                    mode = np.mean(np.argwhere(neighbor4_cnt == neighbor4_cnt.max()))
                # if 四个数存在相同：
                else:
                    mode = np.argwhere(neighbor4_cnt == neighbor4_cnt.max())[0] # 四个数都不同的时候，取左上角？


                downsample[i][j] = mode

    return downsample


def debug_brainweb():
    image = io.imread('/home/jwliu/disk/dataset/BrainWeb/imgs/msles1_c141.bmp')
    print(image.shape)


    ## use data loader - multiple images test

    # data_path = get_data_path('brainweb')
    # t_loader = brainWebLoader(data_path, split='train', is_transform=True, augmentations=None)
    # trainLoader = data.DataLoader(t_loader, batch_size=1, num_workers=2, shuffle=False)
    #
    # for i, (img, lbl) in enumerate(trainLoader):
    #     img = Variable(img.cuda(2))
    #     lbl = Variable(lbl.cuda(2))
    #
    #     print(i)
    #     print(img.size())
    #     print(lbl.size())

    # one image test
    img = np.array(image, dtype=np.uint8)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    # img = img.astype(float) / 255.0
    img = torch.from_numpy(img).int()
    # img = Variable(img.cuda(2))
    print(img.size())
    print(img.size()[0])
    print(img.size()[1])
    print(img.size()[2])
    print(img.size()[3])

    out = img.numpy()
    out = np.squeeze(out)

    pool1 = pool(img, type='mode')
    pool1_ex = pool1.unsqueeze(0)
    pool1_ex = pool1_ex.unsqueeze(1)
    print(pool1_ex.size())



        # img2 = img[i][0].numpy()*255
        # img2 = img.numpy()
        #
        # img3 = np.squeeze(img2)
        # print(img3.shape)
        #
        # img2 = color.rgb2gray(img2)
        # lbl2 = lbl.numpy()
        #
        # img2 = np.squeeze(img2.transpose(1, 2, 0))
        # lbl2 = np.squeeze(lbl2.transpose(1, 2, 0))


        # print(img2.shape)
        # print(lbl2.shape)
    #
    #     plt.figure()
    #     io.imshow(img2)
    #     plt.figure()
    #     io.imshow(lbl2)
    #     plt.show()

    # img = img[:, :, ::-1]
    # print(img.shape)
    # # img = color.gray2rgb(img)
    # img = np.expand_dims(img, axis=0)
    # print(img.shape)
    # img = img.astype(float) / 255.0
    # # NHWC -> NCHW
    # # img = img.transpose(2, 0, 1)
    # print(img.shape)



# if __name__ == '__main__':


    # debug_brainweb()
    # a = np.array([[1,2,3],[4,5,6],[7,8,9]])
    # b = torch.from_numpy(a)
    # print(a.shape)
    #
    # print(b)






