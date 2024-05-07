import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import glob
import scipy
import cv2 as cv
import imageio
import warnings
import tifffile as tiff
from .read_mrc import read_mrc
#from skimage.measure import compare_mse, compare_nrmse, compare_psnr, compare_ssim

from skimage import metrics


# ---------------------------------------------------------------------------------------
#                                training strategy
# ---------------------------------------------------------------------------------------
class ReduceLROnPlateau():
    def __init__(self, model, curmonitor=np.Inf, factor=0.1, patience=10, mode='min',
                 min_delta=1e-4, cooldown=0, min_lr=0, verbose=1,
                 **kwargs):

        self.curmonitor = curmonitor
        if factor > 1.0:
            raise ValueError('ReduceLROnPlateau does not support a factor > 1.0.')

        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.model = model
        self.verbose = verbose
        self.monitor_op = None
        self._reset()

    def _reset(self):
        if self.mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def update_monitor(self, curmonitor):
        self.curmonitor = curmonitor

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, curmonitor):
        curlr = K.get_value(self.model.optimizer.learning_rate)
        self.curmonitor = curmonitor
        if self.curmonitor is None:
            warnings.warn('errro input of monitor', RuntimeWarning)
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(self.curmonitor, self.best):
                self.best = self.curmonitor
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.learning_rate))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.learning_rate, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing '
                                  'learning rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
        return curlr

    def in_cooldown(self):
        return self.cooldown_counter > 0


# ---------------------------------------------------------------------------------------
#                               Data processing tools
# ---------------------------------------------------------------------------------------
def data_loader(images_path, data_path, gt_path, height, width, batch_size, norm_flag=1, resize_flag=1, scale=2, wf=0):
    #print(f' Line 1:  INSIDE SR MODULE DATA_LOADER : data_path : {data_path} : GT_PATH : {gt_path}')
    batch_images_path = np.random.choice(images_path, size=batch_size, replace=True) # this is where the change is, replace set to False
    #print(f'    SR MODULE : batch_images_path  :  {batch_images_path} ')
    image_batch = []
    gt_batch = []
    for path in batch_images_path:
        if path[-3:] == 'tif':
            curBatch = tiff.imread(path)
            #print(f'   in if shatemenet 1 : this is the income after Line 1: {gt_path}')
            #print(f'path_gt = path.replace(data_path, gt_path) = data_path :  {data_path} :  gt_path : {gt_path} ')
            path_gt = path.replace(data_path, gt_path)
            #print(f'   in if shatemenet 1 : after change new path_gt : {path_gt}')
            gt = imageio.imread(path_gt).astype(float)
        else:
            img_path = glob.glob(path + '/*.tif')
            img_path.sort()
            curBatch = []
            for cur in img_path:
                img = imageio.imread(cur).astype(float)
                if resize_flag == 1:
                    img = cv.resize(img, (height * scale, width * scale))
                curBatch.append(img)
            #print(f'  in else statement 1 : \n gt_path : {gt_path} :  \n :  data_path : {data_path} ')
            path_gt = path.replace(data_path, gt_path) + '.tif'
            #print(f' Line 2 : data_path : {data_path}  \n : path_gt  :  {path_gt}  : \n  code: path_gt = path.replace(data_path, gt_path) + .tif')
            gt = imageio.imread(path_gt).astype(float)
            #print(f' inside data loader : gt = imageio.imread(path_gt).astype(float); {gt.shape} ')

        if norm_flag:
            curBatch = prctile_norm(np.array(curBatch))
            gt = prctile_norm(gt)
            # print(curBatch.shape , '   ----> current batch shape is printed')
            # print(gt.shape , '   ----> GT shape is printed')
        else:
            curBatch = np.array(curBatch) / 65535
            gt = gt / 65535#
            print(f' inside data loader : gt = gt / 65535; {gt.shape} ')   #############################################################
            #print(f'  Line 3 : data_path : {data_path} \n : path_gt  :  {path_gt}  \n : code: path_gt = path.replace(data_path, gt_path) + .tif')
            # print(curBatch.shape , '   ----> current batch shape is printed')
            # print(gt.shape , '   ----> GT shape is printed')
        image_batch.append(curBatch)
        #print(f' inside data loader : gt_batch.append(gt); {gt.shape} ')
        gt_batch.append(gt)

    image_batch = np.array(image_batch)
    # print(image_batch.shape , 'this is the IMAGE_BATCH FROM DATA LOADER')
    #print(f' inside data loader : gt_batch = np.array(gt_batch); {gt_batch.__len__} ')
    gt_batch = np.array(gt_batch)
    print(gt_batch.shape , 'this is the gt_batch FROM DATA LOADER')

    image_batch = np.transpose(image_batch, (0, 2, 3, 1))
    print(  f'this is the image_batch FROM DATA LOADER np.transpose(image_batch, (0, 2, 3, 1)), SR MODULE : {image_batch.shape} ')

    gt_batch = gt_batch.reshape((batch_size, width*scale, height*scale, 1))
    print( f'this is the gt_batch FROM DATA LOADER being reshaped ((batch_size, width*scale, height*scale, 1)) : SR MODULE : {gt_batch.shape}  batch_size : {batch_size}, width*scale : {width}, height*scale {scale}, 1 : {batch_size * width*scale * height*scale}')

    if wf == 1:
        image_batch = np.mean(image_batch, 3)
        for b in range(batch_size):
            image_batch[b, :, :] = prctile_norm(image_batch[b, :, :])
        image_batch = image_batch[:, :, :, np.newaxis]

    return image_batch, gt_batch


def data_loader_rDL(images_path, data_path, gt_path, batch_size=1):
    batch_images_path = np.random.choice(images_path, size=batch_size, replace=False)
    image_batch = []
    gt_batch = []
    for path in batch_images_path:
        if path[-3:] == 'tif':
            image = tiff.imread(path)
            path_gt = path.replace(data_path, gt_path)
            gt = tiff.imread(path_gt)
        else:
            imgfile = glob.glob(path + '/*.tif')
            imgfile.sort()
            image = []
            gt = []
            for file in imgfile:
                img = imageio.imread(file).astype(float)
                image.append(img)
            path_gt = path.replace(data_path, gt_path)
            imgfile = glob.glob(path_gt + '/*.tif')
            imgfile.sort()
            for file in imgfile:
                img = imageio.imread(file).astype(float)
                gt.append(img)
        image_batch.append(image)
        gt_batch.append(gt)

    image_batch = np.array(image_batch).astype(float)
    gt_batch = np.array(gt_batch).astype(float)
    # print(image_batch.shape , 'this is the image_batch coming from RDL dataLOader')
    # print(gt_batch.shape , 'this is the gt_batch coming from RDL dataLOader')

    return image_batch, gt_batch


def prctile_norm(x, min_prc=0, max_prc=100):
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-7)
    return y


def cal_comp(gt, pr, mses=None, nrmses=None, psnrs=None, ssims=None):
    if ssims is None:
        ssims = []
    if psnrs is None:
        psnrs = []
    if nrmses is None:
        nrmses = []
    if mses is None:
        mses = []
    gt, pr = np.squeeze(gt), np.squeeze(pr)
    gt = gt.astype(np.float32)
    if gt.ndim == 2:
        n = 1
        gt = np.reshape(gt, (1, gt.shape[0], gt.shape[1]))
        pr = np.reshape(pr, (1, pr.shape[0], pr.shape[1]))
    else:
        n = np.size(gt, 0)

    for i in range(n):
        mses.append(metrics.mean_squared_error(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i]))))
        nrmses.append(metrics.normalized_root_mse(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i])))) # skimage.measure.compare_nrmse to skimage.metrics.normalized_root_mse.   URL: https://scikit-image.org/docs/stable/api/skimage.metrics.html
        psnrs.append(metrics.peak_signal_noise_ratio(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i])))) # This function was renamed from skimage.measure.compare_psnr to skimage.metrics.peak_signal_noise_ratio. remove the 1 at the end.
        ssims.append(metrics.structural_similarity(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i])), data_range = 1)) # Changed in version 0.16: This function was renamed from skimage.measure.compare_ssim to skimage.metrics.structural_similarity. datarange ste to 1 as dividing by 655... to normalize it. 
    return mses, nrmses, psnrs, ssims
