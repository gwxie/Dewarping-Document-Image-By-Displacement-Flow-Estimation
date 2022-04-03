import pickle

import torch
from torch.utils import data

import numpy as np

import seaborn as sns
import sys, os

import cv2
import time
import re
from multiprocessing import Pool

import random
import scipy.spatial.qhull as qhull
from scipy.interpolate import griddata
from scipy.optimize import fsolve
from torch.autograd import Variable
import torch.nn.functional as F

import dataloader

def adjust_position(x_min, y_min, x_max, y_max, new_shape):
    if (new_shape[0] - (x_max - x_min)) % 2 == 0:
        f_g_0_0 = (new_shape[0] - (x_max - x_min)) // 2
        f_g_0_1 = f_g_0_0
    else:
        f_g_0_0 = (new_shape[0] - (x_max - x_min)) // 2
        f_g_0_1 = f_g_0_0 + 1

    if (new_shape[1] - (y_max - y_min)) % 2 == 0:
        f_g_1_0 = (new_shape[1] - (y_max - y_min)) // 2
        f_g_1_1 = f_g_1_0
    else:
        f_g_1_0 = (new_shape[1] - (y_max - y_min)) // 2
        f_g_1_1 = f_g_1_0 + 1

    # return f_g_0_0, f_g_0_1, f_g_1_0, f_g_1_1
    return f_g_0_0, f_g_1_0, new_shape[0] - f_g_0_1, new_shape[1] - f_g_1_1


class SaveFlatImage(object):
    def __init__(self, path, date, date_time, _re_date, data_split, data_path_validate, data_path_test, batch_size, preproccess=False):
        self.path = path
        self.date = date
        self.date_time = date_time
        self._re_date = _re_date
        self.data_split = data_split
        self.preproccess = preproccess
        # self.validate_groun_truth_path =data_path_validate+'validate/scan/'
        # self.test_groun_truth_path ='your_groun_truth_path/'
        self.batch_size = batch_size
        self.perturbed_test_img_path = data_path_test

    def interp_weights(self, xyz, uvw):
        tri = qhull.Delaunay(xyz)
        simplex = tri.find_simplex(uvw)
        vertices = np.take(tri.simplices, simplex, axis=0)
        # pixel_triangle = pixel[tri.simplices]
        temp = np.take(tri.transform, simplex, axis=0)
        delta = uvw - temp[:, 2]
        bary = np.einsum('njk,nk->nj', temp[:, :2, :], delta)
        return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    def interpolate(self, values, vtx, wts):
        return np.einsum('njk,nj->nk', np.take(values, vtx, axis=0), wts)

    # return np.einsum('nj,nj->n', np.take(values, vtx), wts)
    def flatByRegressWithClassiy_multiProcess(self, perturbed_label, perturbed_label_classify, im_name, epoch, scheme='validate', is_scaling=False, perturbed_img=None):
        # for i_val_i in range(perturbed_label.shape[0]):
        #     self.flatByRegressWithClassiy_triangular_v2_RGB(perturbed_label[i_val_i], perturbed_label_classify[i_val_i], im_name[i_val_i], epoch + 1, scheme, is_scaling)

        process_pool = Pool(self.batch_size)
        for i_val_i in range(perturbed_label.shape[0]):
            process_pool.apply_async(func=self.flatByRegressWithClassiy_triangular_v2_RGB, args=(perturbed_label[i_val_i], perturbed_label_classify[i_val_i], im_name[i_val_i], epoch, scheme, is_scaling, perturbed_img[i_val_i]))
            # process_pool.apply_async(func=self.flatByRegressWithClassiy_triangular_v3_RGB, args=(perturbed_label[i_val_i], perturbed_label_classify[i_val_i], im_name[i_val_i], epoch, scheme, is_scaling, perturbed_img[i_val_i]))
            
        process_pool.close()
        process_pool.join()

    def flatByRegressWithClassiy_triangular_v3_RGB(self, perturbed_label, perturbed_label_classify, im_name, epoch, scheme='validate', is_scaling=False, perturbed_img=None):
        # if self.preproccess:
        #     perturbed_label[np.sum(perturbed_label, 2) != -2] *= 10
        # perturbed_label = cv2.GaussianBlur(perturbed_label, (3, 3), 0)
        # perturbed_label = cv2.blur(perturbed_label, (13, 13))
        # perturbed_label = cv2.GaussianBlur(perturbed_label, (5, 5), 0)
        # perturbed_label_backups = perturbed_label.copy()
        # perturbed_label_classify = np.around(cv2.GaussianBlur(perturbed_label_classify.astype(np.float32), (33, 33), 0)).astype(np.uint8)


        if (scheme == 'test' or scheme == 'eval') and is_scaling:
            perturbed_img_path = self.perturbed_test_img_path + im_name
            perturbed_img = cv2.imread(perturbed_img_path, flags=cv2.IMREAD_COLOR)
            perturbed_img = dataloader.resize_image(perturbed_img, 1024*2, 960*2)

            flat_shape = perturbed_img.shape[:2]
            perturbed_label = cv2.resize(perturbed_label * 2, (flat_shape[1], flat_shape[0]), interpolation=cv2.INTER_LINEAR)
            perturbed_label_classify = cv2.resize(perturbed_label_classify.astype(np.float32), (flat_shape[1], flat_shape[0]), interpolation=cv2.INTER_LINEAR)
            perturbed_label_classify[perturbed_label_classify <= 0.5] = 0
            perturbed_label_classify = np.array(perturbed_label_classify).astype(np.uint8)
            # perturbed'_label_classify = np.array(perturbed_label_classify).astype(np.float32)
        else:
            if perturbed_img is None:
                if scheme == 'test' or scheme == 'eval':
                    perturbed_img_path = self.perturbed_test_img_path + im_name
                elif scheme == 'validate':
                    RGB_name = im_name.replace('gw', 'png')
                    perturbed_img_path = '.YOUR_PATH_/validate/png/' + RGB_name
                perturbed_img = cv2.imread(perturbed_img_path, flags=cv2.IMREAD_COLOR)
            flat_shape = perturbed_img.shape[:2]

        flat_img = np.full_like(perturbed_img, 256, dtype=np.uint16)
        
        '''scaling
        origin_pixel_position = np.argwhere(np.zeros(flat_shape, dtype=np.uint32) == 0).reshape(flat_shape[0], flat_shape[1], 2)
        flow = perturbed_label + origin_pixel_position 

        grid_shape = [512, 512]
        flow = cv2.resize(flow, (flat_shape[1], flat_shape[0]), interpolation=cv2.INTER_LINEAR)
        perturbed_label_classify = cv2.resize(perturbed_label_classify.astype(np.float32), (grid_shape[1], grid_shape[0]), interpolation=cv2.INTER_LINEAR)
        perturbed_label_classify[perturbed_label_classify <= 0.5] = 0
        perturbed_label_classify = np.array(perturbed_label_classify).astype(np.uint8)
        '''
        
        perturbed_img_ = perturbed_img[perturbed_label_classify==1]
        origin_pixel_position = np.argwhere(np.zeros(flat_shape, dtype=np.uint32) == 0).reshape(flat_shape[0], flat_shape[1], 2)
        flow = perturbed_label + origin_pixel_position
        flat_img=griddata(flow[perturbed_label_classify==1],perturbed_img_,(origin_pixel_position[:,:,0],origin_pixel_position[:,:,1]),method='linear')
        # cv2.imwrite('./dewarp2.png', img)  
    
        perturbed_img = perturbed_img.reshape(flat_shape[0], flat_shape[1], 3)
        flat_img = flat_img.astype(np.uint8)
        img_figure = np.concatenate(
            (perturbed_img, flat_img), axis=1)

        i_path = os.path.join(self.path, self.date + self.date_time + ' @' + self._re_date,
                              str(epoch)) if self._re_date is not None else os.path.join(self.path, self.date + self.date_time, str(epoch))
        if scheme == 'test':
            i_path += '/test'
        if not os.path.exists(i_path):
            os.makedirs(i_path)

        im_name = im_name.replace('gw', 'png')
        cv2.imwrite(i_path + '/' + im_name, img_figure)

    def flatByRegressWithClassiy_triangular_v2_RGB(self, perturbed_label, perturbed_label_classify, im_name, epoch, scheme='validate', is_scaling=False, perturbed_img=None):
        # if self.preproccess:
        #     perturbed_label[np.sum(perturbed_label, 2) != -2] *= 10
        # perturbed_label = cv2.GaussianBlur(perturbed_label, (3, 3), 0)
        # perturbed_label = cv2.blur(perturbed_label, (13, 13))
        # perturbed_label = cv2.GaussianBlur(perturbed_label, (5, 5), 0)
        # perturbed_label_backups = perturbed_label.copy()
        # perturbed_label_classify = np.around(cv2.GaussianBlur(perturbed_label_classify.astype(np.float32), (33, 33), 0)).astype(np.uint8)


        if (scheme == 'test' or scheme == 'eval') and is_scaling:
            perturbed_img_path = self.perturbed_test_img_path + im_name
            perturbed_img = cv2.imread(perturbed_img_path, flags=cv2.IMREAD_COLOR)
            perturbed_img = dataloader.resize_image(perturbed_img, 1024*2, 960*2)

            flat_shape = perturbed_img.shape[:2]
            perturbed_label = cv2.resize(perturbed_label * 2, (flat_shape[1], flat_shape[0]), interpolation=cv2.INTER_LINEAR)
            perturbed_label_classify = cv2.resize(perturbed_label_classify.astype(np.float32), (flat_shape[1], flat_shape[0]), interpolation=cv2.INTER_LINEAR)
            perturbed_label_classify[perturbed_label_classify <= 0.5] = 0
            perturbed_label_classify = np.array(perturbed_label_classify).astype(np.uint8)
            # perturbed'_label_classify = np.array(perturbed_label_classify).astype(np.float32)
        else:
            if perturbed_img is None:
                if scheme == 'test' or scheme == 'eval':
                    perturbed_img_path = self.perturbed_test_img_path + im_name
                elif scheme == 'validate':
                    RGB_name = im_name.replace('gw', 'png')
                    perturbed_img_path = '.YOUR_PATH_/validate/png/' + RGB_name
                perturbed_img = cv2.imread(perturbed_img_path, flags=cv2.IMREAD_COLOR)
            flat_shape = perturbed_img.shape[:2]

        flat_img = np.full_like(perturbed_img, 256, dtype=np.uint16)

        '''remove the background of input image(perturbed_img) and forward mapping(perturbed_label)'''
        origin_pixel_position = np.argwhere(np.zeros(flat_shape, dtype=np.uint32) == 0).reshape(flat_shape[0] * flat_shape[1], 2)
        perturbed_label = perturbed_label.reshape(flat_shape[0] * flat_shape[1], 2)
        perturbed_img = perturbed_img.reshape(flat_shape[0] * flat_shape[1], 3)
        perturbed_label_classify = perturbed_label_classify.reshape(flat_shape[0] * flat_shape[1])
        perturbed_label[perturbed_label_classify != 0, :] += origin_pixel_position[perturbed_label_classify != 0, :]
        pixel_position = perturbed_label[perturbed_label_classify != 0, :]
        pixel = perturbed_img[perturbed_label_classify != 0, :]
        '''construct Delaunay triangulations in all scattered pixels and then using interpolation'''
        vtx, wts = self.interp_weights(pixel_position, origin_pixel_position)
        # wts[np.abs(wts)>=1]=0
        wts_sum = np.abs(wts).sum(-1)
        wts = wts[wts_sum <= 1, :]
        vtx = vtx[wts_sum <= 1, :]
        flat_img.reshape(flat_shape[0] * flat_shape[1], 3)[wts_sum <= 1, :] = self.interpolate(pixel, vtx, wts)
        flat_img = flat_img.reshape(flat_shape[0], flat_shape[1], 3)

        '''validate test eval'''
        if scheme == 'eval':
            flat_x_min, flat_y_min, flat_x_max, flat_y_max = -1, -1, flat_shape[0], flat_shape[1]

            for x in range(flat_shape[0] // 2, flat_x_max):
                if np.sum(flat_img[x, :]) == 768 * flat_shape[1] and flat_x_max - 1 > x:
                    flat_x_max = x
                    break
            for x in range(flat_shape[0] // 2, flat_x_min, -1):
                if np.sum(flat_img[x, :]) == 768 * flat_shape[1] and x > 0:
                    flat_x_min = x
                    break
            for y in range(flat_shape[1] // 2, flat_y_max):
                if np.sum(flat_img[:, y]) == 768 * flat_shape[0] and flat_y_max - 1 > y:
                    flat_y_max = y
                    break
            for y in range(flat_shape[1] // 2, flat_y_min, -1):
                if np.sum(flat_img[:, y]) == 768 * flat_shape[0] and y > 0:
                    flat_y_min = y
                    break

            flat_x = round((flat_x_max - flat_x_min) * 0.2)  # 0.2
            flat_y = round((flat_y_max - flat_y_min) * 0.2)  # 0.2
            flat_img = flat_img[flat_x_min:flat_x_max, flat_y_min:flat_y_max]
            flat_shape = flat_img.shape[:2]
            flat_x_min, flat_y_min, flat_x_max, flat_y_max = 0, 0, flat_shape[0], flat_shape[1]
            for x in range(flat_shape[0] // 2, flat_x_max):
                if np.count_nonzero(np.sum(flat_img[x, :], 1) == 768) >= flat_x and flat_x_max - 1 > x:
                    flat_x_max = x
                    break
            for x in range(flat_shape[0] // 2, flat_x_min, -1):
                if np.count_nonzero(np.sum(flat_img[x, :], 1) == 768) >= flat_x and x > 0:
                    flat_x_min = x
                    break
            for y in range(flat_shape[1] // 2, flat_y_max):
                if np.count_nonzero(np.sum(flat_img[:, y], 1) == 768) >= flat_y and flat_y_max - 1 > y:
                    flat_y_max = y
                    break
            for y in range(flat_shape[1] // 2, flat_y_min, -1):
                if np.count_nonzero(np.sum(flat_img[:, y], 1) == 768) >= flat_y and y > 0:
                    flat_y_min = y
                    break
            if (flat_x_max - flat_x_min) % 8 != 0:
                flat_x_max += (flat_x_max - flat_x_min) % 8
            if (flat_y_max - flat_y_min) % 8 != 0:
                flat_y_max += (flat_y_max - flat_y_min) % 8

            ''''''
            flat_img = flat_img.astype(np.uint8)
            flat_img_crop = flat_img[flat_x_min:flat_x_max, flat_y_min:flat_y_max]

            ''''''
            scan_img_path = '.YOUR_PATH_/scan/'
            groundTrue_img = cv2.imread(
                scan_img_path + re.match(r'(\d+)_(\d+)( copy.png)', im_name, re.IGNORECASE).group(1) + '.png',
                flags=cv2.IMREAD_COLOR)
            gt_shape = groundTrue_img.shape[:2]

            def func(p):
                x, y = p[0], p[1]
                return [x * y - 598400, x / y - gt_shape[0] / gt_shape[1]]

            im_lr, im_ud = np.round(fsolve(func, [gt_shape[0], gt_shape[1]])).astype(np.int)
            if im_lr % 2 != 0:
                im_lr += 1
            if im_ud % 2 != 0:
                im_ud += 1

            i_path = os.path.join(self.path, self.date + self.date_time + ' @' + self._re_date,
                                  str(epoch)) if self._re_date is not None else os.path.join(self.path,
                                                                                             self.date + self.date_time,
                                                                                             str(epoch), '/eval')

            im_name = im_name.replace(' copy.png', '.jpg')
            # groundTrue_img = cv2.resize(groundTrue_img, (im_ud, im_lr), interpolation=cv2.INTER_NEAREST)
            # groundTrue_img = cv2.cvtColor(groundTrue_img, cv2.COLOR_RGB2GRAY)
            # if not os.path.exists(i_path + '/gray/'):
            #     os.makedirs(i_path + '/gray/')
            # cv2.imwrite(i_path + '/gray/' + im_name, groundTrue_img)

            flat_img = cv2.resize(flat_img_crop, (im_ud, im_lr), interpolation=cv2.INTER_CUBIC)
            # try:
            #     flat_img = cv2.resize(flat_img_crop, (im_ud, im_lr), interpolation=cv2.INTER_NEAREST)
            # except:
            #     print('resize error')

            # flat_img = cv2.cvtColor(flat_img, cv2.COLOR_RGB2GRAY)

            if not os.path.exists(i_path):
                os.makedirs(i_path)
            cv2.imwrite(i_path + '/' + im_name, flat_img)

        else:
            perturbed_img = perturbed_img.reshape(flat_shape[0], flat_shape[1], 3)
            ''''''
            flat_img = flat_img.astype(np.uint8)

            '''groun_truth_img
            try:
                if scheme == 'validate':
                    groun_truth_name = re.match(r'(\w+\d*)_(\d+)_(\d+)_(\w+)', im_name, re.IGNORECASE).group(1, 2)
                    groun_truth_name = '_'.join(groun_truth_name)+'.png'
                    groun_truth = np.zeros_like(perturbed_img, dtype=np.uint8)
                    groun_truth_img = cv2.imread(self.validate_groun_truth_path+groun_truth_name, flags=cv2.IMREAD_COLOR)
                    if groun_truth_img is None:
                        return
                     #shrink
                    if groun_truth_img.shape[:2] > flat_shape:
                        img_shrink = 512
                        im_lr = groun_truth_img.shape[0]
                        im_ud = groun_truth_img.shape[1]
                        if im_lr > im_ud:
                            im_ud = int(im_ud / im_lr * img_shrink)
                            im_lr = img_shrink
                        else:
                            im_lr = int(im_lr / im_ud * img_shrink)
                            im_ud = img_shrink
                        groun_truth_img = cv2.resize(groun_truth_img, (im_ud, im_lr), interpolation=cv2.INTER_CUBIC)

                    groun_truth_img_shape = groun_truth_img.shape[:2]
                    if (flat_shape[0] - groun_truth_img_shape[0]) % 2 == 0:
                        f_g_0_0 = (flat_shape[0] - groun_truth_img_shape[0]) // 2
                        f_g_0_1 = f_g_0_0
                    else:
                        f_g_0_0 = (flat_shape[0] - groun_truth_img_shape[0]) // 2
                        f_g_0_1 = f_g_0_0 + 1

                    if (flat_shape[1] - groun_truth_img_shape[1]) % 2 == 0:
                        f_g_1_0 = (flat_shape[1] - groun_truth_img_shape[1]) // 2
                        f_g_1_1 = f_g_1_0
                    else:
                        f_g_1_0 = (flat_shape[1] - groun_truth_img_shape[1]) // 2
                        f_g_1_1 = f_g_1_0 + 1
                    groun_truth[f_g_0_0:flat_shape[0]-f_g_0_1, f_g_1_0:flat_shape[1]-f_g_1_1] = groun_truth_img
                elif scheme == 'test':
                    groun_truth_name = re.match(r'(\d+)_(\d+)( copy.png)', im_name, re.IGNORECASE).group(1)
                    groun_truth = cv2.imread(self.test_groun_truth_path+groun_truth_name+'.png', flags=cv2.IMREAD_COLOR)
                    groun_truth = cv2.resize(groun_truth, (flat_shape[1], flat_shape[0]), interpolation=cv2.INTER_CUBIC)
            except:
                print('save img error: *************'+groun_truth_name+'*************')
                return
            img_figure = np.concatenate(
                (perturbed_img, flat_img, groun_truth), axis=1)
            '''
            img_figure = np.concatenate(
                (perturbed_img, flat_img), axis=1)

            i_path = os.path.join(self.path, self.date + self.date_time + ' @' + self._re_date,
                                  str(epoch)) if self._re_date is not None else os.path.join(self.path, self.date + self.date_time, str(epoch))
            if scheme == 'test':
                i_path += '/test'
            if not os.path.exists(i_path):
                os.makedirs(i_path)

            im_name = im_name.replace('gw', 'png')
            cv2.imwrite(i_path + '/' + im_name, img_figure)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, m=1):
        self.val = val
        self.sum += val * m
        self.count += n
        self.avg = self.sum / self.count

class FlatImg(object):
    def __init__(self, args, path, date, date_time, _re_date, model,\
                 reslut_file, n_classes, optimizer, \
                 model_D=None, optimizer_D=None, \
                 loss_fn=None, loss_classify_fn=None, data_loader=None, data_loader_hdf5=None, dataPackage_loader = None, data_split=None, \
                 data_path=None, data_path_validate=None, data_path_test=None, data_preproccess=True):     #, valloaderSet, v_loaderSet
        self.args = args
        self.path = path
        self.date = date
        self.date_time = date_time
        self._re_date = _re_date
        # self.valloaderSet = valloaderSet
        # self.v_loaderSet = v_loaderSet
        self.model = model
        self.model_D = model_D
        self.reslut_file = reslut_file
        self.n_classes = n_classes
        self.optimizer = optimizer
        self.optimizer_D = optimizer_D
        self.loss_fn = loss_fn
        self.loss_classify_fn = loss_classify_fn
        self.data_loader = data_loader
        self.data_loader_hdf5 = data_loader_hdf5
        self.dataPackage_loader = dataPackage_loader
        self.data_split = data_split
        self.data_path = data_path
        self.data_path_validate = data_path_validate
        self.data_path_test = data_path_test
        self.data_preproccess = data_preproccess
        self.save_flat_mage = SaveFlatImage(self.path, self.date, self.date_time, self._re_date, self.data_split, self.data_path_validate, self.data_path_test, self.args.batch_size, self.data_preproccess)

        self.validate_loss = AverageMeter()
        self.validate_loss_regress = AverageMeter()
        self.validate_loss_classify = AverageMeter()
        self.lambda_loss = 0.1
        self.lambda_loss_classify = 1

    def saveDataPackage(self, data_size = '640'):

        if not os.path.exists(self.data_path_validate + 'clip' + data_size + '/'):
            os.makedirs(self.data_path_validate + 'clip' + data_size + '/')

        if not os.path.exists(self.data_path_validate + 'label' + data_size + '/'):
            os.makedirs(self.data_path_validate + 'label' + data_size + '/')
        trainloader = self.loadTrainData(data_split=self.data_split, is_shuffle=True)
        begin_train = time.time()
        for i, (images, labels) in enumerate(trainloader):
            with open(self.data_path_validate + 'clip' + data_size + '/' + str(i) + '.im', 'wb') as f:
                pickle_perturbed_im = pickle.dumps(images)
                f.write(pickle_perturbed_im)

            with open(self.data_path_validate + 'label' + data_size + '/' + str(i) + '.lbl', 'wb') as f:
                pickle_perturbed_lbl = pickle.dumps(labels)
                f.write(pickle_perturbed_lbl)

        trian_t = time.time() - begin_train

        m, s = divmod(trian_t, 60)
        h, m = divmod(m, 60)
        print("All Train Time : %02d:%02d:%02d\n" % (h, m, s))

    def loadTrainDataHDF5(self, data_split, is_shuffle=True):
        train_loader = self.data_loader_hdf5(self.data_path, split='train', group='train')
        trainloader = data.DataLoader(train_loader, batch_size=self.args.batch_size, num_workers=1, pin_memory=True, drop_last=True, shuffle=is_shuffle)
        # trainloader = data.DataLoader(train_loader, batch_size=self.args.batch_size, num_workers=0, drop_last=True, shuffle=is_shuffle)   #  pin_memory=True,

        # trainloader = data.DataLoader(train_loader, batch_size=self.args.batch_size, num_workers=self.args.parallel.__len__()//2, drop_last=True,
        #                               shuffle=is_shuffle)
        return trainloader

    def loadTrainData(self, data_split, is_shuffle=True):
        train_loader = self.data_loader(self.data_path, split='train', img_shrink=self.args.img_shrink, preproccess=self.data_preproccess)
        trainloader = data.DataLoader(train_loader, batch_size=self.args.batch_size, num_workers=self.args.batch_size, drop_last=True, pin_memory=True,
                                      shuffle=is_shuffle)
        return trainloader

    # def loadTrainDataPackage(self, data_split, is_shuffle=True, data_size='640'):
    #     train_loader = self.dataPackage_loader(self.data_path, split=data_split, data_size=data_size)
    #     trainloader = data.DataLoader(train_loader, batch_size=1, num_workers=1, shuffle=is_shuffle)
    #
    #     return trainloader

    def loadValidateAndTestData(self, is_shuffle=True, sub_dir='shrink_512/crop/'):
        v1_loader = self.data_loader(self.data_path_validate, split='validate', img_shrink=self.args.img_shrink, is_return_img_name=True, preproccess=self.data_preproccess)
        valloader1 = data.DataLoader(v1_loader, batch_size=self.args.batch_size, num_workers=self.args.batch_size, pin_memory=True, \
                                       shuffle=is_shuffle)

        '''val sets'''
        v_loaderSet = {
            'v1_loader': v1_loader,
        }
        valloaderSet = {
            'valloader1': valloader1,
        }
        # sub_dir = 'crop/crop/'

        t1_loader = self.data_loader(self.data_path_test, split='test', img_shrink=self.args.img_shrink, is_return_img_name=True)
        testloader1 = data.DataLoader(t1_loader, batch_size=self.args.batch_size, num_workers=self.args.batch_size, pin_memory=True, \
                                       shuffle=False)

        '''test sets'''
        t_loaderSet = {
            't1_loader': v1_loader,
        }
        testloaderSet = {
            'testloader1': testloader1,
        }

        self.valloaderSet = valloaderSet
        self.v_loaderSet = v_loaderSet

        self.testloaderSet = testloaderSet
        self.t_loaderSet = t_loaderSet
        # return v_loaderSet, valloaderSet

    def loadTestData(self, is_shuffle=True):


        t1_loader = self.data_loader(self.data_path_test, split='test', img_shrink=None, is_return_img_name=True)
        testloader1 = data.DataLoader(t1_loader, batch_size=self.args.batch_size, num_workers=self.args.batch_size, pin_memory=True, \
                                       shuffle=False)

        '''test sets'''
        testloaderSet = {
            'testloader1': testloader1,
        }

        self.testloaderSet = testloaderSet

    def saveModel_epoch(self, epoch):
        epoch += 1
        state = {'epoch': epoch,
                 'model_state': self.model.state_dict(),
                 'optimizer_state': self.optimizer.state_dict(),    # AN ERROR HAS OCCURED
                 }
        i_path = os.path.join(self.path, self.date + self.date_time + ' @' + self._re_date,
                              str(epoch)) if self._re_date is not None else os.path.join(self.path, self.date + self.date_time, str(epoch))
        if not os.path.exists(i_path):
            os.makedirs(i_path)

        if self._re_date is None:
            torch.save(state, i_path + '/' + self.date + self.date_time + "{}-{}".format(self.args.arch,
                                                                                                self.args.dataset) + ".pkl")  # "./trained_model/{}_{}_best_model.pkl"
        else:
            torch.save(state,
                       i_path + '/' + self._re_date + "@" + self.date + self.date_time + "{}-{}".format(
                           self.args.arch,
                           self.args.dataset) + ".pkl")


    def evalModelGreyC1(self, epoch, train_time, is_scaling=False):
        begin_test = time.time()
        with torch.no_grad():
            for i_valloader, valloader in enumerate(self.testloaderSet.values()):

                for i_val, (images_val, im_name) in enumerate(valloader):
                    try:
                        # save_img_ = True
                        # save_img_ = random.choices([True, False], weights=[0.2, 0.8])[0]

                        images_val = Variable(images_val.cuda(self.args.gpu))

                        outputs, outputs_classify = self.model(images_val, is_softmax=True)
                        outputs_classify = outputs_classify.squeeze(1)
                        # outputs, outputs_classify = self.model(images_val, is_softmax=True)

                        pred_regress = outputs.data.cpu().numpy().transpose(0, 2, 3, 1)
                        # pred_classify = outputs_classify.data.max(1)[1].cpu().numpy()       #     ==outputs.data.argmax(dim=0).cpu().numpy()
                        pred_classify = outputs_classify.data.round().int().cpu().numpy()  # (4, 1280, 1024)  ==outputs.data.argmax(dim=0).cpu().numpy()
                        perturbed_img = images_val.data.cpu().numpy().transpose(0, 2, 3, 1)

                        self.save_flat_mage.flatByRegressWithClassiy_multiProcess(pred_regress, pred_classify, im_name, epoch + 1, scheme='eval', is_scaling=is_scaling, perturbed_img=perturbed_img)
                    except:
                        print('* save image tested error :'+im_name[0])

            test_time = time.time() - begin_test

            print('test time : {test_time:.3f}\t'.format(
                test_time=test_time))

            print('test time : {test_time:.3f}\t'.format(
                test_time=test_time),
                file=self.reslut_file)

    def validateOrTestModelV2GreyC1(self, epoch, trian_t, validate_test='v', is_scaling=False, save_rate=0.001):

        if validate_test == 'v_l3v3':
            loss_l1_list = []
            loss_overall_list = []
            loss_local_list = []
            begin_test = time.time()
            with torch.no_grad():
                for i_valloader, valloader in enumerate(self.valloaderSet.values()):
                    for i_val, (images_val, labels_val, labels_classify_val, im_name) in enumerate(valloader):
                        try:
                            # save_img_ = random.choices([True, False], weights=[0.4, 0.6])[0]
                            save_img_ = random.choices([True, False], weights=[0.005, 0.995])[0]
                            # save_img_ = True

                            images_val = Variable(images_val.cuda(self.args.gpu))
                            labels_val = Variable(labels_val.cuda(self.args.gpu))
                            labels_classify_val = Variable(labels_classify_val.cuda(self.args.gpu))

                            outputs, outputs_classify = self.model(images_val, is_softmax=False)
                            outputs_classify = outputs_classify.squeeze(1)
                            # outputs, outputs_classify = self.model(images_val, is_softmax=True)

                            loss_overall, loss_local, loss_l1 = self.loss_fn(outputs, labels_val, outputs_classify, labels_classify_val, size_average=False)
                            # loss_overall, loss_local, loss_l1 = self.loss_fn(outputs, labels_val, labels_classify_val, size_average=True)
                            loss_regress = loss_overall + loss_local + loss_l1
                            loss_classify = self.loss_classify_fn(outputs_classify, labels_classify_val)

                            outputs_classify = F.sigmoid(outputs_classify)
                            # loss_classify = F.nll_loss(torch.log(outputs_classify), labels_classify_val)
                            loss = self.lambda_loss * loss_regress + self.lambda_loss_classify * loss_classify
                            self.validate_loss.update(loss.item())
                            self.validate_loss_regress.update(loss_regress.item())
                            self.validate_loss_classify.update(loss_classify.item())

                            pred_regress = outputs.data.cpu().numpy().transpose(0, 2, 3, 1)         # (4, 1280, 1024, 2)
                            # pred_classify = outputs_classify.data.max(1)[1].cpu().numpy()  # (4, 1280, 1024)  ==outputs.data.argmax(dim=0).cpu().numpy()
                            pred_classify = outputs_classify.data.round().int().cpu().numpy()  # (4, 1280, 1024)  ==outputs.data.argmax(dim=0).cpu().numpy()
                            perturbed_img = images_val.data.cpu().numpy().transpose(0, 2, 3, 1)         # (4, 1280, 1024, 3)

                            if save_img_:
                                self.save_flat_mage.flatByRegressWithClassiy_multiProcess(pred_regress,
                                                                                          pred_classify, im_name,
                                                                                          epoch + 1,
                                                                                          scheme='validate', is_scaling=is_scaling, perturbed_img=perturbed_img)
                            loss_l1_list.append(loss_l1.item())
                            loss_overall_list.append(loss_overall.item())
                            loss_local_list.append(loss_local.item())
                        except:
                            print('* save image validated error :'+im_name[0])
                test_time = time.time() - begin_test

                # if always_save_model:
                #     self.saveModel(epoch, save_path=self.path)
                list_len = len(loss_l1_list)
                print('train time : {trian_t:.3f}\t'
                      'validate time : {test_time:.3f}\t'
                      'Validate Loss : {v_l.avg:.4f}\t'
                      '[l1:{overall_avg:.2f} l:{local_avg:.4f} cs:{l1_avg:.4f}\t'
                      '[{loss_regress.avg:.4f}  {loss_classify.avg:.4f}]\n'.format(
                       trian_t=trian_t, test_time=test_time,
                       v_l=self.validate_loss,
                       overall_avg=sum(loss_overall_list) / list_len, local_avg=sum(loss_local_list) / list_len, l1_avg=sum(loss_l1_list) / list_len,
                       loss_regress=self.validate_loss_regress, loss_classify=self.validate_loss_classify))
                print('train time : {trian_t:.3f}\t'
                      'validate time : {test_time:.3f}\t'
                      'Validate Loss : {v_l.avg:.4f}\t'
                      '[o:{overall_avg:.2f} l:{local_avg:.4f} ln:{l1_avg:.4f}\t'
                      '[{loss_regress.avg:.4f}  {loss_classify.avg:.4f}]\n'.format(
                       trian_t=trian_t, test_time=test_time,
                       v_l=self.validate_loss,
                       overall_avg=sum(loss_overall_list) / list_len, local_avg=sum(loss_local_list) / list_len, l1_avg=sum(loss_l1_list) / list_len,
                       loss_regress=self.validate_loss_regress, loss_classify=self.validate_loss_classify), file=self.reslut_file)

                self.validate_loss.reset()
                self.validate_loss_regress.reset()
                self.validate_loss_classify.reset()

        else:
            begin_test = time.time()
            with torch.no_grad():
                for i_valloader, valloader in enumerate(self.testloaderSet.values()):

                    for i_val, (images_val, im_name) in enumerate(valloader):
                        # try:
                        save_img_ = random.choices([True, False], weights=[1, 0])[0]
                        # save_img_ = random.choices([True, False], weights=[0.2, 0.8])[0]

                        if save_img_:
                            images_val = Variable(images_val.cuda(self.args.gpu))

                            outputs, outputs_classify = self.model(images_val, is_softmax=True)
                            outputs_classify = outputs_classify.squeeze(1)

                            pred_regress = outputs.data.cpu().numpy().transpose(0, 2, 3, 1)
                            # pred_classify = outputs_classify.data.max(1)[1].cpu().numpy()       #     ==outputs.data.argmax(dim=0).cpu().numpy()
                            pred_classify = outputs_classify.data.round().int().cpu().numpy()  # (4, 1280, 1024)  ==outputs.data.argmax(dim=0).cpu().numpy()
                            perturbed_img = images_val.data.cpu().numpy().transpose(0, 2, 3, 1)

                            self.save_flat_mage.flatByRegressWithClassiy_multiProcess(pred_regress,
                                                                                      pred_classify, im_name,
                                                                                      epoch + 1,
                                                                                      scheme='test', is_scaling=is_scaling, perturbed_img=perturbed_img)
                        # except:
                        #     print('* save image tested error :' + im_name[0])

                test_time = time.time() - begin_test

                print('test time : {test_time:.3f}'.format(
                    test_time=test_time))

                print('test time : {test_time:.3f}'.format(
                    test_time=test_time),
                    file=self.reslut_file)


    def testModelV2GreyC1_index(self, epoch, train_time, index):
        begin_test = time.time()
        with torch.no_grad():
            for i_valloader, valloader in enumerate(self.testloaderSet.values()):

                for i_val, (images_val, im_name) in enumerate(valloader):
                    try:
                        # save_img_ = True
                        # save_img_ = random.choices([True, False], weights=[0.4, 0.6])[0]
                        # save_img_ = random.choices([True, False], weights=[0.2, 0.8])[0]
                        if im_name[0] in index:
                            images_val = Variable(images_val.cuda(self.args.gpu))

                            outputs, outputs_classify = self.model(images_val, is_softmax=True)
                            outputs_classify = outputs_classify.squeeze(1)
                            # outputs, outputs_classify = self.model(images_val, is_softmax=True)

                            pred_regress = outputs.data.cpu().numpy().transpose(0, 2, 3, 1)
                            # pred_classify = outputs_classify.data.max(1)[1].cpu().numpy()       #     ==outputs.data.argmax(dim=0).cpu().numpy()
                            pred_classify = outputs_classify.data.round().int().cpu().numpy()  # (4, 1280, 1024)  ==outputs.data.argmax(dim=0).cpu().numpy()
                            perturbed_img = images_val.data.cpu().numpy().transpose(0, 2, 3, 1)

                            # self.save_flat_mage.flatByRegressWithClassiy_triangular_v2_RGB_v2(pred_regress[0], pred_classify[0], im_name[0], epoch+1, groun_truth_path=self.data_path_test+'scan/scan/', scheme='test')
                            self.save_flat_mage.flatByRegressWithClassiy_triangular_v2_RGB(pred_regress[0], pred_classify[0], im_name[0], epoch+1, scheme='test', perturbed_img=perturbed_img)    # 'scan/scan/'
                        else:
                            continue
                    except:
                        print('* save image tested error :'+im_name[0])

            test_time = time.time() - begin_test

            print('test time : {test_time:.3f}\t'
                  'All Train Time : {train_time.sum:.3f}(s)\t'.format(
                test_time=test_time, train_time=train_time))

            print('test time : {test_time:.3f}\t'
                  'All Train Time : {train_time.sum:.3f}(s)\t'.format(
                test_time=test_time, train_time=train_time),
                file=self.reslut_file)

    def drawHeatmap(self, epoch, image, im_name='image.png'):
        epoch += 1
        # img_figure = sns.heatmap(pred_regress[0, :, :, 0], mask=pred_classify[0, :, :]-1, square=True)
        img_figure = sns.heatmap(image, square=True)
        i_path = os.path.join(self.path, self.date + self.date_time + ' @' + self._re_date, str(epoch)) \
            if self._re_date is not None else os.path.join(self.path, self.date + self.date_time, str(epoch))
        fig = img_figure.get_figure()
        fig.savefig(i_path + '/' + '0.png')
        # fig.savefig(i_path + '/' + im_name)
        # cv2.imwrite(i_path + '/' + im_name, img_figure)
