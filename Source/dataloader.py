import os
import pickle
import random
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np

import re
import cv2

from torch.utils import data
def getDatasets(dir):
	if not os.path.exists(dir):
	    raise Exception(dir+' -- path no find')
	return os.listdir(dir)

'''
Resize the input image into 1024x960 (zooming in or out along the longest side and keeping the aspect ration, then filling zero for padding. )
'''
def resize_image(origin_img, long_edge=1024, short_edge=960):
	# long_edge, short_edge = 2048, 1920
	# long_edge, short_edge = 1024, 960
	# long_edge, short_edge = 512, 480

	im_lr = origin_img.shape[0]
	im_ud = origin_img.shape[1]
	new_img = np.zeros([long_edge, short_edge, 3], dtype=np.uint8)
	new_shape = new_img.shape[:2]
	if im_lr > im_ud:
		img_shrink, base_img_shrink = long_edge, long_edge
		im_ud = int(im_ud / im_lr * base_img_shrink)
		im_ud += 32-im_ud%32
		im_ud = min(im_ud, short_edge)
		im_lr = img_shrink
		origin_img = cv2.resize(origin_img, (im_ud, im_lr), interpolation=cv2.INTER_CUBIC)
		new_img[:, (new_shape[1]-im_ud)//2:new_shape[1]-(new_shape[1]-im_ud)//2] = origin_img
		# mask = np.full(new_shape, 255, dtype='uint8')
		# mask[:, (new_shape[1] - im_ud) // 2:new_shape[1] - (new_shape[1] - im_ud) // 2] = 0
	else:
		img_shrink, base_img_shrink = short_edge, short_edge
		im_lr = int(im_lr / im_ud * base_img_shrink)
		im_lr += 32-im_lr%32
		im_lr = min(im_lr, long_edge)
		im_ud = img_shrink
		origin_img = cv2.resize(origin_img, (im_ud, im_lr), interpolation=cv2.INTER_CUBIC)
		new_img[(new_shape[0] - im_lr) // 2:new_shape[0] - (new_shape[0] - im_lr) // 2, :] = origin_img
	return new_img

class PerturbedDatastsForRegressAndClassify_pickle_color_v2C1(data.Dataset):
	def __init__(self, root, split='1-1', img_shrink=None, is_return_img_name=False, preproccess=False):
		self.root = os.path.expanduser(root)
		self.split = split
		self.img_shrink = img_shrink
		self.is_return_img_name = is_return_img_name
		self.preproccess = preproccess
		# self.mean = np.array([104.00699, 116.66877, 122.67892])
		self.images = collections.defaultdict(list)
		self.labels = collections.defaultdict(list)

		datasets = ['validate', 'test', 'train']

		if self.split == 'test':
			img_file_list = getDatasets(os.path.join(self.root))

			self.images[self.split] = sorted(img_file_list, key=lambda num: (
			int(re.match(r'(\d+)_(\d+)( copy.png)', num, re.IGNORECASE).group(1)), int(re.match(r'(\d+)_(\d+)( copy.png)', num, re.IGNORECASE).group(2))))
		elif self.split in datasets:
			img_file_list = []
			img_file_list_ = getDatasets(os.path.join(self.root))
			for id_ in img_file_list_:
				img_file_list.append(id_.rstrip())

			self.images[self.split] = sorted(img_file_list, key=lambda num: (
			re.match(r'(\w+\d*)_(\d+)_(\d+)_(\w+)', num, re.IGNORECASE).group(1), int(re.match(r'(\w+\d*)_(\d+)_(\d+)_(\w+)', num, re.IGNORECASE).group(2))
			, int(re.match(r'(\w+\d*)_(\d+)_(\d+)_(\w+)', num, re.IGNORECASE).group(3)), re.match(r'(\w+\d*)_(\d+)_(\d+)_(\w+)', num, re.IGNORECASE).group(4)))
		else:
			raise Exception('load data error')
		# self.checkImg()

	def checkImg(self):
		if self.split == 'validate':
			for im_name in self.images[self.split]:
				# if 'SinglePage' in im_name:
				im_path = pjoin(self.root, im_name)
				try:
					with open(im_path, 'rb') as f:
						perturbed_data = pickle.load(f)

					im_shape = perturbed_data.shape
				except:
					print(im_name)
					# os.remove(im_path)

	def __len__(self):
		return len(self.images[self.split])

	def __getitem__(self, item):
		if self.split == 'test':
			im_name = self.images[self.split][item]
			im_path = pjoin(self.root, im_name)

			im = cv2.imread(im_path, flags=cv2.IMREAD_COLOR)

			im = resize_image(im)
			im = self.transform_im(im)

			if self.is_return_img_name:
				return im, im_name
			return im
		else:
			im_name = self.images[self.split][item]

			im_path = pjoin(self.root, im_name)

			with open(im_path, 'rb') as f:
				perturbed_data = pickle.load(f)

			im = perturbed_data[:, :, 0:3]
			lbl = perturbed_data[:, :, 3:5]
			lbl_classify = perturbed_data[:, :, 5]

			im = im.transpose(2, 0, 1)
			lbl = lbl.transpose(2, 0, 1)
			# lbl[lbl == -1] = 0

			im = torch.from_numpy(im).float()
			lbl = torch.from_numpy(lbl).float()
			lbl_classify = torch.from_numpy(lbl_classify).float()

			if self.is_return_img_name:
				return im, lbl, lbl_classify, im_name

			return im, lbl, lbl_classify

	def transform_im(self, im):
		im = im.transpose(2, 0, 1)
		im = torch.from_numpy(im).float()

		return im
