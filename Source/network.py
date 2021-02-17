import torch.nn as nn
import torch
# from xgw.segmentation.xgw_models.xgw_model_utils import *
import torch.nn.init as tinit
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, stride=1):
	return nn.Conv2d(in_channels, out_channels, kernel_size=3,  stride=stride, padding=1)

def transitionUpGN(in_channels, out_dim, act_fn, BatchNorm, GN_num=32):
	model = nn.Sequential(
		# nn.ConvTranspose2d(in_channels, out_dim, kernel_size=2, stride=2),
		nn.ConvTranspose2d(in_channels, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
		BatchNorm(GN_num, out_dim) if out_dim > GN_num else nn.InstanceNorm2d(out_dim),
		# nn.BatchNorm2d(out_dim),
		act_fn,
	)
	return model
def dilation_conv(in_channels, out_dim, stride=1, dilation=4, groups=1):
	model = nn.Sequential(
		nn.Conv2d(in_channels, out_dim, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups),
	)
	return model

def dilation_conv_gn_act(in_channels, out_dim, act_fn, BatchNorm, GN_num=32, dilation=4):
	model = nn.Sequential(
		nn.Conv2d(in_channels, out_dim, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
		BatchNorm(GN_num, out_dim),
		# nn.BatchNorm2d(out_dim),
		act_fn,
	)
	return model
def upsamplingBilinear(scale_factor=2):
	model = nn.Sequential(
		nn.UpsamplingBilinear2d(scale_factor=scale_factor),
		# nn.BatchNorm2d(out_dim),
		# act_fn,
	)
	return model

class ConvBlockResidualGN(nn.Module):
	def __init__(self, in_channels, out_channels, act_fn, BatchNorm, GN_num=32, is_dropout=False):
		super(ConvBlockResidualGN, self).__init__()
		self.conv1 = conv3x3(in_channels, out_channels)
		self.bn1 = BatchNorm(GN_num, out_channels)
		# self.bn1 = BatchNorm(out_channels)
		# self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = act_fn
		self.conv2 = conv3x3(out_channels, out_channels)
		self.bn2 = BatchNorm(GN_num, out_channels)
		# self.bn2 = BatchNorm(out_channels)
		# self.bn2 = nn.BatchNorm2d(out_channels)
		# self.downsample = downsample
		self.is_dropout = is_dropout
		self.drop_out = nn.Dropout2d(p=0.2)

	def forward(self, x):
		residual = x
		out = self.relu(self.bn1(self.conv1(x)))
		if self.is_dropout:
			out = self.drop_out(out)
		out = self.bn2(self.conv2(out))
		# if self.downsample is not None:
		# 	residual = self.downsample(x)
		out += residual
		out = self.relu(out)
		if self.is_dropout:
			out = self.drop_out(out)
		return out

class ResidualBlock34DilatedV4GN(nn.Module):
	def __init__(self, in_channels, out_channels, BatchNorm, GN_num=32, stride=1, downsample=None, is_activation=True, is_top=False, is_dropout=False):
		super(ResidualBlock34DilatedV4GN, self).__init__()
		self.stride = stride
		self.is_activation = is_activation
		self.downsample = downsample
		self.is_top = is_top
		if self.stride != 1 or self.is_top:
			self.conv1 = conv3x3(in_channels, out_channels, self.stride)
		else:
			self.conv1 = dilation_conv(in_channels, out_channels, dilation=1)
		# self.bn1 = BatchNorm(GN_num, out_channels)
		self.bn1 = BatchNorm(GN_num, out_channels) if out_channels > GN_num else nn.InstanceNorm2d(out_channels)
		# self.bn1 = BatchNorm(out_channels)
		# self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		if self.stride != 1 or self.is_top:
			self.conv2 = conv3x3(out_channels, out_channels)
		else:
			self.conv2 = dilation_conv(out_channels, out_channels, dilation=3)
		# self.bn2 = BatchNorm(GN_num, out_channels)
		self.bn2 = BatchNorm(GN_num, out_channels) if out_channels > GN_num else nn.InstanceNorm2d(out_channels)
		# self.bn2 = BatchNorm(out_channels)
		if self.stride == 1 and not self.is_top:
			self.conv3 = dilation_conv(out_channels, out_channels, dilation=1)
			# self.bn3 = BatchNorm(GN_num, out_channels)
			self.bn3 = BatchNorm(GN_num, out_channels) if out_channels > GN_num else nn.InstanceNorm2d(out_channels)
			# self.bn3 = BatchNorm(out_channels)
		# self.bn2 = nn.BatchNorm2d(out_channels)
		self.is_dropout = is_dropout
		self.drop_out = nn.Dropout(p=0.2)
		# self.drop_out = nn.Dropout2d(p=0.2)

	def forward(self, x):
		residual = x

		out1 = self.relu(self.bn1(self.conv1(x)))
		# if self.is_dropout:
		# 	out1 = self.drop_out(out1)
		out = self.bn2(self.conv2(out1))
		if self.downsample is not None:
			residual = self.downsample(x)
		out += residual
		out = self.relu(out)

		if self.stride == 1 and not self.is_top:
			if self.is_dropout:
				out = self.drop_out(out)
			out = self.bn3(self.conv3(out))
			out += out1
			if self.is_activation:
				out = self.relu(out)

		return out

class ResNetV2StraightV2GN(nn.Module):
	def __init__(self, num_filter, map_num, BatchNorm, GN_num=[32, 32, 32, 32], block_nums=[3, 4, 6, 3], block=ResidualBlock34DilatedV4GN, stride=[1, 2, 2, 2], dropRate=[0.2, 0.2, 0.2, 0.2], is_sub_dropout=False):
		super(ResNetV2StraightV2GN, self).__init__()
		self.in_channels = num_filter * map_num[0]
		self.dropRate = dropRate
		self.stride = stride
		self.is_sub_dropout = is_sub_dropout
		# self.is_dropout = is_dropout
		self.drop_out = nn.Dropout(p=dropRate[0])		# nn.Dropout2d(p=dropRate[0])
		self.drop_out_2 = nn.Dropout(p=dropRate[1])		# nn.Dropout2d(p=dropRate[1])
		self.drop_out_3 = nn.Dropout(p=dropRate[2])		# nn.Dropout2d(p=dropRate[2])
		self.drop_out_4 = nn.Dropout(p=dropRate[3])		# nn.Dropout2d(p=dropRate[3])
		self.relu = nn.ReLU(inplace=True)
		# self.conv = conv3x3(3, 16)
		# self.bn = nn.BatchNorm2d(16)
		# self.relu = nn.ReLU(inplace=True)
		self.block_nums = block_nums
		self.layer1 = self.blocklayer(block, num_filter * map_num[0], self.block_nums[0], BatchNorm, GN_num=GN_num[0], stride=self.stride[0])
		self.layer2 = self.blocklayer(block, num_filter * map_num[1], self.block_nums[1], BatchNorm, GN_num=GN_num[1], stride=self.stride[1])
		self.layer3 = self.blocklayer(block, num_filter * map_num[2], self.block_nums[2], BatchNorm, GN_num=GN_num[2], stride=self.stride[2])
		self.layer4 = self.blocklayer(block, num_filter * map_num[3], self.block_nums[3], BatchNorm, GN_num=GN_num[3], stride=self.stride[3])

	def blocklayer(self, block, out_channels, block_nums, BatchNorm, GN_num=32, stride=1):
		downsample = None
		if (stride != 1) or (self.in_channels != out_channels):
			downsample = nn.Sequential(
				# conv7x7(self.in_channels, out_channels, stride=stride),
				conv3x3(self.in_channels, out_channels, stride=stride),
				BatchNorm(GN_num, out_channels))
				# BatchNorm(out_channels))
				# nn.BatchNorm2d(out_channels))
		layers = []
		layers.append(block(self.in_channels, out_channels, BatchNorm, GN_num, stride, downsample, is_top=True, is_dropout=False))
		self.in_channels = out_channels
		for i in range(1, block_nums):
			layers.append(block(out_channels, out_channels, BatchNorm, GN_num, is_activation=True, is_top=False, is_dropout=self.is_sub_dropout))
		return nn.Sequential(*layers)

	def forward(self, x):
	# def forward(self, x, is_skip=False):

		out1 = self.layer1(x)

		# if self.dropRate[0] > 0:
		# 	out1 = self.drop_out(out1)

		out2 = self.layer2(out1)

		# if self.dropRate[1] > 0:
		# 	out2 = self.drop_out_2(out2)

		out3 = self.layer3(out2)

		# if self.dropRate[2] > 0:
		# 	out3 = self.drop_out_3(out3)

		out4 = self.layer4(out3)

		# if self.dropRate[3] > 0:
		# 	out4 = self.drop_out_4(out4)

		# if is_skip:
		# 	return out4, out1
		return out4

class DilatedResnetForFlatByClassifyWithRgressV2v6v4c1GN(nn.Module):

	def __init__(self, n_classes, num_filter, BatchNorm, in_channels=3):
		super(DilatedResnetForFlatByClassifyWithRgressV2v6v4c1GN, self).__init__()
		self.in_channels = in_channels
		self.n_classes = n_classes
		self.num_filter = num_filter
		# act_fn = nn.PReLU()
		act_fn = nn.ReLU(inplace=True)
		# act_fn = nn.LeakyReLU(0.2)

		map_num = [1, 2, 4, 8, 16]
		GN_num = self.num_filter * map_num[0]

		is_dropout = False
		print("\n------load DilatedResnetForFlatByClassifyWithRgressV2v6v4c1GN------\n")

		self.resnet_head = nn.Sequential(
			nn.Conv2d(self.in_channels, self.num_filter * map_num[0], kernel_size=7, stride=1, padding=3),
			nn.InstanceNorm2d(self.num_filter * map_num[0]),
			# BatchNorm(1, self.num_filter * map_num[0]),
			# BatchNorm(self.num_filter * map_num[0]),
			# nn.BatchNorm2d(self.num_filter * map_num[0]),
			act_fn,
			# nn.Dropout(p=0.2),
			nn.Conv2d(self.num_filter * map_num[0], self.num_filter * map_num[0], kernel_size=7, stride=2, padding=3),
			nn.InstanceNorm2d(self.num_filter * map_num[0]),
			# BatchNorm(1, self.num_filter * map_num[0]),
			# BatchNorm(self.num_filter * map_num[0]),
			# nn.BatchNorm2d(self.num_filter * map_num[0]),
			act_fn,
			# nn.Dropout(p=0.2),
			# nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
			nn.Conv2d(self.num_filter * map_num[0], self.num_filter * map_num[0], kernel_size=7, stride=2, padding=3),
			nn.InstanceNorm2d(self.num_filter * map_num[0]),
			# BatchNorm(1, self.num_filter * map_num[0]),
			# BatchNorm(self.num_filter * map_num[0]),
			# nn.BatchNorm2d(self.num_filter * map_num[0]),
			act_fn,
			# nn.Dropout(p=0.2),

		)
		# self.resnet_down = ResNetGN(num_filter, map_num, BatchNorm, GN_num=[32, 32, 32, 32], block_nums=[3, 4, 6, 3], block=ResidualBlock34GN, dropRate=[0, 0, 0, 0])
		self.resnet_down = ResNetV2StraightV2GN(num_filter, map_num, BatchNorm, GN_num=[GN_num, GN_num, GN_num, GN_num], block_nums=[3, 4, 6, 3], block=ResidualBlock34DilatedV4GN, dropRate=[0, 0, 0, 0], is_sub_dropout=False)

		map_num_i = 3
		self.bridge_1 = nn.Sequential(
			dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								 act_fn, BatchNorm, GN_num=GN_num, dilation=1),
			# conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i], act_fn),
		)
		self.bridge_2 = nn.Sequential(
			dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, GN_num=GN_num, dilation=2),
			# conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i], act_fn),
		)
		self.bridge_3 = nn.Sequential(
			dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, GN_num=GN_num, dilation=5),
			# conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i], act_fn),
		)
		self.bridge_4 = nn.Sequential(
			dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, GN_num=GN_num, dilation=8),
			dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, GN_num=GN_num, dilation=3),
			dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, GN_num=GN_num, dilation=2),
		)
		self.bridge_5 = nn.Sequential(
			dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, GN_num=GN_num, dilation=12),
			dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, GN_num=GN_num, dilation=7),
			dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, GN_num=GN_num, dilation=4),
		)
		self.bridge_6 = nn.Sequential(
			dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, GN_num=GN_num, dilation=18),
			dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, GN_num=GN_num, dilation=12),
			dilation_conv_gn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, GN_num=GN_num, dilation=6),
		)

		self.bridge_concate = nn.Sequential(
			nn.Conv2d(self.num_filter * map_num[map_num_i] * 6, self.num_filter * map_num[4], kernel_size=1, stride=1, padding=0),
			BatchNorm(GN_num, self.num_filter * map_num[4]),
			# BatchNorm(self.num_filter * map_num[4]),
			# nn.BatchNorm2d(self.num_filter * map_num[4]),
			act_fn,
		)

		self.regess_4 = ConvBlockResidualGN(self.num_filter * map_num[4], self.num_filter * (map_num[4]), act_fn, BatchNorm, GN_num=GN_num, is_dropout=False)
		# self.regess_4 = ConvBlockV2(self.num_filter * map_num[4], self.num_filter * (map_num[3]), act_fn, BatchNorm, is_dropout=False)
		# self.regess_4 = MergeBlockV2(self.num_filter * map_num[4], self.num_filter * (map_num[3]), n_classes, act_fn, BatchNorm, is_dropout=False)
		# self.trans_4 = upsamplingBilinear(scale_factor=2)
		self.trans_4 = transitionUpGN(self.num_filter * (map_num[4]), self.num_filter * map_num[3], act_fn, BatchNorm, GN_num=GN_num)

		self.regess_3 = ConvBlockResidualGN(self.num_filter * (map_num[3]), self.num_filter * (map_num[3]), act_fn, BatchNorm, GN_num=GN_num, is_dropout=False)
		# self.regess_3 = ConvBlockV2(self.num_filter * (map_num[3]), self.num_filter * (map_num[2]), act_fn, BatchNorm, is_dropout=False)
		# self.regess_3 = MergeBlockV2(self.num_filter * (map_num[3]), self.num_filter * (map_num[2]), n_classes, act_fn, BatchNorm, is_dropout=False)
		# self.trans_3 = upsamplingBilinear(scale_factor=2)
		self.trans_3 = transitionUpGN(self.num_filter * (map_num[3]), self.num_filter * map_num[2], act_fn, BatchNorm, GN_num=GN_num)

		self.regess_2 = ConvBlockResidualGN(self.num_filter * (map_num[2]), self.num_filter * (map_num[2]), act_fn, BatchNorm, GN_num=GN_num, is_dropout=False)
		# self.regess_2 = ConvBlockV2(self.num_filter * (map_num[2]), self.num_filter * (map_num[2]), act_fn, BatchNorm, is_dropout=False)
		# self.regess_2 = MergeBlockV2(self.num_filter * (map_num[2]), self.num_filter * (map_num[2]), n_classes, act_fn, BatchNorm, is_dropout=False)
		# self.trans_2 = upsamplingBilinear(scale_factor=2)
		self.trans_2 = transitionUpGN(self.num_filter * map_num[2], self.num_filter * map_num[1], act_fn, BatchNorm, GN_num=GN_num)

		self.regess_1 = ConvBlockResidualGN(self.num_filter * (map_num[1]), self.num_filter * (map_num[1]), act_fn, BatchNorm, GN_num=GN_num, is_dropout=False)
		# self.regess_1 = ConvBlockV2(self.num_filter * (map_num[2]), self.num_filter * (map_num[1]), act_fn, BatchNorm, is_dropout=False)
		# self.regess_1 = MergeBlockV2(self.num_filter * (map_num[2]), self.num_filter * (map_num[1]), n_classes, act_fn, BatchNorm, is_dropout=False)
		self.trans_1 = upsamplingBilinear(scale_factor=2)
		# self.trans_1 = transitionUpGN(self.num_filter * map_num[1], self.num_filter * map_num[1], act_fn, BatchNorm, GN_num=GN_num)

		self.regess_0 = ConvBlockResidualGN(self.num_filter * (map_num[1]), self.num_filter * (map_num[1]), act_fn, BatchNorm, GN_num=GN_num, is_dropout=False)
		# self.regess_0 = ConvBlockV2(self.num_filter * (map_num[1]), self.num_filter * (map_num[0]), act_fn, BatchNorm, is_dropout=False)
		# self.regess_0 = MergeBlockV2(self.num_filter * (map_num[1]), self.num_filter * (map_num[0]), n_classes, act_fn, BatchNorm, is_dropout=False)
		# self.trans_0 = transitionUpGN(self.num_filter * map_num[1], self.num_filter * map_num[0], act_fn, BatchNorm, GN_num=GN_num)
		self.trans_0 = upsamplingBilinear(scale_factor=2)
		self.up = nn.Sequential(
			nn.Conv2d(self.num_filter * map_num[1], self.num_filter * map_num[0], kernel_size=3, stride=1, padding=1),
			# BatchNorm(GN_num, self.num_filter * map_num[0]),
			nn.InstanceNorm2d(self.num_filter * map_num[0]),
			# nn.BatchNorm2d(self.num_filter * map_num[0]),
			act_fn,
			# nn.Dropout2d(p=0.2),
		)
		self.out_regress = nn.Sequential(
			# nn.Conv2d(map_num[0], map_num[0], kernel_size=3, stride=1, padding=1, bias=False),
			# nn.InstanceNorm2d(map_num[0]),
			# nn.PReLU(),
			nn.Conv2d(self.num_filter * map_num[0], self.num_filter * map_num[0], kernel_size=3, stride=1, padding=1),
			# BatchNorm(GN_num, self.num_filter * map_num[0]),
			nn.InstanceNorm2d(self.num_filter * map_num[0]),
			nn.PReLU(),
			# nn.Dropout2d(p=0.2),
			nn.Conv2d(self.num_filter * map_num[0], n_classes, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm2d(n_classes),
			# BatchNorm(1, n_classes),
			# BatchNorm(n_classes),
			nn.PReLU(),
			nn.Conv2d(n_classes, n_classes, kernel_size=3, stride=1, padding=1),
			#nn.Conv2d(map_num[0], n_classes, kernel_size=3, stride=1, padding=1, bias=False),
		)

		self.out_classify = nn.Sequential(
			# nn.Conv2d(map_num[0], map_num[0], kernel_size=3, stride=1, padding=1, bias=False),
			# nn.InstanceNorm2d(map_num[0]),
			# act_fn,
			# nn.Dropout(p=0.2),
			nn.Conv2d(self.num_filter * map_num[0], self.num_filter * map_num[0], kernel_size=3, stride=1, padding=1),
			# BatchNorm(GN_num, self.num_filter * map_num[0]),
			nn.InstanceNorm2d(self.num_filter * map_num[0]),
			act_fn,
			nn.Dropout2d(p=0.2),		# nn.Dropout(p=0.2),
			nn.Conv2d(self.num_filter * map_num[0], n_classes, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm2d(n_classes),
			# BatchNorm(1, n_classes),
			# BatchNorm(n_classes),
			act_fn,
			# nn.Dropout(p=0.2),
			nn.Conv2d(n_classes, 1, kernel_size=3, stride=1, padding=1),
			#nn.Conv2d(self.num_filter * map_num[0], n_classes, kernel_size=3, stride=1, padding=1, bias=False),
		)

		self.out_classify_softmax = nn.Sigmoid()
		#self.out_classify_softmax = nn.Softmax2d()

		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				tinit.xavier_normal_(m.weight, gain=0.2)
				# if m.bias is not None:
				# 	tinit.constant_(m.bias, 0)
					# tinit.constant_(m.bias, 0.1)
			if isinstance(m, nn.ConvTranspose2d):
				assert m.kernel_size[0] == m.kernel_size[1]
				tinit.xavier_normal_(m.weight, gain=0.2)
				# if m.bias is not None:
				# 	tinit.constant_(m.bias, 0)
					# tinit.constant_(m.bias, 0.1)

	def forward(self, x, is_softmax):
		resnet_head = self.resnet_head(x)
		resnet_down = self.resnet_down(resnet_head)
		#
		'''bridge'''
		bridge_1 = self.bridge_1(resnet_down)
		bridge_2 = self.bridge_2(resnet_down)
		bridge_3 = self.bridge_3(resnet_down)
		bridge_4 = self.bridge_4(resnet_down)
		bridge_5 = self.bridge_5(resnet_down)
		bridge_6 = self.bridge_6(resnet_down)

		# bridge_concate = torch.cat([bridge_1, bridge_2, bridge_3], dim=1)
		# bridge_concate = torch.cat([bridge_1, bridge_2, bridge_3, bridge_4], dim=1)
		# bridge_concate = torch.cat([bridge_1, bridge_2, bridge_3, bridge_4, bridge_5], dim=1)
		bridge_concate = torch.cat([bridge_1, bridge_2, bridge_3, bridge_4, bridge_5, bridge_6], dim=1)
		bridge = self.bridge_concate(bridge_concate)
		# bridge = self.bridge(pool_3)
		# bridge = self.bridge(pool_4)
		# bridge = self.bridge(pool_6)

		regess_4 = self.regess_4(bridge)
		trans_4 = self.trans_4(regess_4)

		regess_3 = self.regess_3(trans_4)
		trans_3 = self.trans_3(regess_3)

		regess_2 = self.regess_2(trans_3)
		trans_2 = self.trans_2(regess_2)

		regess_1 = self.regess_1(trans_2)
		trans_1 = self.trans_1(regess_1)

		regess_0 = self.regess_0(trans_1)
		trans_0 = self.trans_0(regess_0)
		up = self.up(trans_0)

		out_regress = self.out_regress(up)
		out_classify = self.out_classify(up)

		if is_softmax:
			out_classify = self.out_classify_softmax(out_classify)

		return out_regress, out_classify

class ResnetDilatedRgressAndClassifyV2v6v4c1GN(nn.Module):
	def __init__(self, n_classes, num_filter, BatchNorm='GN', in_channels=3):
		super(ResnetDilatedRgressAndClassifyV2v6v4c1GN, self).__init__()
		self.in_channels = in_channels
		self.n_classes = n_classes
		self.num_filter = num_filter
		BatchNorm = nn.GroupNorm


		self.dilated_unet = DilatedResnetForFlatByClassifyWithRgressV2v6v4c1GN(self.n_classes, self.num_filter, BatchNorm, in_channels=self.in_channels)

	def forward(self, x, is_softmax=False):
		return self.dilated_unet(x, is_softmax)



