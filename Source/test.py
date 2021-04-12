'''
2019/6/25
guowang xie
'''
import sys, os
import time

import argparse
import torch
import re

import warnings
from network import ResnetDilatedRgressAndClassifyV2v6v4c1GN
import utils as utils

from perturbed_dataset import PerturbedDatastsForRegressAndClassify_pickle_color_v2C1

def train(args):
    global _re_date
    if args.resume is not None:
        re_date = re.compile(r'\d{4}-\d{1,2}-\d{1,2}')
        _re_date = re_date.search(args.resume).group(0)
        reslut_file = open(path + '/' + date + date_time + ' @' + _re_date + '_' + args.arch + '.log', 'w')
    else:
        _re_date = None
        reslut_file = open(path+'/'+date+date_time+'_'+args.arch+'.log', 'w')

    # Setup Dataloader

    data_split = 'data1024_greyV2'
    data_path = '/dataset/unwarp_new/train/'
    data_path_validate = '/dataset/unwarp_new/train/'+data_split+'/'
    data_path_test = '/dataset/'

    test_shrink_sub_dir = 'shrink_1024_960/crop/'

    args.arch = 'flat_img_classifyAndRegress_grey'
    args.dataset = data_split
    print(args)
    print(args, file=reslut_file)
    print('data_split :' + data_split)
    print('data_split :' + data_split, file=reslut_file)

    n_classes = 2

    '''network'''
    model = ResnetDilatedRgressAndClassifyV2v6v4c1GN(n_classes=n_classes, num_filter=32, BatchNorm='GN', in_channels=3) #


    if args.parallel is not None:
        device_ids = list(map(int, args.parallel))
        args.gpu = device_ids[0]
        if args.gpu < 8:
            torch.cuda.set_device(args.gpu)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        warnings.warn('no gpu , go sleep !')
        exit()

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.8, weight_decay=1e-12)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.l_rate, weight_decay=1e-10)     # 1e-12
    else:
        assert 'please choice optimizer'
        exit('error')

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            '''
            model_parameter_dick = {}
            for k in checkpoint['model_state']:
                model_parameter_dick['module.'+k] = checkpoint['model_state'][k]
            model.load_state_dict(model_parameter_dick)
            '''

            # optimizer.load_state_dict(checkpoint['optimizer_state'])          # 1 why runing error 2 alter the optimizer of original program,because which optimizer was changing as the operaion
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))


    FlatImg = utils.FlatImg(args=args, path=path, date=date, date_time=date_time, _re_date=_re_date, data_split=data_split, model=model, \
                            reslut_file=reslut_file, n_classes=n_classes, optimizer=optimizer, \
                            loss_fn=None, loss_classify_fn=None, data_loader=PerturbedDatastsForRegressAndClassify_pickle_color_v2C1, data_loader_hdf5=None, \
                            data_path=data_path, data_path_validate=data_path_validate, data_path_test=data_path_test, data_preproccess=False)          # , valloaderSet=valloaderSet, v_loaderSet=v_loaderSet
    ''' load data '''
    # train_loader = data_loader(data_path, split=data_split, img_shrink=args.img_shrink)
    # trainloader = data.DataLoader(train_loader, batch_size=args.batch_size, num_workers=args.batch_size//2, shuffle=True)

    # trainloader = FlatImg.loadTrainData(data_split=data_split, is_shuffle=True)
    # FlatImg.loadValidateAndTestData(is_shuffle=True, sub_dir=test_shrink_sub_dir)
    FlatImg.loadTestData(is_shuffle=True, sub_dir=test_shrink_sub_dir)


    train_time = AverageMeter()

    if args.schema == 'test':
        epoch = checkpoint['epoch'] if args.resume is not None else 0
        model.eval()
        FlatImg.validateOrTestModelV2GreyC1(epoch, 0, validate_test='t')
        exit()
    elif args.schema == 'eval':
        epoch = checkpoint['epoch'] if args.resume is not None else 0
        model.eval()
        FlatImg.evalModelGreyC1(epoch, 0, is_scaling=False)
        exit()
    elif args.schema == 'scaling':
        epoch = checkpoint['epoch'] if args.resume is not None else 0
        model.eval()
        FlatImg.validateOrTestModelV2GreyC1(epoch, 0, validate_test='t', is_scaling=True)
        exit()

    m, s = divmod(train_time.sum, 60)
    h, m = divmod(m, 60)
    print("All Train Time : %02d:%02d:%02d\n" % (h, m, s))
    print("All Train Time : %02d:%02d:%02d\n" % (h, m, s), file=reslut_file)

    reslut_file.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, args_lr, args_momentum, args_parallel):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args_lr * (0.2 ** (epoch // 30))   # ** == math.pow      // == math.floor
    # momentum = args_momentum * (0.6 ** (epoch // 30))
    for param_group in optimizer.param_groups:
    # for param_group in optimizer.param_groups if args_parallel is None else optimizer.module.param_groups:
        param_group['l_rate'] = lr
        # param_group['momentum'] = momentum


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='flat_unet',
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')

    parser.add_argument('--dataset', nargs='?', type=str, default='v5',
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')

    parser.add_argument('--img_shrink', nargs='?', type=int, default=None,
                        help='short edge of the input image')

    parser.add_argument('--n_epoch', nargs='?', type=int, default=300,
                        help='# of the epochs')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimization')
    parser.add_argument('--l_rate', nargs='?', type=float, default=0.0002,
                        help='Learning Rate')

    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')            # python segmentation_train.py --resume=./trained_model/fcn8s_pascla_2018-8-04_model.pkl

    parser.add_argument('--print-freq', '-p', default=320, type=int,
                        metavar='N', help='print frequency (default: 10)')  # print frequency

    parser.add_argument('--output-path', default='./flat/', type=str,
                        help='the path is used to  save output --img or result.')  # GPU id ---choose the GPU id that will be used

    parser.add_argument('--batch_size', nargs='?', type=int, default=6,
                        help='Batch Size')#16

    parser.add_argument('--schema', type=str, default='test',
                        help='train or test')

    parser.set_defaults(resume='/2019-06-25 11:52:54/49/2019-06-25 11:52:54flat_img_classifyAndRegress_grey-data1024_greyV2.pkl')

    parser.add_argument('--parallel', default='3', type=list,
                        help='choice the gpu id for parallel ')

    args = parser.parse_args()

    if args.resume is not None:
        # args.resume = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.resume)
        if not os.path.isfile(args.resume):
            raise Exception(args.resume+' -- not exist')

    global path, date, date_time                # if load optimizerAndLoss_verified  ,this should be changed
    date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    date_time = time.strftime(' %H:%M:%S', time.localtime(time.time()))
    path = os.path.join(args.output_path, date)
    # _re_date = None

    if not os.path.exists(path):
        os.makedirs(path)
        # os.mkdir(path)

    train(args)
