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
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
sys.path.append('ROOT')

from network import ResnetDilatedRgressAndClassifyV2v6v4c1GN
import utils as utils

from dataloader import PerturbedDatastsForRegressAndClassify_pickle_color_v2C1

def train(args):
    global _re_date
    if args.resume is not None:
        re_date = re.compile(r'\d{4}-\d{1,2}-\d{1,2}')
        _re_date = re_date.search(args.resume.name).group(0)
        reslut_file = open(path + '/' + date + date_time + ' @' + _re_date + '_' + args.arch + '.log', 'w')
    else:
        _re_date = None
        reslut_file = open(path+'/'+date+date_time+'_'+args.arch+'.log', 'w')

    # Setup Dataloader

    data_path_test = str(args.data_path_test)+'/'

    print(args)

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

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:'+str(args.gpu))
            model.load_state_dict(checkpoint['model_state'])
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    FlatImg = utils.FlatImg(args=args, path=path, date=date, date_time=date_time, _re_date=_re_date, data_split=None, model=model, \
                            reslut_file=reslut_file, n_classes=n_classes, optimizer=None, \
                            loss_fn=None, loss_classify_fn=None, data_loader=PerturbedDatastsForRegressAndClassify_pickle_color_v2C1, data_loader_hdf5=None, \
                            data_path=None, data_path_validate=None, data_path_test=data_path_test, data_preproccess=False)          # , valloaderSet=valloaderSet, v_loaderSet=v_loaderSet
    ''' load data '''
    FlatImg.loadTestData()

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

    reslut_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='Dewarping-Document-Image-By-Displacement-Flow-Estimation',
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')

    parser.add_argument('--dataset', nargs='?', type=str, default='v5',
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')

    parser.add_argument('--l_rate', nargs='?', type=float, default=0.0002,
                        help='Learning Rate')

    parser.add_argument('--resume', default=ROOT / '2019-06-25 11:52:54/49/2019-06-25 11:52:54flat_img_classifyAndRegress_grey-data1024_greyV2.pkl', type=str, 
                        help='Path to previous saved model to restart from') 
    
    parser.add_argument('--data_path_test', default=ROOT / 'dataset/shrink_1024_960/crop/', type=str,
                        help='the path of test images.')  # test image path

    parser.add_argument('--output-path', default=ROOT / 'flat/', type=str,
                        help='the path is used to  save output --img or result.')  # GPU id ---choose the GPU id that will be used

    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')#16

    parser.add_argument('--schema', type=str, default='test',
                        help='train or test')

    # parser.set_defaults(resume='2019-06-25 11:52:54/49/2019-06-25 11:52:54flat_img_classifyAndRegress_grey-data1024_greyV2.pkl')

    parser.add_argument('--parallel', default='6', type=list,
                        help='choice the gpu id for parallel ')

    args = parser.parse_args()

    if args.resume is not None:
        # args.resume = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.resume)
        if not os.path.isfile(args.resume):
            raise Exception(args.resume+' -- no find')
            
    if args.data_path_test is None:
        raise Exception('-- No test path')
    else:    
        if not os.path.exists(args.data_path_test):
            raise Exception(args.data_path_test+' -- no find')

    global path, date, date_time                # if load optimizerAndLoss_verified  ,this should be changed
    date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    date_time = time.strftime(' %H:%M:%S', time.localtime(time.time()))
    path = os.path.join(args.output_path, date)
    if not os.path.exists(path):
        os.makedirs(path)

    train(args)
