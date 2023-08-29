'''
code adapted from PyTorch examples
'''
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.bayesian.resnet_flipout_large as resnet
import models.deterministic.resnet_large as det_resnet
from torchsummary import summary
from utils import util
import csv
import numpy as np
from utils.util import get_rho
from torch.utils.tensorboard import SummaryWriter

torchvision.set_image_backend('accimage')

model_names = sorted(
    name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
    and name.startswith("resnet") and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data',
                    metavar='DIR',
                    default='data/imagenet',
                    help='path to dataset')
parser.add_argument('-a',
                    '--arch',
                    metavar='ARCH',
                    default='resnet50',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet50)')
parser.add_argument('-j',
                    '--workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=90,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--val_batch_size', default=1000, type=int)
parser.add_argument('-b',
                    '--batch-size',
                    default=32,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.001,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p',
                    '--print-freq',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained',
                    dest='pretrained',
                    action='store_true',
                    default=True,
                    help='use pre-trained model')
parser.add_argument('--world-size',
                    default=-1,
                    type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank',
                    default=-1,
                    type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url',
                    default='tcp://224.66.41.62:23456',
                    type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend',
                    default='nccl',
                    type=str,
                    help='distributed backend')
parser.add_argument('--seed',
                    default=None,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed',
                    action='store_true',
                    help='Use multi-processing distributed training to launch '
                    'N processes per node, which has N GPUs. This is the '
                    'fastest way to use PyTorch for either single node or '
                    'multi node data parallel training')
parser.add_argument('--mode', type=str, required=True, help='train | test')
parser.add_argument('--save-dir',
                    dest='save_dir',
                    help='The directory used to save the trained models',
                    default='./checkpoint/bayesian',
                    type=str)
parser.add_argument(
    '--tensorboard',
    type=bool,
    default=True,
    metavar='N',
    help='use tensorboard for logging and visualization of training progress')
parser.add_argument(
    '--log_dir',
    type=str,
    default='./logs/imagenet/bayesian',
    metavar='N',
    help='use tensorboard for logging and visualization of training progress')
parser.add_argument('--num_monte_carlo',
                    type=int,
                    default=50,
                    metavar='N',
                    help='number of Monte Carlo samples')
parser.add_argument(
    '--moped',
    type=bool,
    default=True,
    help='set prior and initialize approx posterior with Empirical Bayes')
parser.add_argument('--delta',
                    type=float,
                    default=0.2,
                    help='delta value for variance scaling in MOPED')

best_acc1 = 0
len_trainset = 1281167
len_valset = 50000
num_classes = 1000


def MOPED_layer(layer, det_layer, delta):
    """
    Set the priors and initialize surrogate posteriors of Bayesian NN with Empirical Bayes
    MOPED (Model Priors with Empirical Bayes using Deterministic DNN)

    Reference:
    [1] Ranganath Krishnan, Mahesh Subedar, Omesh Tickoo.
        Specifying Weight Priors in Bayesian Deep Neural Networks with Empirical Bayes. AAAI 2020.
    [2] Ranganath Krishnan, Mahesh Subedar, Omesh Tickoo.
        Efficient Priors for Scalable Variational Inference in Bayesian Deep Neural Networks. ICCV workshops 2019.
    """

    if (str(layer) == 'Conv2dFlipout()'
            or str(layer) == 'Conv2dReparameterization()'):
        #set the priors
        print(str(layer))
        layer.prior_weight_mu = det_layer.weight.data
        if layer.prior_bias_mu is not None:
            layer.prior_bias_mu = det_layer.bias.data

        #initialize surrogate posteriors
        layer.mu_kernel.data = det_layer.weight.data
        layer.rho_kernel.data = get_rho(det_layer.weight.data, delta)
        if layer.mu_bias is not None:
            layer.mu_bias.data = det_layer.bias.data
            layer.rho_bias.data = get_rho(det_layer.bias.data, delta)

    elif (isinstance(layer, nn.Conv2d)):
        print(str(layer))
        layer.weight.data = det_layer.weight.data
        if layer.bias is not None:
            layer.bias.data = det_layer.bias.data

    elif (str(layer) == 'LinearFlipout()'
          or str(layer) == 'LinearReparameterization()'):
        print(str(layer))
        layer.prior_weight_mu = det_layer.weight.data
        if layer.prior_bias_mu is not None:
            layer.prior_bias_mu = det_layer.bias.data

        #initialize the surrogate posteriors

        layer.mu_weight.data = det_layer.weight.data
        layer.rho_weight.data = get_rho(det_layer.weight.data, delta)
        if layer.mu_bias is not None:
            layer.mu_bias.data = det_layer.bias.data
            layer.rho_bias.data = get_rho(det_layer.bias.data, delta)

    elif str(layer).startswith('Batch'):
        #initialize parameters
        print(str(layer))
        layer.weight.data = det_layer.weight.data
        if layer.bias is not None:
            layer.bias.data = det_layer.bias.data
        layer.running_mean.data = det_layer.running_mean.data
        layer.running_var.data = det_layer.running_var.data
        layer.num_batches_tracked.data = det_layer.num_batches_tracked.data


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker,
                 nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    # define loss function (criterion) and optimizer
    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cpu()

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    tb_writer = None
    if args.tensorboard:
        logger_dir = os.path.join(args.log_dir, 'tb_logger')
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)
        tb_writer = SummaryWriter(logger_dir)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    print('len trainset: ', len(train_dataset))
    print('len valset: ', len(val_dataset))
    len_trainset = len(train_dataset)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.mode == 'train':

        if (args.moped):
            print("MOPED enabled")
            det_model = torch.nn.DataParallel(
                det_resnet.__dict__[args.arch](pretrained=True))
            det_model.cuda()

            for (idx_1, layer_1), (det_idx_1, det_layer_1) in zip(
                    enumerate(model.children()),
                    enumerate(det_model.children())):
                MOPED_layer(layer_1, det_layer_1, args.delta)
                for (idx_2, layer_2), (det_idx_2, det_layer_2) in zip(
                        enumerate(layer_1.children()),
                        enumerate(det_layer_1.children())):
                    MOPED_layer(layer_2, det_layer_2, args.delta)
                    for (idx_3, layer_3), (det_idx_3, det_layer_3) in zip(
                            enumerate(layer_2.children()),
                            enumerate(det_layer_2.children())):
                        MOPED_layer(layer_3, det_layer_3, args.delta)
                        for (idx_4, layer_4), (det_idx_4, det_layer_4) in zip(
                                enumerate(layer_3.children()),
                                enumerate(det_layer_3.children())):
                            MOPED_layer(layer_4, det_layer_4, args.delta)
                            for (idx_5,
                                 layer_5), (det_idx_5, det_layer_5) in zip(
                                     enumerate(layer_4.children()),
                                     enumerate(det_layer_4.children())):
                                MOPED_layer(layer_5, det_layer_5, args.delta)
                                for (idx_6,
                                     layer_6), (det_idx_6, det_layer_6) in zip(
                                         enumerate(layer_5.children()),
                                         enumerate(det_layer_5.children())):
                                    MOPED_layer(layer_6, det_layer_6,
                                                args.delta)

        model.state_dict()
        del det_model

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, args)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args,
                  tb_writer)

            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, epoch, args,
                            tb_writer)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if is_best:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                    },
                    is_best,
                    filename=os.path.join(
                        args.save_dir,
                        'bayesian_flipout_{}_imagenet.pth'.format(args.arch)))

    elif args.mode == 'test':

        checkpoint_file = args.save_dir + '/bayesian_flipout_{}_imagenet.pth'.format(
            args.arch)
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['state_dict'])
        evaluate(model, val_loader, args)


def train(train_loader, model, criterion, optimizer, epoch, args, tb_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    global opt_th
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        '''
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        '''
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output, kl = model(images)

        cross_entropy_loss = criterion(output, target)
        scaled_kl = (kl.data[0] / args.batch_size)
        elbo_loss = cross_entropy_loss + scaled_kl
        loss = cross_entropy_loss + scaled_kl

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        if tb_writer is not None:
            tb_writer.add_scalar('train/cross_entropy_loss',
                                 cross_entropy_loss.item(), epoch)
            tb_writer.add_scalar('train/kl_div', scaled_kl.item(), epoch)
            tb_writer.add_scalar('train/elbo_loss', elbo_loss.item(), epoch)
            tb_writer.add_scalar('train/loss', loss.item(), epoch)
            tb_writer.add_scalar('train/accuracy', acc1.item(), epoch)
            tb_writer.flush()


def validate(val_loader, model, criterion, epoch, args, tb_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    preds_list = []
    labels_list = []
    unc_list = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output, kl = model(images)
            cross_entropy_loss = criterion(output, target)
            scaled_kl = (kl.data[0] / args.batch_size)
            elbo_loss = cross_entropy_loss + scaled_kl
            loss = cross_entropy_loss + scaled_kl

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))

    return top1.avg


def evaluate(model, val_loader, args):
    pred_probs_mc = []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        pred_probs_mc = []
        output_list = []
        labels_list = []
        model.eval()
        begin = time.time()
        for batch_idx, (data, target) in enumerate(val_loader):
            #print('Batch idx {}, data shape {}, target shape {}'.format(batch_idx, data.shape, target.shape))
            if torch.cuda.is_available():
                data, target = data.cuda(non_blocking=True), target.cuda(
                    non_blocking=True)
            else:
                data, target = data.cpu(non_blocking=True), target.cpu(
                    non_blocking=True)
            data = torch.cat([data for _ in range(args.num_monte_carlo)], 0)
            output_mc = []
            output, _ = model.forward(data)
            output_mc.append(output)
            output_ = torch.stack(output_mc)
            output_ = output_.reshape(args.num_monte_carlo, -1, num_classes)
            output_list.append(output_)
            labels_list.append(target)

        end = time.time()
        print("inference throughput: ", len_valset / (end - begin),
              " images/s")

        output = torch.stack(output_list)
        output = output.permute(1, 0, 2, 3)
        output = output.contiguous().view(args.num_monte_carlo, len_valset, -1)
        output = torch.nn.functional.softmax(output, dim=2)
        labels = torch.cat(labels_list)
        pred_mean = output.mean(dim=0)
        Y_pred = torch.argmax(pred_mean, axis=1)

        print('Test accuracy:',
              (Y_pred.data.cpu().numpy() == labels.data.cpu().numpy()).mean() *
              100)

        #np.save(args.log_dir+'/bayesian_flipout_imagenet_probs.npy', output.data.cpu().numpy())
        #np.save(args.log_dir+'/bayesian_flipout_imagenet_labels.npy', labels.data.cpu().numpy())


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
