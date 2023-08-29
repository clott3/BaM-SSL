import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import bayesian_torch.models.deterministic.resnet as resnet
import numpy as np

model_names = sorted(
    name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
    and name.startswith("resnet") and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='CIFAR10')
parser.add_argument('--arch',
                    '-a',
                    metavar='ARCH',
                    default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet20)')
parser.add_argument('-j',
                    '--workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=512,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 512)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=5e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq',
                    '-p',
                    default=50,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 20)')
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
                    help='use pre-trained model')
parser.add_argument('--half',
                    dest='half',
                    action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir',
                    dest='save_dir',
                    help='The directory used to save the trained models',
                    default='./checkpoint/deterministic',
                    type=str)
parser.add_argument(
    '--save-every',
    dest='save_every',
    help='Saves checkpoints at every specified number of epochs',
    type=int,
    default=10)
parser.add_argument('--mode', type=str, required=True, help='train | test')
parser.add_argument(
    '--tensorboard',
    type=bool,
    default=True,
    metavar='N',
    help='use tensorboard for logging and visualization of training progress')
parser.add_argument(
    '--log_dir',
    type=str,
    default='./logs/cifar/deterministic',
    metavar='N',
    help='use tensorboard for logging and visualization of training progress')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    tb_writer = None
    if args.tensorboard:
        logger_dir = os.path.join(args.log_dir, 'tb_logger')
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)
        tb_writer = SummaryWriter(logger_dir)

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
        root='./data',
        train=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]),
        download=True),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
        root='./data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cpu()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], last_epoch=args.start_epoch - 1)

    if args.arch in ['resnet110']:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.1

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    if args.mode == 'train':
        for epoch in range(args.start_epoch, args.epochs):

            # train for one epoch
            print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
            train(args, train_loader, model, criterion, optimizer, epoch,
                  tb_writer)
            lr_scheduler.step()

            prec1 = validate(args, val_loader, model, criterion, epoch,
                             tb_writer)

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            if epoch > 0 and epoch % args.save_every == 0:
                if is_best:
                    save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'best_prec1': best_prec1,
                        },
                        is_best,
                        filename=os.path.join(
                            args.save_dir, '{}_cifar.pth'.format(args.arch)))

    elif args.mode == 'test':
        checkpoint_file = args.save_dir + '/{}_cifar.pth'.format(args.arch)
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_file)
        else:
            checkpoint = torch.load(checkpoint_file,
                                    map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        evaluate(args, model, val_loader)


def train(args,
          train_loader,
          model,
          criterion,
          optimizer,
          epoch,
          tb_writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            target = target.cuda()
            input_var = input.cuda()
        else:
            target = target.cpu()
            input_var = input.cpu()

        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1))

        if tb_writer is not None:
            tb_writer.add_scalar('train/loss', loss.item(), epoch)
            tb_writer.add_scalar('train/accuracy', prec1.item(), epoch)
            tb_writer.flush()


def validate(args, val_loader, model, criterion, epoch, tb_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                target = target.cuda()
                input_var = input.cuda()
                target_var = target.cuda()
            else:
                target = target.cpu()
                input_var = input.cpu()
                target_var = target.cpu()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i,
                          len(val_loader),
                          batch_time=batch_time,
                          loss=losses,
                          top1=top1))

            if tb_writer is not None:
                tb_writer.add_scalar('val/loss', loss.item(), epoch)
                tb_writer.add_scalar('val/accuracy', prec1.item(), epoch)
                tb_writer.flush()

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def evaluate(args, model, val_loader):
    model.eval()
    correct = 0
    output_list = []
    labels_list = []
    with torch.no_grad():
        begin = time.time()
        for data, target in val_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data.cpu(), target.cpu()
            output = model(data)
            output = torch.nn.functional.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            output_list.append(output)
            labels_list.append(target)
        end = time.time()
        print("inference throughput: ", 10000 / (end - begin), " images/s")
    output = torch.cat(output_list)
    target = torch.cat(labels_list)
    print('\nTest Accuracy: {:.2f}%\n'.format(100. * correct /
                                              len(val_loader.dataset)))
    target_labels = target.cpu().data.numpy()
    np.save('./probs_cifar_det.npy', output.data.cpu().numpy())
    np.save('./cifar_test_labels.npy', target.data.cpu().numpy())


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


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


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
