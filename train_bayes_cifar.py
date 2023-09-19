## This code is adapted from https://github.com/kekmodel/FixMatch-pytorch

import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict, Counter
import sys
import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy, ECELoss, ThresholdScheduler
from models.ema import ModelEMA

logger = logging.getLogger(__name__)
best_acc = 0

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))

def save_checkpoint_epoch(state, checkpoint, epoch):
    filename = 'ep'+str(epoch)+'.pth.tar'
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss

def create_model(args):
    import models.wideresnet_ssl as models
    if args.bayes:
        model = models.BayesCEWideResNet(num_classes=args.num_classes,
                                        depth=args.model_depth,
                                        widen_factor=args.model_width,
                                        dropout=0.,
                                        prior_mu=args.prior_mu,
                                        prior_sigma=args.prior_sig
                                        )
    else:
        model = models.SupCEWideResNet(num_classes=args.num_classes,
                                    depth=args.model_depth,
                                    widen_factor=args.model_width,
                                    dropout=0.,
                                    )
    print("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters())/1e6))
    return model

def bayes_predict(args, bayeslayer, reps, T=1):
    ''' creates args.bayes_samples models and get mean and std of softmax(output) '''
    with torch.no_grad():
        with autocast():
            outputs = [torch.softmax(bayeslayer(reps)[0]/T,dim=-1) for _ in range(args.bayes_samples)]
        outputs = torch.stack(outputs)
        mean_output = torch.mean(outputs, 0)
        std_output = torch.std(outputs,0)
        # max_prob, preds = torch.max(mean_preds, dim=-1)
    return mean_output, std_output

def sharpen(p, T):
    sharp_p = p**(1./T)
    sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
    return sharp_p

def main(args):

    global best_acc

    args.total_steps = args.epochs * args.eval_step
    args.path_to_npy = './split_files_numpy/'
    args.device = torch.device('cuda', args.gpu_id)

    fix_seed(args.global_seed)

    os.makedirs(args.out, exist_ok=True)
    args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.model_depth = 28
        args.model_width = 2
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.model_depth = 28
        args.model_width = 8

    labeled_dataset, unlabeled_dataset, test_dataset, labeled_idx = DATASET_GETTERS[args.dataset](
            args, './data',trans='fixmatch')
    np.save(args.out + '/idx.npy',labeled_idx) # save labeled idx for reproducibility
    
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=RandomSampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=RandomSampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.test_batch_size,
        num_workers=args.num_workers)

    model = create_model(args)
    model.to(args.device)

    # set bias and batchnorm to not have weight decay
    no_decay = ['bias', 'bn']
    if args.bayes:
        grouped_parameters = [
            {'params': [p for n, p in model.encoder.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
            {'params': [p for n, p in model.encoder.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    else:
        grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    
    # initialize SGD optimizer and cosine learning rate scheduler
    optimizer = optim.SGD(grouped_parameters,
                            lr=args.lr,
                            momentum=0.9, nesterov=True)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.bayes: # We use a separate Adam optimizer for the bayes layer
        bayes_optimizer = optim.Adam(model.fc.parameters(), lr=args.bayes_lr)
        bayes_scheduler = get_cosine_schedule_with_warmup(
            bayes_optimizer, args.warmup, args.total_steps)
        # we have a quantile scheduler for the std threshold
        quantile_sch = ThresholdScheduler(args.quansch_warmup,args.init_quan,args.final_quan,args.total_steps,args.eval_step)
        
    else:
        bayes_optimizer = None
        bayes_scheduler = None
        quantile_sch = None

    # initialize EMA model
    ema_model = ModelEMA(args, model, args.ema_decay, args.device)

    args.start_epoch = 0
    quan_queue = []

    # automatically resume training from last checkpoint
    if os.path.isfile(args.out + '/checkpoint.pth.tar'):
        checkpoint = torch.load(args.out + '/checkpoint.pth.tar', map_location='cpu')
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict']) # load model
        ema_model.ema.load_state_dict(checkpoint['ema_state_dict']) # load ema model
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if bayes_optimizer is not None:
            bayes_optimizer.load_state_dict(checkpoint['bayes_optimizer'])
        if bayes_scheduler is not None:
            bayes_scheduler.load_state_dict(checkpoint['bayes_scheduler'])
        quan_queue = checkpoint['quan_queue']
        for _ in range(args.start_epoch):
            for _ in range(args.eval_step):
                if quantile_sch is not None:
                    args.quantile = quantile_sch.get_threshold()


    print(f"***** Running training from Epoch {args.start_epoch} *****")
    print(f"  Task = {args.dataset}@{args.num_labeled}")
    print(f"  Num Epochs = {args.epochs}")
    print(f"  Batch size = {args.batch_size}")
    print(f"  Total optimization steps = {args.total_steps}")

    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, 
          bayes_optimizer=bayes_optimizer, bayes_scheduler=bayes_scheduler,
          quantile_sch=quantile_sch, quan_queue=quan_queue)

def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, 
          bayes_optimizer=None,bayes_scheduler=None,
          quantile_sch=None, quan_queue=[]):

    global best_acc
    scaler = GradScaler()
    ece_score = ECELoss()
    
    test_accs = []
    end = time.time()

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    
    model.zero_grad()
    model.train()

    for epoch in range(args.start_epoch, args.epochs):
        
        # Initialize all the running statistics meters
        stime = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        pseudolab_acc = AverageMeter()
        impurity_rate = AverageMeter()
        ece_meter = AverageMeter()
        pmax_meter = AverageMeter()
        if args.bayes:
            losses_kl = AverageMeter()

        p_bar = tqdm(range(args.eval_step))
        
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x, u_lb_idx = next(labeled_iter)
            except:
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x, u_lb_idx = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s), targets_u_true, u_idx = next(unlabeled_iter)
            except:
                unlabeled_iter = iter(unlabeled_trainloader)

                (inputs_u_w, inputs_u_s), targets_u_true, u_idx = next(unlabeled_iter)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]

            inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s))
            inputs = inputs.to(args.device)

            targets_x = targets_x.to(args.device)
            targets_u_true = targets_u_true.to(args.device)

            if quantile_sch is not None:
                args.quantile = quantile_sch.get_threshold()

            ema_model.update(model)

            with autocast():
                ftime = time.time()
                if args.bayes:
                    rep = model.encoder(inputs)
                    logits, Lkl = model.fc(rep)
                else:
                    logits = model(inputs)

                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)

                del logits

                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                if args.bayes:
                    Lx += Lkl / logits_x.shape[0] * args.kl
                    
                    rep_u = rep.detach()[batch_size:]
                    rep_u_w, rep_u_s = rep_u.chunk(2)
                    
                    # sample many models to get prediction and uncertainty
                    if args.uda:
                        mean_output_u_w, std_output_u_w = bayes_predict(args, model.fc, rep_u_w, T=args.uda_T) # softmax applied inside
                    else:
                        mean_output_u_w, std_output_u_w = bayes_predict(args, model.fc, rep_u_w) # softmax applied inside

                    max_probs_u_w, targets_u = torch.max(mean_output_u_w,dim=-1)
                    pred_std = torch.gather(std_output_u_w,1,targets_u.view(-1,1)).squeeze(1)

                    if args.uda:
                        pseudo_label = sharpen(mean_output_u_w, args.uda_T)
                    
                    # create acceptance mask for samples with std less than threshold
                    mask = pred_std.le(args.std_threshold) 
                    mask = mask.float()

                    # Update threshold based on the quantile
                    if args.quantile != -1 and epoch > args.q_warmup:
                        new_threshold = torch.quantile(pred_std, args.quantile).item()
                        if args.q_queue:
                            quan_queue.append(new_threshold)
                            if len(quan_queue)>50: quan_queue.pop(0) # maintain last 50 values
                            new_threshold = np.mean(quan_queue)
                        args.std_threshold = new_threshold
                    
                else:
                    if args.uda:
                        pseudo_label = torch.softmax(logits_u_w.detach()/args.uda_T, dim=-1)
                    else:
                        pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
                    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                    # non-bayesian mask is determined by the threshold on max probability
                    mask = max_probs.ge(args.threshold).float()
                
                # Compute Loss for unlabeled samples and combine loss
                if args.uda:
                    Lu = (ce_loss(logits_u_s, pseudo_label, use_hard_labels=False) * mask).mean()
                else:
                    Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()
                loss = args.lambda_x * Lx + args.lambda_u * Lu

                # Compute all the tracking statistics
                pseudoacc = (targets_u == targets_u_true).float().mean() # track pseudo label accuracy
                # Track ECE and max probability of pseudo labels
                if args.bayes:
                    ece = ece_score(mean_output_u_w, targets_u_true)
                    pmax_mean = torch.mean(max_probs_u_w)
                else:
                    ece = ece_score(pseudo_label, targets_u_true)
                    pmax_mean = torch.mean(max_probs)
                # Track impurity rate
                if mask.sum() > 0:
                    impurity = ((targets_u == targets_u_true) * mask).sum() / mask.sum()
                    impurity_rate.update(impurity.item())

            optimizer.zero_grad()
            if bayes_optimizer is not None:
                bayes_optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            if bayes_optimizer is not None:
                scaler.step(bayes_optimizer)
            scaler.update()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            if args.bayes:
                losses_kl.update(Lkl.item())

            scheduler.step()
            if bayes_scheduler is not None:
                bayes_scheduler.step()
           
            model.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            pseudolab_acc.update(pseudoacc.item())
            ece_meter.update(ece.item())
            pmax_meter.update(pmax_mean.item())
            bayes_lr = bayes_scheduler.get_last_lr()[0] if bayes_scheduler is not None else args.bayes_lr

            if args.bayes:
                p_bar.set_description("Train Ep:{epoch}. It:{batch:4}. LR:{lr:.3f}. Lkl:{loss_kl:.3f}. Lx:{loss_x:.3f}. Lu:{loss_u:.4f}. ".format(
                    epoch=epoch + 1,
                    batch=batch_idx + 1,
                    lr=scheduler.get_last_lr()[0],
                    loss_kl=losses_kl.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg))
            else:
                p_bar.set_description("Train Ep:{epoch}. It:{batch:4}. LR:{lr:.3f}. Lx:{loss_x:.3f}. Lu:{loss_u:.4f}. ".format(
                    epoch=epoch + 1,
                    batch=batch_idx + 1,
                    lr=scheduler.get_last_lr()[0],
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg))

            p_bar.update()

        p_bar.close()

        # TEST model
        if epoch == 0 or epoch % args.eval_every == 0:
            test_model = copy.deepcopy(ema_model.ema)

            if args.bayes:
                test_loss, test_acc, test_ece = test_bayes(args, test_loader, test_model, epoch)
            else:
                test_loss, test_acc, test_ece = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('train/9.pseudo_acc', pseudolab_acc.avg, epoch)
            args.writer.add_scalar('train/10.impurity_rate', impurity_rate.avg, epoch)
            args.writer.add_scalar('train/11.learning_rate', scheduler.get_last_lr()[0], epoch)
            if args.bayes:
                args.writer.add_scalar('train/12.train_loss_kl', losses_kl.avg, epoch)
                if bayes_scheduler is not None:
                    args.writer.add_scalar('train/15.bayes_lr', bayes_scheduler.get_last_lr()[0], epoch)
                if args.quantile != -1:
                    args.writer.add_scalar('train/16.std_threshold', args.std_threshold, epoch)
                
            args.writer.add_scalar('train/17.ECE', ece_meter.avg, epoch)
            args.writer.add_scalar('train/18.max_prob_mean', pmax_meter.avg, epoch)
          
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
            args.writer.add_scalar('test/4.test_ECE', test_ece, epoch)
        
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_dict = {
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'quan_queue': quan_queue
            }
            
            if bayes_optimizer is not None:
                save_dict['bayes_optimizer'] = bayes_optimizer.state_dict()
            if bayes_scheduler is not None:
                save_dict['bayes_scheduler'] = bayes_scheduler.state_dict()
    
            save_checkpoint(save_dict, is_best, args.out)

            if args.save_ep and epoch%50 == 0:
                save_checkpoint_epoch(save_dict, args.out, epoch)
            test_accs.append(test_acc)
            print(f'Best top-1 acc:{best_acc:.2f}')
        
            line_to_print = (
                    f'epoch: {epoch+1} | train_loss: {losses.avg:.3f} | '
                    f'test_acc: {test_acc:.3f} | lr: {scheduler.get_last_lr()[0]:.6f}  | '
                    f'mask: {mask_probs.avg:.3f} '
                    f'pseudo_acc: {pseudolab_acc.avg:.3f} | impurity_rate: {impurity_rate.avg:.3f} | '
                    f'time per epoch: {time.time()-stime}'
                )
            print(line_to_print)
            sys.stdout.flush()

    args.writer.close()

def test(args, test_loader, model, epoch, pseudo=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    ecemeter = AverageMeter()
    prefix = 'Pseudo ' if pseudo else ''

    test_loader = tqdm(test_loader)
    ece_score = ECELoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            with autocast():
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)

            ece = ece_score(outputs.softmax(dim=-1),targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            ecemeter.update(ece.item(), inputs.shape[0])

            batch_time.update(time.time() - end)
            end = time.time()
            test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                batch=batch_idx + 1,
                iter=len(test_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            ))
        test_loader.close()

    print("{}top-1 acc: {:.2f}".format(prefix,top1.avg))
    print("{}top-5 acc: {:.2f}".format(prefix,top5.avg))
    print("{}ECE: {:.4f}".format(prefix,ecemeter.avg))

    return losses.avg, top1.avg, ecemeter.avg


def test_bayes(args, test_loader, model, epoch, bayes_fc=None, pseudo=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    ecemeter = AverageMeter()
    prefix = 'Pseudo ' if pseudo else ''

    test_loader = tqdm(test_loader)
    ece_score = ECELoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            with autocast():
                reps = model.encoder(inputs)
                if bayes_fc is None:
                    mean_output, std_output = bayes_predict(args,model.fc, reps)
                else:
                    mean_output, std_output = bayes_predict(args,bayes_fc, reps)
                ece = ece_score(mean_output, targets)

                outputs, kl = model(inputs)
                loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(mean_output, targets, topk=(1, 5))

            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            ecemeter.update(ece.item(), inputs.shape[0])

            batch_time.update(time.time() - end)
            end = time.time()
            test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                batch=batch_idx + 1,
                iter=len(test_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            ))
        test_loader.close()

    print("{}top-1 acc:{:.2f}".format(prefix,top1.avg))
    print("{}top-5 acc:{:.2f}".format(prefix,top5.avg))
    print("{}ECE:{:.4f}".format(prefix,ecemeter.avg))

    return losses.avg, top1.avg, ecemeter.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu_id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar100', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--num_labeled', type=int, default=400,
                        help='number of labeled data')
    parser.add_argument("--expand_labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--epochs', default=1024, type=int)
    parser.add_argument('--eval_step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')

    parser.add_argument('--ema_decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda_u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--lambda_x', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label max probability threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')

    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument('--global_seed', default=0, type=int,
                        help="random seed")

    parser.add_argument('--bayes',action='store_true')
    parser.add_argument('--bayes_samples',type=int,default=50)
    parser.add_argument('--kl',type=float,default=1.)
    parser.add_argument('--bayes_lr',type=float,default=0.01)
    parser.add_argument('--std_threshold', default=0.02, type=float,
                        help='pseudo label threshold in std of bayes output')

    parser.add_argument('--prior_mu',type=float,default=0)
    parser.add_argument('--prior_sig',type=float,default=1)

    parser.add_argument('--quantile', default=-1, type=float)
    parser.add_argument('--q_warmup', default=-1, type=int)
    parser.add_argument('--q_queue', action='store_true', default=True)
    parser.add_argument('--save_ep', action='store_true', default=True)

    parser.add_argument('--quansch_warmup',type=int, default=10)
    parser.add_argument('--init_quan',type=float, default=0.1)
    parser.add_argument('--final_quan',type=float, default=0.75)

    parser.add_argument('--uda', action='store_true')
    parser.add_argument('--uda_T',type=float, default=0.4)
    parser.add_argument('--test_batch_size',type=int, default=64)
    parser.add_argument('--eval_every', default=1, type=int,
                    help='number of eval steps to run')
    args = parser.parse_args()

    if args.uda: # default UDA uses a lower threshold
        args.threshold = 0.8

    main(args)
 