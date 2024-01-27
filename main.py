import json
import time
import os
import torch
from apex import amp
import shutil
from data import get_dataset,get_logger
from torch.utils.data import DataLoader
from model.create import create_segmenter
import torch.nn as nn
import numpy as np
from optim.Ranger import Ranger
from torch.optim.lr_scheduler import LambdaLR
from optim.metrics import averageMeter
from optim.metrics import runningScore

def save_ckpt(logdir, model, epoch_iter, prefix=''):
    state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state, os.path.join(logdir, prefix + 'model_' + str(epoch_iter) + '.pth'))
class eeemodelLoss(nn.Module):
    def __init__(self):
        super(eeemodelLoss, self).__init__()
        self.class_weight_semantic = torch.from_numpy(np.array(
            [1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])).float()
        self.class_weight_binary = torch.from_numpy(np.array([1.5121, 10.2388])).float()

        self.binary_loss = nn.CrossEntropyLoss(weight=self.class_weight_binary)
        self.semantic_loss = nn.CrossEntropyLoss(weight=self.class_weight_semantic)

    def forward(self, inputs, targets):
        semantic_gt, binary_gt = targets
        semantic_out, sal_out = inputs

        return self.semantic_loss(semantic_out, semantic_gt)*10 + 5 * self.binary_loss(sal_out, binary_gt)


def run(args):
    torch.cuda.set_device(args.cuda)
    with open(args.configs, 'r') as fp:
        cfg = json.load(fp)

    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}-{cfg["dataset"]}-{cfg["model_name"]}-'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy(args.configs, logdir)

    trainset, _, testset = get_dataset(cfg)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True)
    test_loader = DataLoader(testset, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
                             pin_memory=True)

    print("训练数据的长度: " + str(len(train_loader)))
    print("测试数据的长度: " + str(len(test_loader)))

    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')
    device = torch.device(f'cuda:{args.cuda}')

    our_model = create_segmenter(cfg)
    our_model.to(device)

    params_list = our_model.parameters()
    optimizer = Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)

    train_criterion = eeemodelLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    train_loss_meter = averageMeter()
    test_loss_meter = averageMeter()

    running_metrics_test = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    best_test = 0

    amp.register_float_function(torch, 'sigmoid')
    our_model, optimizer = amp.initialize(our_model, optimizer, opt_level=args.opt_level)

    for ep in range(cfg['epochs']):

        our_model.train()
        train_loss_meter.reset()
        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()
            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            targets = sample['label'].to(device)
            binary_label = sample['binary_label'].to(device)

            predict = our_model(image, depth)
            targets = (targets, binary_label)
            loss = train_criterion(predict, targets)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item())

        scheduler.step(ep)

        with torch.no_grad():
            our_model.eval()
            running_metrics_test.reset()
            test_loss_meter.reset()
            for i, sample in enumerate(test_loader):

                image = sample['image'].to(device)
                # Here, depth is TIR.
                depth = sample['depth'].to(device)
                label = sample['label'].to(device)
                predict = our_model(image, depth)[0]

                loss = criterion(predict, label)
                test_loss_meter.update(loss.item())
                predict = predict.max(1)[1].cpu().numpy()  # [1, h, w]
                label = label.cpu().numpy()
                running_metrics_test.update(label, predict)

        train_loss = train_loss_meter.avg
        test_loss = test_loss_meter.avg

        test_macc = running_metrics_test.get_scores()[0]["class_acc: "]
        test_miou = running_metrics_test.get_scores()[0]["mIou: "]
        test_avg = (test_macc + test_miou) / 2
        logger.info(
            f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] loss={train_loss:.3f}/{test_loss:.3f}, mPA={test_macc:.3f}, miou={test_miou:.3f}, avg={test_avg:.3f}')
        if test_avg > best_test:
            best_test = test_avg
            save_ckpt(logdir, our_model,ep+1)
            logger.info(
            	f'Save Iter = [{ep + 1:3d}],  mPA={test_macc:.3f}, miou={test_miou:.3f}, avg={test_avg:.3f}')



if __name__ == '__main__':
    import argparse

    print(torch.cuda.is_available())
    print(torch.version.cuda)
    parser = argparse.ArgumentParser(description="configs")
    parser.add_argument("--configs", type=str, default="configs/LASNet.json", help="Configuration file to use")
    parser.add_argument("--opt_level", type=str, default='O1')
    parser.add_argument("--inputs", type=str.lower, default='rgb', choices=['rgb', 'rgbd'])
    parser.add_argument("--resume", type=str, default='',
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument("--cuda", type=int, default=0, help="set cuda device id")

    args = parser.parse_args()

    print("Starting Training!")
    run(args)