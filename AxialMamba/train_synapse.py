#!/usr/bin/env python
# -*- coding:utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from utils.utils import DiceLoss
from torch.utils.data import DataLoader
from my_datasets.dataset_Synapse import Synapsedataset, RandomGenerator
import argparse
from tqdm import tqdm
import os
from torchvision import transforms
from utils.test_Synapse import inference
from models.Axialmamba4 import MAAmamba_unet8


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=12, help="batch size")
parser.add_argument("--lr", default=0.0001, help="learning rate")
parser.add_argument("--max_epochs", default=100)
parser.add_argument("--img_size", default=256)
parser.add_argument("--save_path", default=r".\Synapse_log\checkpiont")
parser.add_argument("--n_gpu", default=1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--list_dir", default=r".\path\lists_Synapse")
parser.add_argument("--root_dir", default=r".\datas\Synapse")
parser.add_argument("--volume_path", default=r".\path\Synapse\test")
parser.add_argument("--z_spacing", default=10)
parser.add_argument("--num_classes", default=9)
parser.add_argument('--test_save_dir', default=r'.\Synapse_log\test', help='saving prediction as nii!')
parser.add_argument("--patches_size", default=16)
parser.add_argument("--n-skip", default=1)
args = parser.parse_args()

model = MAAmamba_unet8(num_classes=args.num_classes, in_channels=1, width=args.img_size)  # 9

if args.checkpoint:
    model.load_state_dict(torch.load(args.checkpoint))

train_dataset = Synapsedataset(args.root_dir, args.list_dir, split="train", transform=
transforms.Compose(
    [RandomGenerator(output_size=[args.img_size, args.img_size])]))
Train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
db_test = Synapsedataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
testloader = DataLoader(db_test, batch_size=1, shuffle=False)

if args.n_gpu > 1:
    model = nn.DataParallel(model)

model = model.cuda()
model.train()
ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(args.num_classes)
save_interval = args.n_skip  # int(max_epoch/6)

iterator = tqdm(range(0, args.max_epochs), ncols=70)

iter_num = 0

Loss = []
Test_Accuracy = []

Best_dcs = 0.7
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s   %(levelname)s   %(message)s')

max_iterations = args.max_epochs * len(Train_loader)
base_lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
# optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

for epoch in iterator:
    model.train()
    train_loss = 0
    for i_batch, sampled_batch in enumerate(Train_loader):
        image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

        outputs = model(image_batch)

        loss_ce = ce_loss(outputs, label_batch[:].long())
        loss_dice = dice_loss(outputs, label_batch[:], softmax=True)
        loss = loss_dice * 0.5 + loss_ce * 0.5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        # lr_ = base_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1

        logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
        train_loss += loss.item()
    Loss.append(train_loss / len(train_dataset))

    # loss visualization

    # fig1, ax1 = plt.subplots(figsize=(11, 8))
    # ax1.plot(range(epoch + 1), Loss)
    # ax1.set_title("Average trainset loss vs epochs")
    # ax1.set_xlabel("Epoch")
    # ax1.set_ylabel("Current loss")
    # plt.savefig('loss_vs_epochs_gauss.png')

    # plt.clf()
    # plt.close()

    if (epoch + 1) % save_interval == 0:
        avg_dcs, avg_hd = inference(args, model, testloader, args.test_save_dir)
        # avg_dcs, avg_hd = test()

        if avg_dcs > Best_dcs:
            save_mode_path = os.path.join(args.save_path,
                                          'epoch={}_lr={}_avg_dcs={}_avg_hd={}.pth'.format(epoch, lr_, avg_dcs, avg_hd))
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            # temp = 1
            Best_dcs = avg_dcs

        Test_Accuracy.append(avg_dcs)

        # val visualization

        # fig2, ax2 = plt.subplots(figsize=(11, 8))
        # ax2.plot(range(int((epoch + 1) // save_interval)), Test_Accuracy)
        # ax2.set_title("Average val dataset dice score vs epochs")
        # ax2.set_xlabel("Epoch")
        # ax2.set_ylabel("Current dice score")
        # plt.savefig('val_dsc_vs_epochs_gauss.png')

        # plt.clf()
        # plt.close()

    if epoch >= args.max_epochs - 1:
        save_mode_path = os.path.join(args.save_path, 'epoch={}_lr={}_avg_dcs={}.pth'.format(epoch, lr_, avg_dcs))
        torch.save(model.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))
        iterator.close()
        break