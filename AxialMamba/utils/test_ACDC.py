#!/usr/bin/env python
# -*- coding:utf-8 -*-

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import test_single_volume
import os
import os
import numpy as np
from tqdm import tqdm
import torch
import datetime
def inference(args, model, testloader, test_save_path=None):

    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            h, w = sampled_batch["image"].size()[2:]
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
            metric_list += np.array(metric_i)
            logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        metric_list = metric_list / len(testloader)
        for i in range(1, args.num_classes):
            logging.info('Mean class %d mean_dice %f mean_hd95 %f mean_iou %f' %
                         (i, metric_list[i - 1][0], metric_list[i - 1][1], metric_list[i - 1][2]))
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        miou = np.mean([m[2] for m in metric_list], axis=0)
        logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f mean_iou : %f' %
                     (performance, mean_hd95, miou))
        logging.info("Testing Finished!")
        return performance, mean_hd95, miou


def inference2(args, model, testloader, test_save_path=None):
    # 创建日志文件路径（带时间戳）
    # timestamp = datetime.datetime.now()
    log_file = os.path.join(test_save_path,
                            f"test_log.txt") if test_save_path else f"test_log.txt"

    # 初始化指标
    metric_list = []
    total_metrics = {
        'dice': [],
        'hd95': [],
        'iou': []
    }

    # 创建并写入日志文件
    with open(log_file, 'w') as f:
        # 写入测试基本信息
        f.write(f"{'=' * 50}\n")
        f.write(f"Test started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test configuration:\n")
        f.write(f"  - Total batches: {len(testloader)}\n")
        f.write(f"  - Image size: {args.img_size}x{args.img_size}\n")
        f.write(f"  - Z spacing: {args.z_spacing}\n")
        f.write(f"{'=' * 50}\n\n")

        # 开始测试
        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in tqdm(enumerate(testloader)):
                h, w = sampled_batch["image"].size()[2:]
                image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

                # 单体积测试并收集指标
                metrics = test_single_volume(image, label, model,
                                             classes=args.num_classes,
                                             patch_size=[args.img_size, args.img_size],
                                             test_save_path=test_save_path,
                                             case=case_name,
                                             z_spacing=args.z_spacing)

                # 更新总指标
                metric_list.append(metrics)
                for i in range(1, args.num_classes + 1):
                    total_metrics['dice'].append(metrics[i - 1][0])
                    total_metrics['hd95'].append(metrics[i - 1][1])
                    total_metrics['iou'].append(metrics[i - 1][2])

        # 计算平均指标
        avg_metrics = {
            'dice': np.mean(total_metrics['dice']),
            'hd95': np.mean(total_metrics['hd95']),
            'iou': np.mean(total_metrics['iou'])
        }

        # 写入分类指标
        f.write(f"{'=' * 50}\n")
        f.write(f"Per-class metrics:\n")
        for i in range(1, args.num_classes + 1):
            f.write(f"Class {i:2d}: Dice={total_metrics['dice'][i - 1]:.4f} | "
                    f"HD95={total_metrics['hd95'][i - 1]:.2f} | "
                    f"IOU={total_metrics['iou'][i - 1]:.4f}\n")

        # 写入总体性能
        f.write(f"\n{'=' * 50}\n")
        f.write(f"Overall Performance:\n")
        f.write(f"Mean Dice: {avg_metrics['dice']:.4f}\n")
        f.write(f"Mean HD95: {avg_metrics['hd95']:.2f}\n")
        f.write(f"Mean IOU: {avg_metrics['iou']:.4f}\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Test completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 同时输出到控制台（可选）
    print(f"\nResults saved to: {log_file}")
    print(
        f"Overall Performance: Dice={avg_metrics['dice']:.4f}, HD95={avg_metrics['hd95']:.2f}, IOU={avg_metrics['iou']:.4f}")

    return avg_metrics['dice'], avg_metrics['hd95'], avg_metrics['iou']


