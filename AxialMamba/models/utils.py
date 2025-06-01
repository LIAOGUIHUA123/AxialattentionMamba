import torch
import csv
from datetime import datetime
from pathlib import Path
import os
import logging

class SegmentationLogger:
    def __init__(self, save_dir: str = "training_logs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 初始化日志文件
        self.log_file = self.save_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._init_log_file()

        # # 初始化最佳模型追踪
        # self.best_val_loss = float('inf')
        # self.best_model_path = self.save_dir / "best_model.pth"

    def _init_log_file(self):
        """初始化日志文件并写入表头"""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "epoch", "train_loss",
                "val_loss", "val_acc", "val_precision",
                "val_recall", "val_f1", "val_iou"
            ])


    def record_metrics(self, epoch: int, train_loss: float,
                       val_loss: float, val_acc: float,
                       val_precision: float, val_recall: float,
                       val_f1: float, val_iou: float, val_mae:float):
        """记录训练指标"""
        metrics = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            epoch,
            train_loss,
            val_loss,
            val_acc,
            val_precision,
            val_recall,
            val_f1,
            val_iou,
            val_mae
        ]

        # 写入日志文件
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metrics)


class SegmentationLogger2:
    def __init__(self, save_dir: str = "training_logs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 初始化日志文件
        self.log_file = self.save_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._init_log_file()

        # # 初始化最佳模型追踪
        # self.best_val_loss = float('inf')
        # self.best_model_path = self.save_dir / "best_model.pth"

    def _init_log_file(self):
        """初始化日志文件并写入表头"""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "epoch", "avg_dsc",
                "avg_hd", "avg_iou",

            ])


    def record_metrics(self, epoch: int, avg_dsc: float,
                       avg_hd: float, avg_iou: float,
                       ):
        """记录训练指标"""
        metrics = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            epoch,
            avg_dsc,
            avg_hd,
            avg_iou
        ]

        # 写入日志文件
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metrics)
