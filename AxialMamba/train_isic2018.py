import sys
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from models.metric import *
from models.utils import SegmentationLogger
import os
from models.Axialmamba4 import MAAmamba_unet8
from my_datasets.ISIC_dataset import get_loader

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")
    batch_size = 8
    crop_size = (256, 256)
    learning_rate = 0.001
    epochs = 60
    save_dir = r'.\isic_log'
    os.makedirs(save_dir, exist_ok=True)

    logger = SegmentationLogger(save_dir=save_dir)
    def dataloader():
        train_loader = get_loader(
            image_path=r'.\isic_data\ISIC2018_Task1_Training',
            image_size=crop_size,
            batch_size=batch_size,
            num_workers=6,
            augmentation_prob=0.,
            mode='train',
            shuffle=True
        )
        valid_loader = get_loader(
            image_path=r'.\ISIC2018_Task1_Validation',
            image_size=crop_size,
            batch_size=batch_size,
            num_workers=6,
            mode='valid',
            augmentation_prob=0.,
            shuffle=False
        )
        return train_loader, valid_loader

    train_loader, val_loader = dataloader()


    net = MAAmamba_unet8(in_channels=3, num_classes=1, width=256).to(device)
    loss_func = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    train_septs = len(train_loader)
    val_septs = len(val_loader)

    all_trainloss = []
    all_valloss = []
    best_val_loss = float('inf')
    best_metrics = {
        'loss': float('inf'),
        'iou': 0.0,
        'dice': 0.0,
        'pa': 0.0,
        'recall': 0.0,
        'precision': 0.0,
        'f1': 0.0,
        'mae': 0.0
    }


    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        num_correct = 0
        num_pixels = 0

        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            images = images.to(device)
            labels = labels.float().to(device)

            logits = net(images)
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).float()
            num_correct += (preds == labels).sum()
            num_pixels += torch.numel(preds)
            loss = loss_func(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss: {running_loss / (step + 1):.3f}"

        net.eval()
        val_loss = 0.0
        metrics = {
            'iou': 0.0,
            'dice': 0.0,
            'pa': 0.0,
            'recall': 0.0,
            'precision': 0.0,
            'f1': 0.0,
            'mae': 0.0
        }

        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images = val_images.to(device)
                val_labels = val_labels.float().to(device)
                val_pre = net(val_images)
                pred = torch.sigmoid(val_pre)
                pred_labels = (pred > 0.5).float()

                # 计算指标
                TP, FP, FN, TN = get_confusion_matrix(pred_labels, val_labels)
                metrics['iou'] += iou_score(TP, FP, FN)
                metrics['dice'] += dice_score(TP, FP, FN)
                metrics['pa'] += pixel_accuracy(TP, TN, FP, FN)
                metrics['recall'] += calculate_recall(pred_labels, val_labels)
                metrics['precision'] += precision_score(TP, FP)
                metrics['f1'] += calculate_f1_score(pred_labels, val_labels)
                metrics['mae'] += compute_mae(pred_labels, val_labels)

                vloss = loss_func(val_pre, val_labels)
                val_loss += vloss.item()
                val_bar.desc = f"val epoch[{epoch + 1}/{epochs}]"


        val_loss = val_loss / val_septs
        running_acc = num_correct.float() / num_pixels
        running_loss = running_loss / train_septs

        for key in metrics:
            metrics[key] = metrics[key] / val_septs

        all_trainloss.append(running_loss)
        all_valloss.append(val_loss)
        logger.record_metrics(
            epoch=epoch,
            train_loss=running_loss,
            val_loss=val_loss,
            val_acc=metrics['pa'],
            val_precision=metrics['precision'],
            val_iou=metrics['iou'],
            val_recall=metrics['recall'],
            val_f1=metrics['f1'],
            val_mae=metrics['mae']
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = metrics.copy()


            model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'metrics': metrics
            }, model_path)
            print(f"Saved: {val_loss:.3f}")

if __name__ == '__main__':
    main()
