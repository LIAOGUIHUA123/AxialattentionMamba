import torch
import os

def get_confusion_matrix(pred_labels, target):
    TP = ((pred_labels == 1) & (target == 1)).sum().item()
    FP = ((pred_labels == 1) & (target == 0)).sum().item()
    FN = ((pred_labels == 0) & (target == 1)).sum().item()
    TN = ((pred_labels == 0) & (target == 0)).sum().item()
    return TP, FP, FN, TN


# 1. 交并比 (IoU)
def iou_score(TP, FP, FN):
    return TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0


# 2. Dice 系数
def dice_score(TP, FP, FN):
    return (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0


# 3. 像素准确率 (PA)
def pixel_accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0


# 4. 平均交并比 (mIoU)
def mean_iou(TP, FP, FN, TN):
    iou_foreground = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    iou_background = TN / (TN + FN + FP) if (TN + FN + FP) > 0 else 0
    return (iou_foreground + iou_background) / 2


# 5. 定义precision

def precision_score(TP, FP):
    return TP / (TP + FP) if (TP + FP) > 0 else 0


# 6.MAE
def compute_mae(pred, target):
    """
    计算二分类图像分割的MAE（平均绝对误差）

    参数：
    pred (torch.Tensor): 模型预测结果，形状为[B, 1, H, W]（logits或概率）
    target (torch.Tensor): 真实标签，形状为[B, 1, H, W]（0或1）
    返回：
    float: MAE值
    """
    # 确保输入在相同设备上
    assert pred.device == target.device, "输入必须在相同设备上"

    # 二值化处理
    pred_bin = pred

    # 计算MAE
    mae = torch.mean(torch.abs(pred_bin - target))

    return mae.item()


# 7. F1-Score
def f1_score(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


# 8. 真正率 (TPR) 和 假正率 (FPR)
def tpr_fpr(TP, TN, FP, FN):
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    return tpr, fpr

#
# 9,召回recall
def calculate_recall(pred, target):
    """
    计算二分类图像分割的召回率

    参数:
    pred (torch.Tensor): 模型的预测输出，形状为 (N, H, W)，其中 N 是批次大小，H 和 W 是图像的高度和宽度
    target (torch.Tensor): 真实标签，形状与 pred 相同

    返回:
    recall (float): 召回率
    """
    # 确保预测和目标都是二值图像
    # pred = (pred > 0.5).float()
    # target = (target > 0.5).float()

    # 计算真正例（TP）的数量
    tp = (pred * target).sum()

    # 计算实际正例（P）的数量
    p = target.sum()

    # 避免除以零
    if p == 0:
        return 0.0

    # 计算召回率
    recall = tp / p

    return recall.item()
# 10,F1
def calculate_f1_score(pred, target):
    """
    计算二分类图像分割的F1分数。

    参数:
    pred : torch.Tensor
        模型的预测输出，形状为(N, H, W)，其中N是批次大小，H和W是图像的高度和宽度。
        预测输出应该是概率图，每个像素的值在0到1之间。
    target : torch.Tensor
        真实标签，形状与pred相同，应该是二值图，即每个像素的值是0或1。

    返回:
    f1_score : float
        F1分数。
    """
    # 将预测输出转换为二值图
    pred_binary = (pred > 0.5).float()

    # 计算真正例（TP），假正例（FP）和假反例（FN）
    tp = (pred_binary * target).sum().item()
    fp = (pred_binary * (1 - target)).sum().item()
    fn = ((1 - pred_binary) * target).sum().item()

    # 计算精确率（Precision）和召回率（Recall）
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    # 计算F1分数
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

    return f1_score

def save_checkpoint(self, epoch, val_loss, is_best=False, filename='checkpoint.pth'):
    """
    保存训练检查点
    Args:
        epoch: 当前epoch
        val_loss: 验证集损失
        is_best: 是否为最优模型
        filename: 基础文件名
    """
    # 构建完整保存路径
    save_path = os.path.join(self.save_dir, filename)

    # 准备保存字典
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        'val_loss': val_loss
    }