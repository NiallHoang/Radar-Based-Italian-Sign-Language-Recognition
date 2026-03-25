import os
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F
from config import CFG


def get_train_samples(train_dir):
    samples=[]
    labels=[]
    sessions=[]
    classes=sorted(os.listdir(train_dir))
    for cls in classes:
        class_path=os.path.join(train_dir,cls)
        if not os.path.isdir(class_path):
            continue
        label=int(cls.split("_")[0])
        for sample in os.listdir(class_path):
            sample_path=os.path.join(class_path,sample)
            if os.path.isdir(sample_path):
                sid=int(sample.split("_")[1])
                session=sid//126
                samples.append(sample_path)
                labels.append(label)
                sessions.append(session)
    return samples,labels,sessions

def get_val_samples(val_dir):
    samples = []
    for sample in os.listdir(val_dir):
        sample_path = os.path.join(val_dir, sample)
        if os.path.isdir(sample_path):
            samples.append(sample_path)
    return samples

def get_test_samples(val_dir):
    samples = []
    sample_ids = []
    for folder_name in sorted(os.listdir(val_dir)):
        folder_path = os.path.join(val_dir, folder_name)
        if os.path.isdir(folder_path):
            samples.append(folder_path)
            sample_ids.append(folder_name)
    return samples, sample_ids


def mixup_batch(x, y, num_classes, alpha=0.4):
    """
    Apply MixUp at batch level: linearly interpolate pairs of samples
    and blend labels accordingly.

    Args:
        x: (B, C, T, R) input tensor
        y: (B,) integer labels
        num_classes: total number of classes
        alpha: Beta distribution parameter (higher = more mixing)

    Returns:
        x_mix: (B, C, T, R) mixed tensor
        y_soft: (B, num_classes) blended soft labels
    """
    if alpha <= 0:
        return x, F.one_hot(y, num_classes).float()

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # keep lam >= 0.5 to avoid label flip

    B = x.size(0)
    indices = torch.randperm(B, device=x.device)

    x_mix = lam * x + (1 - lam) * x[indices]

    y1_onehot = F.one_hot(y, num_classes).float()
    y2_onehot = F.one_hot(y[indices], num_classes).float()
    y_soft = lam * y1_onehot + (1 - lam) * y2_onehot

    return x_mix, y_soft


def soft_cross_entropy(logits, soft_targets, label_smoothing=0.1):
    """
    Cross-entropy loss for soft (blended) targets with label smoothing.

    Args:
        logits: (B, C) raw model outputs
        soft_targets: (B, C) blended one-hot labels
        label_smoothing: smoothing factor
    """
    num_classes = logits.size(1)
    smoothed = (1.0 - label_smoothing) * soft_targets + label_smoothing / num_classes
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(smoothed * log_probs).sum(dim=1).mean()
    return loss

def cutmix_batch_temporal(x, y, num_classes, alpha=1.0):
    """
    CutMix chỉ theo chiều T — giữ toàn bộ R để tránh artifact vật lý.
    """
    if alpha <= 0:
        return x, F.one_hot(y, num_classes).float()

    lam = np.random.beta(alpha, alpha)

    B, C, T, R = x.shape
    indices = torch.randperm(B, device=x.device)

    cut_ratio = 1.0 - lam
    cut_t = int(T * cut_ratio)
    if cut_t == 0:
        return x, F.one_hot(y, num_classes).float()

    cx = np.random.randint(T)
    t1 = np.clip(cx - cut_t // 2, 0, T)
    t2 = np.clip(cx + cut_t // 2, 0, T)

    x_mix = x.clone()
    x_mix[:, :, t1:t2, :] = x[indices, :, t1:t2, :]

    lam_actual = 1.0 - (t2 - t1) / T
    y1 = F.one_hot(y, num_classes).float()
    y2 = F.one_hot(y[indices], num_classes).float()
    return x_mix, lam_actual * y1 + (1 - lam_actual) * y2

def cutmix_antenna_group(x, y, num_classes):
    """
    Antenna-group CutMix: swap toàn bộ 4 channels (RTM, vel, acc, rgrad)
    của 1 antenna từ sample khác. Model buộc phải học từ 2 antenna còn lại.
    """
    B, C, T, R = x.shape
    indices = torch.randperm(B, device=x.device)
    ant = np.random.randint(3)
    ch_group = [ant, ant + 3, ant + 6, ant + 9]

    x_mix = x.clone()
    x_mix[:, ch_group, :, :] = x[indices][:, ch_group, :, :]

    lam = (C - len(ch_group)) / C
    y1 = F.one_hot(y, num_classes).float()
    y2 = F.one_hot(y[indices], num_classes).float()
    return x_mix, lam * y1 + (1 - lam) * y2

def get_mixup_params(epoch, cfg):
    """
    2-phase augmentation schedule (no decay, no finetune):
      Phase 1 (0 .. PHASE1_END-1):  mixup=0.5, cutmix=0.3, alpha=0.4
      Phase 2 (PHASE1_END+):        mixup=0.5, cutmix=0.4, alpha=0.4  (held)

    Returns: (mixup_prob, cutmix_prob, alpha)
    """
    if epoch < cfg.MIXUP_PHASE1_END:
        return 0.5, 0.3, 0.4
    else:
        return 0.5, 0.4, 0.4

def get_lr(epoch, cfg):
    """
    2-phase LR schedule:
      Warmup  (0 .. WARMUP-1):     linear 0 -> LR
      Cosine  (WARMUP .. EPOCHS):   LR -> LR_MIN
    Note: SWA overrides LR from SWA_START in train.py
    """
    warmup_end = cfg.WARMUP_EPOCHS

    if epoch < warmup_end:
        return cfg.LR * (epoch + 1) / warmup_end
    else:
        progress = (epoch - warmup_end) / max(1, cfg.EPOCHS - warmup_end)
        cos_val = 0.5 * (1 + math.cos(math.pi * progress))
        return cfg.LR_MIN + (cfg.LR - cfg.LR_MIN) * cos_val


def plot_training_history(history: dict, fold: int = None,
                          save_dir: str = CFG.RESULTS_DIR, training_history: str = CFG.training_history, logger=None):
    """
    Vẽ Loss & Accuracy của train/val theo epoch.

    Parameters
    ----------
    history : dict  - chứa 4 key:
        {
            'train_loss': [...],
            'val_loss':   [...],
            'train_acc':  [...],
            'val_acc':    [...],
        }
    fold     : int hoặc None  - nếu dùng cross-val, truyền fold index để đặt tên file.
    save_dir : str            - thư mục lưu ảnh.
    training_history : str    - tên file lịch sử huấn luyện.
    logger   : logging.Logger hoặc None - logger để ghi thông tin.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    train_loss = history['train_loss']
    val_loss   = history['val_loss']
    train_acc  = history['train_acc']
    val_acc    = history['val_acc']

    epochs_range = range(1, len(train_loss) + 1)

    best_val_acc_epoch  = int(np.argmax(val_acc))  + 1
    best_val_loss_epoch = int(np.argmin(val_loss)) + 1

    fold_tag = f" - Fold {fold}" if fold is not None else ""
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(f"Training History{fold_tag}", fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

    # ── Plot 1: Loss ──────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(epochs_range, train_loss, 'b-o', markersize=3, linewidth=1.5, label='Train Loss')
    ax1.plot(epochs_range, val_loss,   'r-o', markersize=3, linewidth=1.5, label='Val Loss')

    ax1.axvline(x=best_val_loss_epoch, color='gray', linestyle='--',
                linewidth=1, alpha=0.7, label=f'Best Val Loss (ep {best_val_loss_epoch})')
    ax1.scatter([best_val_loss_epoch], [val_loss[best_val_loss_epoch - 1]],
                color='red', s=80, zorder=5)
    ax1.annotate(f"{val_loss[best_val_loss_epoch - 1]:.4f}",
                 xy=(best_val_loss_epoch, val_loss[best_val_loss_epoch - 1]),
                 xytext=(8, 8), textcoords='offset points', fontsize=9, color='red')

    ax1.set_title("Loss", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1, len(epochs_range)])

    # ── Plot 2: Accuracy ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(epochs_range, train_acc, 'b-o', markersize=3, linewidth=1.5, label='Train Accuracy')
    ax2.plot(epochs_range, val_acc,   'r-o', markersize=3, linewidth=1.5, label='Val Accuracy')

    ax2.axvline(x=best_val_acc_epoch, color='gray', linestyle='--',
                linewidth=1, alpha=0.7, label=f'Best Val Acc (ep {best_val_acc_epoch})')
    ax2.scatter([best_val_acc_epoch], [val_acc[best_val_acc_epoch - 1]],
                color='red', s=80, zorder=5)
    ax2.annotate(f"{val_acc[best_val_acc_epoch - 1]:.4f}",
                 xy=(best_val_acc_epoch, val_acc[best_val_acc_epoch - 1]),
                 xytext=(8, -14), textcoords='offset points', fontsize=9, color='red')

    ax2.set_title("Accuracy", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1, len(epochs_range)])
    ax2.set_ylim([0, 1.05])

    # ── Lưu file ─────────────────────────────────────────────────────────────
    fname = f"{training_history}_fold{fold}.png" if fold is not None else f"{training_history}.png"
    plot_path = os.path.join(save_dir, fname)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    logger.info(f"Plot saved: {plot_path}")

    # ── In summary ────────────────────────────────────────────────────────────
    logger.info(f"\n{'─'*45}")
    logger.info(f"  Total epochs trained : {len(epochs_range)}")
    logger.info(f"  Best Val Accuracy    : {max(val_acc):.4f}  (epoch {best_val_acc_epoch})")
    logger.info(f"  Best Val Loss        : {min(val_loss):.4f}  (epoch {best_val_loss_epoch})")
    logger.info(f"  Final Train Accuracy : {train_acc[-1]:.4f}")
    logger.info(f"  Final Val Accuracy   : {val_acc[-1]:.4f}")
    logger.info(f"{'─'*45}")

def get_tta_views(x):
    return [
        x,                              # 1. original
        torch.roll(x,  1, dims=2),      # 2. time +1
        torch.roll(x, -1, dims=2),      # 3. time -1
        torch.roll(x,  2, dims=2),      # 4. time +2
        torch.roll(x, -2, dims=2),      # 5. time -2
    ]