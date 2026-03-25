import random
import numpy as np
import sys
import logging
import os
import torch
from torch import nn, optim
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import CFG
from utils import (get_train_samples, mixup_batch, cutmix_batch_temporal,
                   cutmix_antenna_group, soft_cross_entropy, get_mixup_params,
                   get_lr, plot_training_history)
from dataset import RadarDataset
from model import RadarModel

def main():
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(CFG.RANDOM_SEED)

    os.makedirs(CFG.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(CFG.RESULTS_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(CFG.RESULTS_DIR, CFG.LOG_FILE)),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Start program...")

    # ───────────────────────────────────────────────────────────────────────────────
    # GET DATA
    # ───────────────────────────────────────────────────────────────────────────────
    samples, labels, sessions = get_train_samples(CFG.TRAIN_DIR)
    logger.info(f"Total samples: {len(samples)}")

    train_s, val_s, train_l, val_l = train_test_split(
        samples, labels,
        test_size=CFG.VALID_SPLIT,
        random_state=CFG.RANDOM_SEED,
        stratify=labels
    )

    logger.info(f"train samples: {len(train_s)} | val samples: {len(val_s)}")

    # ───────────────────────────────────────────────────────────────────────────────
    # DATASET & DATALOADER
    # ───────────────────────────────────────────────────────────────────────────────

    train_ds = RadarDataset(train_s, train_l, train=True)
    val_ds   = RadarDataset(val_s,   val_l,   train=False)

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,  num_workers=CFG.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=CFG.BATCH_SIZE, num_workers=CFG.num_workers)


    # ───────────────────────────────────────────────────────────────────────────────
    # MODEL / LOSS / OPTIMIZER / SCHEDULER
    # ───────────────────────────────────────────────────────────────────────────────

    model     = RadarModel(CFG.NUM_CLASSES).to(CFG.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)

    # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    #     optimizer, start_factor=0.01, total_iters=CFG.WARMUP_EPOCHS
    # )
    # cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=CFG.EPOCHS - CFG.WARMUP_EPOCHS, eta_min=1e-6
    # )
    # scheduler = torch.optim.lr_scheduler.SequentialLR(
    #     optimizer, [warmup_scheduler, cosine_scheduler], milestones=[CFG.WARMUP_EPOCHS]
    # )

    # SWA model
    swa_model = AveragedModel(model)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    best_val_acc = 0.0

    logger.info("Start training...")
    for epoch in range(CFG.EPOCHS):
        # ── Set LR for this epoch ────────────────────────────────────────────
        if epoch >= CFG.SWA_START:
            lr = CFG.SWA_LR
        else:
            lr = get_lr(epoch, CFG)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        
        # ── Get MixUp params for this epoch ──────────────────────────────────
        mixup_prob, cutmix_prob, mixup_alpha = get_mixup_params(epoch, CFG)

        model.train()
        train_correct = train_total = train_loss = 0

        for x, y in train_loader:
            x = x.to(CFG.device)
            y = y.to(CFG.device)

            # Decide augmentation: MixUp, CutMix-temporal, CutMix-antenna, or none
            rand_val = np.random.rand()
            use_soft = False

            if rand_val < mixup_prob:
                x, y_soft = mixup_batch(x, y, CFG.NUM_CLASSES, alpha=mixup_alpha)
                use_soft = True
            elif rand_val < mixup_prob + cutmix_prob * 0.7:
                x, y_soft = cutmix_batch_temporal(x, y, CFG.NUM_CLASSES, alpha=mixup_alpha)
                use_soft = True
            elif rand_val < mixup_prob + cutmix_prob:
                x, y_soft = cutmix_antenna_group(x, y, CFG.NUM_CLASSES)
                use_soft = True

            optimizer.zero_grad()
            pred = model(x)
            if use_soft:
                loss = soft_cross_entropy(pred, y_soft, label_smoothing=0.1)
            else:
                loss = criterion(pred, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = pred.argmax(1)

            if use_soft:
                dominant_label = y_soft.argmax(1)
                train_correct += (preds == dominant_label).sum().item()
            else:
                train_correct += (preds == y).sum().item()
            train_total += x.size(0)

        train_acc = train_correct / train_total
        epoch_train_loss = train_loss / len(train_loader)

        # ── SWA: update averaged weights ─────────────────────────────────────
        if epoch >= CFG.SWA_START:
            swa_model.update_parameters(model)

        model.eval()
        val_correct = val_total = val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(CFG.device); y = y.to(CFG.device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
                preds = pred.argmax(1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total
        epoch_val_loss = val_loss / len(val_loader)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        phase = ("warmup" if epoch < CFG.WARMUP_EPOCHS
                 else "swa" if epoch >= CFG.SWA_START
                 else "cosine")
        logger.info(f"Epoch {epoch+1}/{CFG.EPOCHS} [{phase}]| train_loss={epoch_train_loss:.4f} | val_loss={epoch_val_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | lr={lr:.6f} | mixup_p={mixup_prob:.2f} cutmix_p={cutmix_prob:.2f} alpha={mixup_alpha:.2f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CFG.CHECKPOINT_DIR, f"{CFG.best_model}.pth"))
            logger.info(f"Saved {CFG.best_model}.pth at epoch {epoch+1} (val={best_val_acc:.4f})")
    # ───────────────────────────────────────────────────────────────────────────────
    # SWA: Update BatchNorm & save
    # ───────────────────────────────────────────────────────────────────────────────
    logger.info("Updating SWA BatchNorm statistics...")
    update_bn(train_loader, swa_model, device=CFG.device)
    
    # Evaluate SWA model on validation set
    swa_model.eval()
    swa_correct = swa_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(CFG.device); y = y.to(CFG.device)
            pred = swa_model(x)
            swa_correct += (pred.argmax(1) == y).sum().item()
            swa_total += y.size(0)
    swa_val_acc = swa_correct / swa_total
    logger.info(f"SWA val_acc={swa_val_acc:.4f} | best single val_acc={best_val_acc:.4f}")

    torch.save(swa_model.module.state_dict(), os.path.join(CFG.CHECKPOINT_DIR, f"{CFG.best_model}_swa.pth"))
    logger.info(f"Saved {CFG.best_model}_swa.pth")

    logger.info("Training finished.")
    plot_training_history(history=history, save_dir=CFG.RESULTS_DIR, training_history=CFG.training_history, logger=logger)

if __name__ == "__main__":
    main()