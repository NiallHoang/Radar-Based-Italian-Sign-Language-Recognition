import os
import sys
import torch

class CFG:
    # Data root
    data_root = "/home/tinhanh/projects/ISLR2026/data"
    # Data paths
    TRAIN_DIR = os.path.join(data_root, "train")
    VAL_DIR = os.path.join(data_root, "val")
    TEST_DIR = os.path.join(data_root, "test")

    # Training parameters
    NUM_CLASSES  = 126
    VALID_SPLIT  = 0.2
    RANDOM_SEED  = 42

    TARGET_T     = 64
    BATCH_SIZE   = 64
    EPOCHS       = 100
    WARMUP_EPOCHS    = 10
    LR           = 5e-4
    LR_MIN       = 1e-6
    WD           = 1e-2
    num_workers  = 4
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    
    # MixUp schedule phases (epoch boundaries)
    MIXUP_PHASE1_END = 35   # standard mixup ends
    # SWA config
    SWA_START    = 70        # bắt đầu collect SWA weights
    SWA_LR       = 5e-5      # constant LR trong giai đoạn SWA


    CHECKPOINT_DIR = r"/home/tinhanh/projects/ISLR2026/thien_model_testing1/checkpoints"
    RESULTS_DIR = r"/home/tinhanh/projects/ISLR2026/thien_model_testing1/results"
    LOG_FILE = "train_14.log"
    training_history = "training_history_14"
    best_model = "best_model_14"

    USE_TQDM = sys.stdout.isatty()