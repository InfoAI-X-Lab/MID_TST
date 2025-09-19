from types import SimpleNamespace

cfg = SimpleNamespace()

# -------------------------------
# Model configuration
# -------------------------------
cfg.model = SimpleNamespace()
cfg.model.name = "resnet18"                      # "resnet18" / "lenet5"
cfg.model.num_classes = NUM_CLASSES              # Defined in main script, e.g. 10
cfg.model.feat_dim = FEATURE_DIM                 # Feature dimension, e.g. 512 (optional)

# -------------------------------
# Training configuration
# -------------------------------
cfg.train = SimpleNamespace()
cfg.train.epochs = TOTAL_EPOCHS
cfg.train.batch_size = BATCH_SIZE
cfg.train.lr = LEARNING_RATE
cfg.train.num_workers = NUM_WORKERS
cfg.train.teacher_path = TEACHER_MODEL_PATH      # Path to teacher model checkpoint, e.g. "checkpoints/teacher.pth"
cfg.train.resume_path  = STAGE1_CKPT_PATH        # Path to student Stage-1 checkpoint, e.g. "checkpoints/stage1.pth"
cfg.train.save_dir     = SAVE_DIR                # Directory to save student checkpoints during training
cfg.train.mine_path  = MINE_PATH
cfg.train.mine_path  = MINE_PATH

# -------------------------------
# Noise configuration
# -------------------------------
cfg.noise = SimpleNamespace()
cfg.noise.std_ratio = GAUSSIAN_STD_RATIO         # Gaussian noise scale ratio, e.g. 0.05

# -------------------------------
# Loss configuration
# -------------------------------
cfg.loss = SimpleNamespace()
cfg.loss.alpha = LOSS_ALPHA                      # Weight for CrossEntropy loss
cfg.loss.beta1 = LOSS_BETA1                      # MID (Stage1) / KL (Stage2) loss weight
cfg.loss.beta2 = LOSS_BETA2                      # KL loss weight (Stage1 only)
cfg.loss.gamma = LOSS_GAMMA                      # Weight for Lipschitz regularization
cfg.loss.temperature = DISTILL_TEMP              # Distillation temperature
cfg.loss.lipschitz_lambda = LIPSCHITZ_TARGET     # Target Lipschitz constant (lambda)
cfg.loss.use_mi = False                          # Whether to use Mutual Information loss

# -------------------------------
# Dataset paths
# -------------------------------
cfg.data = SimpleNamespace()
cfg.data.train_path = TRAIN_DATASET_PATH         # Path to training dataset, e.g. "data/train/"
cfg.data.val_path   = VAL_DATASET_PATH           # Path to validation dataset, e.g. "data/val/"
cfg.data.image_size = IMAGE_SIZE                 # Input image size, e.g. 96
