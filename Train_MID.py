from utils.config import cfg
from utils.add_noise_to_model import add_gaussian_noise_to_model
from utils.losses import compute_stage1_loss, compute_stage2_loss

from datasets import get_datasets
from models.Mine import MINE


import os
import argparse
from contextlib import contextmanager
from typing import List

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# from utils.config import cfg
# from datasets import get_datasets
from utils.add_noise_to_model import add_gaussian_noise_to_model
from utils.losses import compute_stage1_loss, compute_stage2_loss
from models.Mine import MINE


def get_cifar10_datasets():
    """Return CIFAR-10 train/val datasets with normalization."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return trainset, valset


def get_model_from_cfg():
    import models
    model_fn = getattr(models, cfg.model.name)
    return model_fn(num_classes=cfg.model.num_classes)


def _param_std(p: torch.nn.Parameter) -> float:
    with torch.no_grad():
        std = p.data.std()
        if not torch.isfinite(std) or std.item() == 0.0:
            return 1e-8
        return std.item()


@contextmanager
def noisy_weights(model: nn.Module, std_ratio: float):
    noises: List[torch.Tensor] = []
    params: List[torch.nn.Parameter] = []
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.data.numel() > 0:
                cur_std = _param_std(p)
                sigma = std_ratio * cur_std
                n = torch.randn_like(p.data) * sigma
                p.add_(n)
                params.append(p)
                noises.append(n)
    try:
        yield
    finally:
        with torch.no_grad():
            for p, n in zip(params, noises):
                p.sub_(n)


def train_stage1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train_dataset, val_dataset = get_datasets(cfg)
    train_dataset, val_dataset = get_cifar10_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True
    )

    teacher_model = get_model_from_cfg().to(device)
    teacher_model.load_state_dict(torch.load(cfg.train.teacher_path, map_location=device))
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_feat = teacher_model.extractor
    teacher_cls = teacher_model.classifier

    student_model = get_model_from_cfg().to(device)
    student_feat = student_model.extractor
    student_cls = student_model.classifier

    mine = MINE().to(device)
    mine.load_state_dict(torch.load(cfg.train.mine_path, map_location=device))
    mine.eval()
    mine_nm = MINE().to(device)
    mine_nm.load_state_dict(torch.load(cfg.train.mine_nm_path, map_location=device))
    mine_nm.eval()
    for p in mine.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(
        list(student_feat.parameters()) + list(student_cls.parameters()),
        lr=cfg.train.lr
    )

    epochs = cfg.train.epochs
    std_ratio = cfg.noise.std_ratio

    for epoch in range(epochs):
        student_model.train()
        total_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"[Stage1] Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with torch.no_grad():
                feat_clean = teacher_feat(images)
                logits_clean = teacher_cls(feat_clean)

            with noisy_weights(student_model, std_ratio=std_ratio):
                feat_noisy = student_feat(images)
                logits_noisy = student_cls(feat_noisy)

            loss, log = compute_stage1_loss(
                student_logits=logits_noisy,
                teacher_logits=logits_clean,
                student_feat=feat_noisy,
                teacher_feat=feat_clean,
                labels=labels,
                inputs=images,
                model_student=student_model,
                mi_estimator=mine,
                mi_estimator_nm=mine_nm,
                alpha=cfg.loss.alpha,
                beta1=cfg.loss.beta1,
                beta2=cfg.loss.beta2,
                gamma=cfg.loss.gamma,
                T=cfg.loss.temperature,
                lambda_val=cfg.loss.lipschitz_lambda,
                use_mi=cfg.loss.use_mi
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Stage1] Epoch {epoch+1} | Loss: {total_loss / max(1, len(train_loader)):.4f}")

        if cfg.train.save_dir:
            os.makedirs(cfg.train.save_dir, exist_ok=True)
            torch.save({
                'student_feature': student_feat.state_dict(),
                'student_classifier': student_cls.state_dict()
            }, os.path.join(cfg.train.save_dir, f"stage1_epoch{epoch+1}.pth"))


def train_stage2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train_dataset, val_dataset = get_datasets(cfg)
    train_dataset, val_dataset = get_cifar10_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True
    )

    student_model = get_model_from_cfg().to(device)
    ckpt = torch.load(cfg.train.resume_path, map_location=device)
    student_model.extractor.load_state_dict(ckpt['student_feature'])
    student_model.classifier.load_state_dict(ckpt['student_classifier'])

    for p in student_model.extractor.parameters():
        p.requires_grad = False
    student_model.extractor.eval()

    optimizer = optim.Adam(student_model.classifier.parameters(), lr=cfg.train.lr)

    epochs = cfg.train.epochs
    std_ratio = cfg.noise.std_ratio

    teacher_model = None
    if getattr(cfg.train, "teacher_path", None):
        teacher_model = get_model_from_cfg().to(device)
        teacher_model.load_state_dict(torch.load(cfg.train.teacher_path, map_location=device))
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

    for epoch in range(epochs):
        student_model.classifier.train()
        total_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"[Stage2] Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with noisy_weights(student_model, std_ratio=std_ratio):
                feat = student_model.extractor(images)
                logits = student_model.classifier(feat)

            if teacher_model is not None:
                with torch.no_grad():
                    t_feat = teacher_model.extractor(images)
                    t_logits = teacher_model.classifier(t_feat)
            else:
                t_logits = None

            loss, log = compute_stage2_loss(
                student_logits=logits,
                teacher_logits=t_logits,
                labels=labels,
                model_student=student_model,
                alpha=cfg.loss.alpha,
                beta=cfg.loss.beta1,
                gamma=cfg.loss.gamma,
                T=cfg.loss.temperature,
                lambda_val=cfg.loss.lipschitz_lambda
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Stage2] Epoch {epoch+1} | Loss: {total_loss / max(1, len(train_loader)):.4f}")

        if cfg.train.save_dir:
            os.makedirs(cfg.train.save_dir, exist_ok=True)
            torch.save({
                'student_feature': student_model.extractor.state_dict(),
                'student_classifier': student_model.classifier.state_dict(),
            }, os.path.join(cfg.train.save_dir, f"stage2_epoch{epoch+1}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, choices=[1, 2], required=True, help='Which stage to train')
    args = parser.parse_args()
    if args.stage == 1:
        train_stage1()
    else:
        train_stage2()
