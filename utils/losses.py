import torch
import torch.nn.functional as F

# -------------------------------------------------
# Cross Entropy Loss
# -------------------------------------------------
def cross_entropy_loss(logits, labels):
    return F.cross_entropy(logits, labels)


# -------------------------------------------------
# KL Divergence Loss with temperature
# -------------------------------------------------
def kl_divergence_loss(student_logits, teacher_logits, T=1.0):
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    teacher_probs     = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')


# -------------------------------------------------
# JSD-based Mutual Information Estimation
# -------------------------------------------------
def jsd_mi_estimation(joint, marginal):
    joint_term   = torch.mean(-F.softplus(-joint))
    marginal_term = torch.mean(F.softplus(marginal))
    return -(joint_term - marginal_term)


# -------------------------------------------------
# Mutual Information Discrepancy (MID) Loss
# -------------------------------------------------
def compute_mid_loss(inputs, features_clean, features_noisy, mi_estimator,mi_estimator_nm):
    """
    MID = (MI_clean - MI_noisy)^2
    :param inputs:           Input images, shape [B, C, H, W]
    :param features_clean:   Teacher features, shape [B, D]
    :param features_noisy:   Student features, shape [B, D]
    :param mi_estimator:     A MINE-like model (inputs, features) → scalar score
    """
    # Estimate MI for clean (teacher)
    shuffled_clean = features_clean[torch.randperm(features_clean.size(0))]
    joint_clean    = mi_estimator(inputs, features_clean)
    marginal_clean = mi_estimator(inputs, shuffled_clean)
    mi_clean       = jsd_mi_estimation(joint_clean, marginal_clean)

    # Estimate MI for noisy (student)
    shuffled_noisy = features_noisy[torch.randperm(features_noisy.size(0))]
    joint_noisy    = mi_estimator_nm(inputs, features_noisy)
    marginal_noisy = mi_estimator_nm(inputs, shuffled_noisy)
    mi_noisy       = jsd_mi_estimation(joint_noisy, marginal_noisy)

    # Mutual Information Discrepancy
    return (mi_clean - mi_noisy).pow(2)


# -------------------------------------------------
# Lipschitz Regularization Loss
# -------------------------------------------------
def lipschitz_regularization(model, lambda_val=1.0):
    """
    ||W^T W - λ²I||² over all Conv and Linear layers
    """
    reg_loss = 0.0
    for name, param in model.named_parameters():
        if 'weight' in name and param.ndim >= 2:
            W = param.view(param.size(0), -1)
            WT_W = torch.matmul(W, W.T)
            I = torch.eye(W.size(0), device=W.device)
            reg_loss += torch.norm(WT_W - lambda_val**2 * I, p='fro')**2
    return reg_loss


# -------------------------------------------------
# Stage 1 Loss: feature + classifier training
# -------------------------------------------------
def compute_stage1_loss(student_logits, teacher_logits,
                        student_feat, teacher_feat,
                        labels, inputs, model_student, mi_estimator,mi_estimator_nm,
                        alpha=0.6, beta1=0.6, beta2=0.6, gamma=0.01,
                        T=1.0, lambda_val=1.0):
    """
    Total = α * CE + β1 * MID + β2 * KL + γ * Reg
    """
    ce_loss  = cross_entropy_loss(student_logits, labels)
    kl_loss  = kl_divergence_loss(student_logits, teacher_logits, T)
    mid_loss = compute_mid_loss(inputs, teacher_feat, student_feat, mi_estimator,mi_estimator_nm)
    reg_loss = lipschitz_regularization(model_student, lambda_val)

    total = alpha * ce_loss + beta1 * mid_loss + beta2 * kl_loss + gamma * reg_loss
    log = {
        "ce": ce_loss.item(),
        "kl": kl_loss.item(),
        "mid": mid_loss.item(),
        "reg": reg_loss.item()
    }
    return total, log


# -------------------------------------------------
# Stage 2 Loss: classifier finetuning
# -------------------------------------------------
def compute_stage2_loss(student_logits, teacher_logits,
                        labels, model_student,
                        alpha=0.6, beta=0.6, gamma=0.01,
                        T=1.0, lambda_val=1.0):
    """
    Total = α * CE + β * KL + γ * Reg
    """
    ce_loss  = cross_entropy_loss(student_logits, labels)
    kl_loss  = kl_divergence_loss(student_logits, teacher_logits, T)
    reg_loss = lipschitz_regularization(model_student, lambda_val)

    total = alpha * ce_loss + beta * kl_loss + gamma * reg_loss
    log = {
        "ce": ce_loss.item(),
        "kl": kl_loss.item(),
        "reg": reg_loss.item()
    }
    return total, log
