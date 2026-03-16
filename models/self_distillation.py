"""
Self-Distillation Module for model compression.
Trains a lightweight student network to mimic teacher representations.
Based on the InGARSS 2025 paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentEncoder(nn.Module):
    """
    Lightweight student network for distillation.
    Smaller than teacher but learns to mimic its representations.
    """

    def __init__(self, input_dim=384, output_dim=256):
        """
        Args:
            input_dim: Input feature dimension (teacher output)
            output_dim: Output dimension (should match teacher)
        """
        super().__init__()

        # Lightweight projection
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DistillationLoss(nn.Module):
    """
    Distillation loss combining reconstruction and feature matching.
    """

    def __init__(self, temperature=4.0, alpha=0.5):
        """
        Args:
            temperature: Temperature for softening distributions
            alpha: Weight for distillation loss vs reconstruction loss
                   loss = alpha * distill_loss + (1-alpha) * recon_loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_features, teacher_features, recon_loss):
        """
        Compute combined distillation and reconstruction loss.

        Args:
            student_features: Student network output (B, D)
            teacher_features: Teacher network output (B, D)
            recon_loss: Reconstruction loss from decoder

        Returns:
            Total loss
        """
        # MSE distillation loss (feature matching)
        distill_loss = F.mse_loss(student_features, teacher_features.detach())

        # Combined loss
        total_loss = self.alpha * distill_loss + (1.0 - self.alpha) * recon_loss

        return total_loss, distill_loss, recon_loss


class TeacherStudentWrapper(nn.Module):
    """
    Wrapper combining teacher encoder and student for joint training.
    """

    def __init__(self, teacher_encoder, student_encoder, decoder, freeze_teacher=True):
        """
        Args:
            teacher_encoder: Teacher spatial encoder (ViT)
            student_encoder: Student network
            decoder: Reconstruction decoder
            freeze_teacher: Whether to freeze teacher during training
        """
        super().__init__()

        self.teacher = teacher_encoder
        self.student = student_encoder
        self.decoder = decoder

        if freeze_teacher:
            for param in self.teacher.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass: get both teacher and student features.

        Args:
            x: Input frame (B, 3, H, W)

        Returns:
            teacher_feat: Teacher features (B, D)
            student_feat: Student features (B, D)
            recon: Reconstructed frame (B, 3, H, W)
        """
        with torch.no_grad():
            teacher_feat = self.teacher(x)

        student_feat = self.student(teacher_feat.detach())
        recon = self.decoder(student_feat)

        return teacher_feat, student_feat, recon

    def get_num_params_teacher(self):
        """Get number of teacher parameters."""
        return sum(p.numel() for p in self.teacher.parameters())

    def get_num_params_student(self):
        """Get number of student parameters."""
        return sum(p.numel() for p in self.student.parameters())

    def get_param_reduction(self):
        """Get parameter reduction percentage."""
        teacher_params = self.get_num_params_teacher()
        student_params = self.get_num_params_student()
        reduction = 100 * (1 - student_params / teacher_params)
        return reduction


class KnowledgeDistillationTrainer:
    """
    Trainer for knowledge distillation.
    """

    def __init__(self, model, optimizer, distill_loss, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.distill_loss_fn = distill_loss
        self.device = device

    def train_step(self, batch):
        """
        Single training step with distillation.

        Args:
            batch: Input batch (B, N, 3, H, W) or (B, 3, H, W)

        Returns:
            losses: Dict with total_loss, distill_loss, recon_loss
        """
        # Ensure 4D input (B, 3, H, W)
        if batch.dim() == 5:
            # (B, N, 3, H, W) -> take middle frame
            batch = batch[:, batch.shape[1] // 2, :, :, :]

        batch = batch.to(self.device)

        # Forward pass
        teacher_feat, student_feat, recon = self.model(batch)

        # Reconstruction loss
        recon_loss = F.mse_loss(recon, batch)

        # Distillation loss
        total_loss, distill_loss, recon_l = self.distill_loss_fn(
            student_feat, teacher_feat, recon_loss
        )

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        losses = {
            'total': total_loss.item(),
            'distill': distill_loss.item(),
            'recon': recon_l.item(),
        }

        return losses
