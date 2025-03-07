import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from ._base import Distiller
from .KD import KD, kd_loss

class DA_MSE(KD):
    """Base distiller with Disagreement Augmentation"""
    def DA(self, images):
        fake_student = copy.deepcopy(self.student)
        images = images.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([images], lr=self.cfg.DA.LR)
        for _ in range(self.cfg.DA.EPOCHS):
            logits_student, _ = fake_student(images)
            logits_student = torch.nn.functional.normalize(logits_student, p=1.0, dim=-1)
            logits_teacher, _ = self.teacher(images)
            logits_teacher = torch.nn.functional.normalize(logits_teacher, p=1.0, dim=-1)
            loss = -1 * F.mse_loss(logits_student, logits_teacher)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return images.detach().clone()
        
    def forward_train(self, image, target, **kwargs):
        if torch.rand(1)[0] < self.cfg.DA.PROB:
            image = self.DA(image)

        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature
        )
        
        if torch.isnan(loss_ce):
            print("NAN CE LOSS", flush=True)
            raise ValueError()
        if torch.isnan(loss_kd):
            print("NAN KD LOSS", flush=True)
            raise ValueError()
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict