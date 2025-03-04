import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from ._base import Distiller
from .KD import KD, kd_loss

class DA(KD):
    """Base distiller with Disagreement Augmentation"""
    def DA(self, images):
        fake_student = copy.deepcopy(self.student)
        images.requires_grad = True
        optimizer = torch.optim.Adam([images], lr=self.cfg.DA.LR)
        for _ in range(self.cfg.DA.EPOCHS):
            logits_student, _ = fake_student(images)
            logits_teacher, _ = self.teacher(images)
            loss = -1 * kd_loss(
                        logits_student, logits_teacher, 1
                    )
            # loss = -1 * F.mse_loss(logits_student, logits_teacher)
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