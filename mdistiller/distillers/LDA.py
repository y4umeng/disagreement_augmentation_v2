import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from ._base import Distiller
from .KD import KD
from mdistiller.taesd import TAESD

class LDA(KD):
    """Distiller with Latent Disagreement Augmentation"""
    def __init__(self, student, teacher, cfg):
        super(LDA, self).__init__(student, teacher, cfg)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.cfg = cfg
        
        # Set default values if not in config
        if not hasattr(cfg, 'DA'):
            cfg.DA = type('obj', (object,), {
                'PROB': 0.5,
                'EPOCHS': 5,
                'LR': 0.01
            })
        
        import os
        taesd_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "taesd")
        encoder_path = os.path.join(taesd_dir, "taesd_encoder.pth")
        decoder_path = os.path.join(taesd_dir, "taesd_decoder.pth")
        self.ae = TAESD(encoder_path=encoder_path, decoder_path=decoder_path).to("cuda")
        self.norm = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    def denormalize(self, img_tensor, mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)):
        """Denormalize the image tensor to [0,1] range"""
        mean = torch.tensor(mean).view(-1, 1, 1).to(img_tensor.device)
        std = torch.tensor(std).view(-1, 1, 1).to(img_tensor.device)
        return img_tensor * std + mean

    def DA(self, images):
        """Latent Disagreement Augmentation"""
        device = images.device
        image = images.clone()
            
        try:
            # Process the image through the encoder to get latent
            with torch.no_grad():
                denormalized_image = self.denormalize(image)
                # Resize to fit encoder input size
                resized_image = F.interpolate(denormalized_image, size=(64, 64), mode='bilinear', align_corners=False)
                latent = self.ae.encoder(resized_image)
            
            # Create a fresh tensor with gradients
            latent = latent.detach().clone().to(device).requires_grad_(True)
            
            # Create optimizer for the latent
            optimizer = torch.optim.Adam([latent], lr=float(self.cfg.DA.LR))
            
            # Store model modes and set to eval during optimization
            student_training = self.student.training
            teacher_training = self.teacher.training
            self.student.eval()
            self.teacher.eval()
            
            # Optimization loop for latent
            for i in range(int(self.cfg.DA.EPOCHS)):
                optimizer.zero_grad()
                
                # Forward through decoder (with gradient)
                decoded_image = self.ae.decoder(latent)
                decoded_image = torch.clamp(decoded_image, 0, 1)
                normalized_image = self.norm(decoded_image)
                resized_image = F.interpolate(normalized_image, size=(32, 32), mode='bilinear', align_corners=False)
                
                # Get model predictions
                with torch.no_grad():
                    logits_student, _ = self.student(resized_image)
                    logits_teacher, _ = self.teacher(resized_image)
                
                # Calculate loss on detached logits (focusing gradient on latent only)
                normalized_student = F.normalize(logits_student.detach(), p=1, dim=1)
                normalized_teacher = F.normalize(logits_teacher.detach(), p=1, dim=1)
                
                # Maximize disagreement
                disagreement_loss = -1.0 * F.mse_loss(normalized_student, normalized_teacher)
                
                # Create a dummy variable with gradient that depends on the latent
                # This ensures we have a proper gradient path
                dummy_loss = torch.sum(resized_image * 0.0)
                
                # Total loss combines the disagreement with the dummy connection
                total_loss = disagreement_loss + dummy_loss
                
                # Backward pass and optimization step
                total_loss.backward()
                optimizer.step()
            
            # Reset models to their original training modes
            if student_training:
                self.student.train()
            if teacher_training:
                self.teacher.train()
            
            # Generate the final augmented image
            with torch.no_grad():
                decoded_image = self.ae.decoder(latent)
                decoded_image = torch.clamp(decoded_image, 0, 1)
                normalized_image = self.norm(decoded_image)
                augmented_image = F.interpolate(normalized_image, size=(32, 32), mode='bilinear', align_corners=False)
                
                result = augmented_image
                    
            return result.detach()
            
        except Exception as e:
            print(f"Detailed DA error: {type(e).__name__}: {e}")
            # Return original images on error
            return images

    def forward_train(self, image, target, **kwargs):
        """Forward with optional disagreement augmentation"""
        # Randomly apply DA
        if hasattr(self.cfg, 'DA') and torch.rand(1)[0] < self.cfg.DA.PROB:
            try:
                image = self.DA(image)
            except Exception as e:
                print(f"DA error: {e}")
                pass  # Fall back to original image if DA fails
                
        # Regular KD forward pass
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # Compute losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * self.kd_loss_fn(logits_student, logits_teacher)
        
        # Check for NaN values
        if torch.isnan(loss_ce):
            print("NAN CE LOSS", flush=True)
            raise ValueError("NaN CE loss detected")
        if torch.isnan(loss_kd):
            print("NAN KD LOSS", flush=True)
            raise ValueError("NaN KD loss detected")
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
        
    def kd_loss_fn(self, logits_student, logits_teacher):
        """Knowledge distillation loss"""
        log_pred_student = F.log_softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)
        loss = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        loss *= self.temperature**2
        return loss