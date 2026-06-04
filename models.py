
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch.nn as nn
from torchvision import models


class StrongModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels,
                 out_classes, freeze_encoder=True, **kwargs):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels,
            classes=out_classes, **kwargs
        )

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Preprocessing parameters for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std",
                             torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean",
                             torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Dice loss for image segmentation
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE,
                                           from_logits=True)

        # Track the best IoU across epochs
        self.best_valid_iou = torch.tensor(0.0)

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        x, y = batch
        image = x
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = y
        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(),
                                               mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp,
                                              fn, tn,
                                              reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        self.log(f"{stage}_per_image_iou", per_image_iou, prog_bar=True)
        self.log(f"{stage}_dataset_iou", dataset_iou, prog_bar=True)

        if stage == "valid":
            print(f"Epoch {self.current_epoch + 1}: valid_per_image_iou={per_image_iou:.4f}, validation_dataset_iou={dataset_iou:.4f}")
            if per_image_iou > self.best_valid_iou:
                self.best_valid_iou = per_image_iou
                print(f"New best valid_per_image_iou: {self.best_valid_iou:.4f}")

    def training_step(self, batch, batch_idx):
        result = self.shared_step(batch, "train")
        self.training_step_outputs.append(result["loss"])
        return result

    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_average", epoch_average,
                 on_epoch=True, prog_bar=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        result = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(result)
        return result

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        result = self.shared_step(batch, "test")
        self.test_step_outputs.append(result)
        return result

    def on_test_epoch_end(self):
        epoch_average = torch.stack([x["loss"] for x in self.test_step_outputs]).mean()
        self.log("test_epoch_average", epoch_average, prog_bar=True)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.00005)


class SwinBinaryClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.swin_t(pretrained=pretrained)
        self.backbone.head = nn.Identity()
        self.classifier = nn.Linear(768, 1)

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)
