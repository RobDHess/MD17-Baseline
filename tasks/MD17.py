import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class Task(pl.LightningModule):
    """Graph Neural Network module"""

    def __init__(
        self,
        model,
        lr,
        force_weight=1,
        shift=0,
        scale=1,
    ):
        super().__init__()
        self.model = model
        self.lr = lr

        self.force_weight = force_weight
        self.shift = shift
        self.scale = scale

        self.energy_train_metric = torchmetrics.MeanAbsoluteError()
        self.energy_valid_metric = torchmetrics.MeanAbsoluteError()
        self.energy_test_metric = torchmetrics.MeanAbsoluteError()
        self.force_train_metric = torchmetrics.MeanAbsoluteError()
        self.force_valid_metric = torchmetrics.MeanAbsoluteError()
        self.force_test_metric = torchmetrics.MeanAbsoluteError()

    def forward(self, graph):
        energy, force = self.pred_energy_and_force(graph)
        return energy, force

    def pred_energy_and_force(self, graph):
        graph.pos = torch.autograd.Variable(graph.pos, requires_grad=True)
        pred_energy = self.model(graph)
        sign = -1.0
        pred_force = (
            sign
            * torch.autograd.grad(
                pred_energy,
                graph.pos,
                grad_outputs=torch.ones_like(pred_energy),
                create_graph=True,
                retain_graph=True,
            )[0]
        )
        return pred_energy.squeeze(-1), pred_force

    def energy_and_force_loss(self, graph, energy, force):
        loss = F.mse_loss(energy, (graph.energy - self.shift) / self.scale)
        loss += self.force_weight * F.mse_loss(force, graph.force / self.scale)
        return loss

    def training_step(self, graph):
        energy, force = self(graph)

        # print("train", energy * self.scale + self.shift - graph.energy)

        loss = self.energy_and_force_loss(graph, energy, force)
        self.energy_train_metric(energy * self.scale + self.shift, graph.energy)
        self.force_train_metric(force * self.scale, graph.force)

        # cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # self.log("lr", cur_lr, prog_bar=True, on_step=True)
        return loss

    def on_train_epoch_end(self):
        self.log("Energy train MAE", self.energy_train_metric, prog_bar=True)
        self.log("Force train MAE", self.force_train_metric, prog_bar=True)

    @torch.inference_mode(False)
    def validation_step(self, graph, batch_idx):
        energy, force = self(graph)
        # print("valid", energy * self.scale + self.shift - graph.energy)
        self.energy_valid_metric(energy * self.scale + self.shift, graph.energy)
        self.force_valid_metric(force * self.scale, graph.force)

    def on_validation_epoch_end(self):
        self.log("Energy valid MAE", self.energy_valid_metric, prog_bar=True)
        self.log("Force valid MAE", self.force_valid_metric, prog_bar=True)

    @torch.inference_mode(False)
    def test_step(self, graph, batch_idx):
        energy, force = self(graph)
        self.energy_test_metric(energy * self.scale + self.shift, graph.energy)
        self.force_test_metric(force * self.scale, graph.force)

    def on_test_epoch_end(self):
        self.log("Energy test MAE", self.energy_test_metric, prog_bar=True)
        self.log("Force test MAE", self.force_test_metric, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.estimated_stepping_batches
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler_config]
