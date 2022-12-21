import torch
from pytorch_lightning import LightningModule


class Encoder(torch.nn.Module):
    def __init__(self, img_size):
        pass

class Decoder(torch.nn.Module):
    def __init__(self, img_size):
        pass

class ImageModel(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.args = self.hparams

    def training_step(self, x, batch_idx):
        pass
    
    def inference_step(self, x):
        pass