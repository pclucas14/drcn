import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from collections import defaultdict

class Encoder(nn.Module):
    """
    Encoder Architecture described in the paper
    """
    def __init__(self, img_size, in_channels, fc4, fc5):
        super().__init__()
        
        # same padding
        self.cnn_1 = nn.Sequential(
            nn.Conv2d(in_channels, 100, 5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.cnn_2 = nn.Sequential(
            nn.Conv2d(100, 150, 5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.cnn_3 = nn.Sequential(
            nn.Conv2d(150, 200, 3, padding=1),
            nn.ReLU(True),
        )

        self.mlp = nn.Sequential(
            nn.Linear((img_size // 4) ** 2 * 200, fc4),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(fc4, fc5),
            nn.ReLU(),
            nn.Dropout(),
        )

    def forward(self, x):
        bs, C, H, W = x.size()
        x = self.cnn_1(x)
        x = self.cnn_2(x)
        x = self.cnn_3(x)
        x = x.view(bs, -1)
        x = self.mlp(x)

        return x


class Decoder(nn.Module):
    """
    Decoder Architecture described in the paper
    """
    def __init__(self, img_size, out_channels, fc4, fc5):
        super().__init__()
        
        # same padding
        self.img_size = img_size
        self.cnn_m1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(100, out_channels, 5, padding=2),
        )
        self.cnn_m2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(150, 100, 5, padding=2),
            nn.ReLU(True),
        )
        self.cnn_m3 = nn.Sequential(
            nn.Conv2d(200, 150, 3, padding=1),
            nn.ReLU(True),
        )
        # No dropout for this one
        self.mlp = nn.Sequential(
            nn.Linear(fc5, fc4),
            nn.ReLU(True),
            nn.Linear(fc4, (img_size // 4) ** 2 * 200),
            nn.ReLU(True),
        )

    def forward(self, x):
        bs, _ = x.size()
        x = self.mlp(x)
        x = x.view(bs, 200, self.img_size // 4, self.img_size // 4)
        x = self.cnn_m3(x)
        x = self.cnn_m2(x)
        x = self.cnn_m1(x)
        return x

class ImageModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # log hyperparameters
        self.save_hyperparameters(vars(args))

        # --- build model
        self.encoder = Encoder(
            args.img_size, args.in_channels, args.fc4, args.fc5,
        )
        if args.method.startswith("drcn"):
            self.decoder = Decoder(
                args.img_size, args.in_channels, args.fc4, args.fc5,
        )
        self.fc_out = nn.Linear(args.fc5, args.n_classes)

        # --- build losses
        self.mse_loss = F.mse_loss
        self.ce_loss = F.cross_entropy

        # --- ** new ** : target data learnable normalization
        if args.learned_tgt_norm:
            self.tgt_shift = nn.Parameter(torch.zeros(1, args.in_channels, 1, 1))
            self.tgt_scale = nn.Parameter(torch.ones(1, args.in_channels, 1, 1))
        else:
            self.tgt_shift, self.tgt_scale = 0, 1

    def encode(self, x, src=True):
        """
        Wrapper over encoder model. Enables for learnable normalization of target data
        """
        if isinstance(src, bool) and src: 
            x = (x - self.tgt_shift) * self.tgt_scale
        elif isinstance(src, torch.Tensor) and (~src).any():
            x[~src] = (x[~src] - self.tgt_shift)  * self.tgt_scale
        return self.encoder(x)

    def configure_optimizers(self):
        optims = {'adam': torch.optim.Adam, 'rmsprop': torch.optim.RMSprop}
        optim = optims[self.args.optim]
        return optim(
            self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
    
    def noisy(self, x):
        """
        Noise function for denoising objective
        """
        gaussian_x = x + torch.rand_like(x) * self.args.noise_std
        mask = torch.bernoulli(torch.ones_like(x) - self.args.noise_p_drop)
        return gaussian_x * mask

    def training_step(self, data, batch_idx):
        # expand data
        x, y, is_source = data
        x_src, y_src = x[is_source], y[is_source]
        x_tgt, _     = x[~is_source], y[~is_source]

        # loss placeholders
        cls_loss = recon_loss = torch.zeros(1, device=x.device)

        if is_source.any():
            src_enc_out = self.encode(x_src, src=True)
            logits = self.fc_out(src_enc_out)
            cls_loss = self.ce_loss(logits, y_src)

            if self.args.method == "drcn-s":
                src_noisy_enc_out = self.encode(self.noisy(x_src), src=True)
                dec_out = self.decoder(src_noisy_enc_out)
                recon_loss = self.mse_loss(dec_out, x_src)

        if self.args.method == "drcn-st":
            enc_out = self.encode(self.noisy(x), src=is_source)
            dec_out = self.decoder(enc_out)
            recon_loss = self.mse_loss(dec_out, x)
        elif self.args.method == "drcn" and (~is_source).any():
            tgt_noisy_enc_out = self.encode(self.noisy(x_tgt), src=False)
            dec_out = self.decoder(tgt_noisy_enc_out)
            recon_loss = self.mse_loss(dec_out, x_tgt)

        lamb = self.args.lamb
        loss = lamb * cls_loss + (1.0 - lamb) * recon_loss
        self.log_dict(
            {
                "train/cls_loss": cls_loss.item(),
                "train/recon_loss": recon_loss.item(),
                "train/loss": loss.item(),
            }, 
            on_epoch=True
        )
        return loss

    def inference_step(self, data, mode):
        x, y, is_source = data
        x_src, y_src = x[is_source], y[is_source]
        x_tgt, y_tgt = x[~is_source], y[~is_source]

        to_log = {}
        enc_out = self.encode(x, src=is_source)
        logits = self.fc_out(enc_out)
        
        if is_source.any():
            src_cls_loss = F.cross_entropy(enc_out[is_source], y_src)
            src_acc = logits[is_source].argmax(1).eq(y_src).float().mean()
            to_log['src_cls_loss'] = src_cls_loss.item()
            to_log['src_acc'] = src_acc.item()

            if self.args.method.startswith('drcn'):
                src_recon = self.decoder(self.encode(self.noisy(x_src), src=True))
                src_recon_loss = self.mse_loss(src_recon, x_src)
                to_log['src_recon_loss'] = src_recon_loss.item()

        if (~is_source).any():
            tgt_cls_loss = F.cross_entropy(enc_out[~is_source], y_tgt)
            tgt_acc = logits[~is_source].argmax(1).eq(y_tgt).float().mean()
            to_log['tgt_cls_loss'] = tgt_cls_loss.item()
            to_log['tgt_acc'] = tgt_acc.item()

            if self.args.method.startswith('drcn'):
                tgt_recon = self.decoder(self.encode(self.noisy(x_tgt), src=False))
                tgt_recon_loss = self.mse_loss(tgt_recon, x_tgt)
                to_log['tgt_recon_loss'] = tgt_recon_loss

        self.log_dict({f'{mode}_{k}': v for k,v in to_log.items()}, on_epoch=True)
        
        return to_log

    def validation_step(self, data, batch_idx):
        return self.inference_step(data, 'val')
    
    def test_step(self, data, batch_idx):
        return self.inference_step(data, 'test')

    def validation_epoch_end(self, outputs):
        # average the values
        result = defaultdict(list)
        for output in outputs:
            for k,v in output.items():
                result[k] += [v]

        result = {k:sum(v) / len(v) for k,v in result.items()}
        print([f'{k}:{v:.5f}' for k,v in result.items()])