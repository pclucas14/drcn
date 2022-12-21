from args import get_args

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from data_module import ImageDataModule
from pl_model import ImageModel

def run(args):
    seed_everything(args.seed, workers=True)

    # data
    dm = ImageDataModule(args)

    # model
    model = ImageModel(args)

    try:
        import wandb
        # online logging 
        logger = pl.loggers.WandbLogger(
            project='drcn-repro',
            name=args.method
        )
    except:
        logger = None
    
    # model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val_src_cls_loss",
        filename="{epoch:02d}-{val_src_cls_loss:.004f}",
        save_top_k=1,
        mode="min",
    )
    trainer = Trainer(
        gpus=1, # for more than 1 gpu, will need a DistributedSampler
        logger=logger,
        accelerator="gpu",
        num_sanity_val_steps=0,
        callbacks=checkpoint_callback,
        default_root_dir=args.output_dir,
        gradient_clip_val=args.max_grad_norm,
        log_every_n_steps=1 if args.debug else 10,
        limit_val_batches=10 if args.debug else 1.0,
        limit_train_batches=5 if args.debug else 1.0,
        max_epochs=2 if args.debug else args.epochs,
        replace_sampler_ddp=args.no_mix_src_and_tgt,
    )
    trainer.fit(model, dm)
    trainer.test(model, dm, ckpt_path='best')

if __name__ == "__main__":
    args = get_args()
    run(args)
