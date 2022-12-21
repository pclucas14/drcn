from args import get_args_parser

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from data_module import ImageDataModule
# from pl_model import ImageModel

def run_multitask(args):
    seed_everything(args.seed, workers=True)

    # data
    dm = PretrainDataModule(args)

    # model + opt
    if args.n_tasks is None:
        args.n_tasks = len(dm.task2id)

    module = EncoderDecoder(**vars(args))

    # legit logging
    wandb_logger = pl.loggers.WandbLogger(
        project=args.wandb_project,
        name=args.exp_name
    )
    wandb_logger.experiment.save("*.py")
    
    # model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="mbart-{epoch:02d}-{val_loss:.004f}",
        save_top_k=1,
        mode="min",
    )
    trainer = Trainer(
        gpus=-1,
        accelerator="gpu",
        logger=wandb_logger,
        num_sanity_val_steps=0,
        default_root_dir=args.output_dir,
        max_epochs=2 if args.debug else args.num_train_epochs,
        gradient_clip_val=args.max_grad_norm,
        log_every_n_steps=1 if args.debug else 10,
        strategy="ddp_find_unused_parameters_false",
        limit_val_batches=10 if args.debug else 1.0,
        limit_train_batches=5 if args.debug else 1.0,
        callbacks=[ProgressCallback(), checkpoint_callback],
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision=int(args.precision)
        if args.precision in ["16", "32"]
        else args.precision,
    )
    trainer.fit(module, dm)


def run_singletask(args, module=None):
    seed_everything(args.seed, workers=True)

    # data : Note this uses the new args
    dm = FinetuneDataModule(args)

    if module is None:
        # build the pretrained model
        ckpt_path = get_checkpoint_path(args.ckpt_path)

        ckpt_args = torch.load(ckpt_path)['hyper_parameters']
        args.old_exp_name = ckpt_args['exp_name']

        for arg_name in ARGS_TO_OVERWRITE:
            setattr(args, arg_name, ckpt_args[arg_name]) 

        if args.switch_to_avg_selector:
            print('setting `n_skills=1` and `selector=Average`')
            args.pretrain_n_skills = args.n_skills
            args.pretrain_selector = args.selector
            args.n_skills = 1
            args.selector = 'average'

        module = Finetuner.load_from_checkpoint(ckpt_path, **vars(args), strict=False)

    # legit logging
    wandb_logger = pl.loggers.WandbLogger(
        project=args.test_wandb_project,
        name=args.exp_name
    )
    wandb_logger.experiment.save("*.py")

    callbacks = [ProgressCallback()]

    if not args.zeroshot:
        # model checkpointing
        checkpoint_callback = ModelCheckpoint(
            mode="max",
            save_top_k=1,
            monitor=f"val/metric_perf",
            filename="mbart-{epoch:02d}-{val/metric_perf:.2f}",
        )
        callbacks += [checkpoint_callback]

    trainer = Trainer(
        gpus=-1,
        accelerator="gpu",
        logger=wandb_logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        max_steps=args.fs_train_steps,
        default_root_dir=args.output_dir,
        gradient_clip_val=args.max_grad_norm,
        max_epochs=2 if args.debug else 1000,
        log_every_n_steps=5 if args.debug else 50,
        limit_val_batches=10 if args.debug else 1.0,
        strategy="ddp", #_find_unused_parameters_false",
        limit_train_batches=5 if args.debug else 1.0,
        check_val_every_n_epoch=1 if args.debug else 10,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        limit_test_batches=getattr(args, 'limit_test_batches', None)

    )

    if args.zeroshot:
        out = trainer.test(module, dm)
        wandb.finish()
        return out
    else:
        trainer.fit(module, dm)
        best_path = checkpoint_callback.best_model_path
        # module = Finetuner.load_from_checkpoint(best_path, **vars(args), strict=False)
        load_result = module.load_state_dict(torch.load(best_path)['state_dict'], strict=False)
        assert len(load_result.unexpected_keys) == 0

        # manually loading best model
        out = trainer.test(module, dm) #, ckpt_path=None)
        test_em_perf, test_metric_perf = (
            out[0][f"test/{args.task}/em_perf"],
            out[0][f"test/{args.task}/metric_perf"],
        )
        wandb.finish()

        return {
            'val_perf': checkpoint_callback.best_model_score.item(),
            'test_perf': test_metric_perf,
            'metric': dm.val_wrapper.metric,
            'test_em_perf': test_em_perf,
        }



if __name__ == "__main__":
    args = get_args_parser().parse_args()
    run_multitask(args)
