"""
train_semisup.py
Main training script for the Kura Clover Semi-Supervised Model (FlexMatch)
"""

import torch
import os
import logging
import argparse
import omegaconf
import torch.distributed as dist
from argparse import ArgumentParser
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Local imports
from src.models import create_smp_model
from src.datasets import LabeledDataset, UnlabeledDataset
from src.trainer import FlexMatchTrainer, SupervisedTrainer
from src.metrics import MeterSet, RunningAvgMeter
from src.losses import get_loss_criterion, read_class_counts
from src.parameters import OptimConfig, EMA
from src.transforms import get_train_transforms, get_val_transforms, get_weak_transforms, set_normalization_values
from src.utils.device import set_torch_device
from src.utils.config import TrainSemiSupervisedConfig, set_run_name
from src.utils.loggers import setup_loggers, rank_log
from src.distributed import set_env_ranks, setup_ddp, shutdown_ddp
from src.utils.wandb_logger import init_wandb

# Create a parser for command line arguments
parser = ArgumentParser(
    prog="train_semisup.py",
    description="Main training script for the semi-supervised Kura clover segmentation model."
)
parser.add_argument('-c', '--config', type=str, help="The path to the training config YAML file.", default='configs/train_semisup_config.yaml')
parser.add_argument('-b', '--backend', type=str, help="The backend to use for torchrun. Defaults to 'gloo'", default='gloo')
args = parser.parse_args()

if not os.path.exists(args.config):
    raise FileNotFoundError(f"The path to the configuration file {args.config} was not found.")

# Set the backend engine for torchrun
backend = args.backend
setup_ddp(backend=backend)

# Set up configuration objects
yaml_conf = OmegaConf.load(args.config)
default_conf = OmegaConf.structured(TrainSemiSupervisedConfig)
conf = OmegaConf.merge(default_conf, yaml_conf)

set_env_ranks(conf)
set_run_name(conf)
setup_loggers(conf)
logger = logging.getLogger()

set_torch_device(conf)
set_normalization_values(conf)

if conf.loss.get('class_sample_count_path', None) is not None:
    samples, inv_weights = read_class_counts(conf.loss.class_sample_count_path)
    conf.loss.samples, conf.loss.weights = samples, inv_weights
    rank_log(conf.is_main, logger.info, f"Set conf.loss.samples and conf.loss.weights from {conf.loss.class_sample_count_path}")
else:
    if OmegaConf.is_missing(conf.loss, 'samples'):
        pass
    else:
        del conf.loss.samples
        del conf.loss.weights

def main(conf: omegaconf.OmegaConf=conf):
    rank_log(conf.is_main, logger.info, "Current Semi-Supervised Training Configuration\n"+OmegaConf.to_yaml(conf))

    # Initialize W&B
    wandb_run = None
    if conf.is_main:
        try:
            wandb_run = init_wandb(conf, run_name=conf.model_run)
        except Exception as e:
            rank_log(conf.is_main, logger.warning, f"W&B init failed: {e}. Continuing without W&B.")

    # Create and wrap model
    model = create_smp_model(conf=conf).to(conf.device)
    
    if dist.is_available() and dist.is_initialized():
        model = DDP(
            model, 
            device_ids=[conf.local_rank] if 'cuda' in conf.device else None, 
            output_device=conf.local_rank if 'cuda' in conf.device else None, 
            find_unused_parameters=True
        )
        rank_log(conf.is_main, logger.info, f"Created DDP model {conf.model.architecture.value} with encoder {conf.model.config.encoder_name}")
    else:
        rank_log(conf.is_main, logger.info, f"Created local model {conf.model.architecture.value} with encoder {conf.model.config.encoder_name} on {conf.device}")
    
    # Augmentation Pipelines
    train_transforms = get_train_transforms(resize=tuple(conf.images.resize))
    val_transforms = get_val_transforms(resize=tuple(conf.images.resize))
    weak_transforms = get_weak_transforms(resize=tuple(conf.images.resize))

    # Create Datasets
    train_l_ds = LabeledDataset(
        root_dir=conf.directories.train_labeled_dir,
        transforms=train_transforms
    )
    
    train_u_ds = UnlabeledDataset(
        root_dir=conf.directories.train_unlabeled_dir,
        weak_transforms=weak_transforms
    )

    val_ds = LabeledDataset(
        root_dir=conf.directories.val_dir,
        transforms=val_transforms
    )

    rank_log(conf.is_main, logger.info, f"Labeled train: {len(train_l_ds)} images")
    rank_log(conf.is_main, logger.info, f"Unlabeled train: {len(train_u_ds)} images")
    rank_log(conf.is_main, logger.info, f"Validation: {len(val_ds)} images")

    # Samplers
    train_l_sampler = DistributedSampler(
        dataset=train_l_ds, 
        num_replicas=conf.world_size,
        rank=conf.rank, 
        shuffle=True, 
        drop_last=True
    )
    
    train_u_sampler = DistributedSampler(
        dataset=train_u_ds, 
        num_replicas=conf.world_size,
        rank=conf.rank, 
        shuffle=True, 
        drop_last=True
    )
    
    val_sampler = DistributedSampler(
        dataset=val_ds,
        num_replicas=conf.world_size,
        rank=conf.rank,
        shuffle=False,
        drop_last=False
    )

    # DataLoaders
    train_l_loader = DataLoader(
        dataset=train_l_ds, 
        batch_size=conf.batch_size.labeled, 
        sampler=train_l_sampler,
        drop_last=True
    )
    
    train_u_loader = DataLoader(
        dataset=train_u_ds, 
        batch_size=conf.batch_size.unlabeled, 
        sampler=train_u_sampler,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=conf.batch_size.labeled,
        sampler=val_sampler,
        drop_last=False
    )

    optim_config = OptimConfig(conf=conf, model=model)
    model, optimizer, scheduler = optim_config.process()
    
    criterion = get_loss_criterion(conf)
    rank_log(conf.is_main, logger.info, f"Instantiated loss criterion {type(criterion)}")

    if conf.optimizer.ema:
        ema = EMA(model, decay=conf.optimizer.ema_decay, verbose=True)
        rank_log(conf.is_main, logger.info, f"EMA enabled with decay={conf.optimizer.ema_decay}")
    else:
        ema = None

    # Length of training epoch (number of iterations)
    train_length = len(train_u_loader)

    # Create MeterSet for loss tracking
    meters = MeterSet({
        'total_loss': RunningAvgMeter(window_length=15),
        'labeled_loss': RunningAvgMeter(window_length=15),
        'unlabeled_loss': RunningAvgMeter(window_length=15),
        'validation_loss': RunningAvgMeter(window_length=15),
        'train_loss': RunningAvgMeter(window_length=15),
    })

    # Stage 1 — Supervised pretraining on labeled data only
    pretrain_epochs = getattr(conf.training, 'supervised_pretrain_epochs', 0)
    if pretrain_epochs > 0:
        rank_log(conf.is_main, logger.info, f"=== STAGE 1: Supervised pretraining for {pretrain_epochs} epochs ===")

        from src.callbacks import CheckpointManager
        pretrain_conf = OmegaConf.merge(conf, OmegaConf.create({'training': {'epochs': pretrain_epochs}}))
        pretrain_checkpoint_manager = CheckpointManager(conf=pretrain_conf, monitor='val_loss', top_k=3)

        pretrain_meters = MeterSet({
            'train_loss': RunningAvgMeter(window_length=15),
            'validation_loss': RunningAvgMeter(window_length=15),
        })

        pretrain_trainer = SupervisedTrainer(
            name="kura_supervised_pretrain",
            meter_set=pretrain_meters,
            tb_writer=None,
            conf=pretrain_conf,
            model=model,
            train_loader=train_l_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_manager=pretrain_checkpoint_manager,
            sanity_check=False,
            ema=ema,
        )
        pretrain_trainer.train()
        rank_log(conf.is_main, logger.info, "=== STAGE 1 complete. Starting semi-supervised training with pretrained weights ===")

    # Stage 2 — Semi-supervised FlexMatch training
    # Initialize FlexMatchTrainer
    trainer = FlexMatchTrainer(
        name="kura_flexmatch_trainer",
        meter_set=meters,
        conf=conf,
        model=model,
        train_loaders=(train_l_loader, train_u_loader),
        train_length=train_length,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_samplers=(train_l_sampler, train_u_sampler),
        ema=ema,
        wandb_run=wandb_run,
        class_map=getattr(conf.metadata, 'target_mapping', None)
    )

    # Start training
    trainer.train()

    shutdown_ddp()

if __name__ == '__main__':
    main()

