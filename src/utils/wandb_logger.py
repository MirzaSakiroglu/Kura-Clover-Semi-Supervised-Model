try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False

def init_wandb(conf, run_name):
    """Initialize a W&B run. Returns the run object."""
    config_dict = {
        "architecture": conf.model.architecture.value,
        "encoder_name": conf.model.config.encoder_name,
        "classes": conf.model.config.classes,
        "epochs": conf.training.epochs,
        "batch_size_labeled": conf.batch_size.labeled,
        "batch_size_unlabeled": conf.batch_size.unlabeled,
        "learning_rate": conf.optimizer.optimizer_params.lr,
        "loss": conf.loss.name.value,
        "tau": conf.flexmatch.tau,
        "flexmatch_warmup": conf.flexmatch.warmup,
    }
    
    # Try to safely get wandb config section if it exists, otherwise default
    wandb_conf = getattr(conf, "wandb", None)
    if hasattr(wandb_conf, "project"):
        project_name = wandb_conf.project
    else:
        project_name = "kura-clover-ssl"
        
    if hasattr(wandb_conf, "tags"):
        tags = wandb_conf.tags
    else:
        tags = ["flexmatch"]
    
    if not _WANDB_AVAILABLE:
        raise ImportError("wandb is not installed. Run: pip install wandb")

    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config_dict,
        tags=tags
    )
    return run

def log_step(run, step, metrics_dict):
    """Log a dict of scalars at a given step."""
    if run is not None:
        metrics_dict["global_step"] = step
        wandb.log(metrics_dict)

def log_epoch(run, epoch, metrics_dict, artifacts=None):
    """Log epoch-level metrics + optional artifacts (images, checkpoints)."""
    if run is not None:
        metrics_dict["epoch"] = epoch
        wandb.log(metrics_dict)
