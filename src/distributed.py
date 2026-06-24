"""
src/distributed.py
Torch distributed functions
BoMeyering 2025
"""
from omegaconf import OmegaConf
import os
import torch.distributed as dist

def setup_ddp(backend: str):
    """ Set up DDP if in a distributed environment """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend, init_method="env://")

def shutdown_ddp():
    """ Shutdown the distributed processes """
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def set_env_ranks(conf: OmegaConf):
    """ Get ranks and world size for a process, defaulting to single-process if not distributed """
    conf.rank = int(os.environ.get("RANK", 0))
    conf.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    conf.world_size = int(os.environ.get("WORLD_SIZE", 1))
    conf.is_main = is_main_process()

def is_main_process() -> bool:
    """ Returns True if process is the main process """
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0