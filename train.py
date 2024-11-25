import logging

logging.getLogger().setLevel(logging.INFO)
from torch.utils.data import DataLoader
from process.h5_data_loader import (
    SpeechGestureDataset,
    RandomSampler,
)
import torch
import yaml
from easydict import EasyDict
from configs.parse_args import parse_args
import time
from utils.model_util import create_gaussian_diffusion
from train.training_multi import TrainLoop_multi
from model.ADM import AdaptiveDiffuseMotion
from diffusion import logger

def train_multi_task(args):
    """从这里开始训练生成 blendshape 和 bvh

    Parameters
    ----------
    args : 传入的参数

    """
    logger.configure(args)
    trn_dataset = SpeechGestureDataset(
        args.h5file,
        motion_dim=args.motion_dim,
        facial_dim=args.facial_dim,
        style_dim=args.style_dim,
        sequence_length=args.n_poses,
        npy_root="./process",
        version=args.version,
        dataset=args.dataset,
    )

    train_loader = DataLoader(
        trn_dataset,
        num_workers=args.num_workers,
        sampler=RandomSampler(0, len(trn_dataset)),
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
    )

    model = AdaptiveDiffuseMotion(
        modeltype="",
        njoints=args.njoints,
        nfeats=1,
        nexpressions=args.nexpressions,
        arch="trans_enc",
        latent_dim=args.latent_dim,
        n_seed=args.n_seed,
        cond_mask_prob=args.cond_mask_prob,
        device=device_name,
        style_dim=args.style_dim,
        source_audio_dim=args.audio_feature_dim,
        audio_feat_dim_latent=args.audio_feat_dim_latent,
    )
    model.to(mydevice)
    diffusion = create_gaussian_diffusion()
    TrainLoop_multi(args, model, diffusion, mydevice, data=train_loader).run_loop()


if __name__ == "__main__":

    args = parse_args()
    device_name = "cuda:" + args.gpu
    mydevice = torch.device(device_name)
    torch.cuda.set_device(int(args.gpu))
    args.no_cuda = args.gpu

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v

    config = EasyDict(config)

    print(config.name)

    time_local = time.localtime()
    name_expend = "%02d%02d_%02d%02d%02d_" % (
        time_local[1],
        time_local[2],
        time_local[3],
        time_local[4],
        time_local[5],
    )
    config.name = name_expend + config.name

    if "AdaptiveDiffuseMotion" in config.name:
        print("training multi task...")
        train_multi_task(config)
    
