import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch_geometric as tg

import utils.transforms as T
import utils.preprocess as P


parser = argparse.ArgumentParser()
# Data settings
parser.add_argument("--dataset", type=str, default="MD17")
parser.add_argument("--target", type=str, default="aspirin CCSD")
parser.add_argument("--root", type=str, default="data")
parser.add_argument("--log", type=bool, default=False)
parser.add_argument("--gpus", type=int, default=-1)
parser.add_argument("--num_workers", type=int, default=-1)

# Traing settings
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--force_weight", type=float, default=1000)

# Model settings
parser.add_argument("--model", type=str, default="MPNN")
parser.add_argument("--num_types", type=int, default=10)
parser.add_argument("--num_basis", type=int, default=30)
parser.add_argument("--depth", type=int, default=1)
parser.add_argument("--dim", type=int, default=64)
parser.add_argument("--message_depth", type=int, default=1)
parser.add_argument("--update_depth", type=int, default=1)
parser.add_argument("--head_depth", type=int, default=1)
parser.add_argument("--norm", type=str, default="none")
parser.add_argument("--act", type=str, default="silu")
parser.add_argument("--aggr", type=str, default="add")
parser.add_argument("--pool", type=str, default="add")
parser.add_argument("--cutoff", type=float, default=4.0)

args = parser.parse_args()

pool_dict = {
    "add": tg.nn.global_add_pool,
    "mean": tg.nn.global_mean_pool,
    "max": tg.nn.global_max_pool,
}

act_dict = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}

norm_dict = {
    "batch": nn.BatchNorm1d,
    "layer": nn.LayerNorm,
    "instance": nn.InstanceNorm1d,
    "none": None,
}

if __name__ == "__main__":
    # Reproducibility
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)
        deterministic = True
    else:
        deterministic = False

    # Devices
    if args.gpus == -1:
        args.gpus = torch.cuda.device_count()
    if args.num_workers == -1:
        args.num_workers = int(os.cpu_count() / 4)

    # Dataset
    if args.dataset == "MD17":
        transform = [
            T.Kcal2meV(),
            tg.transforms.RadiusGraph(args.cutoff, num_workers=args.num_workers),
        ]
        transform = tg.transforms.Compose(transform)

        if "CCSD" in args.target:
            train_dataset = tg.datasets.MD17(
                args.root, name=args.target, train=True, transform=transform
            )
            train_dataset, valid_dataset = torch.utils.data.random_split(
                train_dataset, [950, 50], torch.Generator().manual_seed(42)
            )

            test_dataset = dataset = tg.datasets.MD17(
                args.root, name=args.target, train=False, transform=transform
            )
        else:
            raise NotImplementedError
            train_dataset = tg.datasets.MD17(
                args.root, name=args.target, transform=transform
            )
        out_dim = 1
        from tasks.MD17 import Task

    dataloaders = [
        tg.loader.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=dataset == train_dataset,
        )
        for dataset in [train_dataset, valid_dataset, test_dataset]
    ]

    # Model
    if args.model == "MPNN":
        from nn.mpnn import MPNN

        model = MPNN(
            num_types=args.num_types,
            num_basis=args.num_basis,
            cutoff=args.cutoff,
            dim=args.dim,
            out_dim=out_dim,
            depth=args.depth,
            message_depth=args.message_depth,
            update_depth=args.update_depth,
            head_depth=args.head_depth,
            norm=norm_dict[args.norm],
            act=act_dict[args.act],
            aggr=args.aggr,
            pooler=pool_dict[args.pool],
        )

    # Task
    if args.dataset == "MD17":
        mean = P.compute_mean(dataloaders[0], "energy")
        rms = P.compute_rms(dataloaders[0], "force")

        print("mean:", mean, "rms:", rms)
        model = Task(
            model=model,
            lr=args.lr,
            force_weight=args.force_weight,
            shift=mean,
            scale=rms,
        )

    # Logging
    if args.log:
        logger = pl.loggers.WandbLogger(
            project="MD17Baseline " + args.dataset,
            name="_".join([args.model]),
            config=args,
        )
    else:
        logger = None

    # Let's go!
    print(model)
    print(args)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        deterministic=deterministic,
    )

    trainer.fit(model, dataloaders[0], dataloaders[1])
    # trainer.test(ckpt_path="best", dataloaders=dataloaders[2])
