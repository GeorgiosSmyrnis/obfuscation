#!/usr/bin/env python

from typing import Any, Callable, Tuple, List, Optional

import argparse
import torch
import os

import logging

import numpy as np
import webdataset as wds

import tensorflow as tf

import torch.nn.functional as F

import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import Accuracy, MeanMetric

from embedding_mapper import EmbeddingMapper, TextWrapper

from tqdm import tqdm

from utils import get_model_and_clf, get_optimizer, get_criterion, get_loaders, sigmoid_with_limit

import warnings
warnings.simplefilter("ignore")

IMAGENET_TRAIN_SIZE = 1281167
IMAGENET_VAL_SIZE = 50000

SIMILARITY_OUTPUT_DIR = "/work2/08002/gsmyrnis/frontera/iccv2023/obfuscation/"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, default=None, help="imagenet train dir")
    parser.add_argument("--val-data", type=str, default=None, help="imagenet val dir")
    parser.add_argument("--model", type=str, default=None, help="model")
    parser.add_argument("--epochs", type=int, default=15, help="Num epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-1, help="learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.5, help="lr decay")
    parser.add_argument("--lr_decay_epochs", type=int, default=5, help="decay epochs")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--shuffle-buffer", type=int, default=100, help="shuffle buffer for WDS")
    parser.add_argument("--num-workers", type=int, default=16, help="num workers")
    parser.add_argument("--num-train-obfuscations", type=int, default=16, help="total num of obfuscations")
    parser.add_argument("--epochs-per-eval", type=int, default=5, help="epochs to run per eval")
    parser.add_argument("--log-dir", type=str, default=None, help="checkpoint directory")

    parser.add_argument("--method", type=str, default="linear", help="method")
    parser.add_argument("--keep_mapper", action="store_true", help="keep mapper?")
    parser.add_argument("--time", type=int, default=100, help="Total time steps for diffusion.")

    parser.add_argument("--debug", action="store_true", help="Enable debug.")
    
    args = parser.parse_args()
    return args

def train(
    model: torch.nn.Module,
    embed_map: EmbeddingMapper,
    clf: torch.nn.Linear,
    dl: tf.data.Dataset,
    criterion: Callable[..., torch.Tensor],
    opt: torch.optim.Optimizer,
    method: str,
    args: argparse.Namespace,
    epoch: int
) -> Tuple[Any, Any]:

    train_acc = Accuracy(task="multiclass", num_classes=1000)
    train_loss = MeanMetric()

    if torch.cuda.is_available():
        train_acc.cuda()
        train_loss.cuda()

    model.eval()
    embed_map.train()
    clf.train()

    num_steps_per_epoch = IMAGENET_TRAIN_SIZE // args.batch_size

    for i, item in tqdm(enumerate(iter(dl.take(num_steps_per_epoch))), total=num_steps_per_epoch):

        images = torch.from_numpy(item["image"].numpy()).permute(0, 3, 1, 2)
        labels = torch.from_numpy(item["label"].numpy())
        bsz = args.batch_size

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        opt.zero_grad()

        with torch.no_grad():
            all_embeds = model(images)
        
        if method == "linear":
            preds = clf(all_embeds)
            loss = criterion(preds, labels)   
        elif method == "param_gen":
            raise NotImplementedError()
            # gen_weight = sigmoid_with_limit(epoch, 20)
            # clean_embeds = all_embeds[:bsz, ...]
            # real_embeds = all_embeds[bsz:, ...]
            # gen_embeds, _ = embed_map(clean_embeds, contexts=context)
            # clean_preds = clf(clean_embeds)
            # real_preds = clf(real_embeds)
            # gen_preds = clf(gen_embeds)
            # preds = torch.cat([real_preds, gen_preds], dim=0)
            # loss = criterion(clean_preds, real_embeds, real_preds, gen_embeds, gen_preds, gen_weight, labels)
        elif method == "diffusion":
            gen_weight = sigmoid_with_limit(epoch, 10)
            clean_embeds = all_embeds[:bsz, ...]
            real_embeds = all_embeds[bsz:, ...]
            noise_pred, noise_true = embed_map(clean_embeds)
            gen_embeds = embed_map.get_sample(real_embeds, training=True)
            clean_preds = clf(clean_embeds)
            real_preds = clf(real_embeds)
            gen_preds = clf(gen_embeds)
            clean_labels = labels[:bsz, ...]
            obf_labels = labels[bsz:, ...]
            preds = torch.cat([clean_preds, gen_preds], dim=0)
            loss = criterion(clean_preds, real_preds, gen_preds, noise_pred, noise_true, gen_weight, clean_labels, obf_labels)
        elif method == "mlp_text":
            gen_weight = sigmoid_with_limit(epoch, 10)
            clean_embeds = all_embeds[:bsz, ...]
            real_embeds = all_embeds[bsz:, ...]
            clean_labels = labels[:bsz, ...]
            obf_labels = labels[bsz:, ...]
            gen_embeds, text_embeds = embed_map(real_embeds, obf_labels)
            clean_preds = clf(clean_embeds)
            real_preds = clf(real_embeds)
            gen_preds = clf(gen_embeds)            
            preds = torch.cat([clean_preds, gen_preds], dim=0)
            loss = criterion(clean_preds, real_embeds, real_preds, gen_embeds, gen_preds, gen_weight, clean_labels, obf_labels, text_embeds)
        elif method == "diffusion_text":
            gen_weight = sigmoid_with_limit(epoch, 10)
            clean_embeds = all_embeds[:bsz, ...]
            real_embeds = all_embeds[bsz:, ...]
            clean_labels = labels[:bsz, ...]
            obf_labels = labels[bsz:, ...]
            noise_pred, noise_true, text_embeds = embed_map(clean_embeds, obf_labels)
            gen_embeds = embed_map.base_mapper.get_sample(real_embeds, training=True)
            clean_preds = clf(clean_embeds)
            real_preds = clf(real_embeds)
            gen_preds = clf(gen_embeds)
            preds = torch.cat([clean_preds, gen_preds], dim=0)
            loss = criterion(clean_preds, real_preds, gen_preds, noise_pred, noise_true, gen_weight, clean_labels, obf_labels, gen_embeds, text_embeds)
        else:
            raise NotImplementedError()

        loss.backward()
        opt.step()

        train_acc(preds, labels)
        train_loss(loss)

    train_acc_avg = train_acc.compute()
    train_loss_avg = train_loss.compute()

    return train_loss_avg, train_acc_avg


def val(
    model: torch.nn.Module,
    clf: torch.nn.Linear,
    dl_list: List[tf.data.Dataset],
    embed_map: Optional[EmbeddingMapper] = None,
    debug: bool = False,
) -> Tuple[Any, Any]:

    val_criterion = torch.nn.CrossEntropyLoss()

    val_accs = [Accuracy(task="multiclass", num_classes=1000) for _ in range(len(dl_list))]
    val_loss = MeanMetric()

    if torch.cuda.is_available():
        for val_acc in val_accs:
            val_acc.cuda()
        val_loss.cuda()

    if debug:
        text_embedding_mapper = TextWrapper(embed_map)

    model.eval()
    clf.eval()
    if embed_map is not None:
        embed_map.eval()
    with torch.no_grad():
        for obf_idx in range(len(dl_list)):
            dl = dl_list[obf_idx]

            similarities = []

            for i, item in tqdm(enumerate(iter(dl))):
                
                images = torch.from_numpy(item["image"].numpy()).permute(0, 3, 1, 2)
                labels = torch.from_numpy(item["label"].numpy())
                
                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)

                embeds = model(images)
                if embed_map is not None and not debug:
                    embeds = embed_map(embeds)
                preds = clf(embeds)
                loss = val_criterion(preds, labels)
                val_accs[obf_idx](preds, labels)
                val_loss(loss)

                if debug:
                    _, text_embeds = text_embedding_mapper(embeds, labels)
                    image_embeds = embeds / torch.linalg.norm(embeds, axis=1, keepdim=True)
                    text_embeds = text_embeds / torch.linalg.norm(text_embeds, axis=1, keepdim=True)
                    logits = image_embeds @ text_embeds.T
                    similarity = torch.diag(logits)
                    similarities.append(similarity.cpu().numpy())

            if debug:
                plt.figure()
                plt.hist(similarities, bins=np.linspace(0,1,50))
                plt.title(f"Similarities {obf_idx}")
                plt.tight_layout()
                plt.savefig(os.path.join(SIMILARITY_OUTPUT_DIR, f"{obf_idx:02}.png"))

    val_accs_avg = [val_acc.compute() for val_acc in val_accs]
    val_loss_avg = val_loss.compute()

    return val_loss_avg, val_accs_avg


def main():
    args = parse_args()

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    model, embed_map, clf = get_model_and_clf(args)

    train_loader, val_loaders = get_loaders(args)

    criterion = get_criterion(args)

    opt, sched = get_optimizer((embed_map, clf), args)

    with SummaryWriter(args.log_dir) as writer:
        for t in range(1, args.epochs+1):

            train_loss, train_acc = train(model, embed_map, clf, train_loader, criterion, opt, args.method, args, t-1)

            logging.info(
                f"Epoch {t:03}: Train Loss = {train_loss.item():.4f}, Train Acc = {train_acc.item():.3f}"
            )
            writer.add_scalar("Loss/Train", train_loss.item(), global_step=t)
            writer.add_scalar("Acc/Train/0", train_acc.item(), global_step=t)                

            sched.step()

            if t % args.epochs_per_eval == 0:
                if args.keep_mapper:
                    val_loss, val_accs = val(model, clf, val_loaders, embed_map, args.debug)  # type: ignore
                else:
                    val_loss, val_accs = val(model, clf, val_loaders, None, args.debug)  # type: ignore
                logging.info(
                    f"Epoch {args.epochs:03}: Val Loss = {val_loss.item():.4f}, Val Acc = {val_accs[0].item():.3f}"
                )
                writer.add_scalar("Loss/Val", val_loss.item(), global_step=args.epochs)
                for obf_idx in range(1, len(val_loaders)):
                    writer.add_scalar(f"Acc/Val/{obf_idx:02}", val_accs[obf_idx].item(), global_step=args.epochs)

                for obf_idx in range(len(val_loaders)):
                    logging.info(f"Type {obf_idx:03}: Val Acc = {val_accs[obf_idx].item():.3f}")
                torch.save({"embed_map": embed_map.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict()}, os.path.join(args.log_dir, f"checkpoint_{t}.pth"))

    if args.debug:
        val_loss, val_accs = val(model, clf, val_loaders, embed_map, args.debug)
    elif args.keep_mapper:
        val_loss, val_accs = val(model, clf, val_loaders, embed_map=embed_map)  # type: ignore
    else:
        val_loss, val_accs = val(model, clf, val_loaders, None, args.debug)  # type: ignore
    logging.info(
        f"Epoch {args.epochs:03}: Val Loss = {val_loss.item():.4f}, Val Acc = {val_accs[0].item():.3f}"
    )
    writer.add_scalar("Loss/Val", val_loss.item(), global_step=args.epochs)
    for obf_idx in range(1, len(val_loaders)):
        writer.add_scalar(f"Acc/Val/{obf_idx:02}", val_accs[obf_idx].item(), global_step=args.epochs)

    for obf_idx in range(len(val_loaders)):
        logging.info(f"Type {obf_idx:03}: Val Acc = {val_accs[obf_idx].item():.3f}")
    torch.save({"embed_map": embed_map.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict()}, os.path.join(args.log_dir, "checkpoint.pth"))

if __name__ == "__main__":
    main()
