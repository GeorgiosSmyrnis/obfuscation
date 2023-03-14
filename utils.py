from typing import Any, Callable, Tuple, Sequence, List

import os
import random
import torch
from argparse import Namespace

import numpy as np
import tensorflow as tf
import torchvision.transforms.functional as F
import webdataset as wds

from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from imagenet_c import corrupt
from image_obfuscation_benchmark.eval.data_utils import get_data, get_mixed_data
from image_obfuscation_benchmark.eval.data_utils import CLEAN, Split, Normalization, get_obfuscations
import embedding_mapper as mapper

import losses as losses_lib

def sigmoid_with_limit(t: int, limit: int) -> float:
    """Calculate a sigmoid that ramps up to 1 after a certain limit.
    This function calculates a sigmoid function, defined as `sigmoid(x)`, where
    `x = 10 * t / limit - 5`. This results in `x` ranging from -5 at 0 to 5 at
    `limit`, and thus the output starts from a value near 0 at `t = 0` and ramps
    up near 1 at `t = limit`.
    Args:
    t: The point at which to evaluate the sigmoid.
    limit: The point at which the sigmoid must attain the value 1-exp(-5),
        which is close enough to 1.
    Returns:
    The value of the sigmoid at the chosen point.
    """
    if limit > 0:
        result = 1/(1+np.exp(-10.0 * float(t) / float(limit) + 5.0))
    else:
        result = 0.2
    return result


class ImageNet100(datasets.ImageFolder):
    """
    Dataset for ImageNet100. Majority of code taken from torchvision.datasets.ImageNet.
    Works in a similar function and has similar semantics to the original class.
    """
    def __init__(self, root, split, transform=None, **kwargs):
        #checking stuff
        root = os.path.expanduser(root)
        if split != 'train' and split != 'val':
            raise ValueError('Split should be train or val.')

        #contains our desired {wnid: class} dictionary
        META_FILE = "meta.bin"

        #initialize parameters from DatasetFolder
        super(ImageNet100, self).__init__(os.path.join(root, split), **kwargs)
        self.root = root
        self.split = split
        self.transform = transform

        #from the dataset folder class, we inherit two properties
        #self.classes is a list of class names based on the folders present in our subset - actually wnids!
        #self.class_to_idx is a dict {wnid: wnid_index} where wnid_index is a number from 0 to 99

        #Load the {wnid: class_name} dictionary from meta.bin
        wnid_to_classes = torch.load(os.path.join(self.root, META_FILE))[0]
        self.wnids = self.classes #current self.classes is actually wnids!
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids] #get the actual class names (e.g. "bird")
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

        #create a dictionary of UNIQUE {index: class values} where the class is the simplest form of the wnid (e.g. common name and not scientific name)
        #this is for CLIP zero-shot classification using the simplest class name
        self.idx_to_class = {idx: cls
                             for idx, clss in enumerate(self.classes)
                             for i, cls in enumerate(clss) if i == 0}


class ObfuscationTransform:

    def __init__(self, num_train_obfuscations: int, is_train: bool) -> None:
        self.num_train_obfuscations = num_train_obfuscations
        self.num_val_obfuscations = num_train_obfuscations + 3
        self.is_train = is_train

    def __call__(self, img_pil):
        if self.is_train:
            obf_idx = random.randint(0, self.num_train_obfuscations-1)
            img_obf = corrupt(
                img_pil,
                severity=5,
                corruption_number=obf_idx
            )
            img = F.to_tensor(img_pil)
            result = torch.cat(
                [img.unsqueeze(0), F.to_tensor(img_obf).unsqueeze(0), (obf_idx + 1)* torch.ones_like(img).unsqueeze(0)],
                dim=0
            )
        else:
            img = F.to_tensor(img_pil)
            obf_imgs = [img.unsqueeze(0)]
            for i in range(self.num_val_obfuscations):
                img_obf = corrupt(
                    img_pil,
                    severity=5,
                    corruption_number=i
                )
                obf_imgs.append(F.to_tensor(img_obf).unsqueeze(0))
            result = torch.cat(obf_imgs, dim=0)
        
        if result.dtype == torch.uint8:
            result = result.float() / 255.0
        else:
            result = result.float()

        return result

def get_model_and_clf(args):
    embed_dim = 2048 if args.model == "resnet50" else 768
    n_cls = 1000

    weights = "IMAGENET1K_V2" if args.model == "resnet50" else "IMAGENET1K_V1"
    model = getattr(models, args.model)(weights)
    if args.model.startswith("resnet"):
        model.fc = torch.nn.Identity()
    elif args.model.startswith("vit"):
        model.heads = torch.nn.Identity()
    for p in model.parameters():
        p.requires_grad = False

    if args.method == "linear":
        embed_map = mapper.IdentityEmbeddingMapper()
    elif args.method == "param_gen":
        embed_map = mapper.ParameterGenerationEmbeddingMapper(
            encoder_decoder_mlp_sizes=[1024, 512],
            param_generator_mlp_sizes=[128],
            context_mlp_sizes=[128],
            embed_dim=embed_dim,
            latent_dim=128,
            context_dim=32,
            num_contexts=18
        )
    elif args.method == "diffusion" or args.method == "diffusion_only":
        embed_map = mapper.DiffusionEmbeddingMapper(
            mlp_sizes=[1024, 512, 256, 512, 1024],
            embed_dim=embed_dim,
            total_time=args.time
        )
    elif args.method == "mlp_text":
        base_embed_map = mapper.MLPEmbeddingMapper(
            mlp_sizes=[1024, 512, 256, 512, 1024],
            input_dim=embed_dim,
            embed_dim=embed_dim,
            final_activation=None
        )

        embed_map = mapper.TextWrapper(base_mapper=base_embed_map)
    elif args.method == "diffusion_text":
        base_embed_map = mapper.DiffusionEmbeddingMapper(
            mlp_sizes=[1024, 512, 256, 512, 1024],
            embed_dim=embed_dim,
            total_time=args.time
        )
        embed_map = mapper.TextWrapper(base_mapper=base_embed_map)
    else:
        raise NotImplementedError()

    clf = torch.nn.Linear(embed_dim, n_cls)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.parallel.DataParallel(model)
            # embed_map = torch.nn.parallel.DataParallel(model)
        model.cuda()
        embed_map.cuda()
        clf.cuda()

    return model, embed_map, clf

def get_criterion(args: Namespace) -> Callable[..., torch.Tensor]:
    if args.method == "linear":
        return torch.nn.CrossEntropyLoss()
    elif args.method == "param_gen":
        return lambda cl, re, rl, ge, gl, gw, clab, olab: (torch.nn.functional.cross_entropy(cl, clab, reduction="mean") + 0.2 * losses_lib.reconstruction_loss(re, ge, loss_type="MSE") + losses_lib.weighted_crossentropy_loss(olab, rl, gl, gw))
    elif args.method == "diffusion":
        return lambda cl, rl, gl, np, nt, gw, clab, olab: (torch.nn.functional.cross_entropy(cl, clab, reduction="mean") + 0.2 * losses_lib.reconstruction_loss(np, nt, loss_type="MSE") + losses_lib.weighted_crossentropy_loss(olab, rl, gl, gw))
    elif args.method == "mlp_text":
        return lambda cl, re, rl, ge, gl, gw, clab, olab, temb: (torch.nn.functional.cross_entropy(cl, clab, reduction="mean") + 0.2 * losses_lib.reconstruction_loss(re, ge, loss_type="MSE") + losses_lib.weighted_crossentropy_loss(olab, rl, gl, gw) + 0.2 * losses_lib.label_clip_loss(ge, temb))
    elif args.method == "diffusion_text":
        return lambda cl, rl, gl, np, nt, gw, clab, olab, ge, temb: (torch.nn.functional.cross_entropy(cl, clab, reduction="mean") + 0.2 * losses_lib.reconstruction_loss(np, nt, loss_type="MSE") + losses_lib.weighted_crossentropy_loss(olab, rl, gl, gw) + 0.2 * losses_lib.label_clip_loss(ge, temb))
    elif args.method == "diffusion_only":
        return torch.nn.MSELoss()
    else:
        raise NotImplementedError()

def get_transforms(args: Namespace) -> Sequence[Callable[[Any], torch.Tensor]]:
    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        ObfuscationTransform(args.num_train_obfuscations, is_train=True),
        normalize
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        ObfuscationTransform(args.num_train_obfuscations, is_train=False),
        normalize
    ])

    return train_tf, val_tf



# def get_loaders(args: Namespace) -> Tuple[DataLoader, DataLoader]:
    
#     train_tf_img, val_tf_img = get_transforms(args)

#     train_ds = ImageNet100(root=args.train_data, split="train", transform=train_tf_img)
#     train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
#     val_ds = ImageNet100(root=args.val_data, split="val", transform=val_tf_img)
#     val_dl = DataLoader(val_ds, batch_size=10, shuffle=False, num_workers=args.num_workers)

#     return train_dl, val_dl

def get_loaders(args: Namespace) -> Tuple[tf.data.Dataset, List[tf.data.Dataset]]:

    train_ds = get_mixed_data(args.train_data, "train", Normalization.IMAGENET_CHANNEL_WISE_NORM, args.batch_size)
    val_ds_list = []
    for obf in get_obfuscations(Split.from_string("test")):
        val_ds_list.append(get_data(
            args.val_data,
            obf,
            "validation",
            Normalization.IMAGENET_CHANNEL_WISE_NORM,
            args.batch_size
        ))

    return train_ds, val_ds_list


def get_optimizer(models: Tuple[torch.nn.Module, torch.nn.Module], args: Namespace) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.StepLR]:
    opt = torch.optim.SGD(
        params=(list(models[0].parameters())+list(models[1].parameters())),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.StepLR(opt, gamma=args.lr_decay, step_size=args.lr_decay_epochs)
    return opt, sched

def get_optimizer_diffusion(model: torch.nn.Module, args: Namespace) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.StepLR]:
    opt = torch.optim.SGD(
        params=model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.StepLR(opt, gamma=args.lr_decay, step_size=args.lr_decay_epochs)
    return opt, sched
