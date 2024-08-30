import argparse
import csv
import os.path

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from SSB.get_datasets.get_gcd_datasets_funcs import get_gcd_datasets
from SSB.utils import load_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups
from PIL import Image
from SSB.utils import load_index_to_name, load_class_splits
from torchvision.datasets.folder import default_loader, make_dataset, IMG_EXTENSIONS
import ttach


def result(model, test_loader, args):
    model.eval()

    preds, targets = [], []

    image_names = []
    for batch_idx, (images, image_ids) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        tta_model = ttach.ClassificationTTAWrapper(model, ttach.aliases.d4_transform())
        image_names.append(image_ids)
        with torch.no_grad():
            _, logits = tta_model(images)
            preds.append(logits.argmax(1).cpu().numpy())

    image_names=[item for sublist in image_names for item in sublist]
    preds = np.concatenate(preds)
    data = [["img","pred"]]
    if args.dataset_name == 'aircraft':
        for i in range(len(preds)):
            data.append([image_names[i],preds[i]])
        with open('pred_aircraft.csv', mode='w', newline='') as f1:
            writer = csv.writer(f1)
            # 写入数据
            for row in data:
                writer.writerow(row)
    elif args.dataset_name == 'scars':
        for i in range(len(preds)):
            data.append([image_names[i],preds[i]])
        with open('pred_scars.csv', mode='w', newline='') as f2:
            writer = csv.writer(f2)
            # 写入数据
            for row in data:
                writer.writerow(row)

    elif args.dataset_name == 'cub':
            for i in range(len(preds)):
                data.append([image_names[i] , preds[i]])
            with open('pred_cub.csv', mode='w', newline='') as f3:
                writer = csv.writer(f3)
            # 写入数据
                for row in data:
                    writer.writerow(row)

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self,  transform,args):
        self.transform = transform
        if args.dataset_name == 'aircraft':
            self.metadata = pd.read_csv("starting_k/pred_aircraft.csv")
            self.num_images = len(self.metadata)

        elif args.dataset_name == 'cub':
            self.metadata = pd.read_csv("starting_k/pred_cub.csv")
            self.num_images = len(self.metadata)

        elif args.dataset_name == 'scars':
            self.metadata = pd.read_csv("starting_k/pred_scars.csv")
            self.num_images = len(self.metadata)
    def __getitem__(self, index):
        if args.dataset_name == 'aircraft':
            image_id = self.metadata.iloc[index]['img']
            image_path=os.path.join("/root/data/FGVC_Aircraft/fgvc-aircraft-2013b/data/images",image_id)
            img=Image.open(image_path)
            img = self.transform(img)
        elif args.dataset_name == 'cub':
            image_id = self.metadata.iloc[index]['img']
            with open('/root/data/CUB/CUB_200_2011/images.txt', 'r') as f:
                for line in f:
                    if image_id in line.split(' ')[1]:
                        image_id1 = line.split()[1]
            image_path=os.path.join("/root/data/CUB/CUB_200_2011/images",image_id1)
            img=Image.open(image_path).convert('RGB')
            img = self.transform(img)

        elif args.dataset_name == 'scars':
            image_id = self.metadata.iloc[index]['img']
            ind_ = load_index_to_name()
            index_to_class_split = ind_[args.dataset_name]
            class_name_to_index = {name: int(ind) for ind, name in index_to_class_split.items()}
            samples = make_dataset('/root/data/Stanford_Cars/car_data/car_data/train',
                                   class_name_to_index,
                                   extensions=IMG_EXTENSIONS,
                                   is_valid_file=None)
            # samples2 = make_dataset('/root/data/Stanford_Cars/car_data/car_data/test',
            #                        class_name_to_index,
            #                        extensions=IMG_EXTENSIONS,
            #                        is_valid_file=None)
            # samples=samples1+samples2
            for i in range(len(samples)):
                if image_id in samples[i][0]:
                    img=Image.open(samples[i][0]).convert('RGB')
                    img = self.transform(img)

        return (img,image_id)


    def __len__(self):
        return self.num_images

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='cub', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)

    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default='aircraft_simgcd', type=str)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    # args = get_class_splits(args)

    # args.num_labeled_classes = len(args.train_classes)
    # args.num_unlabeled_classes = len(args.unlabeled_classes)
    if 'imagenet' in args.dataset_name:
        args.mlp_out_dim = 1000

    init_experiment(args, runner_name=['simgcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')

    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    # args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes
    # import ipdb; ipdb.set_trace()
    class_splits = load_class_splits(args.dataset_name)
    args.train_classes = class_splits['known_classes']
    args.mlp_out_dim = len(class_splits['known_classes']) \
                     + len(class_splits['unknown_classes']['Easy']) \
                     + len(class_splits['unknown_classes']['Medium']) \
                     + len(class_splits['unknown_classes']['Hard'])

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True


    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    # train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
    #                                                                                      train_transform,
    #                                                                                      test_transform,
    #                                                                                      args)
    # import ipdb; ipdb.set_trace()
    # train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_gcd_datasets(args.dataset_name,
    #                                                                                         train_transform,
    #                                                                                         test_transform,)
    #
    # # --------------------
    # # SAMPLER
    # # Sampler which balances labelled and unlabelled examples in each batch
    # # --------------------
    # label_len = len(train_dataset.labelled_dataset)
    # unlabelled_len = len(train_dataset.unlabelled_dataset)
    # sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    # sample_weights = torch.DoubleTensor(sample_weights)
    # sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))
    #
    # # --------------------
    # # DATALOADERS
    # # --------------------
    # train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
    #                           sampler=sampler, drop_last=True, pin_memory=True)
    # test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
    #                                     batch_size=256, shuffle=False, pin_memory=False)
    # test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
    #                                   batch_size=256, shuffle=False, pin_memory=False)

    inference_dataset=InferenceDataset(test_transform,args)
    inference_loader = DataLoader(inference_dataset, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model = nn.Sequential(backbone, projector)
    checkpoint_path = torch.load("/tmp/pycharm_project_806/dev_outputs_al_ema_6rounds/cub/old100_ratio0.2_20240806-032055_query100/NovelMarginSamplingAdaptive_exp1/checkpoints/model_round6.pt", map_location='cpu')
    model.load_state_dict(checkpoint_path['model'])
    model = model.to(device)

    # ----------------------
    # TRAIN
    # ----------------------
    # train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)
    # train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)
    result(model, inference_loader, args)






