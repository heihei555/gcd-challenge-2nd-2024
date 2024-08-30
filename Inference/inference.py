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
from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups
from PIL import Image
from torchvision.datasets.folder import default_loader, make_dataset, IMG_EXTENSIONS

def result(model,test_dataset, test_loader, args):
    model.eval()

    preds, targets = [], []

    image_names = [item[0].split('/')[-1] for item in test_dataset.samples]
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)

        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())

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

# class InferenceDataset(torch.utils.data.Dataset):
#     def __init__(self,  transform,args):
#         self.transform = transform
#         if args.dataset_name == 'aircraft':
#             self.metadata = pd.read_csv("starting_k/pred_aircraft.csv")
#             self.num_images = len(self.metadata)
#
#         elif args.dataset_name == 'cub':
#             self.metadata = pd.read_csv("starting_k/pred_cub.csv")
#             self.num_images = len(self.metadata)
#
#         elif args.dataset_name == 'scars':
#             self.metadata = pd.read_csv("starting_k/pred_scars.csv")
#             self.num_images = len(self.metadata)
#     def __getitem__(self, index):
#         if args.dataset_name == 'aircraft':
#             image_id = self.metadata.iloc[index]['img']
#             image_path=os.path.join("/root/data/FGVC_Aircraft/fgvc-aircraft-2013b/data/images",image_id)
#             img=Image.open(image_path)
#             img = self.transform(img)
#         elif args.dataset_name == 'cub':
#             image_id = self.metadata.iloc[index]['img']
#             with open('/root/data/CUB/CUB_200_2011/images.txt', 'r') as f:
#                 for line in f:
#                     if image_id in line.split(' ')[1]:
#                         image_id1 = line.split()[1]
#             image_path=os.path.join("/root/data/CUB/CUB_200_2011/images",image_id1)
#             img=Image.open(image_path).convert('RGB')
#             img = self.transform(img)
#
#         elif args.dataset_name == 'scars':
#             image_id = self.metadata.iloc[index]['img']
#             ind_ = load_index_to_name()
#             index_to_class_split = ind_[args.dataset_name]
#             class_name_to_index = {name: int(ind) for ind, name in index_to_class_split.items()}
#             samples = make_dataset('/root/data/Stanford_Cars/car_data/car_data/train',
#                                    class_name_to_index,
#                                    extensions=IMG_EXTENSIONS,
#                                    is_valid_file=None)
#             # samples2 = make_dataset('/root/data/Stanford_Cars/car_data/car_data/test',
#             #                        class_name_to_index,
#             #                        extensions=IMG_EXTENSIONS,
#             #                        is_valid_file=None)
#             # samples=samples1+samples2
#             for i in range(len(samples)):
#                 if image_id in samples[i][0]:
#                     img=Image.open(samples[i][0]).convert('RGB')
#                     img = self.transform(img)
#
#         return (img,image_id)
#
#
#     def __len__(self):
#         return self.num_images

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2'])

    parser.add_argument('--warmup_model_dir', type=str,
                        default='/home/pod/shared-nvme/activeGCD/dev_outputs_base/aircraft/old50_ratio0.5_20240820-103353/checkpoints/model.pt')
    parser.add_argument('--dataset_name', type=str, default='aircraft', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')



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
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['simgcd'])
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)

    
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 1024
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes


    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)


    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)

    # inference_dataset=InferenceDataset(test_transform,args)
    # inference_loader = DataLoader(inference_dataset, num_workers=args.num_workers,
    #                                     batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    model = nn.Sequential(backbone, projector)

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        model.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu')['model'])
    model = model.to(device)

    result(model, test_dataset,test_loader_labelled, args)
