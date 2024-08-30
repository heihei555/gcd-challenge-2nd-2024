import argparse
import csv
import os

import torch
import numpy as np
import torch.nn as nn
import wandb

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
from tqdm import tqdm

from project_utils.cluster_utils import AverageMeter
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_datasets_with_gcdval, get_class_splits
from project_utils.cluster_and_log_utils import *
from project_utils.general_utils import init_experiment, str2bool

from models.dino import *
from methods.loss import *



class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]



def results(model, test_dataset,test_loader, args):

    with torch.no_grad():
        model.eval()
        all_feats_val = []
        targets = np.array([])
        mask = np.array([])
        image_names = [item.split('/')[-1] for item in test_dataset.data['filepath'].tolist()]
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            images, label, _ = batch[:3]
            images = images.cuda()

            features = model(images)
            all_feats_val.append(features.detach().cpu().numpy())
            targets = np.append(targets, label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                             else False for x in label]))

        mask = mask.astype(bool)
        feats = np.concatenate(all_feats_val)
        linked = linkage(feats, method="ward")

        gt_dist = linked[:, 2][-args.num_labeled_classes - args.num_unlabeled_classes]
        preds = fcluster(linked, t=gt_dist, criterion='distance')
        test_all_acc_test, test_old_acc_test, test_new_acc_test = log_accs_from_preds(y_true=targets, y_pred=preds,
                                                                                      mask=mask,
                                                                                      T=0, eval_funcs=args.eval_funcs,
                                                                                      save_name="Test/ACC")
        print('Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(test_all_acc_test, test_old_acc_test,test_new_acc_test))
        data = [["img", "pred"]]

        if args.dataset_name == 'cub':
            for i in range(len(preds)):
                data.append([image_names[i], preds[i]])
            with open('pred_cub.csv', mode='w', newline='') as f3:
                writer = csv.writer(f3)
                # 写入数据
                for row in data:
                    writer.writerow(row)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--exp_root', type=str, default='/home/pod/shared-nvme/CMS-inference/log')

    parser.add_argument('--load_path', type=str, default='/root/model_best.pt')
    parser.add_argument('--dataset_name', type=str, default='cub', help='options: cifar10, cifar100, scars')

    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=bool, default=True)

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--eta_min', type=float, default=1e-3)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=bool, default=False)
    parser.add_argument('--sup_con_weight', type=float, default=0.35)
    parser.add_argument('--temperature', type=float, default=0.25)

    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--k', default=8, type=int)
    parser.add_argument('--inductive', action='store_true',default=False)
    parser.add_argument('--wandb', action='store_true', help='Flag to log at wandb')

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_class_splits(args)

    args.feat_dim = 1024
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.num_mlp_layers = 3
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['cms'])
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')

    if args.wandb:
        wandb.init(project='CMS')
        wandb.config.update(args)

    # --------------------
    # MODEL
    # --------------------
    model = DINO(args)

    if args.load_path is not None:
        checkpoint = torch.load(args.load_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------

    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)

    
    test_loader_unlabelled= DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    # ----------------------
    # TRAIN
    # ----------------------
    results(model, unlabelled_train_examples_test, test_loader_unlabelled, args)