import yaml
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import dgl
from dgl import ops
from sklearn.metrics import roc_auc_score


class Dataset:
    def __init__(self, name, add_self_loops=False, device='cpu'):
        print('Preparing data...')
        with open(f'data/{name}/info.yaml', 'r') as file:
            info = yaml.safe_load(file)

        features_df = pd.read_csv(f'data/{name}/features.csv', index_col=0)
        num_features = features_df[info['num_feature_names']].values
        bin_features = features_df[info['bin_feature_names']].values
        cat_features = features_df[info['cat_feature_names']].values
        targets = features_df[info['target_name']].values

        if info['task'] == 'classification':
            num_classes = len(targets.unique())
            num_targets = 1 if num_classes == 2 else num_classes
        else:
            num_targets = 1

        if num_targets > 1:
            targets = targets.astype(int)

        edges_df = pd.read_csv(f'data/{name}/edgelist.csv')
        edges = edges_df.values

        train_mask_df = pd.read_csv(f'data/{name}/train_mask.csv', index_col=0)
        train_mask = train_mask_df.values.reshape(-1)
        train_idx = np.where(train_mask)[0]
        valid_mask_df = pd.read_csv(f'data/{name}/valid_mask.csv', index_col=0)
        valid_mask = valid_mask_df.values.reshape(-1)
        valid_idx = np.where(valid_mask)[0]
        test_mask_df = pd.read_csv(f'data/{name}/test_mask.csv', index_col=0)
        test_mask = test_mask_df.values.reshape(-1)
        test_idx = np.where(test_mask)[0]

        features = np.concatenate([num_features, bin_features, cat_features], axis=1)
        features = torch.from_numpy(features)
        targets = torch.from_numpy(targets)

        edges = torch.from_numpy(edges)
        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(features), idtype=torch.int)
        if add_self_loops:
            graph = dgl.add_self_loop(graph)

        train_idx = torch.from_numpy(train_idx)
        valid_idx = torch.from_numpy(valid_idx)
        test_idx = torch.from_numpy(test_idx)

        self.name = name
        self.device = device

        self.graph = graph.to(device)
        self.features = features.to(device)
        self.targets = targets.to(device)

        self.train_idx = train_idx.to(device)
        self.valid_idx = valid_idx.to(device)
        self.test_idx = test_idx.to(device)

        self.num_features = features.shape[1]
        self.num_targets = num_targets

        if info['task'] == 'classification':
            if num_targets == 1:
                self.loss_fn = F.binary_cross_entropy_with_logits
                self.metric = 'ROC AUC'
            else:
                self.loss_fn = F.cross_entropy
                self.metric = 'accuracy'

        elif info['task'] == 'regression':
            self.loss_fn = F.mse_loss
            self.metric = 'MSE'
        else:
            raise ValueError(f'Uknown task type: {info["task"]}.')

    def compute_metrics(self, preds):
        if self.metric == 'ROC AUC':
            train_metric = roc_auc_score(y_true=self.targets[self.train_idx].cpu().numpy(),
                                         y_score=preds[self.train_idx].cpu().numpy()).item()

            valid_metric = roc_auc_score(y_true=self.targets[self.valid_idx].cpu().numpy(),
                                         y_score=preds[self.valid_idx].cpu().numpy()).item()

            test_metric = roc_auc_score(y_true=self.targets[self.test_idx].cpu().numpy(),
                                        y_score=preds[self.test_idx].cpu().numpy()).item()

        elif self.metric == 'accuracy':
            preds = preds.argmax(axis=1)
            train_metric = (preds[self.train_idx] == self.targets[self.train_idx]).float().mean().item()
            valid_metric = (preds[self.valid_idx] == self.targets[self.valid_idx]).float().mean().item()
            test_metric = (preds[self.test_idx] == self.targets[self.test_idx]).float().mean().item()

        elif self.metric == 'MSE':
            train_metric = ((preds[self.train_idx] - self.targets[self.train_idx]) ** 2).mean().item()
            valid_metric = ((preds[self.valid_idx] - self.targets[self.valid_idx]) ** 2).mean().item()
            test_metric = ((preds[self.test_idx] - self.targets[self.test_idx]) ** 2).mean().item()

        else:
            raise ValueError(f'Unknown metric: {self.metric}.')

        metrics = {
            f'train {self.metric}': train_metric,
            f'valid {self.metric}': valid_metric,
            f'test {self.metric}': test_metric
        }

        return metrics

    @staticmethod
    def compute_sgc_features(graph, node_features, num_props=5):
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

        degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1 / degree_edge_products ** 0.5

        for _ in range(num_props):
            node_features = ops.u_mul_e_sum(graph, node_features, norm_coefs)

        return node_features
