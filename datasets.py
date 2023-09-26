import yaml
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import dgl
from sklearn.preprocessing import (FunctionTransformer, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer,
                                   QuantileTransformer, OneHotEncoder)
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, r2_score


class Dataset:
    transforms = {
        'none': FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x),
        'standard-scaler': StandardScaler(),
        'min-max-scaler': MinMaxScaler(),
        'robust-scaler': RobustScaler(unit_variance=True),
        'power-transform-yeo-johnson': PowerTransformer(method='yeo-johnson', standardize=True),
        'quantile-transform-normal': QuantileTransformer(output_distribution='normal', subsample=1_000_000,
                                                         random_state=0),
        'quantile-transform-uniform': QuantileTransformer(output_distribution='uniform', subsample=1_000_000,
                                                          random_state=0)
    }

    def __init__(self, name, add_self_loops=False, num_features_transform='none', regression_target_transform='none',
                 device='cpu'):
        print('Preparing data...')
        with open(f'data/{name}/info.yaml', 'r') as file:
            info = yaml.safe_load(file)

        features_df = pd.read_csv(f'data/{name}/features.csv', index_col=0)
        num_features = features_df[info['num_feature_names']].values.astype(np.float32)
        bin_features = features_df[info['bin_feature_names']].values.astype(np.float32)
        cat_features = features_df[info['cat_feature_names']].values.astype(np.float32)
        targets = features_df[info['target_name']].values.astype(np.float32)

        if num_features.shape[1] > 0:
            num_features = self.transforms[num_features_transform].fit_transform(num_features)
            num_features = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(num_features)

        if cat_features.shape[1] > 0:
            cat_features = OneHotEncoder(sparse_output=False, dtype=np.float32).fit_transform(cat_features)

        if info['task'] == 'regression':
            targets_orig = targets
            targets_transform = self.transforms[regression_target_transform]
            targets = targets_transform.fit_transform(targets.reshape(-1, 1)).reshape(-1)

        if info['task'] == 'classification':
            classes = np.unique(targets)
            num_classes = len(classes) if -1 not in classes else len(classes) - 1   # -1 is used for unlabeled nodes
            num_targets = 1 if num_classes == 2 else num_classes
        else:
            num_targets = 1

        if num_targets > 1:
            targets = targets.astype(np.int64)

        edges_df = pd.read_csv(f'data/{name}/edgelist.csv')
        edges = edges_df.values

        train_mask_df = pd.read_csv(f'data/{name}/train_mask.csv', index_col=0)
        train_mask = train_mask_df.values.reshape(-1)
        train_idx = np.where(train_mask)[0]
        val_mask_df = pd.read_csv(f'data/{name}/valid_mask.csv', index_col=0)
        val_mask = val_mask_df.values.reshape(-1)
        val_idx = np.where(val_mask)[0]
        test_mask_df = pd.read_csv(f'data/{name}/test_mask.csv', index_col=0)
        test_mask = test_mask_df.values.reshape(-1)
        test_idx = np.where(test_mask)[0]

        features = np.concatenate([num_features, bin_features, cat_features], axis=1)
        features = torch.from_numpy(features)
        targets = torch.from_numpy(targets)
        if info['task'] == 'regression':
            targets_orig = torch.from_numpy(targets_orig)

        edges = torch.from_numpy(edges)
        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(features), idtype=torch.int)
        if add_self_loops:
            graph = dgl.add_self_loop(graph)

        train_idx = torch.from_numpy(train_idx)
        val_idx = torch.from_numpy(val_idx)
        test_idx = torch.from_numpy(test_idx)

        self.name = name
        self.task = info['task']
        self.device = device

        self.graph = graph.to(device)
        self.features = features.to(device)
        self.targets = targets.to(device)
        if info['task'] == 'regression':
            self.targets_orig = targets_orig.to(device)

        self.train_idx = train_idx.to(device)
        self.val_idx = val_idx.to(device)
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
            self.metric = 'R2'
            self.targets_transform = targets_transform
        else:
            raise ValueError(f'Uknown task type: {info["task"]}.')

    def compute_metrics(self, preds):
        if self.metric == 'ROC AUC':
            train_metric = roc_auc_score(y_true=self.targets[self.train_idx].cpu().numpy(),
                                         y_score=preds[self.train_idx].cpu().numpy()).item()

            val_metric = roc_auc_score(y_true=self.targets[self.val_idx].cpu().numpy(),
                                       y_score=preds[self.val_idx].cpu().numpy()).item()

            test_metric = roc_auc_score(y_true=self.targets[self.test_idx].cpu().numpy(),
                                        y_score=preds[self.test_idx].cpu().numpy()).item()

        elif self.metric == 'accuracy':
            preds = preds.argmax(axis=1)
            train_metric = (preds[self.train_idx] == self.targets[self.train_idx]).float().mean().item()
            val_metric = (preds[self.val_idx] == self.targets[self.val_idx]).float().mean().item()
            test_metric = (preds[self.test_idx] == self.targets[self.test_idx]).float().mean().item()

        elif self.metric == 'R2':
            targets_orig = self.targets_orig.cpu().numpy()
            preds_orig = self.targets_transform.inverse_transform(preds.cpu().numpy().reshape(-1, 1)).reshape(-1)

            train_idx = self.train_idx.cpu().numpy()
            val_idx = self.val_idx.cpu().numpy()
            test_idx = self.test_idx.cpu().numpy()

            train_metric = r2_score(y_true=targets_orig[train_idx],
                                    y_pred=preds_orig[train_idx]).item()

            val_metric = r2_score(y_true=targets_orig[val_idx],
                                  y_pred=preds_orig[val_idx]).item()

            test_metric = r2_score(y_true=targets_orig[test_idx],
                                   y_pred=preds_orig[test_idx]).item()

        else:
            raise ValueError(f'Unknown metric: {self.metric}.')

        metrics = {
            f'train {self.metric}': train_metric,
            f'val {self.metric}': val_metric,
            f'test {self.metric}': test_metric
        }

        return metrics
