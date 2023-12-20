import argparse
from tqdm import tqdm

import torch
from torch.cuda.amp import autocast, GradScaler

from model import Model
from datasets import Dataset
from utils import Logger, get_parameter_groups, get_lr_scheduler_with_warmup


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default=None, help='Experiment name. If None, model name is used.')
    parser.add_argument('--save_dir', type=str, default='experiments', help='Base directory for saving information.')
    parser.add_argument('--dataset', type=str, default='vk-users',
                        choices=['vk-users-r', 'vk-users-c', 'hm-products', 'hm-products-group', 'avazu-devices',
                                 'yp-maps', 'yp-fraud', 'yp-fraud-v2', 'yp-fraud-v3', 'yp-games', 'yp-games-v2', 'yp-games-c',
                                 'amazon-users-p', 'amazon-users-s', 'amazon-users-v', 'tolokers-tab', 'questions-tab'])

    # numerical features preprocessing
    parser.add_argument('--numerical_features_transform', type=str, default='quantile-transform-normal',
                        choices=['none', 'standard-scaler', 'min-max-scaler', 'robust-scaler',
                                 'power-transform-yeo-johnson', 'quantile-transform-normal',
                                 'quantile-transform-uniform'])

    # PLR embeddings for numerical features
    parser.add_argument('--plr', default=False, action='store_true', help='Use PLR embeddings for numerical features.')
    parser.add_argument('--plr_n_frequencies', type=int, default=48)
    parser.add_argument('--plr_frequency_scale', type=float, default=0.01)
    parser.add_argument('--plr_d_embedding', type=int, default=16)
    parser.add_argument('--plr_lite', default=False, action='store_true')

    # regression target transform
    parser.add_argument('--regression_target_transform', type=str, default='none',
                        choices=['none', 'standard-scaler', 'min-max-scaler', 'robust-scaler',
                                 'power-transform-yeo-johnson', 'quantile-transform-normal',
                                 'quantile-transform-uniform'])

    # model architecture
    parser.add_argument('--model', type=str, default='GT-sep',
                        choices=['ResNet', 'GCN', 'SAGE', 'GAT', 'GAT-sep', 'GT', 'GT-sep'])
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--hidden_dim_multiplier', type=float, default=1)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--normalization', type=str, default='LayerNorm', choices=['None', 'LayerNorm', 'BatchNorm'])

    # regularization
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0)

    # training parameters
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--num_warmup_steps', type=int, default=None,
                        help='If None, warmup_proportion is used instead.')
    parser.add_argument('--warmup_proportion', type=float, default=0, help='Only used if num_warmup_steps is None.')

    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--amp', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')

    args = parser.parse_args()

    if args.name is None:
        args.name = args.model

    return args


def train_step(model, dataset, optimizer, scheduler, scaler, amp=False):
    model.train()

    with autocast(enabled=amp):
        preds = model(graph=dataset.graph, x=dataset.features)
        loss = dataset.loss_fn(input=preds[dataset.train_idx], target=dataset.targets[dataset.train_idx])

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()


@torch.no_grad()
def evaluate(model, dataset, amp=False):
    model.eval()

    with autocast(enabled=amp):
        preds = model(graph=dataset.graph, x=dataset.features)

    metrics = dataset.compute_metrics(preds)

    return metrics


def main():
    args = get_args()

    torch.manual_seed(0)

    dataset = Dataset(name=args.dataset,
                      add_self_loops=(args.model in ['GCN', 'GAT', 'GT']),
                      num_features_transform=args.numerical_features_transform,
                      regression_target_transform=args.regression_target_transform,
                      device=args.device)

    logger = Logger(args, metric=dataset.metric)

    for run in range(1, args.num_runs + 1):
        model = Model(model_name=args.model,
                      num_layers=args.num_layers,
                      input_dim=dataset.num_features,
                      hidden_dim=args.hidden_dim,
                      output_dim=dataset.num_targets,
                      hidden_dim_multiplier=args.hidden_dim_multiplier,
                      num_heads=args.num_heads,
                      normalization=args.normalization,
                      dropout=args.dropout,
                      use_plr=args.plr,
                      num_numeric_features=dataset.num_numeric_features,
                      plr_n_frequencies=args.plr_n_frequencies,
                      plr_frequency_scale=args.plr_frequency_scale,
                      plr_d_embedding=args.plr_d_embedding,
                      use_plr_lite=args.plr_lite)

        model.to(args.device)

        parameter_groups = get_parameter_groups(model)
        optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
        scaler = GradScaler(enabled=args.amp)
        scheduler = get_lr_scheduler_with_warmup(optimizer=optimizer, num_warmup_steps=args.num_warmup_steps,
                                                 num_steps=args.num_steps, warmup_proportion=args.warmup_proportion)

        logger.start_run(run=run)
        with tqdm(total=args.num_steps, desc=f'Run {run}', disable=args.verbose) as progress_bar:
            for step in range(1, args.num_steps + 1):
                train_step(model=model, dataset=dataset, optimizer=optimizer, scheduler=scheduler,
                           scaler=scaler, amp=args.amp)
                metrics = evaluate(model=model, dataset=dataset, amp=args.amp)
                logger.update_metrics(metrics=metrics, step=step)

                progress_bar.update()
                progress_bar.set_postfix({metric: f'{value:.2f}' for metric, value in metrics.items()})

        logger.finish_run()
        model.cpu()

    logger.print_metrics_summary()


if __name__ == '__main__':
    main()
