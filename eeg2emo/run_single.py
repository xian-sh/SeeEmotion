# run_single.py
import logging
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import (generate_ablation_configs, get_ablation_configs, get_dataset_single_configs,
                    get_model_configs, parse_args)
from engine import create_model, evaluate, train_one_epoch
from loader import SingleDataLoader
from util import EarlyStopping, _set_global_seed, setup_logger


def run_single_ablation_study(args, dataset_name, model_config, base_dataset_config, log):
    """Run single-subject ablation experiments."""
    ablation_configs = get_ablation_configs()
    ablation_types = list(ablation_configs.keys()) if args.ablation_type == 'all' else [args.ablation_type]
    all_results = []

    for ablation_type in ablation_types:
        log.info('\n%s', '=' * 60)
        log.info('Starting ablation study: %s', ablation_configs[ablation_type]['name'])
        log.info('%s', '=' * 60)
        experiment_configs = generate_ablation_configs(base_dataset_config, ablation_type, ablation_configs)

        for dataset_config in experiment_configs:
            experiment_name = dataset_config['experiment_name']
            log.info('\n--- Experiment: %s ---', experiment_name)
            exp_dir = f'./ckpts_{dataset_name.lower()}1/logs_single/{ablation_type}'
            os.makedirs(exp_dir, exist_ok=True)
            records = run_single_experiment_with_config(args, dataset_name, model_config, dataset_config, log)
            for record in records:
                record['ablation_type'] = ablation_type
                record['experiment_name'] = experiment_name
                all_results.append(record)

    df_all = pd.DataFrame(all_results)
    ablation_csv = f'./ckpts_{dataset_name.lower()}1/logs_single/ablation_results_{args.ablation_type}.csv'
    df_all.to_csv(ablation_csv, index=False, float_format='%.4f')
    log.info('Ablation results saved to: %s', ablation_csv)
    generate_ablation_analysis(df_all, ablation_csv.replace('.csv', '_analysis.txt'))
    return df_all


def run_single_experiment_with_config(args, dataset_name, model_config, dataset_config, log):
    """Train and evaluate one configuration on each subject separately."""
    data_factory = SingleDataLoader(dataset_name, dataset_config)
    subject_list = data_factory.get_subject_list()
    records = []
    epochs = args.epochs or 50

    for subject_id in subject_list[:3]:
        try:
            train_set, test_set = data_factory.load_subject_data(subject_id)
            train_loader = DataLoader(train_set, batch_size=dataset_config['batch_size'], shuffle=True, drop_last=True)
            test_loader = DataLoader(test_set, batch_size=dataset_config['test_batch_size'], shuffle=False)

            model = create_model(dataset_name, model_config, dataset_config, args.device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
            early_stopping = EarlyStopping(patience=10, mode='max')

            best_acc, best_f1, best_epoch = 0.0, 0.0, 0
            for epoch in range(1, epochs + 1):
                train_one_epoch(model, train_loader, optimizer, criterion, args.device, epoch=epoch, total_epochs=epochs)
                val_acc, val_f1, _, _, _, _ = evaluate(model, test_loader, args.device)
                if val_f1 > best_f1:
                    best_acc, best_f1, best_epoch = val_acc, val_f1, epoch
                if early_stopping(val_f1):
                    break

            records.append({
                'subject': subject_id,
                'acc': best_acc,
                'f1': best_f1,
                'best_epoch': best_epoch,
                'epochs_trained': epoch,
            })
            log.info('Subject %s | Acc: %.4f | F1: %.4f | Best epoch: %s', subject_id, best_acc, best_f1, best_epoch)
        except Exception as exc:
            log.error('Error in subject %s: %s', subject_id, exc)

    return records


def generate_ablation_analysis(df, output_path):
    """Write a compact ablation summary."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('Ablation Study Analysis Report\n')
        f.write('=' * 50 + '\n\n')
        if df.empty:
            f.write('No valid results.\n')
            return
        for ablation_type in df['ablation_type'].unique():
            subset = df[df['ablation_type'] == ablation_type]
            f.write(f'Ablation Type: {ablation_type}\n')
            f.write('-' * 30 + '\n')
            summary = subset.groupby('experiment_name').agg({'acc': ['mean', 'std'], 'f1': ['mean', 'std']}).round(4)
            f.write(str(summary))
            f.write('\n\n')
            best_exp = subset.groupby('experiment_name')['f1'].mean().idxmax()
            best_f1 = subset.groupby('experiment_name')['f1'].mean().max()
            f.write(f'Best Configuration: {best_exp} (F1: {best_f1:.4f})\n\n')


def main():
    args = parse_args()
    _set_global_seed(42)
    dataset_name = args.dataset
    model_config = get_model_configs()[args.run]
    base_dataset_config = get_dataset_single_configs()[dataset_name]

    log_dir = f'./ckpts_{dataset_name.lower()}1/logs_single/'
    os.makedirs(log_dir, exist_ok=True)
    log_file = f'ablation_{args.ablation_type}_{model_config["model_name"]}.log' if args.ablation_study else model_config['log_name']
    setup_logger(f'{log_dir}/{log_file}')
    log = logging.getLogger()

    if args.ablation_study:
        run_single_ablation_study(args, dataset_name, model_config, base_dataset_config, log)
    else:
        records = run_single_experiment_with_config(args, dataset_name, model_config, base_dataset_config, log)
        result_csv = f'{log_dir}/single_results_{model_config["model_name"]}.csv'
        pd.DataFrame(records).to_csv(result_csv, index=False, float_format='%.4f')
        log.info('Single-subject results saved to: %s', result_csv)


if __name__ == '__main__':
    main()
