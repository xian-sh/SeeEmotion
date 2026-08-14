# run_cross.py
import logging
import os

import pandas as pd
import torch

from config import (generate_ablation_configs, get_ablation_configs, get_dataset_cross_configs,
                    get_model_configs, parse_args)
from engine import create_model, evaluate, train_one_epoch
from loader import CrossDataLoader
from util import EarlyStopping, _set_global_seed, setup_logger


def run_cross_ablation_study(args, dataset_name, model_config, base_dataset_config, log):
    """Run cross-subject ablation experiments."""
    ablation_configs = get_ablation_configs()
    ablation_types = list(ablation_configs.keys()) if args.ablation_type == 'all' else [args.ablation_type]
    all_results = []

    for ablation_type in ablation_types:
        log.info('\n%s', '=' * 60)
        log.info('Starting cross-subject ablation: %s', ablation_configs[ablation_type]['name'])
        log.info('%s', '=' * 60)
        experiment_configs = generate_ablation_configs(base_dataset_config, ablation_type, ablation_configs)

        for dataset_config in experiment_configs:
            experiment_name = dataset_config['experiment_name']
            log.info('\n--- Cross-subject experiment: %s ---', experiment_name)
            result = run_cross_experiment_with_config(args, dataset_name, model_config, dataset_config, log)
            result['ablation_type'] = ablation_type
            result['experiment_name'] = experiment_name
            all_results.append(result)

    df_all = pd.DataFrame(all_results)
    ablation_csv = f'./ckpts_{dataset_name.lower()}/logs_cross/ablation_results_{args.ablation_type}.csv'
    df_all.to_csv(ablation_csv, index=False, float_format='%.4f')
    log.info('Cross-subject ablation results saved to: %s', ablation_csv)
    return df_all


def run_cross_experiment_with_config(args, dataset_name, model_config, dataset_config, log):
    """Train and evaluate one cross-subject configuration."""
    data_loader = CrossDataLoader(dataset_config, dataset_name)
    train_loader, val_loader, test_loader, subject_info = data_loader.get_dataloaders()
    model = create_model(dataset_name, model_config, dataset_config, args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.01)
    early_stopping = EarlyStopping(patience=10, mode='max')
    epochs = args.epochs or 30

    best_acc, best_f1, best_epoch = 0.0, 0.0, 0
    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, args.device, epoch=epoch, total_epochs=epochs)
        val_acc, val_f1, _, _, _, _ = evaluate(model, val_loader, args.device)
        if val_f1 > best_f1:
            best_acc, best_f1, best_epoch = val_acc, val_f1, epoch
        if early_stopping(val_f1):
            break

    test_acc, test_f1, test_cm, _, _, _ = evaluate(model, test_loader, args.device)
    log.info('Best val F1: %.4f at epoch %s | Test Acc: %.4f | Test F1: %.4f', best_f1, best_epoch, test_acc, test_f1)
    log.info('Test confusion matrix:\n%s', test_cm)

    return {
        'val_acc': best_acc,
        'val_f1': best_f1,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'best_epoch': best_epoch,
        'epochs_trained': epoch,
        'train_subjects': len(subject_info[0]),
        'test_subjects': len(subject_info[2]),
    }


def main():
    args = parse_args()
    _set_global_seed(42)
    dataset_name = args.dataset
    model_config = get_model_configs()[args.run]
    base_dataset_config = get_dataset_cross_configs()[dataset_name]

    log_dir = f'./ckpts_{dataset_name.lower()}/logs_cross/'
    os.makedirs(log_dir, exist_ok=True)
    log_file = f'ablation_{args.ablation_type}_{model_config["model_name"]}.log' if args.ablation_study else model_config['log_name']
    setup_logger(f'{log_dir}/{log_file}')
    log = logging.getLogger()

    if args.ablation_study:
        run_cross_ablation_study(args, dataset_name, model_config, base_dataset_config, log)
    else:
        result = run_cross_experiment_with_config(args, dataset_name, model_config, base_dataset_config, log)
        result_csv = f'{log_dir}/cross_results_{model_config["model_name"]}.csv'
        pd.DataFrame([result]).to_csv(result_csv, index=False, float_format='%.4f')
        log.info('Cross-subject results saved to: %s', result_csv)


if __name__ == '__main__':
    main()
