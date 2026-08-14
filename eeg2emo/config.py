# config.py
import argparse

try:
    from backbones import ATCNet, EEGConformer, EEGNet, Labram, SyncNet, TIDNet, TSception, DGCNN, LMDA, TSLANet, EEGMiner
except ImportError:
    ATCNet = EEGConformer = EEGNet = Labram = SyncNet = TIDNet = TSception = DGCNN = LMDA = TSLANet = EEGMiner = None


def parse_args():
    parser = argparse.ArgumentParser(description='EEG emotion recognition training')
    parser.add_argument('--dataset', type=str, default='MMER', choices=['EAV', 'MMER', 'SEED'], help='Dataset name')
    parser.add_argument('--run', type=int, default=1, help='Model configuration index')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--save_compute_report', action='store_true', help='Generate a compute report')
    parser.add_argument('--ablation_study', action='store_true', help='Run an ablation study')
    parser.add_argument('--ablation_type', type=str, default='all', choices=['all', 'harmonic', 'depth', 'dilation'], help='Ablation type')
    parser.add_argument('--epochs', type=int, default=None, help='Override the default epoch count')
    return parser.parse_args()


def get_ablation_configs():
    """Return EEG-only ablation settings."""
    return {
        'harmonic': {
            'name': 'Fourier Harmonic Components',
            'params': {
                'H_values': [12, 12, 12, 12, 12],
                'embedding_dims': [288, 288, 288, 288, 288],
                'output_dims': [4, 6, 8, 16, 20],
            },
        },
        'depth': {
            'name': 'Temporal Architecture Depth',
            'params': {'K_values': [1, 3, 5, 7]},
        },
        'dilation': {
            'name': 'Multi-Scale Receptive Field',
            'params': {'dilation_periods': [3, 5, 7]},
        },
    }


def get_dataset_cross_configs():
    return {
        'EAV': {
            'eeg_path': r'/home/devuser/hjj/seeemotion/data/EAV/EEG',
            'num_subjects': 42,
            'num_classes': 5,
            'eeg_channels': 30,
            'sequence_length': 500,
            'sampling_rate': 100.0,
            'window_seconds': 5,
            'overlap': 0.0,
            'batch_size': 8,
            'val_batch_size': 4,
            'test_batch_size': 4,
            'train_count': 25,
            'val_count': 8,
            'test_count': 9,
            'hidden_dim': 64,
            'depth': 5,
            'merger': True,
            'merger_channels': 8,
            'merger_pos_dim': 288,
            'use_dual_path': 0,
            'dilation_period': 5,
        },
        'MMER': {
            'eeg_path': r'/home/devuser/hjj/seeemotion/data/MMER/EEG',
            'label_path': r'/home/devuser/hjj/seeemotion/data/MMER/Labels_Emotion',
            'num_subjects': 14,
            'subject_list': [1, 5, 11, 12, 19, 20, 22, 23, 24, 25, 29, 32, 33, 38],
            'num_classes': 3,
            'eeg_channels': 18,
            'sequence_length': 600,
            'sampling_rate': 300.0,
            'window_seconds': 2,
            'batch_size': 8,
            'val_batch_size': 8,
            'test_batch_size': 4,
            'chunk_len': 600,
            'stride': 600,
            'train_count': 10,
            'val_count': 2,
            'test_count': 2,
            'hidden_dim': 32,
            'depth': 5,
            'merger': True,
            'merger_channels': 6,
            'merger_pos_dim': 288,
            'use_dual_path': 0,
            'dilation_period': 5,
        },
        'SEED': {
            'eeg_path': r'/home/devuser/hjj/seeemotion/data/SEED/Preprocessed_EEG',
            'data_path': r'/home/devuser/hjj/seeemotion/data/SEED/Preprocessed_EEG',
            'num_subjects': 15,
            'num_classes': 3,
            'eeg_channels': 62,
            'sequence_length': 800,
            'sampling_rate': 200.0,
            'window_seconds': 4,
            'batch_size': 8,
            'val_batch_size': 8,
            'test_batch_size': 4,
            'chunk_len': 800,
            'stride': 800,
            'train_count': 10,
            'val_count': 3,
            'test_count': 2,
            'hidden_dim': 64,
            'depth': 5,
            'merger': True,
            'merger_channels': 16,
            'merger_pos_dim': 288,
            'use_dual_path': 0,
            'dilation_period': 5,
        },
    }


def get_dataset_single_configs():
    configs = get_dataset_cross_configs()
    configs['EAV'].update({
        'subject_range': (1, 43),
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'data_format': 'eav_dataset',
        'merger_channels': 6,
        'depth': 3,
    })
    configs['MMER'].update({
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'data_format': 'mmer_dataset',
        'fs': 300,
        'time_window': 20,
        'merger_channels': 8,
        'depth': 1,
    })
    configs['SEED'].update({
        'subject_range': (1, 16),
        'train_ratio': 0.3,
        'val_ratio': 0.3,
        'data_format': 'seed_dataset',
        'merger_channels': 8,
    })
    return configs


def generate_ablation_configs(base_config, ablation_type, ablation_configs):
    """Build concrete dataset configs for one ablation type."""
    configs = []
    params = ablation_configs[ablation_type]['params']

    if ablation_type == 'harmonic':
        for h, emb_dim, out_dim in zip(params['H_values'], params['embedding_dims'], params['output_dims']):
            config = base_config.copy()
            config['merger_pos_dim'] = emb_dim
            config['merger_channels'] = out_dim
            config['experiment_name'] = f'harmonic_H{h}_emb{emb_dim}_out{out_dim}'
            configs.append(config)
    elif ablation_type == 'depth':
        for depth in params['K_values']:
            config = base_config.copy()
            config['depth'] = depth
            config['experiment_name'] = f'depth_K{depth}'
            configs.append(config)
    elif ablation_type == 'dilation':
        for period in params['dilation_periods']:
            config = base_config.copy()
            config['dilation_period'] = period
            config['experiment_name'] = f'dilation_D{period}'
            configs.append(config)

    return configs


def get_model_configs():
    configs = {1: {'log_name': 'run_our.log', 'model_name': 'our', 'use_custom': True}}
    optional_models = [
        (2, 'ATCNet', ATCNet),
        (3, 'EEGConformer', EEGConformer),
        (4, 'EEGNet', EEGNet),
        (5, 'Labram', Labram),
        (6, 'SyncNet', SyncNet),
        (7, 'TIDNet', TIDNet),
        (8, 'TSception', TSception),
        (9, 'DGCNN', DGCNN),
        (10, 'LMDA', LMDA),
        (11, 'TSLANet', TSLANet),
        (12, 'EEGMiner', EEGMiner),
    ]
    for idx, name, model_cls in optional_models:
        if model_cls is not None:
            configs[idx] = {'log_name': f'run_{name.lower()}.log', 'model_name': name, 'model': model_cls}
    return configs
