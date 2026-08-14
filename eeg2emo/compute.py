# compute.py
import os
import time
import psutil
import json
import threading
import logging
import numpy as np
import torch
from pathlib import Path

try:
    import GPUtil

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: GPUtil not available. Install with: pip install GPUtil")


def count_parameters_and_flops(model, input_shape):
    """Simplified parameter and FLOP calculation."""
    total_params = sum(p.numel() for p in model.parameters())

    input_size = np.prod(input_shape)
    estimated_flops = total_params * input_size * 2

    return total_params, estimated_flops


def convert_to_serializable(obj):
    """Convert NumPy and Torch objects to JSON-serializable values."""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    else:
        return obj


class ComputeReportForm:
    """Compute Report Form (CRF) recorder, simplified version."""

    def __init__(self, save_path="./compute_report"):
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True)

        self.hardware_info = self._get_hardware_info()

        self.training_stats = {
            'total_training_time': 0.0,
            'total_epochs': 0,
            'gpu_utilization_history': [],
            'cpu_utilization_history': [],
            'memory_usage_history': [],
            'flops_per_forward': 0,
            'total_flops': 0,
            'training_samples': 0,
            'model_parameters': 0,
            'peak_gpu_memory': 0,
            'peak_cpu_memory': 0
        }

        self.performance_comparison = {}

        self.monitoring = False
        self.monitor_thread = None

        self.training_start_time = None

    def _get_hardware_info(self):
        """Collect hardware information."""
        info = {
            'cpu_info': {
                'model': 'Unknown',
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True),
                'frequency': psutil.cpu_freq().max if psutil.cpu_freq() else 'Unknown'
            },
            'memory_info': {
                'total_gb': round(psutil.virtual_memory().total / (1024 ** 3), 2)
            },
            'gpu_info': []
        }

        try:
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            info['cpu_info']['model'] = line.split(':')[1].strip()
                            break
        except:
            pass

        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    info['gpu_info'].append({
                        'name': gpu.name,
                        'memory_total_gb': round(gpu.memoryTotal / 1024, 2),
                        'driver_version': getattr(gpu, 'driver', 'Unknown'),
                        'uuid': getattr(gpu, 'uuid', f'gpu_{gpu.id}')
                    })
            except Exception as e:
                logging.warning(f"Error getting GPU info with GPUtil: {e}")

        if torch.cuda.is_available() and not info['gpu_info']:
            for i in range(torch.cuda.device_count()):
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                    info['gpu_info'].append({
                        'name': gpu_name,
                        'memory_total_gb': round(gpu_memory, 2),
                        'driver_version': 'Unknown',
                        'uuid': f'cuda:{i}'
                    })
                except Exception as e:
                    logging.warning(f"Error getting GPU {i} info: {e}")

        return info

    def start_monitoring(self, interval=5.0):
        """Start resource monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_resources(self, interval):
        """Resource monitoring loop."""
        while self.monitoring:
            try:
                timestamp = time.time()

                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.training_stats['cpu_utilization_history'].append({
                    'timestamp': timestamp,
                    'usage_percent': float(cpu_percent)
                })

                memory = psutil.virtual_memory()
                self.training_stats['memory_usage_history'].append({
                    'timestamp': timestamp,
                    'usage_percent': float(memory.percent),
                    'used_gb': round(float(memory.used) / (1024 ** 3), 2)
                })

                self.training_stats['peak_cpu_memory'] = max(
                    self.training_stats['peak_cpu_memory'],
                    float(memory.used) / (1024 ** 3)
                )

                if GPU_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        for i, gpu in enumerate(gpus):
                            self.training_stats['gpu_utilization_history'].append({
                                'timestamp': timestamp,
                                'gpu_id': i,
                                'usage_percent': float(gpu.load * 100),
                                'memory_used_gb': round(float(gpu.memoryUsed) / 1024, 2),
                                'memory_percent': round(float(gpu.memoryUtil) * 100, 2),
                                'temperature': getattr(gpu, 'temperature', 'Unknown')
                            })

                            self.training_stats['peak_gpu_memory'] = max(
                                self.training_stats['peak_gpu_memory'],
                                float(gpu.memoryUsed) / 1024
                            )
                    except Exception as e:
                        pass

                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        try:
                            memory_used = torch.cuda.memory_allocated(i) / (1024 ** 3)
                            memory_cached = torch.cuda.memory_reserved(i) / (1024 ** 3)

                            if not GPU_AVAILABLE:
                                self.training_stats['gpu_utilization_history'].append({
                                    'timestamp': timestamp,
                                    'gpu_id': i,
                                    'usage_percent': 'Unknown',
                                    'memory_used_gb': round(float(memory_used), 2),
                                    'memory_cached_gb': round(float(memory_cached), 2)
                                })

                            self.training_stats['peak_gpu_memory'] = max(
                                self.training_stats['peak_gpu_memory'],
                                float(memory_used)
                            )
                        except Exception as e:
                            pass

                time.sleep(interval)

            except Exception as e:
                logging.warning(f"Error in resource monitoring: {e}")
                time.sleep(interval)

    def calculate_model_flops(self, model, input_shape, device):
        """Calculate model FLOPs with a simplified estimate."""
        try:
            model.eval()

            total_params, estimated_flops = count_parameters_and_flops(model, input_shape)

            self.training_stats['flops_per_forward'] = int(estimated_flops)
            logging.info(f"Estimated FLOPs per forward pass: {estimated_flops:,}")

            return estimated_flops

        except Exception as e:
            logging.warning(f"FLOP calculation failed: {e}")
            return 0

    def record_training_start(self):
        """Record training start."""
        self.training_start_time = time.time()
        self.start_monitoring()

    def record_training_end(self):
        """Record training end."""
        if self.training_start_time is not None:
            self.training_stats['total_training_time'] = time.time() - self.training_start_time
        else:
            logging.warning("Training start time was not recorded. Setting training time to 0.")
            self.training_stats['total_training_time'] = 0.0
        self.stop_monitoring()

    def update_model_info(self, model, num_samples):
        """Update model information."""
        self.training_stats['model_parameters'] = int(sum(p.numel() for p in model.parameters()))
        self.training_stats['training_samples'] = int(num_samples)

    def update_performance_comparison(self, baseline_acc, baseline_f1, our_acc, our_f1, baseline_name="Best Baseline"):
        """Update performance comparison."""
        self.performance_comparison = {
            'baseline_name': str(baseline_name),
            'baseline_accuracy': float(baseline_acc),
            'baseline_f1': float(baseline_f1),
            'our_accuracy': float(our_acc),
            'our_f1': float(our_f1),
            'accuracy_improvement': float(our_acc - baseline_acc),
            'f1_improvement': float(our_f1 - baseline_f1),
            'relative_accuracy_improvement': float(
                (our_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0.0,
            'relative_f1_improvement': float((our_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0.0
        }

    def calculate_efficiency_metrics(self):
        """Calculate efficiency metrics."""
        metrics = {}

        if self.training_stats['total_training_time'] > 0:
            metrics['samples_per_second'] = float(
                self.training_stats['training_samples'] / self.training_stats['total_training_time'])
            metrics['epochs_per_hour'] = float(
                self.training_stats['total_epochs'] / (self.training_stats['total_training_time'] / 3600))

        if self.training_stats['gpu_utilization_history']:
            gpu_usage = []
            for entry in self.training_stats['gpu_utilization_history']:
                usage = entry.get('usage_percent')
                if isinstance(usage, (int, float)):
                    gpu_usage.append(float(usage))

            if gpu_usage:
                metrics['avg_gpu_utilization'] = float(np.mean(gpu_usage))
                metrics['max_gpu_utilization'] = float(np.max(gpu_usage))

        if self.training_stats['cpu_utilization_history']:
            cpu_usage = [float(entry['usage_percent']) for entry in self.training_stats['cpu_utilization_history']]
            if cpu_usage:
                metrics['avg_cpu_utilization'] = float(np.mean(cpu_usage))
                metrics['max_cpu_utilization'] = float(np.max(cpu_usage))

        metrics['peak_gpu_memory_gb'] = float(self.training_stats['peak_gpu_memory'])
        metrics['peak_cpu_memory_gb'] = float(self.training_stats['peak_cpu_memory'])

        if self.training_stats['flops_per_forward'] > 0:
            total_flops = (self.training_stats['flops_per_forward'] *
                           self.training_stats['training_samples'] *
                           self.training_stats['total_epochs'])
            self.training_stats['total_flops'] = int(total_flops)

            if self.training_stats['total_training_time'] > 0:
                metrics['gflops_per_second'] = float(total_flops / self.training_stats['total_training_time'] / 1e9)

            if self.training_stats['model_parameters'] > 0:
                metrics['flops_per_parameter'] = float(total_flops / self.training_stats['model_parameters'])

        return metrics

    def generate_report(self, dataset_name, model_name, results=None):
        """Generate the compute report."""
        try:
            efficiency_metrics = self.calculate_efficiency_metrics()

            report = {
                'experiment_info': {
                    'dataset': str(dataset_name),
                    'model': str(model_name),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                },
                'hardware_configuration': convert_to_serializable(self.hardware_info),
                'training_statistics': {
                    'total_training_time_hours': round(float(self.training_stats['total_training_time']) / 3600, 3),
                    'total_epochs': int(self.training_stats['total_epochs']),
                    'training_samples': int(self.training_stats['training_samples']),
                    'model_parameters': int(self.training_stats['model_parameters']),
                    'estimated_flops_per_forward_pass': int(self.training_stats['flops_per_forward']),
                    'estimated_total_flops': int(self.training_stats['total_flops']),
                },
                'resource_utilization': {
                    'peak_gpu_memory_gb': round(float(self.training_stats['peak_gpu_memory']), 2),
                    'peak_cpu_memory_gb': round(float(self.training_stats['peak_cpu_memory']), 2),
                },
                'efficiency_metrics': convert_to_serializable(efficiency_metrics),
                'performance_comparison': convert_to_serializable(self.performance_comparison),
            }

            if results:
                report['final_results'] = convert_to_serializable(results)

            report_path = self.save_path / f"compute_report_{dataset_name}_{model_name}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            simplified_report = {
                'hardware': {
                    'gpu_type': self.hardware_info['gpu_info'][0]['name'] if self.hardware_info[
                        'gpu_info'] else 'CPU Only',
                    'num_gpus': len(self.hardware_info['gpu_info']),
                    'gpu_memory_gb': self.hardware_info['gpu_info'][0]['memory_total_gb'] if self.hardware_info[
                        'gpu_info'] else 0,
                    'cpu_cores': int(self.hardware_info['cpu_info']['cores']),
                    'cpu_memory_gb': float(self.hardware_info['memory_info']['total_gb'])
                },
                'computation': {
                    'training_time_hours': round(float(self.training_stats['total_training_time']) / 3600, 3),
                    'model_parameters': int(self.training_stats['model_parameters']),
                    'estimated_total_flops': int(self.training_stats['total_flops']),
                    'peak_gpu_memory_gb': round(float(self.training_stats['peak_gpu_memory']), 2),
                },
                'efficiency': convert_to_serializable(efficiency_metrics),
                'performance_vs_baseline': convert_to_serializable(self.performance_comparison)
            }

            simplified_path = self.save_path / f"CRF_submission_{dataset_name}_{model_name}.json"
            with open(simplified_path, 'w') as f:
                json.dump(simplified_report, f, indent=2, ensure_ascii=False)

            return report_path, simplified_path

        except Exception as e:
            logging.error(f"Error generating report: {e}")
            error_report = {
                'error': str(e),
                'dataset': str(dataset_name),
                'model': str(model_name),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            error_path = self.save_path / f"error_report_{dataset_name}_{model_name}.json"
            with open(error_path, 'w') as f:
                json.dump(error_report, f, indent=2)

            return None, None

    def save_training_logs(self, dataset_name, model_name):
        """Save detailed training logs."""
        try:
            logs = {
                'gpu_utilization': convert_to_serializable(self.training_stats['gpu_utilization_history']),
                'cpu_utilization': convert_to_serializable(self.training_stats['cpu_utilization_history']),
                'memory_usage': convert_to_serializable(self.training_stats['memory_usage_history'])
            }

            log_path = self.save_path / f"training_logs_{dataset_name}_{model_name}.json"
            with open(log_path, 'w') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)

            return log_path

        except Exception as e:
            logging.error(f"Error saving training logs: {e}")
            return None


def create_crf_instance(dataset_name, save_path_suffix=""):
    """Create a CRF instance."""
    save_path = f"./compute_reports_{dataset_name.lower()}{save_path_suffix}"
    return ComputeReportForm(save_path=save_path)


def get_baseline_scores():
    """Return baseline scores for each dataset."""
    return {
        'EAV': {'ATCNet': 0.6060, 'EEGMiner': 0.6708},
        'MMER': {'EEGConformer': 0.6679, 'EEGMiner': 0.6311},
        'SEED': {'ATCNet': 0.7704, 'EEGConformer': 0.7778}
    }


def setup_crf_for_experiment(dataset_name, model_name, save_path_suffix=""):
    """Set up CRF for an experiment."""
    crf = create_crf_instance(dataset_name, save_path_suffix)
    logging.info("Compute Report Form (CRF) initialized - using simplified FLOP calculation")
    return crf


def finalize_crf_report(crf, dataset_name, model_name, test_acc, test_f1):
    """Finalize the CRF report."""
    if not crf:
        return None, None

    baseline_scores = get_baseline_scores()
    if dataset_name in baseline_scores:
        best_baseline = max(baseline_scores[dataset_name].items(), key=lambda x: x[1])
        baseline_name, baseline_acc = best_baseline
        crf.update_performance_comparison(
            baseline_acc=baseline_acc,
            baseline_f1=baseline_acc,
            our_acc=test_acc,
            our_f1=test_f1,
            baseline_name=baseline_name
        )

    try:
        report_path, simplified_path = crf.generate_report(dataset_name, model_name)

        if report_path and simplified_path:
            log_path = crf.save_training_logs(dataset_name, model_name)

            logging.info(f"Compute Report Form (CRF) generated:")
            logging.info(f"  Full report: {report_path}")
            logging.info(f"  Submission report: {simplified_path}")
            if log_path:
                logging.info(f"  Training logs: {log_path}")

            return simplified_path, report_path
        else:
            logging.error("Failed to generate CRF reports")
            return None, None

    except Exception as e:
        logging.error(f"Error generating CRF report: {e}")
        return None, None
