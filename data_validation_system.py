# -*- coding: utf-8 -*-
"""
Data Validation and Verification System
=====================================

Purpose:
- Establish rigorous data validation process
- Ensure consistency between reported data and actual results
- Document implementation details and validation methodology
- Prevent discrepancies in future iterations

Features:
1. Real-time data validation during training
2. Statistical consistency checks
3. Results verification against expected ranges
4. Data integrity validation
5. Reproducibility verification

Author: Data Validation System
Date: 2026-04-07
"""

import sys
import os
import numpy as np
import json
import hashlib
from datetime import datetime
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ==============================================================================
# CONFIGURATION
# ==============================================================================

VALIDATION_CONFIG = {
    'expected_ranges': {
        'satisfaction': {'min': 0.4, 'max': 1.0},
        'reward': {'min': -20.0, 'max': 20.0},
        'actor_loss': {'min': 0.0, 'max': 10.0},
        'critic_loss': {'min': 0.0, 'max': 10.0},
        'entropy': {'min': 0.0, 'max': 2.0},
        'kl_divergence': {'min': 0.0, 'max': 1.0},
        'handover_latency': {'min': 0.0, 'max': 50.0},
        'packet_loss_rate': {'min': 0.0, 'max': 10.0},
        'qos_violation_rate': {'min': 0.0, 'max': 100.0},
    },
    'statistical_checks': {
        'reward_std_threshold': 30.0,  # 变异系数阈值 (%)
        'satisfaction_std_threshold': 15.0,  # 变异系数阈值 (%)
        'min_episodes_for_stats': 5,  # 统计分析的最小episode数
    },
    'data_integrity': {
        'checksum_enabled': True,
        'duplicate_check': True,
        'range_check': True,
    },
    'reproducibility': {
        'seed_check': True,
        'environment_check': True,
        'hyperparameter_check': True,
    },
}


# ==============================================================================
# VALIDATION CLASSES
# ==============================================================================

class DataValidator:
    """Data validation and verification system"""

    def __init__(self, config=VALIDATION_CONFIG):
        self.config = config
        self.validation_results = []
        self.statistical_metrics = {}
        self.data_integrity = {}
        self.reproducibility = {}

    def validate_episode_data(self, episode_data):
        """Validate single episode data"""
        validation_result = {
            'episode': episode_data.get('episode', -1),
            'valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {},
        }

        # Validate each metric against expected ranges
        for metric, expected_range in self.config['expected_ranges'].items():
            if metric in episode_data:
                value = episode_data[metric]
                if value < expected_range['min'] or value > expected_range['max']:
                    validation_result['valid'] = False
                    validation_result['errors'].append(
                        f"{metric} value {value:.3f} out of expected range [{expected_range['min']}, {expected_range['max']}]"
                    )
                validation_result['metrics'][metric] = value

        # Check for required fields
        required_fields = ['episode', 'reward', 'satisfaction']
        for field in required_fields:
            if field not in episode_data:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Missing required field: {field}")

        return validation_result

    def validate_statistical_consistency(self, all_episode_data):
        """Validate statistical consistency across episodes"""
        if len(all_episode_data) < self.config['statistical_checks']['min_episodes_for_stats']:
            return {
                'valid': False,
                'message': f"Not enough episodes for statistical analysis (need at least {self.config['statistical_checks']['min_episodes_for_stats']})"
            }

        metrics = defaultdict(list)
        for episode_data in all_episode_data:
            for metric in self.config['expected_ranges']:
                if metric in episode_data:
                    metrics[metric].append(episode_data[metric])

        stats = {}
        for metric, values in metrics.items():
            if values:
                stats[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'cv': float(np.std(values) / np.mean(values) * 100) if np.mean(values) > 0 else 0,
                }

        # Check变异系数
        for metric in ['reward', 'satisfaction']:
            if metric in stats:
                threshold = self.config['statistical_checks'][f'{metric}_std_threshold']
                if stats[metric]['cv'] > threshold:
                    stats[metric]['warning'] = f"High variability: CV = {stats[metric]['cv']:.1f}% > {threshold}%"

        return {
            'valid': True,
            'statistics': stats
        }

    def validate_data_integrity(self, all_episode_data):
        """Validate data integrity"""
        integrity = {
            'valid': True,
            'checks': {
                'duplicate_check': True,
                'range_check': True,
                'checksum_valid': True,
            },
            'errors': [],
        }

        # Check for duplicates
        if self.config['data_integrity']['duplicate_check']:
            episodes = [d.get('episode', -1) for d in all_episode_data]
            if len(episodes) != len(set(episodes)):
                integrity['valid'] = False
                integrity['checks']['duplicate_check'] = False
                integrity['errors'].append("Duplicate episode numbers found")

        # Check ranges
        if self.config['data_integrity']['range_check']:
            for episode_data in all_episode_data:
                for metric, expected_range in self.config['expected_ranges'].items():
                    if metric in episode_data:
                        value = episode_data[metric]
                        if not (expected_range['min'] <= value <= expected_range['max']):
                            integrity['valid'] = False
                            integrity['checks']['range_check'] = False
                            integrity['errors'].append(
                                f"{metric} out of range in episode {episode_data.get('episode', -1)}"
                            )

        # Checksum validation
        if self.config['data_integrity']['checksum_enabled']:
            data_str = json.dumps(all_episode_data, sort_keys=True)
            checksum = hashlib.md5(data_str.encode()).hexdigest()
            integrity['checksum'] = checksum

        return integrity

    def validate_reproducibility(self, environment_config, hyperparameters, seed):
        """Validate reproducibility"""
        reproducibility = {
            'valid': True,
            'checks': {
                'seed_check': True,
                'environment_check': True,
                'hyperparameter_check': True,
            },
            'details': {
                'seed': seed,
                'environment_config': environment_config,
                'hyperparameters': hyperparameters,
            },
        }

        # Seed check
        if self.config['reproducibility']['seed_check']:
            if not isinstance(seed, int) or seed < 0:
                reproducibility['valid'] = False
                reproducibility['checks']['seed_check'] = False

        # Environment check
        if self.config['reproducibility']['environment_check']:
            required_env_fields = ['num_bs', 'num_uav', 'max_steps']
            for field in required_env_fields:
                if field not in environment_config:
                    reproducibility['valid'] = False
                    reproducibility['checks']['environment_check'] = False

        # Hyperparameter check
        if self.config['reproducibility']['hyperparameter_check']:
            required_hp_fields = ['actor_lr', 'critic_lr', 'clip_epsilon']
            for field in required_hp_fields:
                if field not in hyperparameters:
                    reproducibility['valid'] = False
                    reproducibility['checks']['hyperparameter_check'] = False

        return reproducibility

    def run_full_validation(self, all_episode_data, environment_config, hyperparameters, seed):
        """Run full validation suite"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'episode_validation': [],
            'statistical_validation': {},
            'data_integrity': {},
            'reproducibility': {},
            'overall_valid': True,
        }

        # Validate each episode
        for episode_data in all_episode_data:
            result = self.validate_episode_data(episode_data)
            results['episode_validation'].append(result)
            if not result['valid']:
                results['overall_valid'] = False

        # Validate statistical consistency
        stats_result = self.validate_statistical_consistency(all_episode_data)
        results['statistical_validation'] = stats_result
        if not stats_result.get('valid', False):
            results['overall_valid'] = False

        # Validate data integrity
        integrity_result = self.validate_data_integrity(all_episode_data)
        results['data_integrity'] = integrity_result
        if not integrity_result.get('valid', False):
            results['overall_valid'] = False

        # Validate reproducibility
        reproducibility_result = self.validate_reproducibility(environment_config, hyperparameters, seed)
        results['reproducibility'] = reproducibility_result
        if not reproducibility_result.get('valid', False):
            results['overall_valid'] = False

        # Store results
        self.validation_results.append(results)
        self.statistical_metrics = stats_result.get('statistics', {})
        self.data_integrity = integrity_result
        self.reproducibility = reproducibility_result

        return results


# ==============================================================================
# VERIFICATION TOOLS
# ==============================================================================

class ResultsVerifier:
    """Verify results against expected performance"""

    def __init__(self):
        self.expected_performance = {
            'small': {
                'mappo': {'satisfaction': 0.85, 'reward': 10.0},
                'enhanced': {'satisfaction': 0.80, 'reward': 8.0},
                'traditional': {'satisfaction': 0.75, 'reward': 6.0},
            },
            'medium': {
                'mappo': {'satisfaction': 0.80, 'reward': 8.0},
                'enhanced': {'satisfaction': 0.75, 'reward': 6.0},
                'traditional': {'satisfaction': 0.70, 'reward': 4.0},
            },
            'large': {
                'mappo': {'satisfaction': 0.75, 'reward': 6.0},
                'enhanced': {'satisfaction': 0.70, 'reward': 4.0},
                'traditional': {'satisfaction': 0.65, 'reward': 2.0},
            },
        }

    def verify_performance(self, results, scenario):
        """Verify performance against expected values"""
        verification = {
            'scenario': scenario,
            'valid': True,
            'comparisons': {},
        }

        if scenario not in self.expected_performance:
            verification['valid'] = False
            verification['error'] = f"Unknown scenario: {scenario}"
            return verification

        expected = self.expected_performance[scenario]

        for algorithm, metrics in results.items():
            if algorithm not in expected:
                continue

            comparison = {}
            for metric, value in metrics.items():
                if metric in expected[algorithm]:
                    expected_value = expected[algorithm][metric]
                    deviation = (value - expected_value) / expected_value * 100
                    comparison[metric] = {
                        'actual': value,
                        'expected': expected_value,
                        'deviation': deviation,
                        'within_tolerance': abs(deviation) < 20.0,  # 20% tolerance
                    }
                    if not comparison[metric]['within_tolerance']:
                        verification['valid'] = False

            verification['comparisons'][algorithm] = comparison

        return verification

    def generate_verification_report(self, verification_results):
        """Generate verification report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'verification_results': verification_results,
            'summary': {
                'total_scenarios': len(verification_results),
                'valid_scenarios': sum(1 for r in verification_results if r['valid']),
                'invalid_scenarios': sum(1 for r in verification_results if not r['valid']),
            },
        }

        return report


# ==============================================================================
# VISUALIZATION TOOLS
# ==============================================================================

def generate_validation_report(results, output_dir='validation_results'):
    """Generate validation report with visualizations"""
    os.makedirs(output_dir, exist_ok=True)

    # Create validation summary plot
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'Data Validation Report\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                 fontsize=16, fontweight='bold')

    # Subplot 1: Metric distributions
    ax1 = plt.subplot(2, 2, 1)
    if 'statistical_validation' in results and 'statistics' in results['statistical_validation']:
        stats = results['statistical_validation']['statistics']
        metrics = list(stats.keys())[:4]  # Top 4 metrics
        for metric in metrics:
            if metric in stats:
                data = {
                    'Mean': stats[metric]['mean'],
                    'Std Dev': stats[metric]['std'],
                    'Min': stats[metric]['min'],
                    'Max': stats[metric]['max'],
                }
                x = np.arange(len(data))
                ax1.bar(x, list(data.values()), label=metric)
                ax1.set_xticks(x)
                ax1.set_xticklabels(list(data.keys()))
                ax1.set_title('Metric Statistics')
                ax1.set_ylabel('Value')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

    # Subplot 2: Validation status
    ax2 = plt.subplot(2, 2, 2)
    validation_status = {
        'Episode Validation': sum(1 for r in results['episode_validation'] if r['valid']),
        'Statistical Validation': 1 if results['statistical_validation'].get('valid', False) else 0,
        'Data Integrity': 1 if results['data_integrity'].get('valid', False) else 0,
        'Reproducibility': 1 if results['reproducibility'].get('valid', False) else 0,
    }
    labels = list(validation_status.keys())
    values = list(validation_status.values())
    colors = ['green' if v > 0 else 'red' for v in values]
    ax2.bar(labels, values, color=colors)
    ax2.set_title('Validation Status')
    ax2.set_ylabel('Valid (1) / Invalid (0)')
    ax2.set_ylim(0, 1.1)
    for i, v in enumerate(values):
        ax2.text(i, v + 0.05, str(v), ha='center')

    # Subplot 3: Error summary
    ax3 = plt.subplot(2, 2, 3)
    error_counts = defaultdict(int)
    for episode_result in results['episode_validation']:
        for error in episode_result['errors']:
            error_counts[error[:50]] += 1
    if error_counts:
        errors = list(error_counts.keys())[:5]  # Top 5 errors
        counts = list(error_counts.values())[:5]
        ax3.barh(errors, counts)
        ax3.set_title('Top Validation Errors')
        ax3.set_xlabel('Count')
    else:
        ax3.text(0.5, 0.5, 'No errors found', ha='center', va='center')
        ax3.set_title('Validation Errors')

    # Subplot 4: Overall status
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    overall_status = "PASS" if results['overall_valid'] else "FAIL"
    status_color = "green" if results['overall_valid'] else "red"
    status_text = f"OVERALL VALIDATION: {overall_status}"
    ax4.text(0.5, 0.5, status_text, ha='center', va='center',
             fontsize=18, fontweight='bold', color=status_color)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'validation_report_{timestamp}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def save_validation_results(results, output_dir='validation_results'):
    """Save validation results to JSON"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'validation_results_{timestamp}.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'version': '1.0',
            'configuration': VALIDATION_CONFIG,
            'results': results,
            'timestamp': timestamp,
        }, f, indent=2, ensure_ascii=False, default=str)

    return output_file


# ==============================================================================
# MAIN VALIDATION PIPELINE
# ==============================================================================

def run_validation_pipeline(experiment_data, environment_config, hyperparameters, seed):
    """Run complete validation pipeline"""
    print("\n" + "="*80)
    print("DATA VALIDATION AND VERIFICATION PIPELINE")
    print("="*80)

    # Initialize validator
    validator = DataValidator()

    # Run validation
    validation_results = validator.run_full_validation(
        experiment_data,
        environment_config,
        hyperparameters,
        seed
    )

    # Generate report
    report_path = generate_validation_report(validation_results)
    results_path = save_validation_results(validation_results)

    # Print summary
    print("\nValidation Summary:")
    print(f"Overall Valid: {validation_results['overall_valid']}")
    print(f"Validation Report: {report_path}")
    print(f"Results Saved: {results_path}")

    # Check specific issues
    if not validation_results['overall_valid']:
        print("\nValidation Issues Found:")
        for i, episode_result in enumerate(validation_results['episode_validation']):
            if not episode_result['valid']:
                print(f"Episode {i}: {len(episode_result['errors'])} errors")

    return validation_results


# ==============================================================================
# SAMPLE USAGE
# ==============================================================================

def sample_usage():
    """Sample usage of the validation system"""
    # Sample episode data
    sample_data = []
    for i in range(10):
        sample_data.append({
            'episode': i + 1,
            'reward': np.random.normal(5.0, 2.0),
            'satisfaction': np.random.normal(0.8, 0.1),
            'actor_loss': np.random.normal(0.5, 0.2),
            'critic_loss': np.random.normal(0.3, 0.1),
            'entropy': np.random.normal(0.8, 0.2),
        })

    # Sample configs
    env_config = {
        'num_bs': 4,
        'num_uav': 10,
        'max_steps': 100,
    }

    hp_config = {
        'actor_lr': 3e-4,
        'critic_lr': 1e-3,
        'clip_epsilon': 0.1,
    }

    seed = 42

    # Run validation
    results = run_validation_pipeline(sample_data, env_config, hp_config, seed)
    return results


# ==============================================================================
# INTEGRATION WITH EXPERIMENT SYSTEM
# ==============================================================================

def integrate_with_experiment(experiment_results, env_config, hp_config, seed):
    """Integrate validation with experiment system"""
    # Convert experiment results to validation format
    validation_data = []
    for ep_idx, (reward, satisfaction, actor_loss, critic_loss, entropy) in enumerate(zip(
        experiment_results.get('rewards', []),
        experiment_results.get('satisfactions', []),
        experiment_results.get('actor_losses', []),
        experiment_results.get('critic_losses', []),
        experiment_results.get('entropies', [])
    )):
        validation_data.append({
            'episode': ep_idx + 1,
            'reward': reward,
            'satisfaction': satisfaction,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy': entropy,
        })

    # Run validation
    return run_validation_pipeline(validation_data, env_config, hp_config, seed)


if __name__ == "__main__":
    # Run sample usage
    sample_usage()
