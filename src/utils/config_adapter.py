"""Config adapter to convert between config formats.

Adapts the existing flat config structure to work with the implementation.
"""


def adapt_config(config):
    """Adapt flat config structure to nested structure expected by implementation.

    Args:
        config (dict): Original config dictionary

    Returns:
        dict: Adapted config dictionary
    """
    # Create adapted config
    adapted = {
        'env': {
            'name': config.get('env_name', 'CarRacing-v3'),
            'render_mode': config.get('render_mode', None),
            'num_actions': len(config['action_space']['actions'])
        },
        'actions': config['action_space']['actions'],
        'preprocessing': {
            'grayscale': config['preprocessing']['grayscale'],
            'frame_stack': config['preprocessing']['frame_stack'],
            'normalize': config['preprocessing']['normalize'],
            'frame_size': [96, 96]
        },
        'network': {
            'type': 'cnn',
            'conv_layers': [],
            'fc_layers': [config['network_architecture']['fc_hidden']],
            'activation': config['network_architecture']['activation'],
            'dueling': config['network_architecture']['dueling']
        },
        'agent': {
            'algorithm': 'dqn',
            'gamma': config['dqn']['gamma'],
            'learning_rate': config['learning_rate'],
            'replay_buffer_size': config['training']['replay_buffer_size'],
            'batch_size': config['training']['batch_size'],
            'learning_starts': config['training']['learning_starts'],
            'epsilon_start': config['dqn']['epsilon_start'],
            'epsilon_end': config['dqn']['epsilon_end'],
            'epsilon_decay_steps': config['dqn']['epsilon_decay_steps'],
            'target_update_frequency': config['training']['target_update_frequency'],
            'train_frequency': config['training']['train_frequency'],
            'grad_clip': config['dqn']['grad_clip'],
            'double_dqn': config['dqn']['double_dqn']
        },
        'training': {
            'total_episodes': config['training']['num_episodes'],
            'max_steps_per_episode': config['training']['max_steps_per_episode'],
            'eval_frequency': config['evaluation']['eval_frequency'],
            'eval_episodes': config['evaluation']['num_eval_episodes'],
            'save_frequency': config['checkpoint']['save_frequency'],
            'checkpoint_dir': config['checkpoint']['save_dir'],
            'log_dir': config['logging']['log_dir'],
            'video_dir': 'results/videos',
            'seed': config.get('seed', 42),
            'device': 'auto'
        },
        'logging': {
            'tensorboard': config['logging']['tensorboard'],
            'console': True,
            'save_metrics': True,
            'metrics_file': 'results/metrics.csv'
        }
    }

    # Convert conv layers from [out_channels, kernel_size, stride] to dict format
    for layer in config['network_architecture']['conv_layers']:
        adapted['network']['conv_layers'].append({
            'filters': layer[0],
            'kernel_size': layer[1],
            'stride': layer[2]
        })

    return adapted
