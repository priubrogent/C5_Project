import wandb
from lora_fine_tune import train

# 1. Define the Sweep Configuration
sweep_config = {
    'method': 'bayes', # options: grid, random, bayes
    'metric': {
        'name': 'mAP_0.50_0.95',
        'goal': 'maximize'   
    },
    'parameters': {
        'epochs':{
            'values': [10,15,20]
        },
        'learning_rate': {
            'distribution': 'log_uniform_values', 
            'min': 1e-5,
            'max': 1e-3
        },
        'batch_size': {
            'values': [16]
        },
        'lora_r': {
            'values': [8, 16]
        },
        'lora_alpha':{
            'distribution': 'int_uniform',
            'min': 8,
            'max': 32
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 0.1
        },
        'warmup_ratio': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.2
        },
        'lr_scheduler': {
            'values': ['linear', 'cosine', 'constant']
        },
        'optimizer': {
            'values': ['adamw_torch_fused', 'rmsprop']
        }
    }
}

# 2. Initialize the sweep
sweep_id = wandb.sweep(
    sweep_config, 
    project="Week1-DETR",
)

# 3. Launch the agent
wandb.agent(sweep_id, function=train)