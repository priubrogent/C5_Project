import wandb
from DeTR.task_e import train # Assuming your refactored code is in task_e.py

# 1. Define the Sweep Configuration
sweep_config = {
    'method': 'random', # options: grid, random, bayes
    'metric': {
        'name': 'eval/loss',
        'goal': 'minimize'   
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-4, 5e-5, 2e-4]
        },
        'batch_size': {
            'values': [16, 32]
        },
        'lora_r': {
            'values': [8, 16]
        },
        'encoder_layers': {
            'values': [3, 6]
        },
        'decoder_layers': {
            'values': [3, 6]
        },
        'weight_decay': {
            'values': [1e-4, 1e-3]
        }
    }
}

# 2. Initialize the sweep
sweep_id = wandb.sweep(
    sweep_config, 
    project="KITTI-MOTS-DETR-Ablation",
    entity="your_wandb_username" 
)

# 3. Launch the agent
# The agent will call the 'train' function from task_e.py
wandb.agent(sweep_id, function=train)