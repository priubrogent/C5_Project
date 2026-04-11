import os
from peft import LoraConfig, get_peft_model
from models import CaptioningModel
from tokenizer import build_tokenizer

def main():
    data_root = "../datasets/vizwiz"
    train_ann = os.path.join(data_root, 'annotations', 'train.json')
    cache_dir = os.path.join(data_root, 'tokenizer_cache')
    
    print("Loading tokenizer to get exact vocab size...")
    tokenizer = build_tokenizer('subword', train_ann, cache_dir)

    print("Initializing Qwen 0.8B Model architecture...")
    model = CaptioningModel(
        encoder_name='clip', 
        decoder_type='qwen',
        decoder_model_name='Qwen/Qwen3.5-4B-Base',
        vocab_size=tokenizer.vocab_size,
        hidden_dim=768 # Projection dimension for CLIP base
    )
    
    # Apply the exact LoRA configuration used during training/inference
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj" 
    ]
    
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model.decoder.qwen = get_peft_model(model.decoder.qwen, config)

    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "="*50)
    print("MODEL PARAMETER BREAKDOWN")
    print("="*50)
    print(f"Total Parameters (Inference):  {total_params:,}")
    print(f"Trainable Parameters (LoRA):   {trainable_params:,}")
    print(f"Percentage Trained:            {(trainable_params / total_params) * 100:.3f}%")
    print("="*50)

if __name__ == "__main__":
    main()