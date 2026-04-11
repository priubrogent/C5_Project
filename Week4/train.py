import argparse
import json
import os

#os.environ["CUDA_VISIBLE_DEVICES"]='7'
#os.environ["CUDA_VISIBLE_DEVICES"]='0'

import random
import time
from pathlib import Path

from peft import LoraConfig, get_peft_model, PeftModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from transformers import get_cosine_schedule_with_warmup

import evaluate as hf_evaluate

from dataset import VizWizDataset, collate_fn
from models import CaptioningModel
from tokenizer import build_tokenizer

# ImageNet normalization constants (for denormalization)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_DEFAULT_DATA_ROOT = '/data/113-2/users/gasbert/master/C5/vizwiz_dataset'
_DEFAULT_OUT_ROOT  = '/data/113-2/users/gasbert/master/C5/Week4_outputs'

from transformers import AutoTokenizer

def load_metrics():
    bleu = hf_evaluate.load('bleu')
    rouge = hf_evaluate.load('rouge')
    meteor = hf_evaluate.load('meteor')
    return bleu, rouge, meteor


def compute_metrics(bleu, rouge, meteor, predictions, references):
    bleu1  = bleu.compute(predictions=predictions, references=references, max_order=1)['bleu'] * 100
    bleu2  = bleu.compute(predictions=predictions, references=references, max_order=2)['bleu'] * 100
    rougeL = rouge.compute(predictions=predictions, references=[r[0] for r in references])['rougeL'] * 100
    met    = meteor.compute(predictions=predictions, references=references)['meteor'] * 100
    return {'bleu1': bleu1, 'bleu2': bleu2, 'rougeL': rougeL, 'meteor': met}


def get_fixed_examples(dataset, n=10, seed=42):
    """Select n fixed indices from a dataset for qualitative evaluation."""
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(n, len(dataset)))
    return indices


@torch.no_grad()
def log_qualitative_examples(model, dataset, indices, tokenizer, device, epoch, split,
                             out_path, wandb_module=None, n_wandb=10):
    """Generate captions for fixed examples, save all locally and log a subset to W&B."""
    model.eval()
    records = []
    wandb_images = []

    for i, idx in enumerate(indices):
        img_tensor, _, gt_captions = dataset[idx]
        fname = dataset.samples[idx][0]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        gen = model.generate(img_tensor, tokenizer.max_len - 1, tokenizer.sos_idx, tokenizer.eos_idx)
        pred_caption = tokenizer.decode(gen[0].cpu().tolist())

        records.append({
            'epoch':      epoch,
            'image':      fname,
            'image_path': os.path.join(dataset.img_dir, fname),
            'captions':   gt_captions,
            'prediction': pred_caption,
        })

        if wandb_module is not None and i < n_wandb:
            img_denorm = img_tensor[0].cpu() * IMAGENET_STD + IMAGENET_MEAN
            img_denorm = img_denorm.clamp(0, 1)
            pil_img = to_pil_image(img_denorm)
            gt_str = gt_captions[0] if gt_captions else 'N/A'
            wandb_images.append(
                wandb_module.Image(pil_img, caption=f"Pred: {pred_caption}\nGT: {gt_str}")
            )

    with open(out_path, 'w') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    if wandb_module is not None:
        wandb_module.log({f"{split}_qualitative": wandb_images}, step=epoch)


def train_one_epoch(model, optimizer, criterion, dataloader, device, tf_prob, args=None, scheduler=None):
    model.train()
    total_loss, n = 0.0, 0
    for imgs, captions, _ in dataloader:
        imgs, captions = imgs.to(device), captions.to(device)
        optimizer.zero_grad()
        use_tf = random.random() < tf_prob
        logits = model(imgs, captions, teacher_forcing=use_tf)
        loss = criterion(logits, captions[:, 1:])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        if scheduler is not None and args.decoder == 'qwen':
            scheduler.step()
        
        total_loss += loss.item() * imgs.shape[0]
        n += imgs.shape[0]
    return total_loss / n


@torch.no_grad()
def eval_epoch(model, criterion, dataloader, tokenizer, device, bleu, rouge, meteor):
    model.eval()
    total_loss, n = 0.0, 0
    predictions, references = [], []

    for imgs, captions, all_captions in dataloader:
        imgs, captions = imgs.to(device), captions.to(device)
        logits = model(imgs, captions, teacher_forcing=True)
        loss = criterion(logits, captions[:, 1:])
        total_loss += loss.item() * imgs.shape[0]
        n += imgs.shape[0]

        gen = model.generate(imgs, tokenizer.max_len - 1, tokenizer.sos_idx, tokenizer.eos_idx)
        for i in range(imgs.shape[0]):
            pred = tokenizer.decode(gen[i].cpu().tolist())
            if pred.strip():
                predictions.append(pred)
                references.append(all_captions[i])

    metrics = compute_metrics(bleu, rouge, meteor, predictions, references) if predictions else {}
    return total_loss / n, metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--encoder', default='resnet18',
                   choices=['resnet18', 'resnet34', 'resnet50', 'vgg16', 'vgg19', 
                           'vit-b-16', 'vit-b-32', 'clip'])
    p.add_argument('--decoder', default='gru', 
                   choices=['gru', 'lstm', 'gru_attn', 'gpt2', 't5', 'smollm', 'qwen'])
    p.add_argument('--decoder_layers', type=int, default=1)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--embed_dim', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--text_repr', default='char', choices=['char', 'word', 'subword'])
    p.add_argument('--max_len', type=int, default=None)
    p.add_argument('--decoder_model_name', type=str, default=None,
                   help='Hugging Face model name for transformer decoders (gpt2, t5, smollm)')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--optimizer', default='adam', choices=['adam', 'adamw', 'sgd'])
    p.add_argument('--teacher_forcing', action='store_true', default=True)
    p.add_argument('--no_teacher_forcing', dest='teacher_forcing', action='store_false')
    p.add_argument('--scheduled_tf', action='store_true', default=False)
    p.add_argument('--lr_decay', type=float, default=0.5)
    p.add_argument('--lr_patience', type=int, default=5)
    p.add_argument('--es_patience', type=int, default=10)
    p.add_argument('--es_metric', default='val_loss',
                   choices=['val_loss', 'bleu1', 'bleu2', 'rougeL', 'meteor'])
    p.add_argument('--run_name', default='run')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--val_fraction', type=float, default=0.1)
    p.add_argument('--wandb', action='store_true', default=True)
    p.add_argument('--wandb_project', default='c5-week4-captioning')
    p.add_argument('--wandb_entity', default='just-an-arbitrary-team-name')
    p.add_argument('--data_root', default=_DEFAULT_DATA_ROOT)
    p.add_argument('--out_root', default=_DEFAULT_OUT_ROOT)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--freeze_encoder', action='store_true', default=False,
               help='Freeze encoder weights (no gradient updates)')
    p.add_argument('--freeze_decoder', action='store_true', default=False,
                help='Freeze decoder weights (no gradient updates)')
    p.add_argument('--use_lora_encoder', action='store_true', default=False,
                   help='Apply LoRA to the vision encoder (CLIP, ViT)')
    p.add_argument('--use_lora_decoder', action='store_true', default=False,
                   help='Apply LoRA to the transformer decoder')
    p.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    p.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    p.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    p.add_argument('--pretrained_weights', type=str, default=None,
                   help='Path to a .pt file to load partial weights (e.g., just the encoder)')
    p.add_argument('--filtered_annotations', action='store_true', default=False)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_img_dir = os.path.join(args.data_root, 'train')
    val_img_dir   = os.path.join(args.data_root, 'val')
    if args.filtered_annotations:
        train_ann = os.path.join(args.data_root, 'annotations', 'train_filtered.json')
        val_ann   = os.path.join(args.data_root, 'annotations', 'val_filtered.json')
        print("Using filtered annotations!!")
    else:
        train_ann = os.path.join(args.data_root, 'annotations', 'train.json')
        val_ann   = os.path.join(args.data_root, 'annotations', 'val.json')
    cache_dir     = os.path.join(args.data_root, 'tokenizer_cache')

    out_dir = Path(args.out_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)
    print(f"Device: {device}")

    print(f"Building custom tokenizer ({args.text_repr})...")
    tokenizer = build_tokenizer(args.text_repr, train_ann, cache_dir, max_len=args.max_len)

    print("Loading datasets...")
    ds_train = VizWizDataset(train_img_dir, train_ann, tokenizer,
                             split='train', val_fraction=args.val_fraction, seed=args.seed)
    ds_val   = VizWizDataset(train_img_dir, train_ann, tokenizer,
                             split='val', val_fraction=args.val_fraction, seed=args.seed)
    ds_test  = VizWizDataset(val_img_dir, val_ann, tokenizer, split='test', seed=args.seed)
    print(f"  train: {len(ds_train)}, val: {len(ds_val)}, test: {len(ds_test)}")

    dl_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                 collate_fn=collate_fn, pin_memory=True)
    dl_train = DataLoader(ds_train, shuffle=True,  **dl_kw)
    dl_val   = DataLoader(ds_val,   shuffle=False, **dl_kw)
    dl_test  = DataLoader(ds_test,  shuffle=False, **dl_kw)

    model = CaptioningModel(
        encoder_name=args.encoder,
        decoder_type=args.decoder,
        decoder_layers=args.decoder_layers,
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        decoder_model_name=args.decoder_model_name,
    ).to(device)
    
    # --- WEIGHT TRANSFER ---
    if args.pretrained_weights:
        print(f"\nExtracting weights from {args.pretrained_weights}...")
        state_dict = torch.load(args.pretrained_weights, map_location=device)
        
        filtered_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.') and not k.startswith('encoder.proj'):
                # 1. Strip the PEFT LoRA wrapper
                clean_key = k.replace('base_model.model.', '')
                
                # 2. Revert LoRA 'base_layer' naming back to standard PyTorch naming
                clean_key = clean_key.replace('.base_layer.weight', '.weight')
                clean_key = clean_key.replace('.base_layer.bias', '.bias')
                
                # 3. Discard the actual LoRA matrices since your new encoder is frozen
                if 'lora_' not in clean_key:
                    filtered_dict[clean_key] = v
        
        # Load with strict=False
        missing, unexpected = model.load_state_dict(filtered_dict, strict=False)
        print(f"Successfully transferred {len(filtered_dict)} vision encoder tensors.")
        print(f"Unexpected keys loaded: {len(unexpected)}")
        print(f"Missing keys ignored: {len(missing)}\n")
    # --------------------------------
    
    # LORA INTEGRATION
    if args.use_lora_encoder and args.encoder in ['clip', 'vit-b-16', 'vit-b-32']:
        print(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha}) to encoder ({args.encoder})...")
        
        # "q_proj" and "v_proj" catch CLIP layers
        # "query" and "value" catch standard Hugging Face ViT layers
        # "in_proj_weight" catches torchvision ViT implementations
        target_modules_enc = ["q_proj", "v_proj", "query", "value", "in_proj_weight"]

        config_enc = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules_enc,
            lora_dropout=args.lora_dropout,
            bias="none"
        )
        
        # Apply PEFT to the underlying model inside the encoder wrapper
        # (Based on models.py, CLIP/ViT is stored in self.model inside the wrapper)
        if hasattr(model.encoder, 'model'):
            model.encoder.model = get_peft_model(model.encoder.model, config_enc)
        else:
            model.encoder = get_peft_model(model.encoder, config_enc)
            
        print("Encoder LoRA applied successfully!")
    elif args.use_lora_encoder:
        print(f"Warning: LoRA is not implemented for CNN encoders like {args.encoder}. Skipping Encoder LoRA.")
    
    if args.use_lora_decoder and args.decoder in ['gpt2', 't5', 'smollm', 'qwen']:
        print(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha}) to {args.decoder}...")
        
        # Define target modules based on the specific transformer architecture
        if args.decoder == 'gpt2':
            target_modules = ["c_attn", "c_proj"]
        elif args.decoder == 't5':
            target_modules = ["q", "v"]
        elif args.decoder in ['smollm', 'qwen']:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj" 
            ]

        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM" if args.decoder != 't5' else "SEQ_2_SEQ_LM"
        )
        
        # Apply PEFT specifically to the Hugging Face model inside your custom wrapper
        if args.decoder == 'gpt2':
            model.decoder.gpt2 = get_peft_model(model.decoder.gpt2, config)
        elif args.decoder == 't5':
            model.decoder.t5 = get_peft_model(model.decoder.t5, config)
        elif args.decoder == 'smollm':
            model.decoder.smollm = get_peft_model(model.decoder.smollm, config)
        elif args.decoder == 'qwen':
            model.decoder.qwen = get_peft_model(model.decoder.qwen, config)
            
        print("LoRA applied successfully!")
    elif args.use_lora_decoder:
        print("Warning: LoRA is only implemented for transformer decoders (gpt2, t5, smollm). Skipping LoRA.")

    # Freeze encoder if requested
    if args.freeze_encoder:
        print("Freezing encoder weights")
        for p in model.encoder.parameters():
            p.requires_grad = False

    # Freeze decoder if requested
    if args.freeze_decoder:
        print("Freezing decoder weights")
        for name, p in model.named_parameters():
            # Everything that is NOT encoder is considered decoder side
            if not name.startswith("encoder"):
                p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)

    print(f"  Model: {args.encoder} + {args.decoder}x{args.decoder_layers} | "
        f"text={args.text_repr} | params={n_trainable/1e6:.1f}M")

    wandb_module = None
    if args.wandb:
        import wandb as wandb_module
        wandb_module.init(project=args.wandb_project, entity=args.wandb_entity,
                          name=args.run_name, config=vars(args), mode="online")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    bleu, rouge, meteor = load_metrics()

    # Fixed indices for qualitative evaluation — same every epoch for tracking
    train_qual_indices = get_fixed_examples(ds_train, n=100, seed=args.seed)
    val_qual_indices   = get_fixed_examples(ds_val,   n=100, seed=args.seed)
    print(f"Qualitative eval: {len(train_qual_indices)} train, {len(val_qual_indices)} val examples")

    # Inference Mode
    if len(trainable_params) == 0:
        print("\nNo trainable parameters — running in inference-only mode.\n")

        # Evaluate on validation set
        t0 = time.time()
        val_loss, val_metrics = eval_epoch(model, criterion, dl_val, tokenizer, device,
                                           bleu, rouge, meteor)
        elapsed = time.time() - t0
        
        epoch = 1

        row = {'epoch': epoch,
               **{k: round(v, 2) for k, v in val_metrics.items()},
               'time_inferene_s': round(elapsed, 1)}
        
        if wandb_module is not None:
            wandb_module.log(row)

        # Evaluate on test set
        test_loss, test_metrics = eval_epoch(
            model, nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx),
            dl_test, tokenizer, device, bleu, rouge, meteor
        )

        if wandb_module is not None:
            wandb_module.log({'test_' + k: v for k, v in test_metrics.items()})
            wandb_module.finish()

        print(f"\n{'='*60}")
        print(f"TEST RESULTS [{args.run_name}] (FROZEN MODEL)")
        print(f"  test_loss = {test_loss:.4f}")
        print(f"  BLEU-1    = {test_metrics.get('bleu1', 0):.2f}%")
        print(f"  BLEU-2    = {test_metrics.get('bleu2', 0):.2f}%")
        print(f"  ROUGE-L   = {test_metrics.get('rougeL', 0):.2f}%")
        print(f"  METEOR    = {test_metrics.get('meteor', 0):.2f}%")
        print(f"{'='*60}")

        log_qualitative_examples(model, ds_val, val_qual_indices, tokenizer, device,
                                 epoch, 'val', out_dir / 'val_samples.json', wandb_module)

        return


    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
        
    # Training with warmup and cosine decay for Qwen
    if args.decoder == 'qwen':  
        total_steps = len(dl_train) * args.epochs
        warmup_steps = int(total_steps * 0.1) 
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_decay, patience=args.lr_patience)
    

    # Early stopping: auto-detect direction from metric name
    es_mode = 'min' if args.es_metric == 'val_loss' else 'max'
    best_val_loss   = float('inf')
    best_es_value   = float('inf') if es_mode == 'min' else float('-inf')
    es_counter      = 0
    history         = []

    tf_mode = 'scheduled' if args.scheduled_tf else ('on' if args.teacher_forcing else 'off')
    print(f"Early stopping: metric={args.es_metric} mode={es_mode} patience={args.es_patience}")
    print(f"Teacher forcing: {tf_mode}")

    for epoch in range(1, args.epochs + 1):
        if args.scheduled_tf:
            tf_prob = 1.0 - (epoch - 1) / max(1, args.epochs - 1)
        elif args.teacher_forcing:
            tf_prob = 1.0
        else:
            tf_prob = 0.0

        t0 = time.time()
        train_loss = train_one_epoch(model, optimizer, criterion, dl_train, device, tf_prob, args, scheduler)
        val_loss, val_metrics = eval_epoch(model, criterion, dl_val, tokenizer, device,
                                           bleu, rouge, meteor)
        elapsed = time.time() - t0
        if args.decoder != 'qwen':
            scheduler.step(val_loss)

        row = {'epoch': epoch, 'tf_prob': round(tf_prob, 3),
               'train_loss': round(train_loss, 4),
               'val_loss': round(val_loss, 4),
               **{k: round(v, 2) for k, v in val_metrics.items()},
               'time_s': round(elapsed, 1)}
        history.append(row)

        # best loss checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / 'best_loss_model.pt')

        # best early-stopping metric checkpoint
        es_value = val_loss if args.es_metric == 'val_loss' else val_metrics.get(args.es_metric, best_es_value)
        improved = (es_value < best_es_value) if es_mode == 'min' else (es_value > best_es_value)
        if improved:
            best_es_value = es_value
            es_counter    = 0
            torch.save(model.state_dict(), out_dir / 'best_metric_model.pt')
        else:
            es_counter += 1

        print(f"Epoch {epoch:3d}/{args.epochs}  tf={tf_prob:.2f}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"BLEU-1={val_metrics.get('bleu1', 0):.1f}%  "
              f"BLEU-2={val_metrics.get('bleu2', 0):.1f}%  "
              f"ROUGE-L={val_metrics.get('rougeL', 0):.1f}%  "
              f"METEOR={val_metrics.get('meteor', 0):.1f}%  "
              f"ES={es_counter}/{args.es_patience}  [{elapsed:.0f}s]")

        if wandb_module is not None:
            wandb_module.log(row)

        log_qualitative_examples(model, ds_train, train_qual_indices, tokenizer, device,
                                 epoch, 'train', out_dir / 'train_samples.json', wandb_module)
        log_qualitative_examples(model, ds_val, val_qual_indices, tokenizer, device,
                                 epoch, 'val', out_dir / 'val_samples.json', wandb_module)

        with open(out_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        if es_counter >= args.es_patience:
            print(f"\nEarly stopping triggered at epoch {epoch} "
                  f"(no improvement in {args.es_metric} for {args.es_patience} epochs).")
            break

    print("\nRunning final test evaluation (best metric checkpoint)...")
    model.load_state_dict(torch.load(out_dir / 'best_metric_model.pt'))
    test_loss, test_metrics = eval_epoch(model, criterion, dl_test, tokenizer, device,
                                         bleu, rouge, meteor)

    print(f"\n{'='*60}")
    print(f"TEST RESULTS [{args.run_name}]")
    print(f"  test_loss = {test_loss:.4f}")
    print(f"  BLEU-1    = {test_metrics.get('bleu1', 0):.2f}%")
    print(f"  BLEU-2    = {test_metrics.get('bleu2', 0):.2f}%")
    print(f"  ROUGE-L   = {test_metrics.get('rougeL', 0):.2f}%")
    print(f"  METEOR    = {test_metrics.get('meteor', 0):.2f}%")
    print(f"{'='*60}")

    with open(out_dir / 'test_results.json', 'w') as f:
        json.dump({'test_loss': test_loss, **test_metrics}, f, indent=2)

    if wandb_module is not None:
        wandb_module.log({'test_' + k: v for k, v in test_metrics.items()})
        wandb_module.finish()



if __name__ == '__main__':
    main()
