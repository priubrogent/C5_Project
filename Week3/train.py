import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

import evaluate as hf_evaluate

from dataset import VizWizDataset, collate_fn
from models import CaptioningModel
from tokenizer import build_tokenizer

# ImageNet normalization constants (for denormalization)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_DEFAULT_DATA_ROOT = '/hhome/priubrogent/mcvpol/vizwiz_dataset'
_DEFAULT_OUT_ROOT  = '/hhome/priubrogent/mcvpol/C5/Week3/outputs'


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
def log_qualitative_examples(model, dataset, indices, tokenizer, device, epoch, split, wandb):
    """Generate captions for fixed examples and log to wandb."""
    model.eval()
    images_with_captions = []

    for idx in indices:
        img_tensor, _, gt_captions = dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Generate caption
        gen = model.generate(img_tensor, tokenizer.max_len - 1, tokenizer.sos_idx, tokenizer.eos_idx)
        pred_caption = tokenizer.decode(gen[0].cpu().tolist())

        # Denormalize image for visualization
        img_denorm = img_tensor[0].cpu() * IMAGENET_STD + IMAGENET_MEAN
        img_denorm = img_denorm.clamp(0, 1)
        pil_img = to_pil_image(img_denorm)

        # Format caption: predicted + ground truth
        gt_str = gt_captions[0] if gt_captions else "N/A"
        caption_text = f"Pred: {pred_caption}\nGT: {gt_str}"

        images_with_captions.append(wandb.Image(pil_img, caption=caption_text))

    wandb.log({f"{split}_qualitative": images_with_captions}, step=epoch)


def train_one_epoch(model, optimizer, criterion, dataloader, device, tf_prob):
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
                   choices=['resnet18', 'resnet34', 'resnet50', 'vgg16', 'vgg19'])
    p.add_argument('--decoder', default='gru', choices=['gru', 'lstm', "gru_attn"])
    p.add_argument('--decoder_layers', type=int, default=1)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--embed_dim', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--text_repr', default='char', choices=['char', 'word', 'subword'])
    p.add_argument('--max_len', type=int, default=None)
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
    p.add_argument('--wandb', action='store_true', default=False)
    p.add_argument('--wandb_project', default='c5-week3-captioning')
    p.add_argument('--wandb_entity', default=None)
    p.add_argument('--data_root', default=_DEFAULT_DATA_ROOT)
    p.add_argument('--out_root', default=_DEFAULT_OUT_ROOT)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_img_dir = os.path.join(args.data_root, 'train')
    val_img_dir   = os.path.join(args.data_root, 'val')
    train_ann     = os.path.join(args.data_root, 'annotations', 'train.json')
    val_ann       = os.path.join(args.data_root, 'annotations', 'val.json')
    cache_dir     = os.path.join(args.data_root, 'tokenizer_cache')

    out_dir = Path(args.out_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)
    print(f"Device: {device}")

    print(f"Building tokenizer ({args.text_repr})...")
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
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {args.encoder} + {args.decoder}x{args.decoder_layers} | "
          f"text={args.text_repr} | params={n_params/1e6:.1f}M")

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_decay, patience=args.lr_patience)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    bleu, rouge, meteor = load_metrics()

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=args.run_name, config=vars(args))

    # Select fixed examples for qualitative evaluation (always same 10)
    train_qual_indices = get_fixed_examples(ds_train, n=10, seed=args.seed)
    val_qual_indices = get_fixed_examples(ds_val, n=10, seed=args.seed)
    print(f"Selected {len(train_qual_indices)} train and {len(val_qual_indices)} val examples for qualitative eval")

    # Early stopping: auto-detect direction from metric name
    es_mode = 'min' if args.es_metric == 'val_loss' else 'max'
    best_val_loss   = float('inf')
    best_es_value   = float('inf') if es_mode == 'min' else float('-inf')
    es_counter      = 0
    history         = []

    # Fix the same 1000 val indices for qualitative tracking across all epochs
    val_sample_indices = random.sample(range(len(ds_val)), min(1000, len(ds_val)))

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
        train_loss = train_one_epoch(model, optimizer, criterion, dl_train, device, tf_prob)
        val_loss, val_metrics = eval_epoch(model, criterion, dl_val, tokenizer, device,
                                           bleu, rouge, meteor)
        elapsed = time.time() - t0
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

        if args.wandb:
            import wandb
            wandb.log(row)
            # Log qualitative examples
            log_qualitative_examples(model, ds_train, train_qual_indices, tokenizer, device, epoch, "train", wandb)
            log_qualitative_examples(model, ds_val, val_qual_indices, tokenizer, device, epoch, "val", wandb)

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

    if args.wandb:
        import wandb
        wandb.log({'test_' + k: v for k, v in test_metrics.items()})
        wandb.finish()


if __name__ == '__main__':
    main()
