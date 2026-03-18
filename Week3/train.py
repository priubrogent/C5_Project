import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import evaluate as hf_evaluate

from dataset import VizWizDataset, collate_fn
from models import CaptioningModel
from tokenizer import build_tokenizer

DATA_ROOT     = '/home/priubrogent/01_MCV/C5/vizwiz_dataset'
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, 'train')
VAL_IMG_DIR   = os.path.join(DATA_ROOT, 'val')
TRAIN_ANN     = os.path.join(DATA_ROOT, 'annotations', 'train.json')
VAL_ANN       = os.path.join(DATA_ROOT, 'annotations', 'val.json')
CACHE_DIR     = os.path.join(DATA_ROOT, 'tokenizer_cache')
OUT_ROOT      = '/home/priubrogent/01_MCV/C5/00_Project/Week3/outputs'


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


def train_one_epoch(model, optimizer, criterion, dataloader, device, teacher_forcing):
    model.train()
    total_loss, n = 0.0, 0
    for imgs, captions, _ in dataloader:
        imgs, captions = imgs.to(device), captions.to(device)
        optimizer.zero_grad()
        logits = model(imgs, captions, teacher_forcing=teacher_forcing)
        loss = criterion(logits, captions[:, 1:])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * imgs.shape[0]
        n += imgs.shape[0]
    return total_loss / n


@torch.no_grad()
def eval_epoch(model, criterion, dataloader, tokenizer, device,
               bleu, rouge, meteor, max_eval_samples=1000):
    model.eval()
    total_loss, n = 0.0, 0
    predictions, references = [], []
    n_eval = 0

    for imgs, captions, all_captions in dataloader:
        imgs, captions = imgs.to(device), captions.to(device)
        logits = model(imgs, captions, teacher_forcing=True)
        loss = criterion(logits, captions[:, 1:])
        total_loss += loss.item() * imgs.shape[0]
        n += imgs.shape[0]

        if n_eval < max_eval_samples:
            gen = model.generate(imgs, tokenizer.max_len - 1, tokenizer.sos_idx, tokenizer.eos_idx)
            for i in range(imgs.shape[0]):
                pred = tokenizer.decode(gen[i].cpu().tolist())
                if pred.strip():
                    predictions.append(pred)
                    references.append(all_captions[i])
            n_eval += imgs.shape[0]

    metrics = compute_metrics(bleu, rouge, meteor, predictions, references) if predictions else {}
    return total_loss / n, metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--encoder', default='resnet18',
                   choices=['resnet18', 'resnet34', 'resnet50', 'vgg16', 'vgg19'])
    p.add_argument('--decoder', default='gru', choices=['gru', 'lstm'])
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
    p.add_argument('--lr_decay', type=float, default=0.5)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--run_name', default='run')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--val_fraction', type=float, default=0.1)
    p.add_argument('--max_eval_samples', type=int, default=2000)
    p.add_argument('--wandb', action='store_true', default=False)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(OUT_ROOT) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)
    print(f"Device: {device}")

    print(f"Building tokenizer ({args.text_repr})...")
    tokenizer = build_tokenizer(args.text_repr, TRAIN_ANN, CACHE_DIR, max_len=args.max_len)

    print("Loading datasets...")
    ds_train = VizWizDataset(TRAIN_IMG_DIR, TRAIN_ANN, tokenizer,
                             split='train', val_fraction=args.val_fraction, seed=args.seed)
    ds_val   = VizWizDataset(TRAIN_IMG_DIR, TRAIN_ANN, tokenizer,
                             split='val', val_fraction=args.val_fraction, seed=args.seed)
    ds_test  = VizWizDataset(VAL_IMG_DIR, VAL_ANN, tokenizer, split='test', seed=args.seed)
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
        optimizer, mode='min', factor=args.lr_decay, patience=args.patience)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    bleu, rouge, meteor = load_metrics()

    if args.wandb:
        import wandb
        wandb.init(project='c5-week3-captioning', name=args.run_name, config=vars(args))

    best_val_loss = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, optimizer, criterion, dl_train,
                                     device, args.teacher_forcing)
        val_loss, val_metrics = eval_epoch(model, criterion, dl_val, tokenizer, device,
                                           bleu, rouge, meteor, args.max_eval_samples)
        elapsed = time.time() - t0
        scheduler.step(val_loss)

        row = {'epoch': epoch, 'train_loss': round(train_loss, 4),
               'val_loss': round(val_loss, 4),
               **{k: round(v, 2) for k, v in val_metrics.items()},
               'time_s': round(elapsed, 1)}
        history.append(row)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"BLEU-1={val_metrics.get('bleu1', 0):.1f}%  "
              f"BLEU-2={val_metrics.get('bleu2', 0):.1f}%  "
              f"ROUGE-L={val_metrics.get('rougeL', 0):.1f}%  "
              f"METEOR={val_metrics.get('meteor', 0):.1f}%  "
              f"[{elapsed:.0f}s]")

        if args.wandb:
            import wandb
            wandb.log(row)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / 'best_model.pt')

        with open(out_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print("\nRunning final test evaluation...")
    model.load_state_dict(torch.load(out_dir / 'best_model.pt'))
    test_loss, test_metrics = eval_epoch(model, criterion, dl_test, tokenizer, device,
                                         bleu, rouge, meteor, max_eval_samples=len(ds_test))

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
