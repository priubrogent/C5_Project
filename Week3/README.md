# Week 3 — Image Captioning

This week implements an image captioning pipeline trained on the [VizWiz dataset](https://vizwiz.org/tasks-and-datasets/image-captioning/). The system uses a CNN encoder to extract visual features and an RNN decoder to generate captions, with support for multiple encoder architectures, decoder types, and tokenization strategies.

---

## Project Structure

```
Week3/
├── dataset.py          # VizWiz dataset loader and tokenization utilities
├── tokenizer.py        # Character, word, and subword (BPE) tokenizers
├── models.py           # Encoder and decoder architectures
├── train.py            # Training loop with evaluation and W&B logging
├── qualitative_eval.py # Inference and visual output generation
└── dataset_size.py     # Dataset statistics utility
```

---

## Architecture

### Encoders (`models.py`)

Three encoder families are supported:

| Encoder    | Backbone              | Output dim |
| ---------- | --------------------- | ---------- |
| `resnet18` | HuggingFace ResNet-18 | 512        |
| `resnet34` | HuggingFace ResNet-34 | 512        |
| `resnet50` | HuggingFace ResNet-50 | 2048       |
| `vgg16`    | torchvision VGG-16    | 512        |
| `vgg19`    | torchvision VGG-19    | 512        |

A linear projection maps any encoder output to the chosen `hidden_dim`.

For attention-based decoders, `AttentionSupportive_ResNetEncoder` returns the full spatial feature map `(B, N, C)` instead of a pooled vector.

### Decoders (`models.py`)

Three decoder types are available via `--decoder`:

- **`gru`** — single or multi-layer GRU, initialized with the pooled image feature.
- **`lstm`** — same as above using LSTM cells.
- **`gru_attn`** — GRU with additive (Bahdanau) attention over the spatial encoder features. At each decoding step, a context vector is computed from encoder spatial outputs and the current decoder hidden state, then concatenated with the token embedding as input to the GRU.

The full `CaptioningModel` class supports:

- Configurable `embed_dim`, `hidden_dim`, `decoder_layers`, `dropout`
- Teacher forcing (always on, always off, or scheduled linear decay)
- Greedy decoding via `.generate()`

---

## Tokenizers (`tokenizer.py`)

Three tokenization strategies are available via `--text_repr`:

| Type      | Class              | Vocab size   | Max len | Notes                                 |
| --------- | ------------------ | ------------ | ------- | ------------------------------------- |
| `char`    | `CharTokenizer`    | ~80 (fixed)  | 150     | No training required                  |
| `word`    | `WordTokenizer`    | ~data-driven | 35      | Built from train captions, min_freq=2 |
| `subword` | `SubwordTokenizer` | 4000 (BPE)   | 50      | Trained with HuggingFace `tokenizers` |

All tokenizers share a common interface: `encode(text) -> List[int]`, `decode(indices) -> str`, and expose `sos_idx`, `eos_idx`, `pad_idx`, `vocab_size`, and `max_len`. Word and subword tokenizers are cached to disk after the first build.

---

## Dataset (`dataset.py`)

`VizWizDataset` loads images and captions from the VizWiz annotation JSON. It supports a train/val split (default 10% val, seeded for reproducibility) and a separate test split from the official val set.

Images are resized to 224×224 and normalized with ImageNet statistics. At training time, one caption is sampled at random per image; all captions are returned for evaluation.

---

## Training (`train.py`)

```bash
python train.py \
    --encoder resnet50 \
    --decoder gru \
    --text_repr subword \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-3 \
    --optimizer adamw \
    --run_name my_run \
    --wandb
```

### Key arguments

| Argument            | Default    | Description                                          |
| ------------------- | ---------- | ---------------------------------------------------- |
| `--encoder`         | `resnet18` | `resnet18`, `resnet34`, `resnet50`, `vgg16`, `vgg19` |
| `--decoder`         | `gru`      | `gru`, `lstm`, `gru_attn`                            |
| `--decoder_layers`  | `1`        | Number of RNN layers                                 |
| `--hidden_dim`      | `512`      | Hidden and embedding dimension                       |
| `--text_repr`       | `char`     | `char`, `word`, `subword`                            |
| `--optimizer`       | `adam`     | `adam`, `adamw`, `sgd`                               |
| `--teacher_forcing` | on         | Use `--no_teacher_forcing` to disable                |
| `--scheduled_tf`    | off        | Linearly decay teacher forcing probability           |
| `--es_metric`       | `val_loss` | Early stopping metric                                |
| `--es_patience`     | `10`       | Early stopping patience (epochs)                     |

Training saves `best_loss_model.pt` and `best_metric_model.pt` checkpoints, a `history.json` with per-epoch metrics, and qualitative caption samples every epoch. Final test evaluation is run automatically on the best checkpoint.

Metrics reported: **BLEU-1**, **BLEU-2**, **ROUGE-L**, **METEOR** (via HuggingFace `evaluate`).

---

## Qualitative Evaluation (`qualitative_eval.py`)

Generates annotated image grids and individual PNGs with predicted and ground-truth captions, useful for slides and reports.

```bash
python qualitative_eval.py --run_dir outputs/resnet50_gru_subword_adamw_1e3
python qualitative_eval.py --run_dir outputs/resnet50_gru_subword_adamw_1e3 \
    --splits val test --n_samples 12 --cols 4 --checkpoint best_loss_model.pt
```

Outputs are saved under `<run_dir>/qualitative/`:

- `grid_val.png` / `grid_test.png` — combined image grids
- `individual_val/` / `individual_test/` — per-image annotated PNGs
- `predictions.json` — all predictions and ground-truth captions

---

## Dependencies

- PyTorch, torchvision
- HuggingFace `transformers`, `evaluate`, `tokenizers`
- `matplotlib`, `Pillow`
- `wandb` (optional, for experiment tracking)
