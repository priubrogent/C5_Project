import torch
import torch.nn as nn
from transformers import ResNetModel
import torchvision.models as tv_models


class ResNetEncoder(nn.Module):

    HF_NAMES = {
        'resnet18': 'microsoft/resnet-18',
        'resnet34': 'microsoft/resnet-34',
        'resnet50': 'microsoft/resnet-50',
    }
    OUT_DIMS = {'resnet18': 512, 'resnet34': 512, 'resnet50': 2048}

    def __init__(self, name: str, output_dim: int):
        super().__init__()
        self.model = ResNetModel.from_pretrained(self.HF_NAMES[name])
        feat_dim = self.OUT_DIMS[name]
        self.proj = nn.Linear(feat_dim, output_dim) if feat_dim != output_dim else nn.Identity()

    def forward(self, x):
        feat = self.model(x).pooler_output.squeeze(-1).squeeze(-1)
        return self.proj(feat)


class VGGEncoder(nn.Module):

    def __init__(self, name: str, output_dim: int):
        super().__init__()
        if name == 'vgg16':
            backbone = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
        else:
            backbone = tv_models.vgg19(weights=tv_models.VGG19_Weights.DEFAULT)
        self.features = backbone.features
        self.proj = nn.Linear(512, output_dim) if 512 != output_dim else nn.Identity()

    def forward(self, x):
        feat = self.features(x).mean(dim=[2, 3])
        return self.proj(feat)


def build_encoder(name: str, output_dim: int) -> nn.Module:
    if name.startswith('resnet'):
        return ResNetEncoder(name, output_dim)
    if name.startswith('vgg'):
        return VGGEncoder(name, output_dim)
    raise ValueError(f"Unknown encoder: {name}")


class CaptioningModel(nn.Module):

    def __init__(self, encoder_name='resnet18', decoder_type='gru',
                 decoder_layers=1, vocab_size=80,
                 embed_dim=512, hidden_dim=512, dropout=0.0):
        super().__init__()

        self.encoder = build_encoder(encoder_name, hidden_dim)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        rnn_cls = nn.GRU if decoder_type == 'gru' else nn.LSTM
        self.decoder = rnn_cls(
            embed_dim, hidden_dim,
            num_layers=decoder_layers,
            batch_first=True,
            dropout=dropout if decoder_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim, vocab_size)

        self.decoder_type = decoder_type
        self.decoder_layers = decoder_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

    def _init_hidden(self, img):
        feat = self.encoder(img)
        h0 = feat.unsqueeze(0).expand(self.decoder_layers, -1, -1).contiguous()
        if self.decoder_type == 'lstm':
            return (h0, torch.zeros_like(h0))
        return h0

    def forward(self, img, captions, teacher_forcing=True):
        hidden = self._init_hidden(img)

        if teacher_forcing:
            inp = self.dropout(self.embed(captions[:, :-1]))
            out, _ = self.decoder(inp, hidden)
            return self.proj(out).permute(0, 2, 1)

        B, T = captions.shape
        current = captions[:, 0]
        outputs = []
        for _ in range(T - 1):
            emb = self.dropout(self.embed(current)).unsqueeze(1)
            out, hidden = self.decoder(emb, hidden)
            logit = self.proj(out.squeeze(1))
            outputs.append(logit)
            current = logit.argmax(dim=-1)
        return torch.stack(outputs, dim=2)

    @torch.no_grad()
    def generate(self, img, max_len, sos_idx, eos_idx):
        self.eval()
        B = img.shape[0]
        device = img.device
        hidden = self._init_hidden(img)
        current = torch.full((B,), sos_idx, dtype=torch.long, device=device)
        generated = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            emb = self.embed(current).unsqueeze(1)
            out, hidden = self.decoder(emb, hidden)
            logit = self.proj(out.squeeze(1))
            current = logit.argmax(dim=-1)
            current = current.masked_fill(finished, 0)
            generated.append(current.clone())
            finished |= (current == eos_idx)
            if finished.all():
                break

        while len(generated) < max_len:
            generated.append(torch.zeros(B, dtype=torch.long, device=device))

        return torch.stack(generated, dim=1)
