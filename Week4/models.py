import torch
import torch.nn as nn
from transformers import (
    ResNetModel,
    ViTModel,
    ViTImageProcessor,
    CLIPVisionModel,
    CLIPImageProcessor,
    GPT2LMHeadModel,
    AutoTokenizer,
    T5ForConditionalGeneration,
    AutoModel,
    AutoModelForCausalLM,
)
import torchvision.models as tv_models
from transformers.modeling_outputs import BaseModelOutput


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


class AttentionSupportive_ResNetEncoder(nn.Module):

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
        outputs = self.model(x)
        feat_map = outputs.last_hidden_state  # (B, C, H, W)

        B, C, H, W = feat_map.shape

        # flatten spatial dimensions
        feat_map = feat_map.view(B, C, H * W).permute(0, 2, 1)  # (B, N, C)
        return self.proj(feat_map)


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


class ViTEncoder(nn.Module):
    """Vision Transformer encoder (ViT-B/16, ViT-B/32)"""

    HF_NAMES = {
        'vit-b-16': 'google/vit-base-patch16-224',
        'vit-b-32': 'google/vit-base-patch32-224-in21k',
    }
    OUT_DIMS = {'vit-b-16': 768, 'vit-b-32': 768}

    def __init__(self, name: str, output_dim: int):
        super().__init__()
        if name not in self.HF_NAMES:
            raise ValueError(f"Unknown ViT model: {name}")
        self.model = ViTModel.from_pretrained(self.HF_NAMES[name])
        feat_dim = self.OUT_DIMS[name]
        self.proj = nn.Linear(feat_dim, output_dim) if feat_dim != output_dim else nn.Identity()

    def forward(self, x):
        outputs = self.model(x)
        #feat = outputs.pooler_output  # (B, 768)
        feat = outputs.last_hidden_state 
        return self.proj(feat)


class CLIPEncoder(nn.Module):
    """CLIP vision encoder"""

    def __init__(self, model_name: str = 'openai/clip-vit-base-patch32', output_dim: int = 512):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_name)
        # CLIP vision models output 768-dim features
        feat_dim = self.model.config.hidden_size
        self.proj = nn.Linear(feat_dim, output_dim) if feat_dim != output_dim else nn.Identity()

    def forward(self, x):
        outputs = self.model(x)
        feat = outputs.pooler_output  # (B, feat_dim)
        return self.proj(feat)


class GPT2Decoder(nn.Module):
    """GPT2-based autoregressive decoder for image captioning"""

    def __init__(self, input_dim: int, vocab_size: int = None, model_name: str = 'gpt2'):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.gpt2_config = self.gpt2.config
        
        # Projection from encoder output to GPT2 embedding dimension
        self.embed_proj = nn.Linear(input_dim, self.gpt2_config.hidden_size)
        
        # If custom vocab size, resize token embeddings
        if vocab_size is not None and vocab_size != self.gpt2_config.vocab_size:
            self.gpt2.resize_token_embeddings(vocab_size)

    def forward(self, encoder_output, input_ids=None, attention_mask=None):
        """
        encoder_output: (B, hidden_dim) from encoder
        input_ids: (B, seq_len) token ids
        """
        # Project encoder output to embedding space and expand for sequence
        proj_output = self.embed_proj(encoder_output)  # (B, gpt2_hidden)
        
        if input_ids is not None:
            # Get GPT2 embeddings
            gpt2_embeds = self.gpt2.transformer.wte(input_ids)  # (B, seq_len, gpt2_hidden)
            
            # Concatenate projected encoder output as prefix
            combined_embeds = torch.cat([proj_output.unsqueeze(1), gpt2_embeds], dim=1)
            
            # Create attention mask if provided
            if attention_mask is not None:
                attention_mask = torch.cat([
                    torch.ones(encoder_output.shape[0], 1, device=encoder_output.device),
                    attention_mask
                ], dim=1)
            
            outputs = self.gpt2(inputs_embeds=combined_embeds, attention_mask=attention_mask)
            return outputs.logits
        else:
            # Generation mode: just return encoder output
            return proj_output

    def generate(self, encoder_output, max_length=50, eos_token_id=None, sos_token_id=None):
        """Generate captions using GPT2"""
        device = encoder_output.device
        batch_size = encoder_output.shape[0]
        
        proj_output = self.embed_proj(encoder_output).unsqueeze(1)  # (B, 1, hidden)
        
        # Start with SOS token if provided
        if sos_token_id is not None:
            input_ids = torch.full((batch_size, 1), sos_token_id, dtype=torch.long, device=device)
        else:
            input_ids = torch.full((batch_size, 1), self.gpt2_config.bos_token_id or 50256, dtype=torch.long, device=device)
        
        generated = []
        for _ in range(max_length):
            gpt2_embeds = self.gpt2.transformer.wte(input_ids)
            combined_embeds = torch.cat([proj_output, gpt2_embeds], dim=1)
            
            outputs = self.gpt2(inputs_embeds=combined_embeds)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated.append(next_token)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return torch.cat(generated, dim=1)


class T5Decoder(nn.Module):
    """T5-based encoder-decoder for image captioning"""

    def __init__(self, input_dim: int, vocab_size: int = None, model_name: str = 't5-base'):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.t5_config = self.t5.config
        
        # Projection from encoder output to T5 embedding dimension
        self.embed_proj = nn.Linear(input_dim, self.t5_config.d_model)
        
        # If custom vocab size, resize token embeddings
        if vocab_size is not None and vocab_size != self.t5_config.vocab_size:
            self.t5.resize_token_embeddings(vocab_size)

    def forward(self, encoder_output, decoder_input_ids=None, attention_mask=None):
        """
        encoder_output: (B, hidden_dim) from encoder
        decoder_input_ids: (B, seq_len) token ids for decoder
        """
        proj_output = self.embed_proj(encoder_output)  # (B, t5_d_model)
        
        # Create encoder hidden state
        # T5 expects encoder_last_hidden_state: (B, seq_len, d_model)
        #encoder_hidden_state = proj_output.unsqueeze(1)  # (B, 1, d_model)
        encoder_hidden_state = proj_output

        if decoder_input_ids is not None:          
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_state)
            outputs = self.t5(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
            )
            return outputs.logits
        else:
            return proj_output

    def generate(self, encoder_output, max_length=50, eos_token_id=None, sos_token_id=None):
        """Generate captions using T5"""
        device = encoder_output.device
        #proj_output = self.embed_proj(encoder_output).unsqueeze(1)  # (B, 1, d_model)
        proj_output = self.embed_proj(encoder_output)

        batch_size = encoder_output.shape[0]
        if sos_token_id is not None:
            decoder_input_ids = torch.full((batch_size, 1), sos_token_id, dtype=torch.long, device=device)
        else:
            decoder_input_ids = torch.full((batch_size, 1), self.t5_config.decoder_start_token_id or 0, dtype=torch.long, device=device)
        
        generated = []
        for _ in range(max_length):
            encoder_outputs = BaseModelOutput(last_hidden_state=proj_output)
            outputs = self.t5(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            generated.append(next_token)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return torch.cat(generated, dim=1)


class SmolLMDecoder(nn.Module):
    """SmolLM-based autoregressive decoder for image captioning"""

    def __init__(self, input_dim: int, vocab_size: int = None, model_name: str = 'HuggingFaceTB/SmolLM-135M'):
        super().__init__()
        self.smollm = AutoModelForCausalLM.from_pretrained(model_name)
        self.smollm_config = self.smollm.config
        
        # Projection from encoder output to SmolLM embedding dimension
        self.embed_proj = nn.Linear(input_dim, self.smollm_config.hidden_size)
        
        # If custom vocab size, resize token embeddings
        if vocab_size is not None and vocab_size != self.smollm_config.vocab_size:
            self.smollm.resize_token_embeddings(vocab_size)

    def forward(self, encoder_output, input_ids, attention_mask=None):

        model_dtype = self.smollm.dtype

        # Get embeddings safely (model-agnostic)
        smollm_embeds = self.smollm.get_input_embeddings()(input_ids)

        # Project encoder output
        projected_encoder = self.embed_proj(encoder_output)
        projected_encoder = projected_encoder.to(model_dtype).unsqueeze(1)

        # Concatenate
        combined_embeds = torch.cat([projected_encoder, smollm_embeds], dim=1)

        # Fix attention mask (IMPORTANT)
        if attention_mask is not None:
            prefix_mask = torch.ones(
                (attention_mask.shape[0], 1),
                device=attention_mask.device
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # Forward through model
        outputs = self.smollm(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )

        return outputs.logits  # (B, seq_len, vocab_size)

    def generate(self, encoder_output, max_length=50, eos_token_id=None, sos_token_id=None):
        """Generate captions using SmolLM"""
        device = encoder_output.device
        batch_size = encoder_output.shape[0]
        
        proj_output = self.embed_proj(encoder_output).to(self.smollm.dtype).unsqueeze(1)  # (B, 1, hidden)
        
        # Start with SOS token if provided
        if sos_token_id is not None:
            input_ids = torch.full((batch_size, 1), sos_token_id, dtype=torch.long, device=device)
        else:
            bos_id = getattr(self.smollm_config, 'bos_token_id', 0)
            input_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        
        generated = []
        for _ in range(max_length):
            smollm_embeds = self.smollm.get_input_embeddings()(input_ids)
            combined_embeds = torch.cat([proj_output, smollm_embeds], dim=1)
            
            outputs = self.smollm(inputs_embeds=combined_embeds)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated.append(next_token)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return torch.cat(generated, dim=1)

    def _get_base_model(self):
        base = self.smollm

        # Unwrap LoRA
        if hasattr(base, "base_model"):
            base = base.base_model

        # Recursively unwrap `.model` until we find embed_tokens
        while hasattr(base, "model") and not hasattr(base, "embed_tokens"):
            base = base.model
        return base


def build_encoder(name: str, output_dim: int, use_attention=False) -> nn.Module:
    if name.startswith('resnet'):
        if use_attention:
            return AttentionSupportive_ResNetEncoder(name, output_dim)
        return ResNetEncoder(name, output_dim)
    if name.startswith('vgg'):
        return VGGEncoder(name, output_dim)
    if name.startswith('vit'):
        return ViTEncoder(name, output_dim)
    if name.startswith('clip'):
        return CLIPEncoder(output_dim=output_dim)
    raise ValueError(f"Unknown encoder: {name}")


class AdditiveAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.W_h = nn.Linear(enc_dim, attn_dim)
        self.W_s = nn.Linear(dec_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: (B, N, enc_dim)
        # decoder_hidden: (B, dec_dim)

        B, N, _ = encoder_outputs.shape

        enc_proj = self.W_h(encoder_outputs)             # (B, N, A)
        dec_proj = self.W_s(decoder_hidden).unsqueeze(1) # (B, 1, A)

        scores = self.v(torch.tanh(enc_proj + dec_proj)).squeeze(-1)  # (B, N)
        attn_weights = torch.softmax(scores, dim=1)                   # (B, N)

        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # (B, enc_dim)

        return context, attn_weights


class CaptioningModel(nn.Module):

    def __init__(self, encoder_name='resnet18', decoder_type='gru',
                 decoder_layers=1, vocab_size=80,
                 embed_dim=512, hidden_dim=512, dropout=0.0, decoder_model_name=None):
        super().__init__()

        self.encoder = build_encoder(
            encoder_name,
            hidden_dim,
            use_attention=(decoder_type == 'gru_attn')
        )
        self.decoder_type = decoder_type
        self.decoder_layers = decoder_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Transformer-based decoders
        if decoder_type == 'gpt2':
            model_name = decoder_model_name or 'gpt2'
            self.decoder = GPT2Decoder(hidden_dim, vocab_size=vocab_size, model_name=model_name)
        elif decoder_type == 't5':
            model_name = decoder_model_name or 't5-base'
            self.decoder = T5Decoder(hidden_dim, vocab_size=vocab_size, model_name=model_name)
        elif decoder_type == 'smollm':
            model_name = decoder_model_name or 'HuggingFaceTB/SmolLM-135M'
            self.decoder = SmolLMDecoder(hidden_dim, vocab_size=vocab_size, model_name=model_name)
        else:
            # CNN-RNN based decoders
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.dropout = nn.Dropout(dropout)

            rnn_cls = nn.GRU if decoder_type == 'gru' else nn.LSTM
            self.decoder = rnn_cls(
                embed_dim, hidden_dim,
                num_layers=decoder_layers,
                batch_first=True,
                dropout=dropout if decoder_layers > 1 else 0.0,
            )
            if decoder_type == 'gru_attn':
                self.attn = AdditiveAttention(hidden_dim, hidden_dim, hidden_dim)
                self.decoder = nn.GRU(
                    embed_dim + hidden_dim,  # input is bigger now
                    hidden_dim,
                    num_layers=decoder_layers,
                    batch_first=True,
                )
            self.proj = nn.Linear(hidden_dim, vocab_size)

    def _init_hidden(self, img):
        feat = self.encoder(img)

        if self.decoder_type == 'gru_attn':
            feat = feat.mean(dim=1)  # (B, D)

        h0 = feat.unsqueeze(0).expand(self.decoder_layers, -1, -1).contiguous()

        if self.decoder_type == 'lstm':
            return (h0, torch.zeros_like(h0))
        return h0

    def forward(self, img, captions, teacher_forcing=True):
        
        # Transformer-based decoder forward pass
        if self.decoder_type in ['gpt2', 'smollm']:
            encoder_output = self.encoder(img)  # (B, hidden_dim)
            if teacher_forcing and captions is not None:
                # Create attention mask (all ones since we're processing full sequence)
                attention_mask = torch.ones(
                    captions[:, :-1].shape,
                    dtype=torch.long,
                    device=captions.device
                )
                logits = self.decoder(encoder_output, captions[:, :-1], attention_mask)
                logits = logits[:, 1:, :]  # remove prefix token output
                return logits.permute(0, 2, 1)
            else:
                return encoder_output
        
        elif self.decoder_type == 't5':
            encoder_output = self.encoder(img)  # (B, hidden_dim)
            if teacher_forcing and captions is not None:
                attention_mask = torch.ones(
                    captions[:, :-1].shape,
                    dtype=torch.long,
                    device=captions.device
                )
                logits = self.decoder(encoder_output, captions[:, :-1], attention_mask)
                return logits.permute(0, 2, 1)  # (B, vocab_size, seq_len)
            else:
                return encoder_output

        elif self.decoder_type == 'gru_attn':
            encoder_outputs = self.encoder(img)  # (B, N, D)

            hidden = encoder_outputs.mean(dim=1).unsqueeze(0).repeat(
                self.decoder_layers, 1, 1
            )

            if teacher_forcing:
                embeddings = self.dropout(self.embed(captions[:, :-1]))
                outputs = []
                h = hidden

                for t in range(embeddings.size(1)):
                    emb_t = embeddings[:, t]

                    context, _ = self.attn(encoder_outputs, h[-1])

                    rnn_input = torch.cat([emb_t, context], dim=1).unsqueeze(1)
                    out, h = self.decoder(rnn_input, h)

                    logits = self.proj(out.squeeze(1))
                    outputs.append(logits)

                return torch.stack(outputs, dim=2)

            else:
                B, T = captions.shape
                current = captions[:, 0]
                outputs = []
                h = hidden

                for _ in range(T - 1):
                    emb = self.dropout(self.embed(current))

                    context, _ = self.attn(encoder_outputs, h[-1])

                    rnn_input = torch.cat([emb, context], dim=1).unsqueeze(1)
                    out, h = self.decoder(rnn_input, h)

                    logit = self.proj(out.squeeze(1))
                    outputs.append(logit)

                    current = logit.argmax(dim=-1)

                return torch.stack(outputs, dim=2)

        # Standard GRU / LSTM path
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

        # Transformer-based decoder generation
        if self.decoder_type in ['gpt2', 'smollm']:
            encoder_output = self.encoder(img)  # (B, hidden_dim)
            gen = self.decoder.generate(encoder_output, max_length=max_len, 
                                       sos_token_id=sos_idx, eos_token_id=eos_idx)
            
            # Pad to max_len if needed
            if gen.shape[1] < max_len:
                padding = torch.zeros(B, max_len - gen.shape[1], 
                                     dtype=torch.long, device=device)
                gen = torch.cat([gen, padding], dim=1)
            return gen[:, :max_len]
        
        elif self.decoder_type == 't5':
            encoder_output = self.encoder(img)  # (B, hidden_dim)
            gen = self.decoder.generate(encoder_output, max_length=max_len,
                                       sos_token_id=sos_idx, eos_token_id=eos_idx)
            
            # Pad to max_len if needed
            if gen.shape[1] < max_len:
                padding = torch.zeros(B, max_len - gen.shape[1],
                                     dtype=torch.long, device=device)
                gen = torch.cat([gen, padding], dim=1)
            return gen[:, :max_len]

        elif self.decoder_type == 'gru_attn':
            encoder_outputs = self.encoder(img)

            hidden = encoder_outputs.mean(dim=1).unsqueeze(0).repeat(
                self.decoder_layers, 1, 1
            )

            current = torch.full((B,), sos_idx, dtype=torch.long, device=device)
            generated = []
            finished = torch.zeros(B, dtype=torch.bool, device=device)

            h = hidden

            for _ in range(max_len):
                emb = self.embed(current)

                context, _ = self.attn(encoder_outputs, h[-1])

                rnn_input = torch.cat([emb, context], dim=1).unsqueeze(1)
                out, h = self.decoder(rnn_input, h)

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
