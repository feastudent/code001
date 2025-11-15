"""
gpt.py - Arquitetura do modelo GPT (Generative Pre-trained Transformer)
Autor: Claude
Descrição: Implementação completa do modelo GPT com atenção multi-cabeça
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple
import inspect


class MultiHeadAttention(nn.Module):
    """
    Mecanismo de Atenção Multi-Cabeça (Multi-Head Attention)

    Este é o coração do Transformer. Permite que o modelo
    "preste atenção" a diferentes partes da sequência simultaneamente.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0, \
            "n_embd deve ser divisível por n_heads"

        # Projeções para Query, Key, Value (todas de uma vez por eficiência)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Projeção de saída
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularização
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Flash attention (PyTorch 2.0+) - muito mais rápido!
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("AVISO: usando atenção manual. Atualize PyTorch para 2.0+ para Flash Attention!")
            # Máscara causal (lower triangular) para não "espiar" o futuro
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dimensionality (n_embd)

        # Calcula query, key, values para todas as cabeças de uma vez
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Separa as múltiplas cabeças: (B, T, C) -> (B, n_heads, T, head_size)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # Atenção
        if self.flash:
            # Flash Attention eficiente (GPU otimizado)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # Atenção manual (para compatibilidade)
            # (B, n_heads, T, head_size) @ (B, n_heads, head_size, T) -> (B, n_heads, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, n_heads, T, T) @ (B, n_heads, T, head_size) -> (B, n_heads, T, head_size)

        # Junta todas as cabeças novamente
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Projeção de saída
        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    """
    Rede Feed-Forward (MLP)

    Uma rede neural simples aplicada a cada posição
    independentemente e identicamente.
    """

    def __init__(self, config):
        super().__init__()
        # Expansão: geralmente 4x o tamanho do embedding
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # Função de ativação GELU (mais suave que ReLU)
        self.gelu = nn.GELU()
        # Projeção de volta para o tamanho original
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Bloco Transformer completo

    Combina atenção multi-cabeça e feed-forward com
    conexões residuais e normalização de camada.
    """

    def __init__(self, config):
        super().__init__()
        # Layer Normalization antes da atenção (pre-norm)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        # Layer Normalization antes do MLP
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x):
        # Conexões residuais (+x) são cruciais para treinar redes profundas
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    Modelo GPT completo

    Este é o modelo principal que combina embeddings,
    múltiplos blocos Transformer, e a cabeça de linguagem.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Componentes do modelo
        self.transformer = nn.ModuleDict(dict(
            # Embedding de tokens (palavras/subpalavras)
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Embedding de posição (onde o token está na sequência)
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # Dropout no início
            drop = nn.Dropout(config.dropout),
            # Blocos Transformer empilhados
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
            # Normalização final
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        # Cabeça de linguagem: projeta embeddings de volta para vocabulário
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Compartilha pesos entre embedding e cabeça de linguagem (reduz parâmetros)
        self.transformer.wte.weight = self.lm_head.weight

        # Inicializa todos os pesos
        self.apply(self._init_weights)
        # Aplica escala especial para projeções residuais
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layers))

        print(f"Modelo GPT inicializado com {self.get_num_params()/1e6:.2f}M parâmetros")

    def _init_weights(self, module):
        """Inicialização de pesos (importante para convergência)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        """
        Retorna o número de parâmetros do modelo

        Args:
            non_embedding: Se True, não conta embeddings de posição
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        """
        Forward pass do modelo

        Args:
            idx: Tensor de índices de tokens (B, T)
            targets: Tensor de targets para cálculo de loss (B, T) [opcional]

        Returns:
            logits: Predições (B, T, vocab_size)
            loss: Loss se targets fornecido, None caso contrário
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Sequência de tamanho {t} não pode exceder block_size {self.config.block_size}"

        # Posições [0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Forward do transformer
        tok_emb = self.transformer.wte(idx)  # Token embeddings (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # Position embeddings (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Passa por todos os blocos Transformer
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        # Calcula logits
        if targets is not None:
            # Se estamos treinando, só calcula logits para eficiência
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Se estamos gerando, podemos calcular só o último token
            logits = self.lm_head(x)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Gera novos tokens autogressivamente

        Args:
            idx: Sequência inicial (B, T)
            max_new_tokens: Quantos tokens gerar
            temperature: Controla aleatoriedade (maior = mais aleatório)
            top_k: Se definido, apenas amostra dos top k tokens mais prováveis

        Returns:
            Sequência estendida com novos tokens
        """
        for _ in range(max_new_tokens):
            # Corta contexto se ficar muito longo
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Foca apenas no último passo de tempo
            logits = logits[:, -1, :] / temperature

            # Opcionalmente limita aos top-k tokens
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Aplica softmax para obter probabilidades
            probs = F.softmax(logits, dim=-1)

            # Amostra do distribuição
            idx_next = torch.multinomial(probs, num_samples=1)

            # Adiciona à sequência
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configura otimizador AdamW com decaimento de peso apropriado

        Separa parâmetros que devem ter decaimento de peso (matrizes)
        dos que não devem (biases e layer norms).
        """
        # Separa parâmetros treináveis
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Parâmetros 2D (matrizes de peso) terão decaimento
        # Parâmetros 1D (biases, layer norms) não terão
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"Otimizador: {len(decay_params)} tensores com decaimento ({num_decay_params} parâmetros)")
        print(f"Otimizador: {len(nodecay_params)} tensores sem decaimento ({num_nodecay_params} parâmetros)")

        # Usa fused AdamW se disponível (mais rápido em CUDA)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Usando {'fused' if use_fused else 'padrão'} AdamW")

        return optimizer

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        Carrega modelo pré-treinado do GPT-2 (compatibilidade)

        Permite usar pesos do GPT-2 da OpenAI como inicialização.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}

        print(f"Carregando pesos do {model_type} pré-treinado...")

        # Apenas block_size pode ser menor, mas nunca maior
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copia enquanto garante que todos os parâmetros estão alinhados
        from llm_gpt.config import ModelConfig

        config_args = {
            'gpt2':         dict(n_layers=12, n_heads=12, n_embd=768),
            'gpt2-medium':  dict(n_layers=24, n_heads=16, n_embd=1024),
            'gpt2-large':   dict(n_layers=36, n_heads=20, n_embd=1280),
            'gpt2-xl':      dict(n_layers=48, n_heads=25, n_embd=1600),
        }[model_type]

        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args.update(override_args)

        config = ModelConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()

        # Copia pesos, tratando transposição do Conv1D
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in sd_hf.keys():
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
