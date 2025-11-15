"""
config.py - Configurações do modelo LLM GPT
Autor: Claude
Descrição: Define os hiperparâmetros e configurações do modelo
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """
    Configuração do modelo GPT

    Parâmetros inspirados no GPT-3, mas escaláveis para diferentes tamanhos
    """
    # Arquitetura do modelo
    vocab_size: int = 50257  # Tamanho do vocabulário (mesmo do GPT-2)
    n_layers: int = 12  # Número de camadas do transformer
    n_heads: int = 12  # Número de cabeças de atenção
    n_embd: int = 768  # Dimensão dos embeddings
    block_size: int = 1024  # Tamanho máximo do contexto (tokens)

    # Regularização
    dropout: float = 0.1  # Taxa de dropout
    bias: bool = True  # Usar bias nas camadas lineares

    # Treinamento
    batch_size: int = 8  # Tamanho do batch
    learning_rate: float = 3e-4  # Taxa de aprendizado
    max_iters: int = 100000  # Número máximo de iterações
    weight_decay: float = 0.1  # Decaimento de peso
    beta1: float = 0.9  # Beta1 do AdamW
    beta2: float = 0.95  # Beta2 do AdamW
    grad_clip: float = 1.0  # Gradient clipping

    # Learning rate decay
    decay_lr: bool = True  # Aplicar decay na learning rate
    warmup_iters: int = 2000  # Iterações de warmup
    lr_decay_iters: int = 100000  # Iterações para decay
    min_lr: float = 3e-5  # Learning rate mínima

    # Avaliação
    eval_interval: int = 500  # Intervalo de avaliação
    eval_iters: int = 200  # Iterações de avaliação

    # Sistema
    device: str = 'cuda'  # 'cuda' ou 'cpu'
    compile: bool = False  # Usar torch.compile (PyTorch 2.0+)


# Configurações pré-definidas para diferentes tamanhos de modelo
@dataclass
class GPTSmallConfig(ModelConfig):
    """GPT Small - ~124M parâmetros (similar ao GPT-2 small)"""
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768


@dataclass
class GPTMediumConfig(ModelConfig):
    """GPT Medium - ~350M parâmetros (similar ao GPT-2 medium)"""
    n_layers: int = 24
    n_heads: int = 16
    n_embd: int = 1024


@dataclass
class GPTLargeConfig(ModelConfig):
    """GPT Large - ~774M parâmetros (similar ao GPT-2 large)"""
    n_layers: int = 36
    n_heads: int = 20
    n_embd: int = 1280


@dataclass
class GPTXLConfig(ModelConfig):
    """GPT XL - ~1.5B parâmetros (similar ao GPT-2 XL)"""
    n_layers: int = 48
    n_heads: int = 25
    n_embd: int = 1600
    block_size: int = 1024


@dataclass
class GPTTinyConfig(ModelConfig):
    """GPT Tiny - ~10M parâmetros (para testes rápidos)"""
    n_layers: int = 4
    n_heads: int = 4
    n_embd: int = 256
    block_size: int = 512
    batch_size: int = 16


def get_config(model_type: str = 'small') -> ModelConfig:
    """
    Retorna a configuração do modelo baseado no tipo

    Args:
        model_type: 'tiny', 'small', 'medium', 'large', 'xl'

    Returns:
        ModelConfig correspondente
    """
    configs = {
        'tiny': GPTTinyConfig(),
        'small': GPTSmallConfig(),
        'medium': GPTMediumConfig(),
        'large': GPTLargeConfig(),
        'xl': GPTXLConfig(),
    }

    if model_type not in configs:
        raise ValueError(f"Tipo de modelo '{model_type}' não reconhecido. "
                        f"Use: {list(configs.keys())}")

    return configs[model_type]
