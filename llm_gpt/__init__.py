"""
LLM GPT - Modelo de Linguagem do Zero
Autor: Claude
Versão: 1.0.0

Um modelo GPT completo e funcional implementado do zero em PyTorch.
"""

__version__ = '1.0.0'
__author__ = 'Claude'

from .config import get_config, ModelConfig
from .model import GPT
from .utils import get_tokenizer, Tokenizer

__all__ = [
    'GPT',
    'get_config',
    'ModelConfig',
    'get_tokenizer',
    'Tokenizer',
]
