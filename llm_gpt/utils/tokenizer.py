"""
tokenizer.py - Tokenizador para o modelo LLM
Autor: Claude
Descrição: Wrapper para usar tiktoken (tokenizador do GPT) com fallback para básico
"""

import os
import pickle
from typing import List, Union


class Tokenizer:
    """
    Classe de tokenização que suporta múltiplos backends

    Usa tiktoken (tokenizador oficial do GPT) por padrão,
    com fallback para tokenizador de caracteres simples.
    """

    def __init__(self, tokenizer_type: str = 'gpt2'):
        """
        Inicializa o tokenizador

        Args:
            tokenizer_type: Tipo do tokenizador ('gpt2', 'char', 'custom')
        """
        self.tokenizer_type = tokenizer_type

        if tokenizer_type == 'gpt2':
            self._init_gpt2_tokenizer()
        elif tokenizer_type == 'char':
            self._init_char_tokenizer()
        else:
            raise ValueError(f"Tipo de tokenizador '{tokenizer_type}' não suportado")

    def _init_gpt2_tokenizer(self):
        """Inicializa tokenizador GPT-2 usando tiktoken"""
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("gpt2")
            self.vocab_size = self.tokenizer.n_vocab
            print(f"✓ Tokenizador GPT-2 carregado (vocab_size={self.vocab_size})")
        except ImportError:
            print("⚠ tiktoken não disponível, usando tokenizador de caracteres")
            self._init_char_tokenizer()

    def _init_char_tokenizer(self):
        """Inicializa tokenizador de caracteres simples (para testes)"""
        # Caracteres básicos + especiais
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'-\n")

        # Adiciona caracteres portugueses
        chars_pt = list("áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ")
        chars.extend(chars_pt)

        # Tokens especiais
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.stoi = {ch: i for i, ch in enumerate(special_tokens + chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

        self.pad_token = special_tokens[0]
        self.unk_token = special_tokens[1]
        self.bos_token = special_tokens[2]
        self.eos_token = special_tokens[3]

        print(f"✓ Tokenizador de caracteres inicializado (vocab_size={self.vocab_size})")

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Codifica texto para lista de IDs

        Args:
            text: Texto para codificar
            add_special_tokens: Se True, adiciona tokens especiais (BOS/EOS)

        Returns:
            Lista de IDs de tokens
        """
        if self.tokenizer_type == 'gpt2':
            ids = self.tokenizer.encode(text, allowed_special="all")
        else:
            # Tokenizador de caracteres
            ids = []
            if add_special_tokens:
                ids.append(self.stoi[self.bos_token])

            for char in text:
                ids.append(self.stoi.get(char, self.stoi[self.unk_token]))

            if add_special_tokens:
                ids.append(self.stoi[self.eos_token])

        return ids

    def decode(self, ids: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        """
        Decodifica IDs de volta para texto

        Args:
            ids: Lista de IDs ou lista de listas de IDs

        Returns:
            Texto decodificado ou lista de textos
        """
        # Verifica se é uma lista de listas (batch)
        if ids and isinstance(ids[0], list):
            return [self.decode(seq) for seq in ids]

        if self.tokenizer_type == 'gpt2':
            return self.tokenizer.decode(ids)
        else:
            # Tokenizador de caracteres
            chars = [self.itos.get(i, self.unk_token) for i in ids]
            # Remove tokens especiais
            chars = [c for c in chars if c not in [self.pad_token, self.bos_token, self.eos_token]]
            return ''.join(chars)

    def save(self, path: str):
        """Salva configuração do tokenizador"""
        if self.tokenizer_type == 'char':
            with open(path, 'wb') as f:
                pickle.dump({
                    'type': self.tokenizer_type,
                    'stoi': self.stoi,
                    'itos': self.itos,
                    'vocab_size': self.vocab_size
                }, f)
            print(f"✓ Tokenizador salvo em {path}")

    @classmethod
    def load(cls, path: str):
        """Carrega tokenizador de arquivo"""
        if not os.path.exists(path):
            print(f"⚠ Arquivo {path} não encontrado, criando novo tokenizador")
            return cls()

        with open(path, 'rb') as f:
            data = pickle.load(f)

        tokenizer = cls.__new__(cls)
        tokenizer.tokenizer_type = data['type']

        if data['type'] == 'char':
            tokenizer.stoi = data['stoi']
            tokenizer.itos = data['itos']
            tokenizer.vocab_size = data['vocab_size']

        print(f"✓ Tokenizador carregado de {path}")
        return tokenizer


def train_tokenizer(data_path: str, vocab_size: int = 5000, tokenizer_type: str = 'bpe'):
    """
    Treina um tokenizador personalizado a partir de dados

    Args:
        data_path: Caminho para arquivo de texto
        vocab_size: Tamanho do vocabulário desejado
        tokenizer_type: Tipo do tokenizador ('bpe', 'char')

    Returns:
        Tokenizador treinado
    """
    if tokenizer_type == 'bpe':
        try:
            import sentencepiece as spm

            # Treina modelo SentencePiece (BPE)
            spm.SentencePieceTrainer.train(
                input=data_path,
                model_prefix='tokenizer_custom',
                vocab_size=vocab_size,
                character_coverage=1.0,
                model_type='bpe',
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
            )

            print(f"✓ Tokenizador BPE treinado com vocab_size={vocab_size}")
            print(f"  Modelos salvos: tokenizer_custom.model, tokenizer_custom.vocab")

            # TODO: Implementar wrapper para SentencePiece
            # Por enquanto, retorna tokenizador básico
            return Tokenizer('char')

        except ImportError:
            print("⚠ sentencepiece não disponível, usando tokenizador de caracteres")
            return Tokenizer('char')
    else:
        return Tokenizer('char')


# Funções auxiliares para uso rápido
def get_tokenizer(tokenizer_type: str = 'gpt2') -> Tokenizer:
    """
    Retorna tokenizador pronto para uso

    Args:
        tokenizer_type: 'gpt2' ou 'char'

    Returns:
        Instância de Tokenizer
    """
    return Tokenizer(tokenizer_type)
