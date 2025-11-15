"""
dataset.py - Dataset e DataLoader para treinamento do LLM
Autor: Claude
Descrição: Gerencia carregamento e processamento de dados de texto
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List
import pickle


class TextDataset(Dataset):
    """
    Dataset para dados de texto tokenizados

    Carrega texto, tokeniza e cria janelas de contexto
    para treinamento autoregressivo.
    """

    def __init__(self, data_path: str, block_size: int, tokenizer):
        """
        Args:
            data_path: Caminho para arquivo .txt ou .bin (tokens pré-processados)
            block_size: Tamanho da janela de contexto
            tokenizer: Instância do tokenizador
        """
        self.block_size = block_size
        self.tokenizer = tokenizer

        # Verifica se já existe versão tokenizada
        cache_path = data_path.replace('.txt', '_tokenized.bin')

        if os.path.exists(cache_path):
            print(f"✓ Carregando dados tokenizados de {cache_path}")
            self.data = np.memmap(cache_path, dtype=np.uint16, mode='r')
        elif data_path.endswith('.bin'):
            print(f"✓ Carregando dados binários de {data_path}")
            self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        else:
            print(f"✓ Tokenizando {data_path}...")
            with open(data_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Tokeniza
            tokens = tokenizer.encode(text)
            tokens = np.array(tokens, dtype=np.uint16)

            # Salva versão tokenizada para reutilizar
            print(f"✓ Salvando dados tokenizados em {cache_path}")
            tokens.tofile(cache_path)

            self.data = tokens

        print(f"✓ Dataset carregado: {len(self.data)} tokens, "
              f"{len(self)} exemplos de treinamento")

    def __len__(self):
        """Número de exemplos de treinamento"""
        # Cada exemplo tem block_size+1 tokens (input + target)
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        """
        Retorna um exemplo de treinamento

        Returns:
            x: Sequência de entrada (block_size tokens)
            y: Sequência alvo (mesmos tokens deslocados em 1)
        """
        # Pega block_size + 1 tokens
        chunk = self.data[idx:idx + self.block_size + 1]
        chunk = torch.from_numpy(chunk.astype(np.int64))

        x = chunk[:-1]  # Entrada: primeiros block_size tokens
        y = chunk[1:]   # Target: mesmos tokens deslocados

        return x, y


class InMemoryTextDataset(Dataset):
    """
    Dataset que mantém todos os dados em memória (para datasets pequenos)

    Mais rápido que TextDataset para dados que cabem na RAM.
    """

    def __init__(self, text: str, block_size: int, tokenizer):
        """
        Args:
            text: Texto completo para treinar
            block_size: Tamanho da janela de contexto
            tokenizer: Instância do tokenizador
        """
        self.block_size = block_size
        self.tokenizer = tokenizer

        print("✓ Tokenizando texto...")
        tokens = tokenizer.encode(text)
        self.data = torch.tensor(tokens, dtype=torch.long)

        print(f"✓ Dataset em memória: {len(self.data)} tokens, "
              f"{len(self)} exemplos")

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def create_dataloaders(
    train_path: str,
    val_path: Optional[str],
    tokenizer,
    block_size: int,
    batch_size: int,
    num_workers: int = 0,
    in_memory: bool = False
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Cria DataLoaders para treinamento e validação

    Args:
        train_path: Caminho para dados de treino
        val_path: Caminho para dados de validação (opcional)
        tokenizer: Tokenizador a usar
        block_size: Tamanho da janela de contexto
        batch_size: Tamanho do batch
        num_workers: Número de workers para carregamento paralelo
        in_memory: Se True, usa InMemoryTextDataset

    Returns:
        train_loader, val_loader (val_loader pode ser None)
    """
    # Dataset de treino
    if in_memory:
        with open(train_path, 'r', encoding='utf-8') as f:
            train_text = f.read()
        train_dataset = InMemoryTextDataset(train_text, block_size, tokenizer)
    else:
        train_dataset = TextDataset(train_path, block_size, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Acelera transferência CPU->GPU
    )

    # Dataset de validação (se fornecido)
    val_loader = None
    if val_path and os.path.exists(val_path):
        if in_memory:
            with open(val_path, 'r', encoding='utf-8') as f:
                val_text = f.read()
            val_dataset = InMemoryTextDataset(val_text, block_size, tokenizer)
        else:
            val_dataset = TextDataset(val_path, block_size, tokenizer)

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader


def prepare_dataset(
    raw_text_path: str,
    output_dir: str = 'data',
    train_split: float = 0.9,
    tokenizer=None
):
    """
    Prepara dataset a partir de arquivo de texto bruto

    Divide em treino/validação e tokeniza.

    Args:
        raw_text_path: Caminho para arquivo .txt bruto
        output_dir: Diretório para salvar dados processados
        train_split: Proporção para treinamento (0-1)
        tokenizer: Tokenizador (se None, usa GPT-2)

    Returns:
        Caminhos para arquivos de treino e validação
    """
    os.makedirs(output_dir, exist_ok=True)

    # Carrega texto
    print(f"✓ Lendo {raw_text_path}...")
    with open(raw_text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"✓ Texto carregado: {len(text)} caracteres")

    # Divide em treino/validação
    split_idx = int(len(text) * train_split)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    # Salva versões divididas
    train_path = os.path.join(output_dir, 'train.txt')
    val_path = os.path.join(output_dir, 'val.txt')

    with open(train_path, 'w', encoding='utf-8') as f:
        f.write(train_text)

    with open(val_path, 'w', encoding='utf-8') as f:
        f.write(val_text)

    print(f"✓ Dataset salvo:")
    print(f"  Treino: {train_path} ({len(train_text)} caracteres)")
    print(f"  Validação: {val_path} ({len(val_text)} caracteres)")

    return train_path, val_path


# Exemplo de gerador de dados sintéticos para testes
def generate_dummy_data(output_path: str = 'data/dummy.txt', num_chars: int = 100000):
    """
    Gera dados sintéticos para testes rápidos

    Args:
        output_path: Onde salvar o arquivo
        num_chars: Quantos caracteres gerar
    """
    import random

    # Palavras em português para dados sintéticos
    palavras = [
        'o', 'a', 'de', 'que', 'e', 'do', 'da', 'em', 'um', 'para',
        'é', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais',
        'casa', 'dia', 'tempo', 'vida', 'mundo', 'ano', 'vez', 'país',
        'pessoa', 'coisa', 'lugar', 'parte', 'caso', 'grupo', 'governo',
        'ser', 'estar', 'ter', 'haver', 'fazer', 'dar', 'poder', 'dizer',
        'ir', 'ver', 'saber', 'querer', 'ficar', 'passar', 'vir',
    ]

    text = []
    current_length = 0

    while current_length < num_chars:
        # Gera uma frase
        sentence_length = random.randint(5, 15)
        sentence = ' '.join(random.choices(palavras, k=sentence_length))
        sentence = sentence.capitalize() + '. '

        text.append(sentence)
        current_length += len(sentence)

    text = ''.join(text)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"✓ Dados sintéticos gerados: {output_path} ({len(text)} caracteres)")
    return output_path
