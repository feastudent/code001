"""
train.py - Script de treinamento do LLM GPT
Autor: Claude
Descrição: Treina o modelo com todas as otimizações modernas
"""

import os
import time
import math
import argparse
from contextlib import nullcontext
import torch
import numpy as np
from tqdm import tqdm

from llm_gpt.config import get_config
from llm_gpt.model import GPT
from llm_gpt.utils import get_tokenizer
from llm_gpt.data import create_dataloaders, prepare_dataset, generate_dummy_data


def get_lr(it, config):
    """
    Calcula learning rate com warmup e decay coseno

    Args:
        it: Iteração atual
        config: Configuração do modelo

    Returns:
        Learning rate para a iteração
    """
    # 1) Warmup linear por warmup_iters passos
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters

    # 2) Se passou do decay, retorna lr mínima
    if it > config.lr_decay_iters:
        return config.min_lr

    # 3) Decay coseno entre warmup e lr_decay_iters
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # de 1 para 0
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, config, device):
    """
    Estima loss em treino e validação

    Args:
        model: Modelo GPT
        train_loader: DataLoader de treino
        val_loader: DataLoader de validação
        config: Configuração
        device: Device (cuda/cpu)

    Returns:
        Dict com losses {'train': x, 'val': y}
    """
    out = {}
    model.eval()

    for split, loader in [('train', train_loader), ('val', val_loader)]:
        if loader is None:
            continue

        losses = []
        for i, (x, y) in enumerate(loader):
            if i >= config.eval_iters:
                break

            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
                logits, loss = model(x, y)
            losses.append(loss.item())

        out[split] = np.mean(losses) if losses else float('inf')

    model.train()
    return out


def train(args):
    """
    Função principal de treinamento

    Args:
        args: Argumentos da linha de comando
    """
    # ========== CONFIGURAÇÃO ==========
    print("=" * 60)
    print("🚀 INICIANDO TREINAMENTO DO LLM GPT")
    print("=" * 60)

    # Carrega configuração
    config = get_config(args.model_size)

    # Override com argumentos da linha de comando
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.max_iters:
        config.max_iters = args.max_iters

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.device = device
    print(f"✓ Device: {device}")

    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memória: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Diretórios
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    # ========== DADOS ==========
    print("\n" + "=" * 60)
    print("📚 PREPARANDO DADOS")
    print("=" * 60)

    # Se não tiver dados, gera dados sintéticos
    if not args.train_data or not os.path.exists(args.train_data):
        print("⚠ Dados de treino não encontrados. Gerando dados sintéticos...")
        args.train_data = generate_dummy_data('data/train.txt', num_chars=500000)
        args.val_data = generate_dummy_data('data/val.txt', num_chars=50000)

    # Tokenizador
    tokenizer = get_tokenizer(args.tokenizer)
    config.vocab_size = tokenizer.vocab_size

    # DataLoaders
    train_loader, val_loader = create_dataloaders(
        train_path=args.train_data,
        val_path=args.val_data,
        tokenizer=tokenizer,
        block_size=config.block_size,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
        in_memory=args.in_memory
    )

    # ========== MODELO ==========
    print("\n" + "=" * 60)
    print("🏗️  CRIANDO MODELO")
    print("=" * 60)

    # Cria ou carrega modelo
    if args.resume and os.path.exists(args.resume):
        print(f"✓ Carregando checkpoint de {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model = GPT(config)
        model.load_state_dict(checkpoint['model'])
        start_iter = checkpoint.get('iter', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    else:
        print(f"✓ Criando novo modelo: {args.model_size}")
        model = GPT(config)
        start_iter = 0
        best_val_loss = float('inf')

    model.to(device)

    # Compila modelo (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print("✓ Compilando modelo com torch.compile (pode demorar)...")
        model = torch.compile(model)

    # ========== OTIMIZADOR ==========
    print("\n" + "=" * 60)
    print("⚙️  CONFIGURANDO OTIMIZADOR")
    print("=" * 60)

    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=device
    )

    if args.resume and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Scaler para mixed precision (acelera treino e economiza memória)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    # ========== TREINAMENTO ==========
    print("\n" + "=" * 60)
    print("🎯 INICIANDO LOOP DE TREINAMENTO")
    print("=" * 60)
    print(f"Configuração:")
    print(f"  Modelo: {args.model_size} ({model.get_num_params()/1e6:.2f}M parâmetros)")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max iterations: {config.max_iters}")
    print(f"  Block size: {config.block_size}")
    print("=" * 60 + "\n")

    # Contexto para autocast (mixed precision)
    ctx = torch.amp.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16) \
          if device == 'cuda' else nullcontext()

    # Loop de treinamento
    iter_num = start_iter
    train_iter = iter(train_loader)
    running_loss = 0.0
    t0 = time.time()

    pbar = tqdm(range(start_iter, config.max_iters), initial=start_iter, total=config.max_iters)

    for iter_num in pbar:
        # ===== Learning rate decay =====
        lr = get_lr(iter_num, config) if config.decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ===== Avaliação periódica =====
        if iter_num % config.eval_interval == 0 and iter_num > 0:
            losses = estimate_loss(model, train_loader, val_loader, config, device)
            print(f"\nIter {iter_num}: train loss {losses['train']:.4f}, "
                  f"val loss {losses.get('val', 0):.4f}")

            # Salva melhor modelo
            if 'val' in losses and losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'iter': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))
                print(f"✓ Melhor modelo salvo (val_loss={best_val_loss:.4f})")

        # ===== Batch de treino =====
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # Forward pass
        with ctx:
            logits, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # Gradient clipping
        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # Update
        scaler.step(optimizer)
        scaler.update()

        # Logging
        running_loss += loss.item()

        if iter_num % 10 == 0:
            lossf = running_loss / 10
            running_loss = 0.0

            # Calcula tokens/segundo
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tokens_per_sec = config.batch_size * config.block_size * 10 / dt

            pbar.set_description(
                f"loss: {lossf:.4f} | lr: {lr:.2e} | {tokens_per_sec:.0f} tok/s"
            )

    # ========== SALVA MODELO FINAL ==========
    print("\n" + "=" * 60)
    print("💾 SALVANDO MODELO FINAL")
    print("=" * 60)

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
        'iter': config.max_iters,
        'best_val_loss': best_val_loss,
    }
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save(checkpoint, final_path)
    print(f"✓ Modelo final salvo em: {final_path}")

    print("\n🎉 TREINAMENTO CONCLUÍDO!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treina modelo LLM GPT')

    # Dados
    parser.add_argument('--train-data', type=str, default=None,
                        help='Caminho para dados de treino (.txt)')
    parser.add_argument('--val-data', type=str, default=None,
                        help='Caminho para dados de validação (.txt)')
    parser.add_argument('--in-memory', action='store_true',
                        help='Carrega dataset em memória (mais rápido para dados pequenos)')

    # Modelo
    parser.add_argument('--model-size', type=str, default='tiny',
                        choices=['tiny', 'small', 'medium', 'large', 'xl'],
                        help='Tamanho do modelo')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                        choices=['gpt2', 'char'],
                        help='Tipo de tokenizador')

    # Treinamento
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Tamanho do batch')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--max-iters', type=int, default=None,
                        help='Número máximo de iterações')

    # Sistema
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='Diretório para salvar checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Caminho para checkpoint para continuar treinamento')
    parser.add_argument('--compile', action='store_true',
                        help='Usar torch.compile (PyTorch 2.0+)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Número de workers para DataLoader')

    args = parser.parse_args()
    train(args)
