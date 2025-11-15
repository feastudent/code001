"""
finetune_example.py - Exemplo de fine-tuning do modelo
Autor: Claude
Descrição: Demonstra como fazer fine-tuning em dados específicos
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from llm_gpt import GPT, get_config, get_tokenizer
from llm_gpt.data import InMemoryTextDataset


def criar_dados_exemplo():
    """Cria dados de exemplo para fine-tuning"""
    # Textos de exemplo sobre programação em Python
    textos = [
        "Python é uma linguagem de programação de alto nível.",
        "Para declarar uma variável em Python, basta atribuir um valor: x = 10.",
        "Funções em Python são definidas com a palavra-chave def.",
        "Listas em Python são criadas usando colchetes: lista = [1, 2, 3].",
        "O loop for em Python é muito versátil: for item in lista: print(item).",
        "Python possui tipagem dinâmica, não é necessário declarar tipos.",
        "A indentação em Python é obrigatória e define blocos de código.",
        "Dicionários em Python usam chaves: pessoa = {'nome': 'João', 'idade': 30}.",
        "List comprehensions são poderosas: quadrados = [x**2 for x in range(10)].",
        "Python tem uma biblioteca padrão extensa e muito útil.",
    ] * 100  # Repete para ter mais dados

    return "\n".join(textos)


def finetune():
    """Executa fine-tuning do modelo"""
    print("="*60)
    print("🎯 EXEMPLO DE FINE-TUNING")
    print("="*60)

    # ===== CONFIGURAÇÃO =====
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Device: {device}")

    # Cria dados
    print("\n📚 Preparando dados...")
    texto = criar_dados_exemplo()
    print(f"✓ Texto: {len(texto)} caracteres")

    # Tokenizador
    tokenizer = get_tokenizer('gpt2')

    # Dataset
    dataset = InMemoryTextDataset(
        text=texto,
        block_size=128,
        tokenizer=tokenizer
    )

    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
    )

    # ===== MODELO =====
    print("\n🏗️  Criando modelo...")
    config = get_config('tiny')
    config.vocab_size = tokenizer.vocab_size
    model = GPT(config)
    model.to(device)
    print(f"✓ Modelo: {model.get_num_params()/1e6:.2f}M parâmetros")

    # Otimizador
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.1
    )

    # ===== TREINAMENTO =====
    print("\n🎯 Iniciando fine-tuning...")
    num_epochs = 10
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # Forward
            logits, loss = model(x, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    # ===== TESTE =====
    print("\n🔮 Testando modelo após fine-tuning...")
    model.eval()

    prompt = "Python é"
    tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)

    with torch.no_grad():
        generated = model.generate(
            tokens,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50
        )

    text = tokenizer.decode(generated[0].tolist())
    print(f"\n📝 Geração:")
    print(text)

    # ===== SALVAR =====
    print("\n💾 Salvando modelo...")
    os.makedirs('examples', exist_ok=True)
    checkpoint = {
        'model': model.state_dict(),
        'config': config,
    }
    torch.save(checkpoint, 'examples/modelo_finetuned.pt')
    print("✓ Salvo em examples/modelo_finetuned.pt")

    print("\n✅ FINE-TUNING CONCLUÍDO!")


if __name__ == '__main__':
    finetune()
