"""
quick_start.py - Exemplo rápido de uso do LLM GPT
Autor: Claude
Descrição: Demonstra como usar o modelo programaticamente
"""

import sys
import os

# Adiciona o diretório pai ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from llm_gpt import GPT, get_config, get_tokenizer


def exemplo_criacao_modelo():
    """Exemplo 1: Criar um modelo do zero"""
    print("\n" + "="*60)
    print("📝 EXEMPLO 1: Criando um modelo do zero")
    print("="*60)

    # Cria configuração
    config = get_config('tiny')  # Modelo pequeno para demonstração
    print(f"✓ Configuração criada: {config.n_layers} camadas, {config.n_heads} cabeças")

    # Cria modelo
    model = GPT(config)
    print(f"✓ Modelo criado: {model.get_num_params()/1e6:.2f}M parâmetros")

    # Mostra arquitetura
    print(f"\nArquitetura:")
    print(f"  - Vocabulário: {config.vocab_size}")
    print(f"  - Contexto: {config.block_size} tokens")
    print(f"  - Embedding: {config.n_embd} dimensões")
    print(f"  - Camadas: {config.n_layers}")
    print(f"  - Cabeças de atenção: {config.n_heads}")

    return model


def exemplo_inferencia(model):
    """Exemplo 2: Fazer inferência com o modelo"""
    print("\n" + "="*60)
    print("🔮 EXEMPLO 2: Gerando texto")
    print("="*60)

    # Tokenizador
    tokenizer = get_tokenizer('gpt2')

    # Prompt
    prompt = "Era uma vez"
    print(f"Prompt: '{prompt}'")

    # Codifica
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    print(f"✓ Tokens codificados: {tokens.shape}")

    # Gera
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            tokens,
            max_new_tokens=50,
            temperature=1.0,
            top_k=50
        )

    # Decodifica
    text = tokenizer.decode(generated[0].tolist())
    print(f"\n📝 Texto gerado:\n{text}")


def exemplo_forward_pass(model):
    """Exemplo 3: Forward pass manual"""
    print("\n" + "="*60)
    print("⚡ EXEMPLO 3: Forward pass manual")
    print("="*60)

    # Cria batch de exemplo
    batch_size = 2
    seq_length = 10
    vocab_size = model.config.vocab_size

    # Tokens aleatórios (simulando dados)
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    y = torch.randint(0, vocab_size, (batch_size, seq_length))

    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")

    # Forward pass
    logits, loss = model(x, y)

    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Loss: {loss.item():.4f}")

    # Probabilidades do próximo token
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, k=5)

    print(f"\nTop 5 tokens mais prováveis:")
    for prob, idx in zip(top_k_probs, top_k_indices):
        print(f"  Token {idx.item()}: {prob.item()*100:.2f}%")


def exemplo_salvar_carregar(model):
    """Exemplo 4: Salvar e carregar modelo"""
    print("\n" + "="*60)
    print("💾 EXEMPLO 4: Salvando e carregando modelo")
    print("="*60)

    # Salva
    checkpoint_path = 'examples/exemplo_checkpoint.pt'
    os.makedirs('examples', exist_ok=True)

    checkpoint = {
        'model': model.state_dict(),
        'config': model.config,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Modelo salvo em {checkpoint_path}")

    # Carrega
    loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
    loaded_model = GPT(loaded_checkpoint['config'])
    loaded_model.load_state_dict(loaded_checkpoint['model'])
    print(f"✓ Modelo carregado de {checkpoint_path}")

    # Verifica se são iguais
    original_params = sum(p.numel() for p in model.parameters())
    loaded_params = sum(p.numel() for p in loaded_model.parameters())
    print(f"✓ Parâmetros: original={original_params}, carregado={loaded_params}")


def exemplo_comparacao_tamanhos():
    """Exemplo 5: Compara diferentes tamanhos de modelo"""
    print("\n" + "="*60)
    print("📊 EXEMPLO 5: Comparando tamanhos de modelo")
    print("="*60)

    tamanhos = ['tiny', 'small', 'medium']

    print(f"\n{'Modelo':<10} {'Parâmetros':>15} {'Camadas':>10} {'Embedding':>12}")
    print("-" * 60)

    for tamanho in tamanhos:
        config = get_config(tamanho)
        model = GPT(config)
        params = model.get_num_params()

        print(f"{tamanho:<10} {params/1e6:>12.2f}M {config.n_layers:>10} {config.n_embd:>12}")


def main():
    """Executa todos os exemplos"""
    print("\n" + "="*60)
    print("🚀 LLM GPT - EXEMPLOS DE USO")
    print("="*60)
    print("Este script demonstra como usar o modelo programaticamente")

    # Exemplo 1: Criar modelo
    model = exemplo_criacao_modelo()

    # Exemplo 2: Inferência
    exemplo_inferencia(model)

    # Exemplo 3: Forward pass
    exemplo_forward_pass(model)

    # Exemplo 4: Salvar/Carregar
    exemplo_salvar_carregar(model)

    # Exemplo 5: Comparar tamanhos
    exemplo_comparacao_tamanhos()

    print("\n" + "="*60)
    print("✅ TODOS OS EXEMPLOS EXECUTADOS COM SUCESSO!")
    print("="*60)
    print("\n💡 Próximos passos:")
    print("  1. Treine um modelo: python train.py --model-size tiny")
    print("  2. Gere texto: python generate.py --checkpoint checkpoints/best_model.pt --interactive")
    print("  3. Leia o README.md para mais informações")
    print("\n")


if __name__ == '__main__':
    main()
