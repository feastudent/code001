"""
test_setup.py - Testa se a instalação está funcionando
Autor: Claude
Descrição: Verifica todas as dependências e componentes
"""

import sys
import importlib


def test_imports():
    """Testa se todos os módulos necessários estão instalados"""
    print("="*60)
    print("🔍 TESTANDO IMPORTAÇÕES")
    print("="*60)

    modules = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'tqdm': 'TQDM',
    }

    optional_modules = {
        'tiktoken': 'tiktoken (tokenizador GPT-2)',
        'transformers': 'transformers (carregar GPT-2 pré-treinado)',
        'sentencepiece': 'sentencepiece (tokenizador BPE)',
    }

    all_ok = True

    # Módulos obrigatórios
    print("\n📦 Módulos obrigatórios:")
    for module, name in modules.items():
        try:
            importlib.import_module(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NÃO INSTALADO!")
            all_ok = False

    # Módulos opcionais
    print("\n📦 Módulos opcionais:")
    for module, name in optional_modules.items():
        try:
            importlib.import_module(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ⚠ {name} - não instalado (opcional)")

    return all_ok


def test_cuda():
    """Testa disponibilidade de CUDA"""
    print("\n" + "="*60)
    print("🖥️  TESTANDO CUDA")
    print("="*60)

    import torch

    if torch.cuda.is_available():
        print(f"  ✓ CUDA disponível")
        print(f"  ✓ Versão CUDA: {torch.version.cuda}")
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ Memória: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print(f"  ⚠ CUDA não disponível - usando CPU")
        print(f"  ℹ️  Treino será mais lento, mas funcional")
        return False


def test_model_creation():
    """Testa criação de modelo"""
    print("\n" + "="*60)
    print("🏗️  TESTANDO CRIAÇÃO DE MODELO")
    print("="*60)

    try:
        from llm_gpt import GPT, get_config

        config = get_config('tiny')
        model = GPT(config)

        params = model.get_num_params()
        print(f"  ✓ Modelo criado com {params/1e6:.2f}M parâmetros")
        return True
    except Exception as e:
        print(f"  ✗ Erro ao criar modelo: {e}")
        return False


def test_tokenizer():
    """Testa tokenizador"""
    print("\n" + "="*60)
    print("📝 TESTANDO TOKENIZADOR")
    print("="*60)

    try:
        from llm_gpt.utils import get_tokenizer

        # Tenta GPT-2
        try:
            tokenizer = get_tokenizer('gpt2')
            print(f"  ✓ Tokenizador GPT-2 carregado")
            print(f"  ✓ Vocabulário: {tokenizer.vocab_size} tokens")
        except:
            print(f"  ⚠ GPT-2 não disponível, usando char")
            tokenizer = get_tokenizer('char')
            print(f"  ✓ Tokenizador char carregado")

        # Testa encode/decode
        text = "Olá, mundo!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        print(f"  ✓ Encode/decode funcionando")
        print(f"    Original: '{text}'")
        print(f"    Tokens: {len(tokens)}")
        print(f"    Decoded: '{decoded}'")

        return True
    except Exception as e:
        print(f"  ✗ Erro no tokenizador: {e}")
        return False


def test_forward_pass():
    """Testa forward pass do modelo"""
    print("\n" + "="*60)
    print("⚡ TESTANDO FORWARD PASS")
    print("="*60)

    try:
        import torch
        from llm_gpt import GPT, get_config

        config = get_config('tiny')
        model = GPT(config)
        model.eval()

        # Cria input dummy
        x = torch.randint(0, config.vocab_size, (2, 10))
        y = torch.randint(0, config.vocab_size, (2, 10))

        # Forward pass
        with torch.no_grad():
            logits, loss = model(x, y)

        print(f"  ✓ Forward pass bem-sucedido")
        print(f"    Input: {x.shape}")
        print(f"    Logits: {logits.shape}")
        print(f"    Loss: {loss.item():.4f}")

        return True
    except Exception as e:
        print(f"  ✗ Erro no forward pass: {e}")
        return False


def test_generation():
    """Testa geração de texto"""
    print("\n" + "="*60)
    print("🎨 TESTANDO GERAÇÃO")
    print("="*60)

    try:
        import torch
        from llm_gpt import GPT, get_config
        from llm_gpt.utils import get_tokenizer

        config = get_config('tiny')
        model = GPT(config)
        model.eval()

        tokenizer = get_tokenizer('char')  # Usa char para garantir

        # Gera texto
        prompt = "Olá"
        tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

        with torch.no_grad():
            generated = model.generate(tokens, max_new_tokens=20)

        text = tokenizer.decode(generated[0].tolist())

        print(f"  ✓ Geração bem-sucedida")
        print(f"    Prompt: '{prompt}'")
        print(f"    Gerado: '{text[:50]}...'")

        return True
    except Exception as e:
        print(f"  ✗ Erro na geração: {e}")
        return False


def main():
    """Executa todos os testes"""
    print("\n" + "="*60)
    print("🧪 TESTE DE INSTALAÇÃO - LLM GPT")
    print("="*60)
    print("Este script verifica se tudo está funcionando corretamente\n")

    results = []

    # Executa testes
    results.append(("Importações", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("Criação de Modelo", test_model_creation()))
    results.append(("Tokenizador", test_tokenizer()))
    results.append(("Forward Pass", test_forward_pass()))
    results.append(("Geração", test_generation()))

    # Resumo
    print("\n" + "="*60)
    print("📊 RESUMO DOS TESTES")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"  {name:<20} {status}")

    print("="*60)
    print(f"\n{passed}/{total} testes passaram")

    if passed == total:
        print("\n🎉 TUDO FUNCIONANDO PERFEITAMENTE!")
        print("\n💡 Próximos passos:")
        print("  1. Execute: python examples/quick_start.py")
        print("  2. Treine um modelo: python train.py --model-size tiny --max-iters 1000")
        print("  3. Gere texto: python generate.py --checkpoint checkpoints/best_model.pt --interactive")
    else:
        print("\n⚠️  ALGUNS TESTES FALHARAM")
        print("Verifique as mensagens de erro acima e instale as dependências necessárias:")
        print("  pip install -r requirements.txt")

    print("")


if __name__ == '__main__':
    main()
