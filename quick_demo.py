"""
quick_demo.py - Demonstração rápida do LLM GPT (compatível com Windows)
Autor: Claude
Descrição: Treina um modelo pequeno e gera texto
"""

import os
import sys
import subprocess


def run_command(cmd, description):
    """Executa comando e mostra progresso"""
    print("\n" + "="*60)
    print(f"🔧 {description}")
    print("="*60)
    print(f"Comando: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, shell=False)

    if result.returncode != 0:
        print(f"\n❌ Erro ao executar: {description}")
        return False

    return True


def main():
    """Executa demo completo"""
    print("="*60)
    print("🚀 LLM GPT - DEMO RÁPIDO")
    print("="*60)
    print("Este script irá:")
    print("  1. Testar a instalação")
    print("  2. Treinar um modelo pequeno (~5-10 min)")
    print("  3. Gerar texto de exemplo")
    print("="*60)

    # Verifica Python
    print(f"\n✓ Python: {sys.version}")

    # 1. Testa instalação
    if not run_command(
        [sys.executable, "test_setup.py"],
        "1️⃣  Testando instalação"
    ):
        print("\n⚠️  Algumas dependências estão faltando.")
        print("Execute: pip install -r requirements.txt")
        return

    # 2. Treina modelo
    input("\n⏸️  Pressione Enter para iniciar o treinamento (ou Ctrl+C para cancelar)...")

    if not run_command(
        [
            sys.executable, "train.py",
            "--model-size", "tiny",
            "--max-iters", "2000",
            "--batch-size", "8"
        ],
        "2️⃣  Treinando modelo pequeno"
    ):
        print("\n❌ Erro no treinamento")
        return

    # 3. Gera texto
    input("\n⏸️  Pressione Enter para gerar texto (ou Ctrl+C para cancelar)...")

    if not run_command(
        [
            sys.executable, "generate.py",
            "--checkpoint", "checkpoints/best_model.pt",
            "--prompt", "Era uma vez",
            "--max-tokens", "100",
            "--temperature", "0.8"
        ],
        "3️⃣  Gerando texto"
    ):
        print("\n❌ Erro na geração")
        return

    # Sucesso!
    print("\n" + "="*60)
    print("✅ DEMO CONCLUÍDO COM SUCESSO!")
    print("="*60)
    print("\n💡 Próximos passos:")
    print("\n1. Modo interativo (chat com o modelo):")
    print(f"   {sys.executable} generate.py --checkpoint checkpoints/best_model.pt --interactive")
    print("\n2. Treinar com seus próprios dados:")
    print(f"   {sys.executable} train.py --train-data seus_dados.txt --model-size small")
    print("\n3. Explorar exemplos:")
    print(f"   {sys.executable} examples/quick_start.py")
    print(f"   {sys.executable} examples/finetune_example.py")
    print("\n4. Ler a documentação:")
    print("   README.md")
    print("")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo cancelado pelo usuário")
        sys.exit(0)
