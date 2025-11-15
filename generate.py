"""
generate.py - Script de geração de texto com o LLM GPT
Autor: Claude
Descrição: Gera texto usando modelo treinado
"""

import os
import argparse
import torch

from llm_gpt.config import get_config
from llm_gpt.model import GPT
from llm_gpt.utils import get_tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 200,
    device: str = 'cuda'
):
    """
    Gera texto a partir de um prompt

    Args:
        model: Modelo GPT
        tokenizer: Tokenizador
        prompt: Texto inicial
        max_new_tokens: Quantos tokens gerar
        temperature: Controla aleatoriedade (0.0 = determinístico, 1.0+ = criativo)
        top_k: Limita amostragem aos k tokens mais prováveis
        device: Device para inferência

    Returns:
        Texto gerado
    """
    model.eval()

    # Codifica prompt
    if prompt:
        tokens = tokenizer.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    else:
        # Se não há prompt, começa com token vazio
        tokens = torch.zeros((1, 1), dtype=torch.long, device=device)

    print(f"\n{'='*60}")
    print(f"📝 PROMPT: {prompt if prompt else '(vazio)'}")
    print(f"{'='*60}")
    print(f"⚙️  temperature={temperature}, top_k={top_k}, max_tokens={max_new_tokens}")
    print(f"{'='*60}\n")

    # Gera tokens
    with torch.no_grad():
        generated = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )

    # Decodifica
    generated_tokens = generated[0].tolist()
    text = tokenizer.decode(generated_tokens)

    return text


def interactive_mode(model, tokenizer, device='cuda', **gen_kwargs):
    """
    Modo interativo: conversa com o modelo

    Args:
        model: Modelo GPT
        tokenizer: Tokenizador
        device: Device para inferência
        **gen_kwargs: Argumentos para geração (temperature, top_k, etc.)
    """
    print("\n" + "="*60)
    print("💬 MODO INTERATIVO")
    print("="*60)
    print("Digite seu prompt e pressione Enter para gerar texto.")
    print("Comandos especiais:")
    print("  /temp X   - Ajusta temperature (ex: /temp 0.8)")
    print("  /topk X   - Ajusta top_k (ex: /topk 100)")
    print("  /tokens X - Ajusta max_new_tokens (ex: /tokens 200)")
    print("  /quit     - Sair")
    print("="*60 + "\n")

    temperature = gen_kwargs.get('temperature', 0.8)
    top_k = gen_kwargs.get('top_k', 200)
    max_new_tokens = gen_kwargs.get('max_new_tokens', 100)

    while True:
        try:
            prompt = input("\n🤖 Você: ").strip()

            if not prompt:
                continue

            # Comandos especiais
            if prompt.startswith('/'):
                parts = prompt.split()
                cmd = parts[0].lower()

                if cmd == '/quit':
                    print("👋 Até logo!")
                    break
                elif cmd == '/temp' and len(parts) > 1:
                    temperature = float(parts[1])
                    print(f"✓ Temperature ajustada para {temperature}")
                    continue
                elif cmd == '/topk' and len(parts) > 1:
                    top_k = int(parts[1])
                    print(f"✓ Top-k ajustado para {top_k}")
                    continue
                elif cmd == '/tokens' and len(parts) > 1:
                    max_new_tokens = int(parts[1])
                    print(f"✓ Max tokens ajustado para {max_new_tokens}")
                    continue
                else:
                    print("⚠ Comando não reconhecido")
                    continue

            # Gera texto
            print("\n💭 Gerando...")
            text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device
            )

            print(f"\n🤖 GPT: {text}\n")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\n👋 Até logo!")
            break
        except Exception as e:
            print(f"\n❌ Erro: {e}")


def main(args):
    """Função principal"""
    # ========== SETUP ==========
    print("="*60)
    print("🚀 LLM GPT - GERAÇÃO DE TEXTO")
    print("="*60)

    # Device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"✓ Device: {device}")

    # ========== CARREGA MODELO ==========
    print("\n" + "="*60)
    print("🏗️  CARREGANDO MODELO")
    print("="*60)

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint não encontrado: {args.checkpoint}")
        print("💡 Treine um modelo primeiro com train.py")
        return

    # Carrega checkpoint
    print(f"✓ Carregando de {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Recria modelo
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    print(f"✓ Modelo carregado: {model.get_num_params()/1e6:.2f}M parâmetros")

    # ========== TOKENIZADOR ==========
    print("\n" + "="*60)
    print("📚 CARREGANDO TOKENIZADOR")
    print("="*60)

    tokenizer = get_tokenizer(args.tokenizer)
    print(f"✓ Tokenizador: {args.tokenizer} (vocab_size={tokenizer.vocab_size})")

    # ========== GERAÇÃO ==========
    if args.interactive:
        # Modo interativo
        interactive_mode(
            model=model,
            tokenizer=tokenizer,
            device=device,
            temperature=args.temperature,
            top_k=args.top_k,
            max_new_tokens=args.max_tokens
        )
    else:
        # Geração única
        if not args.prompt:
            print("❌ Forneça um prompt com --prompt ou use --interactive")
            return

        text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )

        print("\n" + "="*60)
        print("✨ TEXTO GERADO")
        print("="*60)
        print(text)
        print("="*60 + "\n")

        # Salva se solicitado
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"✓ Texto salvo em {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gera texto com modelo LLM GPT')

    # Modelo
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Caminho para checkpoint do modelo (.pt)')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                        choices=['gpt2', 'char'],
                        help='Tipo de tokenizador (deve ser o mesmo usado no treino)')

    # Geração
    parser.add_argument('--prompt', type=str, default=None,
                        help='Prompt inicial para geração')
    parser.add_argument('--max-tokens', type=int, default=100,
                        help='Número máximo de tokens a gerar')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Temperature (0.0=determinístico, 1.0+=criativo)')
    parser.add_argument('--top-k', type=int, default=200,
                        help='Top-k sampling (None=desabilitado)')

    # Modo
    parser.add_argument('--interactive', action='store_true',
                        help='Modo interativo (chat)')
    parser.add_argument('--output', type=str, default=None,
                        help='Salvar texto gerado em arquivo')

    # Sistema
    parser.add_argument('--cpu', action='store_true',
                        help='Forçar uso de CPU (mesmo com GPU disponível)')

    args = parser.parse_args()
    main(args)
