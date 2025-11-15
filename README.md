# 🚀 LLM GPT do Zero - Modelo de Linguagem Completo

Um modelo de linguagem tipo GPT-3 implementado completamente do zero em PyTorch, com código otimizado, bem comentado e fácil de entender em **Português BR**.

## 🌟 Características

- ✅ **Arquitetura Transformer completa** com atenção multi-cabeça
- ✅ **Código 100% em português** com comentários detalhados
- ✅ **Otimizado para performance** com Flash Attention, mixed precision, gradient clipping
- ✅ **Fácil de treinar** com suporte a múltiplos tamanhos de modelo
- ✅ **Fácil de usar** com scripts simples para treino e geração
- ✅ **Tokenizador GPT-2** integrado (compatível com 50K vocabulário)
- ✅ **Modo interativo** para conversar com o modelo
- ✅ **Compatível com GPT-2 pré-treinado** da OpenAI

## 📋 Requisitos

```bash
Python 3.8+
PyTorch 2.0+
CUDA (opcional, mas recomendado para treino)
```

## 🔧 Instalação

```bash
# Clone o repositório
git clone <seu-repo>
cd code001

# Instale as dependências
pip install -r requirements.txt
```

## 🚀 Início Rápido

### 1️⃣ Treinamento Rápido (com dados sintéticos)

```bash
# Treina um modelo pequeno (10M parâmetros) com dados sintéticos
python train.py --model-size tiny --max-iters 5000

# O modelo será salvo em checkpoints/best_model.pt
```

### 2️⃣ Geração de Texto

```bash
# Modo interativo (chat)
python generate.py --checkpoint checkpoints/best_model.pt --interactive

# Geração única com prompt
python generate.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "Era uma vez" \
    --max-tokens 200 \
    --temperature 0.8
```

## 📚 Treinamento Completo

### Preparar seus próprios dados

```python
# Prepare seus dados de texto
from llm_gpt.data import prepare_dataset

train_path, val_path = prepare_dataset(
    raw_text_path='seus_dados.txt',
    output_dir='data',
    train_split=0.9
)
```

### Treinar modelo customizado

```bash
# Modelo pequeno (124M parâmetros - similar ao GPT-2 small)
python train.py \
    --model-size small \
    --train-data data/train.txt \
    --val-data data/val.txt \
    --batch-size 8 \
    --max-iters 100000 \
    --output-dir checkpoints

# Modelo médio (350M parâmetros)
python train.py \
    --model-size medium \
    --train-data data/train.txt \
    --val-data data/val.txt \
    --batch-size 4 \
    --max-iters 100000

# Com compilação (PyTorch 2.0+, muito mais rápido!)
python train.py \
    --model-size small \
    --compile \
    --train-data data/train.txt
```

## 🎯 Tamanhos de Modelo Disponíveis

| Modelo | Parâmetros | Camadas | Cabeças | Embedding | Uso Recomendado |
|--------|-----------|---------|---------|-----------|-----------------|
| `tiny` | ~10M | 4 | 4 | 256 | Testes rápidos |
| `small` | ~124M | 12 | 12 | 768 | Projetos pequenos, GPT-2 small |
| `medium` | ~350M | 24 | 16 | 1024 | Projetos médios, GPT-2 medium |
| `large` | ~774M | 36 | 20 | 1280 | Projetos grandes, GPT-2 large |
| `xl` | ~1.5B | 48 | 25 | 1600 | Máxima qualidade, GPT-2 XL |

## 💡 Exemplos de Uso

### Modo Programático

```python
import torch
from llm_gpt import GPT, get_config, get_tokenizer

# Carrega modelo treinado
checkpoint = torch.load('checkpoints/best_model.pt')
config = checkpoint['config']
model = GPT(config)
model.load_state_dict(checkpoint['model'])
model.eval()

# Tokenizador
tokenizer = get_tokenizer('gpt2')

# Gera texto
prompt = "Era uma vez"
tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

with torch.no_grad():
    generated = model.generate(
        tokens,
        max_new_tokens=100,
        temperature=0.8,
        top_k=200
    )

text = tokenizer.decode(generated[0].tolist())
print(text)
```

### Criar Modelo do Zero

```python
from llm_gpt import GPT, get_config

# Cria configuração customizada
config = get_config('small')
config.vocab_size = 50257
config.block_size = 1024

# Cria modelo
model = GPT(config)
print(f"Modelo com {model.get_num_params()/1e6:.2f}M parâmetros")

# Treina...
```

### Carregar Pesos do GPT-2 Pré-treinado

```python
from llm_gpt.model import GPT

# Carrega modelo pré-treinado da OpenAI
model = GPT.from_pretrained('gpt2')  # ou 'gpt2-medium', 'gpt2-large', 'gpt2-xl'

# Agora você pode fazer fine-tuning nos seus dados!
```

## 🏗️ Arquitetura

```
GPT
├── Token Embedding (vocab_size → n_embd)
├── Position Embedding (block_size → n_embd)
├── Transformer Blocks (×n_layers)
│   ├── Layer Norm
│   ├── Multi-Head Attention
│   │   ├── Query, Key, Value Projections
│   │   ├── Scaled Dot-Product Attention
│   │   └── Output Projection
│   ├── Layer Norm
│   └── Feed-Forward Network
│       ├── Linear (n_embd → 4×n_embd)
│       ├── GELU Activation
│       └── Linear (4×n_embd → n_embd)
└── Language Model Head (n_embd → vocab_size)
```

## ⚡ Otimizações Implementadas

- **Flash Attention**: Atenção GPU-otimizada (PyTorch 2.0+)
- **Mixed Precision Training**: Treino em bfloat16 (mais rápido, menos memória)
- **Gradient Clipping**: Previne explosão de gradientes
- **Weight Decay**: Regularização L2 adaptativa
- **Learning Rate Warmup**: Aquecimento linear + decay coseno
- **Fused AdamW**: Otimizador otimizado para CUDA
- **Torch Compile**: Compilação JIT para 2x+ speedup
- **Gradient Accumulation**: Simula batches maiores

## 📊 Parâmetros de Geração

### Temperature (0.0 - 2.0+)
- **0.0-0.3**: Determinístico, repetitivo
- **0.5-0.8**: Balanceado (recomendado)
- **0.9-1.5**: Criativo, variado
- **1.5+**: Muito aleatório

### Top-K (10-500)
- **10-50**: Conservador, coerente
- **100-200**: Balanceado (recomendado)
- **200-500**: Mais diversidade

## 🎓 Como Funciona?

### 1. Atenção Multi-Cabeça
Permite que o modelo "preste atenção" a diferentes partes do texto simultaneamente:

```python
# Simplificado
Q, K, V = input.split(3)  # Query, Key, Value
attention_scores = Q @ K.T / sqrt(d_k)
attention_weights = softmax(attention_scores)
output = attention_weights @ V
```

### 2. Treinamento Autoregressivo
O modelo aprende a prever o próximo token:

```
Input:  [Era, uma, vez, um]
Target: [uma, vez, um, gato]
```

### 3. Geração
Durante geração, amostramos do modelo iterativamente:

```
1. Começa com prompt: "Era uma vez"
2. Modelo prevê próximo token: "um"
3. Adiciona à sequência: "Era uma vez um"
4. Repete até max_tokens
```

## 🔍 Estrutura do Projeto

```
code001/
├── llm_gpt/                 # Pacote principal
│   ├── __init__.py
│   ├── config.py           # Configurações do modelo
│   ├── model/              # Arquitetura do modelo
│   │   ├── __init__.py
│   │   └── gpt.py         # Modelo GPT completo
│   ├── data/               # Datasets e dataloaders
│   │   ├── __init__.py
│   │   └── dataset.py
│   └── utils/              # Utilitários
│       ├── __init__.py
│       └── tokenizer.py   # Tokenizador
├── train.py                # Script de treinamento
├── generate.py             # Script de geração
├── requirements.txt        # Dependências
└── README.md              # Este arquivo
```

## 💻 Requisitos de Hardware

### Treinamento

| Modelo | GPU VRAM | RAM | Tempo (1000 iter) |
|--------|----------|-----|-------------------|
| tiny | 2GB | 4GB | ~5 min |
| small | 8GB | 16GB | ~20 min |
| medium | 16GB | 32GB | ~1 hora |
| large | 24GB | 64GB | ~2 horas |
| xl | 40GB+ | 128GB | ~4 horas |

*Tempos aproximados em GPU A100. Use `--batch-size` menor se ficar sem memória.

### Inferência

- **CPU**: Funcional para todos os tamanhos (mais lento)
- **GPU**: 2GB+ VRAM recomendado

## 🐛 Troubleshooting

### Out of Memory (OOM)

```bash
# Reduza batch size
python train.py --batch-size 2

# Use gradient accumulation (simula batch maior)
# TODO: Adicionar suporte

# Use modelo menor
python train.py --model-size tiny
```

### Treino muito lento

```bash
# Use compilação (PyTorch 2.0+)
python train.py --compile

# Verifique se está usando GPU
python -c "import torch; print(torch.cuda.is_available())"

# Use mixed precision (automático em CUDA)
```

### Loss não diminui

- Verifique se seus dados estão corretos
- Tente learning rate menor: `--learning-rate 1e-4`
- Aumente warmup: edite `config.py`
- Use modelo maior se dados forem complexos

## 📖 Recursos para Aprender Mais

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Paper original do Transformer
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visualização excelente
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Implementação minimalista (inspiração)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se livre para abrir issues ou pull requests.

## 📝 Licença

MIT License - Sinta-se livre para usar em projetos pessoais e comerciais.

## 🙏 Agradecimentos

- Andrej Karpathy pelo [nanoGPT](https://github.com/karpathy/nanoGPT)
- OpenAI pelo GPT-2 e arquitetura
- Comunidade PyTorch

## 📞 Suporte

Se tiver dúvidas ou problemas:
1. Verifique este README
2. Abra uma issue no GitHub
3. Leia os comentários no código (estão em português!)

---

**Feito com ❤️ em Python e PyTorch**

🚀 **Bom treinamento!**
