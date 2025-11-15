#!/bin/bash
# quick_demo.sh - Demonstração rápida do LLM GPT
# Autor: Claude
# Descrição: Treina um modelo pequeno e gera texto

echo "=================================="
echo "🚀 LLM GPT - DEMO RÁPIDO"
echo "=================================="
echo ""

# Verifica se Python está instalado
if ! command -v python &> /dev/null; then
    echo "❌ Python não encontrado!"
    exit 1
fi

# Testa instalação
echo "1️⃣  Testando instalação..."
python test_setup.py

echo ""
echo "=================================="
echo "2️⃣  Treinando modelo pequeno..."
echo "=================================="
echo "Isso deve levar ~5-10 minutos"
echo ""

python train.py \
    --model-size tiny \
    --max-iters 2000 \
    --batch-size 8

echo ""
echo "=================================="
echo "3️⃣  Gerando texto..."
echo "=================================="
echo ""

python generate.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "Era uma vez" \
    --max-tokens 100 \
    --temperature 0.8

echo ""
echo "=================================="
echo "✅ DEMO CONCLUÍDO!"
echo "=================================="
echo ""
echo "💡 Experimente o modo interativo:"
echo "   python generate.py --checkpoint checkpoints/best_model.pt --interactive"
echo ""
