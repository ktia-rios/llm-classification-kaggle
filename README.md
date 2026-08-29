# LLM Classification Finetuning (Kaggle Chatbot Arena)

Este projeto implementa e compara abordagens para predição de preferências humanas entre respostas geradas por Large Language Models (LLMs), fundamentação direta de Reward Modeling (RLHF).

## Tabela de Experimentos & Benchmarks

| Modelo | Estratégia / Features | CV Log Loss (5-Fold) | Observações |
| :--- | :--- | :--- | :--- |
| **Baseline Ingênuo** | Probabilidades a priori fixas | **1.09723** | Referência mínima (média histórica) |
| **TF-IDF + Logistic Regression** | n-grams (1,2), max_features=25k | **1.12491** | Bag-of-words não captura semântica e gera overconfidence |
| **DeBERTa-v3 / LLM Backbone** | Fine-tuning com Cross-Encoder | *A calcular* | Próxima etapa |

## Principais Insights da EDA
- **Distribuição de Classes:** Classes balanceadas (A: 34.91%, B: 34.19%, Empate: 30.90%).
- **Length Bias:** Respostas vencedoras têm em média ~250 a 270 caracteres a mais que as perdedoras, indicando forte correlação entre verbosidade e preferência humana.

### Modelo Base: DeBERTa-v3 (Cross-Encoder)
- **Modelo Base:** `microsoft/deberta-v3-small`
- **Tamanho Máximo de Sequência:** 512 tokens
- **Otimizador:** AdamW (`lr=1.5e-5`, `weight_decay=0.01`, `warmup_steps=300`)
- **Épocas:** 2
- **Resultado de Validação (Log Loss):** 1.0968

## Resultados e Experimentos

| Abordagem | Arquitetura / Modelo | Validação (Log Loss) | Status |
| :--- | :--- | :--- | :--- |
| **Baseline 0** | Distribuição Uniforme (1/3) | 1.0972 | Concluído |
| **Baseline 1** | TF-IDF + Regressão Logística | 1.1249 | Concluído |
| **Fine-Tuning NLP** | **DeBERTa-v3-small (Cross-Encoder)** | **1.0968** | **Melhor Modelo** |

### Configuração do DeBERTa-v3
- **Sequência Máxima:** 512 tokens
- **Estratégia:** Cross-Encoder (`[CLS] Prompt [SEP] Resposta A [SEP] Resposta B [SEP]`)
- **Otimizador:** AdamW (`lr=1.5e-5`, `weight_decay=0.01`, `warmup_steps=300`)
- **Épocas:** 2 (Batch size efetivo: 16 via Gradient Accumulation)
- **Hardware:** GPU Tesla T4 (Kaggle)