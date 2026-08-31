🤖 Modelagem de Machine Learning — Risco de Crédito

3º Entregável — Modelagem, Validação e Interpretação Disciplina: Machine Learning & Modelling — FIAP

RA	Nome
573087	Gustavo Pugas Linczuk
574157	Gustavo Lino
570875	Lucas Lopes Arias
📌 Objetivo

Continuação do 2º entregável (Análise Exploratória de Dados), aplicando a etapa de Modelagem do CRISP-DM ao problema de predição de inadimplência em crédito pessoal. O objetivo é construir e avaliar modelos de classificação capazes de estimar o risco de um cliente não pagar um empréstimo.

🗂️ Escopo deste entregável
Preparação dos dados — scaling (StandardScaler) e encoding (Label Encoding + One-Hot Encoding)
Treinamento de modelos — Regressão Logística, Random Forest e KNN
Validação — holdout estratificado 80/20, com métricas de classificação (Accuracy, Precision, Recall, F1-score, AUC)
Tuning básico de hiperparâmetros — GridSearchCV (C na Regressão Logística, k no KNN, n_estimators/max_depth na Random Forest)
Interpretação — quais variáveis mais impactam o risco de inadimplência (feature importance + coeficientes)
📊 Dataset

Dataset simulado de crédito pessoal (1.500 registros, 11 variáveis), gerado com seed fixa (np.random.seed(42)) para reproduzir exatamente a base tratada no 2º entregável (limpeza de nulos/outliers e encoding). Principais variáveis: idade, renda mensal, valor do empréstimo, prazo, número de dependentes, taxa de utilização de crédito, posse de imóvel, histórico de pagamento, nível de escolaridade e estado civil. Variável-alvo: inadimplente (0/1).

🧠 Modelos treinados
Modelo	Dados	Por que usar
Regressão Logística	Escalados	Baseline interpretável, coeficientes indicam direção/força do impacto de cada variável
Random Forest	Originais	Captura relações não lineares e interações; robusta a outliers, não exige scaling
KNN	Escalados	Modelo baseado em distância/similaridade entre clientes; bom para tuning didático (k)

Regressão Logística e Random Forest usam class_weight='balanced' para compensar o desbalanceamento (~70/30) da classe inadimplente.

🏆 Resultado

O modelo vencedor (por F1-score, com AUC como critério de desempate) é identificado automaticamente dentro do próprio notebook, a partir da tabela de resultados — garantindo que a conclusão sempre reflita os números reais gerados na execução.

🔑 Principais insights

historico_pagamento, renda_mensal e taxa_utilizacao_credito são consistentemente as variáveis mais relevantes para explicar o risco de inadimplência, tanto no modelo linear (Regressão Logística) quanto no não linear (Random Forest).

🚀 Próximos passos sugeridos
Testar técnicas adicionais de balanceamento de classes (ex.: SMOTE)
Avaliar modelos adicionais (Gradient Boosting / XGBoost) e ensembles
Ajustar o threshold de decisão para priorizar Recall
Validar o modelo com dados reais
🛠️ Como executar
bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook 3_entregavel_modelagem_ML_corrigido.ipynb

Rode todas as células em ordem (Run All). Não é necessário nenhum arquivo externo — o dataset é gerado dentro do próprio notebook, com seed fixa para reprodutibilidade.

📁 Estrutura do repositório
.
├── 3_entregavel_modelagem_ML_corrigido.ipynb   # notebook principal
└── README.md
