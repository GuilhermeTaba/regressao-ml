# Previsão de preços de Airbnb em Nova York

**APS2 — Modelagem, Avaliação e Interpretação**
**Membros:** Arthur Sampaio Bernardes e Guilherme Kenzo Taba Nakamura

📄 **Documentação completa (notebook renderizado):** https://SEU-USUARIO.github.io/NOME-DO-REPO/

## Sobre o projeto

Regressão do preço diário (`price`) de anúncios do dataset [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) (48.895 anúncios, 2019). O pipeline da APS1 (engenharia de atributos, encoding e escalonamento) é reaproveitado, e o alvo é treinado em escala `log1p`.

**Métrica:** MAE em dólares, com validação cruzada KFold (5 folds) apenas no treino. O conjunto de teste (20%) é usado uma única vez, na avaliação final.

## Resultados

| Modelo | MAE validação (USD) |
|---|---|
| Dummy (mediana) | 83,78 |
| Ridge (`alpha=10`) | 60,06 |
| Random Forest ajustada | 56,46 |
| **Random Forest + `dist_centro_km`** | **56,39** |

O melhor modelo obteve **MAE de USD 57,69 no teste**, próximo ao da validação. O erro é baixo nas faixas de preço mais comuns (USD 21,81 em anúncios de até USD 100) e cresce bastante nos anúncios acima de USD 500, que são subestimados.

## Conteúdo do notebook

1. Seleção e implementação dos modelos (Dummy, Ridge, Random Forest), busca de hiperparâmetros com `GridSearchCV`, mudança no pipeline e curvas de aprendizado
2. Avaliação no conjunto de teste: real × predito, resíduos × predito e distribuição dos resíduos
3. Interpretação: coeficientes do Ridge, importância das features da Random Forest e SHAP (global e 3 predições individuais)
4. Discussão crítica: limitações do modelo e dos dados e ações para operacionalização

## Arquivos

| Arquivo | Descrição |
|---|---|
| `REGRESSAO_ML.ipynb` | Notebook com código e saídas |
| `index.html` | Versão renderizada do notebook, publicada no GitHub Pages |

## Como reproduzir

Abra `REGRESSAO_ML.ipynb` no Google Colab e execute todas as células. O dataset é baixado automaticamente pelo `kagglehub`. Principais bibliotecas: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn` e `shap`.