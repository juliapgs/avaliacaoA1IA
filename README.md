# Atividade de Rede Neural Multicamadas (MLP) - Problema XOR

Este repositório contém os códigos, gráficos e apresentação relacionados à atividade de Redes Neurais Artificiais (MLP), desenvolvida como parte da disciplina de Inteligência Artificial do curso de Sistemas de Informação da UNITINS.

## 📌 Problema Apresentado

O **XOR (OU Exclusivo)** é um problema clássico da lógica booleana onde:

- A saída é `1` (verdadeiro) **apenas quando as entradas são diferentes**
- A saída é `0` (falso) **quando as entradas são iguais**

A principal dificuldade do problema XOR é que ele **não é linearmente separável**, ou seja, **não é possível resolvê-lo com uma simples linha reta**.

## 🧠 Rede Neural Multicamadas (MLP)

A **Rede Neural Multicamadas (MLP)** é composta por:

- **Camada de entrada**: recebe os dados
- **Camada(s) oculta(s)**: possui neurônios com funções de ativação não-lineares
- **Camada de saída**: retorna o resultado final da rede

A **camada oculta** é essencial para resolver o problema XOR, pois permite que a rede aprenda transformações **não lineares**, construindo combinações que resolvem o problema corretamente.

## 🎯 Objetivo da Atividade

Testar diferentes configurações da rede neural variando os principais parâmetros e **analisar como essas variações influenciam no aprendizado da rede**.

## ⚙️ Parâmetros Testados

Foram testadas **5 combinações** de configurações, com variações nos seguintes parâmetros:

- **Número de neurônios na camada oculta**: 2, 3, 4, 5 e 6  
- **Taxas de aprendizado**: 0.01, 0.05, 0.1, 0.2 e 0.5  
- **Número de épocas de treino**: de 2000 a 10000  

## 📹 O que é mostrado no vídeo

- Explicação do código utilizado (`mlp_simples.py`)
- Lógica do problema XOR
- Execução prática dos testes
- Tabela com os resultados de **acurácia** e **erro quadrático médio**

