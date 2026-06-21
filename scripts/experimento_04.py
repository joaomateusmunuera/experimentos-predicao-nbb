import os
import time
import pandas as pd
import numpy as np

from rede_neural import run_model_rede_neural
from vanilla import run_model_vanilla
from svm import run_model_svm 
from random_forest import run_model_rf
from naive_bayes import run_model_naive_bayes
from xgboost_model import run_model_xgboost

from experimentos import save_results_csv

modelos = ['xgboost','svm','rede_neural','random_forest'] 
jogos_media = '15' 

# Com 15 temporadas totais, o script de extração gera exatamente 14 janelas (de janela_01 a janela_14)
janelas = [f"janela_{i:02d}" for i in range(1, 15)]

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
start_time = time.time()

# Loop lista de modelos
for modelo in modelos:
    results = [] 
    acuracias_do_experimento = []
    f1_scores_do_experimento = []

    for janela in janelas:
        # Caminhos apontando para a estrutura gerada pelo novo experimento_04
        treino_path = os.path.join(base_path, "data", "experimento_04", jogos_media, janela, "treino.csv")
        teste_path = os.path.join(base_path, "data", "experimento_04", jogos_media, janela, "teste.csv")

        print(f"Rodando modelo: {modelo.upper()} | Média: {jogos_media} | {janela}")

        if modelo == 'rede_neural':
            accuracy, f1, best_params = run_model_rede_neural(treino_path, teste_path, True)
        elif modelo == 'vanilla':
            accuracy, f1, best_params = run_model_vanilla(treino_path, teste_path)
        elif modelo == 'svm':
            accuracy, f1, best_params = run_model_svm(treino_path, teste_path, True)
        elif modelo == 'random_forest':
            accuracy, f1, best_params = run_model_rf(treino_path, teste_path, True)
        elif modelo == 'naive_bayes':
            accuracy, f1, best_params = run_model_naive_bayes(treino_path, teste_path)
        elif modelo == 'xgboost':
            accuracy, f1, best_params = run_model_xgboost(treino_path, teste_path, True)

        acuracias_do_experimento.append(accuracy)
        f1_scores_do_experimento.append(f1)

        results.append({
            'Janela': janela,
            'Acurácia': f'{accuracy:.2f}',
            'Desvio Padrão Acurácia': '-',
            'F1-Score': f'{f1:.2f}',
            'Desvio Padrão F1-Score': '-',
        })

    # Resultados consolidados da execução do modelo atual ao longo de todas as janelas
    media_acuracia = np.mean(acuracias_do_experimento)
    media_f1_score = np.mean(f1_scores_do_experimento)
    desvio_padrao_acuracia = np.std(acuracias_do_experimento)
    desvio_padrao_f1_score = np.std(f1_scores_do_experimento)

    results.append({
        'Janela': 'Média Geral',
        'Acurácia': f'{media_acuracia:.2f}',
        'Desvio Padrão Acurácia': f'{desvio_padrao_acuracia:.2f}',
        'F1-Score': f'{media_f1_score:.2f}',
        'Desvio Padrão F1-Score': f'{desvio_padrao_f1_score:.2f}',
    })

    # Arquivo final salvo na pasta de resultados do experimento_04
    path = os.path.join(base_path, 'results', 'experimento_04', f'{modelo}_experimento_04_{jogos_media}.csv')
    save_results_csv(path, results)

# Temporizador
end_time = time.time()
print(f"\n--- Experimento 04 concluído em {end_time - start_time:.2f} segundos ---")