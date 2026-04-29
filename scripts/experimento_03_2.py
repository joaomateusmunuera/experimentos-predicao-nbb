import os
import time
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

# Seus imports de modelos
from rede_neural import run_model_rede_neural
from svm import run_model_svm 
from random_forest import run_model_rf
from naive_bayes import run_model_naive_bayes
from vanilla import run_model_vanilla
from xgboost_model import run_model_xgboost

# Configurações
modelos = ['vanilla','svm'] 
temporadas = [
    "2008-2009", "2009-2010", "2011-2012", "2012-2013",
    "2013-2014", "2014-2015", "2015-2016", "2016-2017",
    "2018-2019", "2019-2020", "2020-2021", "2021-2022", 
    "2022-2023", "2023-2024", "2024-2025"
]

K_espacamento = 1
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
start_time = time.time()

for modelo in modelos:
    for temporada in temporadas:
        temporada_dir = os.path.join(base_path, "data", "experimento_03", temporada)
        
        if not os.path.exists(temporada_dir):
            continue

        janelas = [d for d in os.listdir(temporada_dir) if os.path.isdir(os.path.join(temporada_dir, d))]
        janelas.sort(key=lambda x: int(x.split('-')[0]))

        print(f"\n>> Processando: {modelo.upper()} | Temporada {temporada}")

        num_sequencial = 1
        acertos_acumulados = 0
        total_jogos_testados = 0
        evolucao_metricas = []
        
        # Inicializa listas de F1 APENAS se for vanilla para economizar memória
        y_true_acumulado = [] if modelo == 'vanilla' else None
        y_pred_acumulado = [] if modelo == 'vanilla' else None

        for i, pasta_janela in enumerate(janelas):
            if i % K_espacamento != 0:
                num_sequencial += 1
                continue

            treino_path = os.path.join(temporada_dir, pasta_janela, f"treino_{num_sequencial}.csv")
            teste_path = os.path.join(temporada_dir, pasta_janela, f"teste_{num_sequencial}.csv")

            if os.path.exists(treino_path) and os.path.exists(teste_path):
                try:
                    # Carrega y_test apenas se for Vanilla (necessário para F1)
                    y_test_val = None
                    if modelo == 'vanilla':
                        from experimentos import read_dados
                        _, _, _, y_test_val = read_dados(treino_path, teste_path)
                    
                    accuracy = None
                    
                    # Chamada dos modelos (Grid Search ativado para os complexos)
                    if modelo == 'naive_bayes':
                        accuracy, _, _ = run_model_naive_bayes(treino_path, teste_path)
                    elif modelo == 'svm':
                        accuracy, _, _ = run_model_svm(treino_path, teste_path, True)
                    elif modelo == 'vanilla':
                        accuracy, _, _ = run_model_vanilla(treino_path, teste_path)
                    elif modelo == 'random_forest':
                        accuracy, _, _ = run_model_rf(treino_path, teste_path, True)
                    elif modelo == 'xgboost':
                        accuracy, _, _ = run_model_xgboost(treino_path, teste_path, True)
                    elif modelo == 'rede_neural':
                        accuracy, _, _ = run_model_rede_neural(treino_path, teste_path, True)

                    if accuracy is not None:
                        total_jogos_testados += 1
                        if accuracy == 1.0:
                            acertos_acumulados += 1
                        
                        acuracia_prog = acertos_acumulados / total_jogos_testados
                        
                        registro = {                            
                            'Jogo_Real_n': i + 1,
                            'Amostra_n': total_jogos_testados,
                            'Resultado': "Acerto" if accuracy == 1.0 else "Erro",
                            'Acuracia_Acumulada': round(acuracia_prog, 4)
                        }

                        # Só calcula F1-Score se for o algoritmo Vanilla
                        if modelo == 'vanilla' and y_test_val is not None:
                            # Determina predição baseada no acerto/erro
                            y_pred_val = y_test_val[0] if accuracy == 1.0 else 1 - y_test_val[0]
                            
                            y_true_acumulado.append(y_test_val[0])
                            y_pred_acumulado.append(y_pred_val)
                            
                            # Calcula F1 com base em toda a matriz de confusão acumulada até aqui
                            f1_prog = f1_score(y_true_acumulado, y_pred_acumulado, average='weighted', zero_division=0)
                            registro['F1_Score_Acumulado'] = round(f1_prog, 4)
                        
                        evolucao_metricas.append(registro)

                except Exception as e:
                    print(f"[!] Erro na janela {pasta_janela}: {e}")

            num_sequencial += 1

        # Salvamento
        if evolucao_metricas:
            output_dir = os.path.join(base_path, 'results', 'experimento_03_2')
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f'evolucao_{modelo}_{temporada}_K{K_espacamento}.csv')
            pd.DataFrame(evolucao_metricas).to_csv(filename, index=False)

print(f"\n--- Experimento concluído em {time.time() - start_time:.2f} segundos ---")