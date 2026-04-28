import os
import time
import pandas as pd
import numpy as np

# Seus imports de modelos
from rede_neural import run_model_rede_neural
from svm import run_model_svm 
from random_forest import run_model_rf
from experimentos import save_results_csv
from naive_bayes import run_model_naive_bayes

# Configurações
#jogos_medias = ['15']
modelos = ['naive_bayes','svm']
temporadas = [
    "2008-2009","2009-2010", "2011-2012", "2012-2013",
    "2013-2014", "2014-2015", "2015-2016", "2016-2017",
    "2018-2019", "2019-2020", "2020-2021", "2021-2022", 
    "2022-2023", "2023-2024","2024-2025"
]

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
start_time = time.time()

# Configuração de espaçamento: 1 testa tudo, 5 pula de 5 em 5, etc.
K_espacamento = 1

for modelo in modelos:
    for temporada in temporadas:
        # Caminho onde os datasets corrigidos estão armazenados [cite: 38-40]
        temporada_dir = os.path.join(base_path, "data", "experimento_03", temporada)
        
        if not os.path.exists(temporada_dir):
            print(f"Pulo: {temporada} não encontrada.")
            continue

        # Ordenação das janelas para garantir a cronologia [cite: 44, 52]
        janelas = [d for d in os.listdir(temporada_dir) if os.path.isdir(os.path.join(temporada_dir, d))]
        janelas.sort(key=lambda x: int(x.split('-')[0]))

        print(f"\n>> Analisando Evolução: {modelo.upper()} | Temporada {temporada} | Step {K_espacamento}")

        num_sequencial = 1
        acertos_acumulados = 0
        total_jogos_testados = 0
        evolucao_acuracia = []

        for i, pasta_janela in enumerate(janelas):
            # Lógica de Stride (Espaçamento)
            deve_processar = (i % K_espacamento == 0)

            treino_path = os.path.join(temporada_dir, pasta_janela, f"treino_{num_sequencial}.csv")
            teste_path = os.path.join(temporada_dir, pasta_janela, f"teste_{num_sequencial}.csv")

            if deve_processar and os.path.exists(treino_path) and os.path.exists(teste_path):
                try:
                    # Execução dos modelos conforme as correções de leak [cite: 36, 62, 68]
                    if modelo == 'naive_bayes':
                        accuracy, _, _ = run_model_naive_bayes(treino_path, teste_path)
                    elif modelo == 'svm':
                        accuracy, _, _ = run_model_svm(treino_path, teste_path, False)

                    if accuracy is not None:
                        total_jogos_testados += 1
                        if accuracy == 1.0:
                            acertos_acumulados += 1
                        
                        # Cálculo da acurácia acumulada (Progressiva)
                        acuracia_atual = acertos_acumulados / total_jogos_testados
                        evolucao_acuracia.append({
                            'Jogo_Real_n': i + 1,
                            'Amostra_n': total_jogos_testados,
                            'Resultado': "Acerto" if accuracy == 1.0 else "Erro",
                            'Acuracia_Acumulada': round(acuracia_atual, 4)
                        })
                except Exception as e:
                    print(f"[!] Erro no processamento: {e}")

            # Incrementa o sufixo para manter sincronia com os arquivos físicos [cite: 45, 56]
            num_sequencial += 1

        # Salvamento dos resultados acumulados por temporada
        if evolucao_acuracia:
            output_dir = os.path.join(base_path, 'results', 'experimento_03_2')
            os.makedirs(output_dir, exist_ok=True)
            
            filename = os.path.join(output_dir, f'evolucao_{modelo}_{temporada}_K{K_espacamento}.csv')
            pd.DataFrame(evolucao_acuracia).to_csv(filename, index=False)
            print(f">> Arquivo de evolução salvo em: {filename}")

end_time = time.time()
print(f"\n--- Experimento concluído em {end_time - start_time:.2f} segundos ---")