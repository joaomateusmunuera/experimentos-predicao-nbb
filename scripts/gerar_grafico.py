import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Configurações
modelos = ['svm']
temporadas = [
    "2008-2009","2009-2010", "2011-2012", "2012-2013",
    "2013-2014", "2014-2015", "2015-2016", "2016-2017",
    "2018-2019", "2019-2020", "2020-2021", "2021-2022", 
    "2022-2023", "2023-2024","2024-2025"
]
k = 1
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

for modelo in modelos:
    for temporada in temporadas:
        file_path = os.path.join(base_path, 'results', 'experimento_03_2', f'evolucao_{modelo}_{temporada}_K{k}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            plt.figure(figsize=(12, 6))
            
            plt.plot(df['Jogo_Real_n'], df['Acuracia_Acumulada'], 
                    label=f'Acurácia Acumulada ({modelo.upper()})', 
                    color='#1f77b4', linewidth=2)

            plt.axhline(y=0.70, color='r', linestyle='--', alpha=0.5, label='Meta 70%')

            max_jogos = df['Jogo_Real_n'].max()
            plt.xticks(np.arange(0, max_jogos + 20, 20))

            plt.title(f'Evolução da Acurácia - Temporada {temporada}', fontsize=14, pad=15)
            plt.xlabel('Número do Jogo na Temporada', fontsize=12)
            plt.ylabel('Acurácia Acumulada', fontsize=12)
            plt.ylim(0, 1.0)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(loc='lower right')

            final_acc = df['Acuracia_Acumulada'].iloc[-1]
            final_jogo = df['Jogo_Real_n'].iloc[-1]
            plt.annotate(f'Final: {final_acc:.4f}', 
                        xy=(final_jogo, final_acc), 
                        xytext=(final_jogo - 40, final_acc + 0.1),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

            plt.tight_layout()

            # --- LÓGICA DE SALVAMENTO ---
            # Define o diretório de destino
            plot_dir = os.path.join(base_path, 'results', 'graficos')
            os.makedirs(plot_dir, exist_ok=True) # Cria a pasta se não existir

            # Define o nome do arquivo (pode ser .png, .pdf, .jpg)
            plot_filename = os.path.join(plot_dir, f'grafico_{modelo}_{temporada}.png')
            
            # Salva a imagem com alta resolução (dpi=300 é ótimo para o TCC)
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f">> Gráfico salvo com sucesso em: {plot_filename}")
            # ----------------------------
        else:
            print(f"Arquivo não encontrado: {file_path}")