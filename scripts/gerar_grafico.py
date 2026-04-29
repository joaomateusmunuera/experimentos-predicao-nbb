import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Configurações
modelos = ['xgboost']
temporadas = [
     "2008-2009", "2009-2010"
   # "2008-2009", "2009-2010", "2011-2012", "2012-2013",
  #  "2013-2014", "2014-2015", "2015-2016", "2016-2017",
  #  "2018-2019", "2019-2020", "2020-2021", "2021-2022", 
  #  "2022-2023", "2023-2024", "2024-2025"
]
k = 1
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

for modelo in modelos:
    for temporada in temporadas:
        # Caminhos dos arquivos
        path_grid = os.path.join(base_path, 'results', 'experimento_03_2', f'evolucao_{modelo}_{temporada}_K{k}.csv')
        path_vanilla = os.path.join(base_path, 'results', 'experimento_03_2', f'evolucao_vanilla_{temporada}_K{k}.csv')

        if os.path.exists(path_grid) and os.path.exists(path_vanilla):
            df_grid = pd.read_csv(path_grid)
            df_vanilla = pd.read_csv(path_vanilla)

            plt.figure(figsize=(14, 7))
            
            # 1. Plotar a linha do modelo otimizado (Grid Search) - AZUL
            plt.plot(df_grid['Jogo_Real_n'], df_grid['Acuracia_Acumulada'], 
                     label=f'{modelo.upper()}', 
                     color='#1f77b4', linewidth=2.5)

            # 2. Plotar a linha do modelo Vanilla (Padrão) - VERMELHA
            plt.plot(df_vanilla['Jogo_Real_n'], df_vanilla['Acuracia_Acumulada'], 
                     label='Vanilla (Referência)', 
                     color='red', linewidth=2, alpha=0.8)

            # --- AJUSTE DO EIXO X ---
            max_jogos = df_grid['Jogo_Real_n'].max()
            plt.xticks(np.arange(0, max_jogos + 20, 20))

            # Customização do gráfico
            plt.title(f'Comparação de Evolução: Vanilla vs {modelo.upper()} - {temporada}', fontsize=14, pad=15)
            plt.xlabel('Número do Jogo na Temporada', fontsize=12)
            plt.ylabel('Acurácia Acumulada', fontsize=12)
            plt.ylim(0.4, 0.8)  # Foco na zona de acerto competitiva
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(loc='lower right', fontsize=10)

            # Anotação do resultado final Grid Search
            final_acc_grid = df_grid['Acuracia_Acumulada'].iloc[-1]
            final_jogo = df_grid['Jogo_Real_n'].iloc[-1]
            plt.annotate(f'Final Grid: {final_acc_grid:.4f}', 
                        xy=(final_jogo, final_acc_grid), 
                        xytext=(final_jogo - 45, final_acc_grid + 0.05),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

            # Anotação do resultado final Vanilla
            final_acc_vanilla = df_vanilla['Acuracia_Acumulada'].iloc[-1]
            plt.annotate(f'Final Vanilla: {final_acc_vanilla:.4f}', 
                        xy=(final_jogo, final_acc_vanilla), 
                        xytext=(final_jogo - 45, final_acc_vanilla - 0.05),
                        arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.1))

            plt.tight_layout()

            # --- LÓGICA DE SALVAMENTO ---
            plot_dir = os.path.join(base_path, 'results', 'graficos')
            os.makedirs(plot_dir, exist_ok=True) 

            plot_filename = os.path.join(plot_dir, f'grafico_{modelo}_{temporada}.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f">> Comparação salva: {plot_filename}")
            
            # Fecha a figura para liberar memória antes da próxima temporada
            plt.close()
        else:
            if not os.path.exists(path_grid):
                print(f"[-] Grid não encontrado: {path_grid}")
            if not os.path.exists(path_vanilla):
                print(f"[-] Vanilla não encontrado: {path_vanilla}")

print("\n--- Todos os gráficos foram gerados com sucesso ---")