import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- CONFIGURAÇÕES ---
modelos = ['svm']
temporadas = [
    "2008-2009", "2009-2010", "2011-2012", "2012-2013",
    "2013-2014", "2014-2015", "2015-2016", "2016-2017",
    "2018-2019", "2019-2020", "2020-2021", "2021-2022", 
    "2022-2023", "2023-2024", "2024-2025"
]

k = 1
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

for modelo in modelos:
    for temporada in temporadas:
        # Caminhos dos arquivos CSV gerados pelo experimento_03_2.py
        path_grid = os.path.join(base_path, 'results', 'experimento_03_2', f'evolucao_{modelo}_{temporada}_K{k}.csv')
        path_vanilla = os.path.join(base_path, 'results', 'experimento_03_2', f'evolucao_vanilla_{temporada}_K{k}.csv')

        if os.path.exists(path_grid) and os.path.exists(path_vanilla):
            df_grid = pd.read_csv(path_grid)
            df_vanilla = pd.read_csv(path_vanilla)

            plt.figure(figsize=(14, 7))
            
            # 1. Plotar a linha do modelo - AZUL
            plt.plot(df_grid['Jogo_Real_n'], df_grid['Acuracia_Acumulada'], 
                     label=f'Acurácia {modelo.upper()}', 
                     color='#1f77b4', linewidth=2.5)

            # 2. Plotar a linha da Acurácia Vanilla - VERMELHA
            plt.plot(df_vanilla['Jogo_Real_n'], df_vanilla['Acuracia_Acumulada'], 
                     label='Acurácia Vanilla', 
                     color='red', linewidth=2, alpha=0.8)

            # 3. Plotar a linha do F1-Score Vanilla - VERMELHO CLARO PONTILHADO

            if 'F1_Score_Acumulado' in df_vanilla.columns:
                plt.plot(df_vanilla['Jogo_Real_n'], df_vanilla['F1_Score_Acumulado'], 
                         label='F1-Score Vanilla', 
                         color='indianred',linewidth=2, alpha=0.7)

            # --- AJUSTES DE EIXO E LIMITES (O segredo para a anotação aparecer) ---
            max_jogos = df_grid['Jogo_Real_n'].max()
            plt.xticks(np.arange(0, max_jogos + 20, 20))
            
            # Expandimos o X em +80 para criar espaço para as caixas de texto à direita
            plt.xlim(-5, max_jogos + 80) 
            plt.ylim(0.20, 0.85) # Range ideal para visualizar o contraste Acc vs F1

            # Customização do gráfico
            plt.title(f'Comparação de Evolução: Vanilla vs {modelo.upper()} - {temporada}', fontsize=14, pad=15)
            plt.xlabel('Número do Jogo na Temporada', fontsize=12)
            plt.ylabel('Métrica Acumulada', fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.6)
            
            # Legenda movida para a esquerda para não obstruir as setas na direita
            plt.legend(loc='lower right', fontsize=10, shadow=True, frameon=True)

            # --- ANOTAÇÕES COM PILHAMENTO FIXO (ORDEM DE ESCADA) ---
            final_jogo = df_grid['Jogo_Real_n'].iloc[-1]
            espaco_lateral = 12

            # 1. SVM (Sempre no topo da escada)
            final_acc_grid = df_grid['Acuracia_Acumulada'].iloc[-1]
            plt.annotate(f'Final {modelo.upper()}: {final_acc_grid:.4f}', 
                        xy=(final_jogo, final_acc_grid), 
                        xytext=(final_jogo + espaco_lateral, 0.70), # Altura fixa no topo
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color='black'),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

            # 2. Vanilla Acc (Sempre no meio da escada)
            final_acc_vanilla = df_vanilla['Acuracia_Acumulada'].iloc[-1]
            plt.annotate(f'Final Acc Vanilla: {final_acc_vanilla:.4f}', 
                        xy=(final_jogo, final_acc_vanilla), 
                        xytext=(final_jogo + espaco_lateral, 0.58), # Altura fixa no meio
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.0", color='red'),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.1))
            
            # 3. Vanilla F1 (Sempre na base da escada)
            if 'F1_Score_Acumulado' in df_vanilla.columns:
                final_f1_vanilla = df_vanilla['F1_Score_Acumulado'].iloc[-1]
                plt.annotate(f'Final F1 Vanilla: {final_f1_vanilla:.4f}', 
                            xy=(final_jogo, final_f1_vanilla), 
                            xytext=(final_jogo + espaco_lateral, 0.45), # Altura fixa na base
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color='indianred'),
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="indianred", alpha=0.6))

            plt.tight_layout()

            # --- SALVAMENTO ---
            plot_dir = os.path.join(base_path, 'results', 'graficos')
            os.makedirs(plot_dir, exist_ok=True) 

            plot_filename = os.path.join(plot_dir, f'grafico_{modelo}_{temporada}.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f">> Gráfico salvo: {plot_filename}")
            
            plt.close()
        else:
            if not os.path.exists(path_grid):
                print(f"[-] Arquivo Grid faltando: {path_grid}")
            if not os.path.exists(path_vanilla):
                print(f"[-] Arquivo Vanilla faltando: {path_vanilla}")

print("\n--- Todos os gráficos foram gerados com sucesso ---")