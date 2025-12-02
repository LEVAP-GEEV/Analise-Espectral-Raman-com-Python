# Seja bem vindo(a) ao PySAR (Python Spectrum Analisys Raman)
# Importando bibliotecas necessárias para o PySAR
import matplotlib.pyplot as plt
import peakutils as pk
import numpy as np
import tkinter as tk
import pandas as pd
import os
import matplotlib
from lmfit.models import LorentzianModel, ConstantModel
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
from pyPreprocessing import baseline_correction as bs, smoothing as smo
from tkinter import filedialog
from datetime import datetime

matplotlib.use('TkAgg')
plt.rcParams['figure.max_open_warning'] = 0
plt.rcParams['figure.figsize'] = [16, 12]

# Obter o tamanho da tela para ajustar o tamanho das figuras
def obter_tamanho_tela():
    try:
        root = tk.Tk()
        root.withdraw()
        largura = root.winfo_screenwidth()
        altura = root.winfo_screenheight()
        root.destroy()

        # Converter pixels para polegadas (considerando DPI ~100)
        largura_polegadas = largura / 100
        altura_polegadas = altura / 100

        return largura_polegadas, altura_polegadas
    except:
        # Fallback para tamanhos padrão
        return 16, 12

# Processar espectro individual na análise comparativa
def processar_espectro_comparativo_iterativo(arquivo, metodo_baseline, parametros_picos):
    try:
        # Carregar dados
        data = np.loadtxt(arquivo)
        x = data[:, 0]
        y = data[:, 1]

        # Detectar automaticamente a região do espectro
        threshold = np.max(y) * 0.01
        signal_indices = np.where(y > threshold)[0]

        if len(signal_indices) == 0:
            raise ValueError("Nenhum sinal significativo detectado no espectro")

        auto_x_min = x[signal_indices[0]]
        auto_x_max = x[signal_indices[-1]]

        margin = (auto_x_max - auto_x_min) * 0.05
        auto_x_min = max(x[0], auto_x_min - margin)
        auto_x_max = min(x[-1], auto_x_max + margin)

        mask = (x >= auto_x_min) & (x <= auto_x_max)
        rs = x[mask]
        intensity_original = y[mask]

        if len(rs) == 0:
            raise ValueError("Nenhum ponto com sinal significativo detectado no espectro")

        print(f"{len(rs)} pontos no intervalo {rs[0]:.1f} - {rs[-1]:.1f} cm⁻¹")

        # Aplicar baseline pré-definida
        if metodo_baseline == "Peakutils (deg=3)":
            baseline_final = pk.baseline(intensity_original, deg=3, max_it=1000, tol=1e-10)
        elif metodo_baseline == "Quantil":
            baseline_final = baseline_quantil_otimizada(intensity_original)
        elif metodo_baseline == "SNIP":
            inten_reshaped = intensity_original.reshape(1, -1)
            baseline_final = bs.generate_baseline(inten_reshaped, mode='SNIP')[0]
        else:
            baseline_final = pk.baseline(intensity_original, deg=3, max_it=1000, tol=1e-10)

        intensity_corrigida = intensity_original - baseline_final

        # Suavização
        inten = intensity_corrigida.reshape(1, -1)
        intensity = smo.smoothing(inten, mode='sav_gol')[0]

        # Sistema iterativo
        result_final = None
        picos_manuais = None
        primeiro_ajuste = True

        while True:
            print(f"\nPROCESSANDO: {os.path.basename(arquivo)}")

            if primeiro_ajuste:
                # Primeira vez: processar automaticamente
                result = processar_com_parametros(rs, intensity_corrigida, intensity_original,
                                                  baseline_final, parametros_picos, metodo_baseline)
                primeiro_ajuste = False
            elif picos_manuais is not None:
                # Seleção de picos manuais: processar com picos manuais
                print(f"Processando {len(picos_manuais)} picos selecionados manualmente...")
                result = processar_com_picos_manuais(rs, intensity_corrigida, intensity_original,
                                                     baseline_final, parametros_picos, metodo_baseline, picos_manuais)
                picos_manuais = None
            else:
                # Modificar parâmetros: processar com novos parâmetros
                print("Reprocessando com parâmetros atuais...")
                result = processar_com_parametros(rs, intensity_corrigida, intensity_original,
                                                  baseline_final, parametros_picos, metodo_baseline)

            # Verificar resultado do ajuste
            if result is None:
                print("Nenhum pico detectado ou falha no ajuste.")
                print("O que você gostaria de fazer?")
                print("1. Selecionar picos manualmente")
                print("2. Pular este espectro")

                escolha = input("\nDigite sua escolha (1-2): ").strip()

                if escolha == "1":
                    picos_manuais = selecionar_picos_manualmente(rs, intensity_corrigida, None, None)
                    continue
                elif escolha == "2":
                    print(f"Pulando espectro: {os.path.basename(arquivo)}")
                    return None, None, None
                else:
                    print("Opção inválida.")
                    continue
            else:
                # Ajuste bem-sucedido
                print(f"\nAJUSTE CONCLUÍDO PARA: {os.path.basename(arquivo)}")
                print("O que você gostaria de fazer?")
                print("1. Aceitar este ajuste e prosseguir")
                print("2. Refinar picos manualmente")
                print("3. Recomeçar do zero")
                print("4. Pular este espectro")

                escolha = input("\nDigite sua escolha (1-4): ").strip()

                if escolha == "1":
                    result_final = result
                    break

                elif escolha == "2":
                    picos_base = []
                    for param_name in result.params:
                        if '_center' in param_name:
                            picos_base.append(result.params[param_name].value)

                    print("Modo de refinamento - Selecione os picos desejados")
                    picos_manuais = selecionar_picos_manualmente(rs, intensity_corrigida, picos_base, result)

                elif escolha == "3":
                    print("Recomeçando do zero...")
                    primeiro_ajuste = True
                    picos_manuais = None

                elif escolha == "4":
                    print(f"Pulando espectro: {os.path.basename(arquivo)}")
                    return None, None, None
                else:
                    print("Opção inválida.")

        return result_final, rs, intensity_original

    except Exception as e:
        print(f"Erro no processamento: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Seleção manual de picos
def selecionar_picos_manualmente(rs, intensity_corrigida, picos_atuais=None, result_anterior=None):
    largura, altura = obter_tamanho_tela()
    fig, ax = plt.subplots(figsize=(largura*0.9, altura*0.85))
    plt.subplots_adjust(bottom=0.15)

    # Plotar o espectro
    ax.plot(rs, intensity_corrigida, 'b-', linewidth=1, label='Espectro')
    ax.set_xlabel('Raman Shift (cm⁻¹)')
    ax.set_ylabel('Intensidade')
    ax.set_title('Instruções - Clique para ADICIONAR, Shift+Clique para REMOVER')
    ax.grid(True, alpha=0.3)

    # Inicializar lista de picos
    if picos_atuais is None:
        picos_selecionados = []
    else:
        picos_selecionados = picos_atuais.copy()

    # Listas para armazenar objetos gráficos
    marcadores = []
    textos = []

    def atualizar_grafico():
        # Limpar marcadores antigos
        for marcador in marcadores:
            try:
                marcador.remove()
            except:
                pass
        for texto in textos:
            try:
                texto.remove()
            except:
                pass
        marcadores.clear()
        textos.clear()

        # Plotar novos marcadores
        for pico in picos_selecionados:
            altura = np.interp(pico, rs, intensity_corrigida)
            # Marcador
            marcador = ax.plot(pico, altura, 'ro', markersize=8, markeredgecolor='black',
                               markerfacecolor='red', markeredgewidth=2)[0]
            marcadores.append(marcador)
            # Texto
            texto = ax.text(pico, altura, f'  {pico:.1f}', fontsize=8,
                            verticalalignment='bottom', fontweight='bold')
            textos.append(texto)

        # Atualizar legenda
        handles, labels = ax.get_legend_handles_labels()
        if picos_selecionados:
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='red', markersize=8,
                                      label='Picos')]
            ax.legend(handles + legend_elements, labels + ['Picos'])
        else:
            ax.legend(handles, labels)

        fig.canvas.draw()

    # Plotar inicialmente
    atualizar_grafico()

    # Função para clique
    def onclick(event):
        if event.inaxes == ax:
            x_pos = event.xdata
            y_pos = event.ydata

            # Verificar se é para remover (Shift + clique)
            if event.key == 'shift':
                # Encontrar pico mais próximo
                if picos_selecionados:
                    distancias = [abs(x_pos - pico) for pico in picos_selecionados]
                    indice_mais_proximo = np.argmin(distancias)
                    distancia_minima = distancias[indice_mais_proximo]

                    if distancia_minima < 4:  # Tolerância de 4 cm⁻¹
                        pico_removido = picos_selecionados.pop(indice_mais_proximo)
                        atualizar_grafico()
            else:
                # Modo adição - verificar se não existe pico muito próximo
                if picos_selecionados:
                    distancias = [abs(x_pos - pico) for pico in picos_selecionados]
                    distancia_minima = min(distancias)
                else:
                    distancia_minima = float('inf')

                if distancia_minima > 5:  # Distância mínima de 5 cm⁻¹
                    picos_selecionados.append(x_pos)
                    atualizar_grafico()

    # Conectar evento
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Botões
    from matplotlib.widgets import Button

    ax_finalizar = plt.axes([0.7, 0.05, 0.2, 0.075])
    ax_limpar = plt.axes([0.45, 0.05, 0.2, 0.075])
    ax_cancelar = plt.axes([0.2, 0.05, 0.2, 0.075])

    btn_finalizar = Button(ax_finalizar, 'Finalizar')
    btn_limpar = Button(ax_limpar, 'Limpar Tudo')
    btn_cancelar = Button(ax_cancelar, 'Cancelar')

    resultado = None

    def finalizar(event):
        nonlocal resultado
        resultado = sorted(picos_selecionados)
        plt.close()

    def limpar_tudo(event):
        nonlocal picos_selecionados
        picos_selecionados.clear()
        print("Todos os picos removidos!")
        atualizar_grafico()

    def cancelar(event):
        nonlocal resultado
        resultado = picos_atuais  # Retornar os picos originais
        print("Seleção cancelada. Mantendo picos originais.")
        plt.close()

    btn_finalizar.on_clicked(finalizar)
    btn_limpar.on_clicked(limpar_tudo)
    btn_cancelar.on_clicked(cancelar)

    plt.show()

    # Desconectar evento
    fig.canvas.mpl_disconnect(cid)

    return resultado

# Processar espectro incluindo picos selecionados manualmente
def processar_com_picos_manuais(rs, intensity_corrigida, intensity_original, baseline_final,
                                parametros_picos, metodo_baseline, picos_manuais=None):
    largura, altura = obter_tamanho_tela()
    try:
        # Suavização
        inten = intensity_corrigida.reshape(1, -1)
        intensity = smo.smoothing(inten, mode='sav_gol')[0]

        print(f"\n===INICIANDO AJUSTE COM PICOS MANUAIS===")
        todos_picos = picos_manuais if picos_manuais else []

        if len(todos_picos) == 0:
            print("NENHUM PICO PARA AJUSTE!")
            return None

        print(f"Total de picos para ajuste: {len(todos_picos)}")
        print(f"Método de baseline: {metodo_baseline}")

        k = min(60, len(todos_picos))  # Número de Lorentzianas

        if len(todos_picos) < k:
            k = len(todos_picos)

        if k == 0:
            print("Nenhum pico para ajuste!")
            return None

        if k > 60:
            print(f"k={k} é muito alto. Limitando a 60 Lorentzianas")
            k = 60

        # Preparar dados para clusterização (usar posições e alturas estimadas)
        peak_data = []
        for pico in todos_picos:
            idx = np.argmin(np.abs(rs - pico))
            altura = intensity[idx]
            peak_data.append([pico, altura])

        peak_data = np.array(peak_data)

        # Clusterização para centros iniciais (se temos muitos picos)
        if k > 1:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=20, max_iter=500).fit(peak_data)
            centers = sorted(kmeans.cluster_centers_, key=lambda x: x[0])
        else:
            centers = peak_data

        model = None
        params = None

        # Configurar modelo
        for i, (center, amp) in enumerate(centers):
            lorentz = LorentzianModel(prefix=f'l{i + 1}_')
            if model is None:
                model = lorentz
                params = lorentz.make_params()
            else:
                model += lorentz
                params.update(lorentz.make_params())

            amp_factor = 5 if k > 40 else 10
            sigma_init = 3 if k > 40 else 5

            params[f'l{i + 1}_center'].set(value=center, min=rs.min(), max=rs.max())
            params[f'l{i + 1}_amplitude'].set(value=amp * amp_factor, min=0, max=amp * 20)
            params[f'l{i + 1}_sigma'].set(value=sigma_init, min=0.5, max=20)

        # Adicionar constante
        constant = ConstantModel(prefix='c_')
        model = constant + model
        params.update(constant.make_params())
        params['c_c'].set(value=np.mean(intensity) * 0.1, min=0)

        print(f"Ajuste com {k} Lorentzianas...")

        # Ajuste
        result = model.fit(data=intensity, params=params, x=rs,
                           method='leastsq',
                           max_nfev=5000,
                           nan_policy='omit')

        # Extrair picos do ajuste atual
        picos_ajuste_atual = []
        alturas_ajuste_atual = []

        for param_name in result.params:
            if '_center' in param_name:
                center = result.params[param_name].value
                # Encontrar altura no ajuste
                idx = np.argmin(np.abs(rs - center))
                altura = result.best_fit[idx]
                picos_ajuste_atual.append(center)
                alturas_ajuste_atual.append(altura)

        # Mostrar gráfico do ajuste
        comps = result.eval_components()

        largura, altura = obter_tamanho_tela()
        fig, ax = plt.subplots(figsize=(largura * 0.9, altura * 0.85))
        plt.subplots_adjust(bottom=0.15)

        # Subplot 1: Baseline e dados corrigidos
        plt.subplot(2, 1, 1)
        plt.plot(rs, intensity_original, 'gray', label='Dados Originais', linewidth=1, alpha=0.6)
        plt.plot(rs, baseline_final, 'orange', label='Baseline', linewidth=2, linestyle='--')
        plt.plot(rs, intensity_corrigida, 'blue', label='Dados Corrigidos', linewidth=1, alpha=0.8)
        plt.xlabel(r'Raman shift (cm$^{-1}$)')
        plt.ylabel(r'Intensity (a.u.)')
        plt.legend()
        plt.title('Correção de Baseline')
        plt.grid(True, alpha=0.3)

        # Subplot 2: Resultado do ajuste
        plt.subplot(2, 1, 2)
        plt.plot(rs, intensity, 'k-', label='Dados Suavizados', linewidth=1, alpha=0.7)
        plt.plot(rs, result.best_fit, 'r-', label='Ajuste', linewidth=2)

        #Mostrar apenas os picos do ajuste atual
        if picos_ajuste_atual:
            plt.scatter(picos_ajuste_atual, alturas_ajuste_atual, color='green', s=60,
                        label=f'Picos do Ajuste ({len(picos_ajuste_atual)})', zorder=5,
                        edgecolors='white', linewidth=1.5)

        # Plot componentes apenas se k for razoável
        if k <= 30:
            for name, comp in comps.items():
                if name != 'c_':
                    plt.plot(rs, comp, '--', alpha=0.3, linewidth=1)

        plt.xlabel(r'Raman shift (cm$^{-1}$)')
        plt.ylabel(r'Intensity (a.u.)')
        plt.legend()
        plt.title(f'Ajuste com {k} Lorentzianas')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return result

    except Exception as e:
        print(f"Erro no processamento: {e}")
        import traceback
        traceback.print_exc()
        return None

# Extrai dados dos picos no formato necessário para comparação
def extrair_dados_picos(params):
    dados = {}
    picos = set()
    for param_name in params:
        if '_' in param_name and param_name != 'c_c':
            pico = param_name.split('_')[0]
            picos.add(pico)

    for pico in picos:
        dados[pico] = {
            'amplitude': params.get(f'{pico}_amplitude').value if f'{pico}_amplitude' in params else 0,
            'center': params.get(f'{pico}_center').value if f'{pico}_center' in params else 0,
            'sigma': params.get(f'{pico}_sigma').value if f'{pico}_sigma' in params else 0,
            'height': params.get(f'{pico}_height').value if f'{pico}_height' in params else 0
        }
    return dados

# Comparar picos entre dois espectros
def comparar_picos_espectros(ref_picos, comp_picos, nome_comp, nome_ref, tolerancia=5):
    dados_comparacao = []

    # Mapear picos da referência
    ref_centers = {pico: ref_picos[pico]['center'] for pico in ref_picos}
    comp_centers = {pico: comp_picos[pico]['center'] for pico in comp_picos}

    # Encontrar picos comuns
    picos_comuns = []
    for ref_pico, ref_center in ref_centers.items():
        for comp_pico, comp_center in comp_centers.items():
            if abs(ref_center - comp_center) <= tolerancia:
                picos_comuns.append((ref_pico, comp_pico, ref_center, comp_center))
                break

    # Picos que desapareceram (estão na ref mas não no comp)
    picos_desaparecidos = []
    for ref_pico in ref_centers:
        encontrou = False
        for comp_pico, comp_center in comp_centers.items():
            if abs(ref_centers[ref_pico] - comp_center) <= tolerancia:
                encontrou = True
                break
        if not encontrou:
            picos_desaparecidos.append(ref_pico)

    # Picos novos (estão no comp mas não na ref)
    picos_novos = []
    for comp_pico in comp_centers:
        encontrou = False
        for ref_pico, ref_center in ref_centers.items():
            if abs(comp_centers[comp_pico] - ref_center) <= tolerancia:
                encontrou = True
                break
        if not encontrou:
            picos_novos.append(comp_pico)

    # Compilar resultados
    for ref_pico, comp_pico, ref_center, comp_center in picos_comuns:
        dados_comparacao.append({
            'Espectro_Comparado': nome_comp,
            'Espectro_Referencia': nome_ref,
            'Tipo': 'COMUM',
            'Pico_Referencia': ref_pico,
            'Pico_Comparado': comp_pico,
            'Center_Referencia': ref_center,
            'Center_Comparado': comp_center,
            'Delta_Center': abs(ref_center - comp_center),
            'Amplitude_Referencia': ref_picos[ref_pico]['amplitude'],
            'Amplitude_Comparado': comp_picos[comp_pico]['amplitude'],
            'Delta_Amplitude': comp_picos[comp_pico]['amplitude'] - ref_picos[ref_pico]['amplitude']
        })

    for pico in picos_desaparecidos:
        dados_comparacao.append({
            'Espectro_Comparado': nome_comp,
            'Espectro_Referencia': nome_ref,
            'Tipo': 'DESAPARECEU',
            'Pico_Referencia': pico,
            'Pico_Comparado': '-',
            'Center_Referencia': ref_picos[pico]['center'],
            'Center_Comparado': '-',
            'Delta_Center': '-',
            'Amplitude_Referencia': ref_picos[pico]['amplitude'],
            'Amplitude_Comparado': '-',
            'Delta_Amplitude': '-'
        })

    for pico in picos_novos:
        dados_comparacao.append({
            'Espectro_Comparado': nome_comp,
            'Espectro_Referencia': nome_ref,
            'Tipo': 'NOVO',
            'Pico_Referencia': '-',
            'Pico_Comparado': pico,
            'Center_Referencia': '-',
            'Center_Comparado': comp_picos[pico]['center'],
            'Delta_Center': '-',
            'Amplitude_Referencia': '-',
            'Amplitude_Comparado': comp_picos[pico]['amplitude'],
            'Delta_Amplitude': '-'
        })
    return pd.DataFrame(dados_comparacao)

# Analisar múltiplos espectros e comparar com a referência
def analisar_comparacao_espectros(espectros_dados, referencia_nome, tolerancia_cm=5):
    resultados = []
    referencia_picos = espectros_dados[referencia_nome]
    for nome_espectro, dados_espectro in espectros_dados.items():
        if nome_espectro == referencia_nome:
            continue

        comparacao = comparar_picos_espectros(
            referencia_picos, dados_espectro, nome_espectro, referencia_nome, tolerancia_cm
        )
        resultados.append(comparacao)
    return pd.concat(resultados, ignore_index=True)

# Gerar um resumo estatístico da análise
def gerar_resumo_analise(df_comparacao):
    resumo_dados = []
    for espectro in df_comparacao['Espectro_Comparado'].unique():
        dados_espectro = df_comparacao[df_comparacao['Espectro_Comparado'] == espectro]

        comuns = dados_espectro[dados_espectro['Tipo'] == 'COMUM']
        novos = dados_espectro[dados_espectro['Tipo'] == 'NOVO']
        desaparecidos = dados_espectro[dados_espectro['Tipo'] == 'DESAPARECEU']

        resumo_dados.append({
            'Espectro': espectro,
            'Picos_Comuns': len(comuns),
            'Picos_Novos': len(novos),
            'Picos_Desaparecidos': len(desaparecidos),
        })
    return pd.DataFrame(resumo_dados)

# Identificar picos que estão presentes em todos os espectros
def identificar_picos_comuns_todos(espectros_dados, tolerancia=5):
    if not espectros_dados:
        return []

    # Pegar todos os espectros
    nomes_espectros = list(espectros_dados.keys())

    # Começar com os picos do primeiro espectro
    picos_base = espectros_dados[nomes_espectros[0]]
    picos_comuns = list(picos_base.keys())

    # Para cada espectro subsequente, filtrar os picos comuns
    for i in range(1, len(nomes_espectros)):
        espectro_atual = espectros_dados[nomes_espectros[i]]
        picos_atuais = list(espectro_atual.keys())

        # Verificar quais picos do base existem no atual (com tolerância)
        picos_comuns_temp = []
        for pico_base in picos_comuns:
            center_base = picos_base[pico_base]['center']
            encontrou = False

            for pico_atual in picos_atuais:
                center_atual = espectro_atual[pico_atual]['center']
                if abs(center_base - center_atual) <= tolerancia:
                    encontrou = True
                    break

            if encontrou:
                picos_comuns_temp.append(pico_base)

        picos_comuns = picos_comuns_temp
    return picos_comuns

# Criar um gráfico mostrando os espectros sobrepostos com destaque para os picos comuns
def criar_grafico_picos_comuns(espectros_dados, picos_comuns, pasta_destino, timestamp, todos_results=None,
                               rs_data=None):
    if not picos_comuns:
        print("Nenhum pico comum encontrado em todos os espectros")
        return None

    # Verificar quais picos realmente existem em todos os espectros
    picos_validos = []
    for pico_comum in picos_comuns:
        valido = True
        for dados_espectro in espectros_dados.values():
            if pico_comum not in dados_espectro:
                valido = False
                break
        if valido:
            picos_validos.append(pico_comum)

    if not picos_validos:
        print("Nenhum pico comum válido encontrado em todos os espectros")
        return None

    # Criar figura com subplots
    largura, altura = obter_tamanho_tela()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(largura * 0.9, altura * 0.85))

    # Cores para diferentes espectros
    cores = plt.cm.Set1(np.linspace(0, 1, len(espectros_dados)))

    # Gráfico 1: Espectros reais sobrepostos com picos comuns destacados
    # Coletar todos os centers dos picos válidos para definir o range do gráfico
    centers_comuns = []
    for pico_valido in picos_validos:
        # Usar o center do primeiro espectro como referência
        center_ref = next(iter(espectros_dados.values()))[pico_valido]['center']
        centers_comuns.append(center_ref)

    # Ordenar os picos válidos por posição
    centers_comuns.sort()

    # Plotar cada espectro REAL (usando os dados dos ajustes)
    for i, (nome_espectro, dados_espectro) in enumerate(espectros_dados.items()):
        nome_limpo = os.path.splitext(nome_espectro)[0]

        # Se temos os resultados completos e dados de x, usar os fits reais
        if todos_results and nome_espectro in todos_results and rs_data and nome_espectro in rs_data:
            result = todos_results[nome_espectro]
            rs = rs_data[nome_espectro]

            # Plotar o ajuste completo do espectro
            y_fit = result.best_fit

            # Normalizar para melhor visualização
            y_normalized = y_fit / np.max(y_fit) * (1 - i * 0.1) if np.max(y_fit) > 0 else y_fit

            ax1.plot(rs, y_normalized, color=cores[i], label=nome_limpo, linewidth=2, alpha=0.8)

            # Marcar os picos válidos neste espectro
            for pico_valido in picos_validos:
                if pico_valido in dados_espectro:
                    center = dados_espectro[pico_valido]['center']
                    # Encontrar o índice mais próximo do center no eixo x
                    idx = np.argmin(np.abs(rs - center))
                    height = y_normalized[idx]

                    ax1.scatter(center, height, color=cores[i], s=80, alpha=0.9,
                                zorder=5, edgecolors='white', linewidth=1)

        else:
            # Fallback: usar espectro simulado (como antes)
            if centers_comuns:
                x_simulado = np.linspace(min(centers_comuns) - 50, max(centers_comuns) + 50, 1000)
            else:
                x_simulado = np.linspace(0, 100, 1000)

            y_simulado = np.zeros_like(x_simulado)

            # Adicionar cada pico válido como uma Lorentziana
            for pico_valido in picos_validos:
                if pico_valido in dados_espectro:
                    center = dados_espectro[pico_valido]['center']
                    amplitude = dados_espectro[pico_valido]['amplitude']
                    sigma = dados_espectro[pico_valido]['sigma']

                    # Lorentziana: y = amplitude / (1 + ((x - center)/sigma)^2)
                    lorentz = amplitude / (1 + ((x_simulado - center) / sigma) ** 2)
                    y_simulado += lorentz

            # Normalizar para melhor visualização
            if np.max(y_simulado) > 0:
                y_simulado = y_simulado / np.max(y_simulado) * (1 - i * 0.1)
            else:
                y_simulado = y_simulado * (1 - i * 0.1)

            # Plotar o espectro simulado
            ax1.plot(x_simulado, y_simulado, color=cores[i], label=nome_limpo, linewidth=2, alpha=0.8)

            # Marcar os picos válidos
            for pico_valido in picos_validos:
                if pico_valido in dados_espectro:
                    center = dados_espectro[pico_valido]['center']
                    amplitude = dados_espectro[pico_valido]['amplitude']
                    sigma = dados_espectro[pico_valido]['sigma']

                    # Calcular altura da Lorentziana no centro
                    height_calculado = amplitude / (np.pi * sigma) if sigma > 0 else amplitude

                    # Normalizar a altura para o gráfico
                    if np.max(y_simulado) > 0:
                        height_normalizado = height_calculado / np.max(y_simulado) * (1 - i * 0.1)
                    else:
                        height_normalizado = height_calculado * (1 - i * 0.1)

                    ax1.scatter(center, height_normalizado, color=cores[i], s=80, alpha=0.9,
                                zorder=5, edgecolors='white', linewidth=1)

    ax1.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    ax1.set_ylabel('Intensidade Normalizada (a.u.)', fontsize=12)
    ax1.set_title(f'Espectros com {len(picos_validos)} Picos Comuns - CURVAS SOBREPOSTAS',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Gráfico 2: variação dos centros dos picos válidos
    picos_data = []
    for i, pico_valido in enumerate(picos_validos):
        centers_por_espectro = []
        amplitudes_por_espectro = []

        for dados_espectro in espectros_dados.values():
            if pico_valido in dados_espectro:
                center_alvo = dados_espectro[pico_valido]['center']
                centers_por_espectro.append(center_alvo)
                amplitudes_por_espectro.append(dados_espectro[pico_valido]['amplitude'])

        if centers_por_espectro:
            picos_data.append({
                'Pico': f'Pico {i + 1}',
                'Pico_Original': pico_valido,
                'Center_Media': np.mean(centers_por_espectro),
                'Center_Std': np.std(centers_por_espectro),
                'Amplitude_Media': np.mean(amplitudes_por_espectro),
                'Centers': centers_por_espectro,
                'Amplitudes': amplitudes_por_espectro
            })

    # Ordenar por centro médio
    picos_data.sort(key=lambda x: x['Center_Media'])

    # Plotar variação dos centros com barras de erro
    for i, pico_info in enumerate(picos_data):
        # Plotar cada medida individual
        for j, center in enumerate(pico_info['Centers']):
            ax2.scatter(center, i, color=cores[j], alpha=0.6, s=40)

        # Plotar média com barra de erro
        ax2.errorbar(pico_info['Center_Media'], i,
                     xerr=pico_info['Center_Std'],
                     fmt='o', color='black', markersize=8,
                     capsize=5, capthick=2, elinewidth=2)

    ax2.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    ax2.set_ylabel('Pico', fontsize=12)
    ax2.set_yticks(range(len(picos_data)))
    ax2.set_yticklabels([pico['Pico'] for pico in picos_data])
    ax2.set_title('Variação dos Centers dos Picos Comuns', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Salvar o gráfico
    caminho_grafico = os.path.join(pasta_destino, f"picos_comuns_sobrepostos_{timestamp}.png")
    plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
    plt.show()

    # Salvar dados dos picos válidos
    dados_comuns = []
    for pico_info in picos_data:
        dados_comuns.append({
            'Pico': pico_info['Pico'],
            'Pico_Original': pico_info['Pico_Original'],
            'Center_Media_cm-1': round(pico_info['Center_Media'], 2),
            'Desvio_Padrao_cm-1': round(pico_info['Center_Std'], 2),
            'Amplitude_Media': round(pico_info['Amplitude_Media'], 2),
            'Centers_Todos': '; '.join(f"{c:.1f}" for c in pico_info['Centers'])
        })

    df_comuns = pd.DataFrame(dados_comuns)
    caminho_comuns = os.path.join(pasta_destino, f"picos_comuns_detalhados_{timestamp}.csv")
    df_comuns.to_csv(caminho_comuns, index=False, sep=';', encoding='utf-8-sig')

    return df_comuns

# Processar o espectro com os parâmetros de detecção fornecidos
def processar_com_parametros(rs, intensity_corrigida, intensity_original, baseline_final, parametros_picos,
                             metodo_baseline):
    try:
        # Suavização
        inten = intensity_corrigida.reshape(1, -1)
        intensity = smo.smoothing(inten, mode='sav_gol')[0]

        # Detecção de picos com parâmetros atuais
        peaks, props = find_peaks(
            intensity,
            height=parametros_picos['height'],
            distance=parametros_picos['distance'],
            prominence=parametros_picos['prominence'],
            width=parametros_picos['width'],
            wlen=parametros_picos['wlen']
        )

        peak_positions = rs[peaks]
        peak_heights = intensity[peaks]

        print(f"{len(peaks)} picos detectados")

        if len(peaks) == 0:
            print("NENHUM PICO DETECTADO com os parâmetros atuais!")
            return None

        print(f"Método de baseline: {metodo_baseline}")

        k = min(60, len(peak_positions))  # Número de Lorentzianas

        if len(peak_positions) < k:
            print(f"Apenas {len(peak_positions)} picos detectados, reduzindo k para este valor")
            k = len(peak_positions)

        if k == 0:
            print("Nenhum pico detectado para ajuste!")
            return None

        if k > 60:
            print(f"k={k} é muito alto. Limitando a 60 Lorentzianas")
            k = 60

        # Clusterização para centros iniciais
        peak_data = np.column_stack((peak_positions, peak_heights))
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=20, max_iter=500).fit(peak_data)
        centers = sorted(kmeans.cluster_centers_, key=lambda x: x[0])

        model = None
        params = None

        # Configurar modelo
        for i, (center, amp) in enumerate(centers):
            lorentz = LorentzianModel(prefix=f'l{i + 1}_')
            if model is None:
                model = lorentz
                params = lorentz.make_params()
            else:
                model += lorentz
                params.update(lorentz.make_params())

            amp_factor = 5 if k > 40 else 10
            sigma_init = 3 if k > 40 else 5

            params[f'l{i + 1}_center'].set(value=center, min=rs.min(), max=rs.max())
            params[f'l{i + 1}_amplitude'].set(value=amp * amp_factor, min=0, max=amp * 20)
            params[f'l{i + 1}_sigma'].set(value=sigma_init, min=0.5, max=20)

        # Adicionar constante
        constant = ConstantModel(prefix='c_')
        model = constant + model
        params.update(constant.make_params())
        params['c_c'].set(value=np.mean(intensity) * 0.1, min=0)

        print(f"Ajuste com {k} Lorentzianas...")

        # Ajuste
        result = model.fit(data=intensity, params=params, x=rs,
                           method='leastsq',
                           max_nfev=5000,
                           nan_policy='omit')

        # Extrair picos do ajuste atual
        picos_ajuste_atual = []
        alturas_ajuste_atual = []

        for param_name in result.params:
            if '_center' in param_name:
                center = result.params[param_name].value
                # Encontrar altura no ajuste
                idx = np.argmin(np.abs(rs - center))
                altura = result.best_fit[idx]
                picos_ajuste_atual.append(center)
                alturas_ajuste_atual.append(altura)

        # Mostrar gráfico do ajuste
        comps = result.eval_components()

        largura, altura = obter_tamanho_tela()
        plt.figure(figsize=(largura * 0.9, altura * 0.85))

        # Subplot 1: Baseline e dados corrigidos
        plt.subplot(2, 1, 1)
        plt.plot(rs, intensity_original, 'gray', label='Dados Originais', linewidth=1, alpha=0.6)
        plt.plot(rs, baseline_final, 'orange', label='Baseline', linewidth=2, linestyle='--')
        plt.plot(rs, intensity_corrigida, 'blue', label='Dados Corrigidos', linewidth=1, alpha=0.8)
        plt.xlabel(r'Raman shift (cm$^{-1}$)')
        plt.ylabel(r'Intensity (a.u.)')
        plt.legend()
        plt.title('Correção de Baseline')
        plt.grid(True, alpha=0.3)

        # Subplot 2: Resultado do ajuste
        plt.subplot(2, 1, 2)
        plt.plot(rs, intensity, 'k-', label='Dados Suavizados', linewidth=1, alpha=0.7)
        plt.plot(rs, result.best_fit, 'r-', label='Ajuste', linewidth=2)

        # Mostrar apenas os picos do ajuste atual
        if picos_ajuste_atual:
            plt.scatter(picos_ajuste_atual, alturas_ajuste_atual, color='green', s=60,
                        label=f'Picos do Ajuste', zorder=5,
                        edgecolors='white', linewidth=1.5)

        # Plot componentes apenas se k for razoável
        if k <= 30:
            for name, comp in comps.items():
                if name != 'c_':
                    plt.plot(rs, comp, '--', alpha=0.3, linewidth=1)

        plt.xlabel(r'Raman shift (cm$^{-1}$)')
        plt.ylabel(r'Intensity (a.u.)')
        plt.legend()
        plt.title(f'Ajuste com {k} Lorentzianas')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return result

    except Exception as e:
        print(f"Erro no processamento: {e}")
        import traceback
        traceback.print_exc()
        return None

# Processa um espectro indiviualmente
def processar_espectro_individual_completo(arquivo):
    print(f"\nANÁLISE INDIVIDUAL: {os.path.basename(arquivo)}")

    # Declarar variáveis no escopo correto
    rs = None
    intensity_original = None
    intensity_corrigida = None
    baseline_final = None
    metodo = None

    try:
        # Carregar dados
        data = np.loadtxt(arquivo)
        x = data[:, 0]
        y = data[:, 1]

        # Encontrar onde o sinal começa (primeiro ponto com intensidade significativa)
        threshold = np.max(y) * 0.01  # 1% da intensidade máxima
        signal_indices = np.where(y > threshold)[0]

        if len(signal_indices) == 0:
            print("Nenhum sinal significativo detectado!")
            return None

        x_min = x[signal_indices[0]]
        x_max = x[signal_indices[-1]]

        # Aplicar limites com margem de segurança
        margin = (x_max - x_min) * 0.05  # 5% de margem
        x_min = max(x[0], x_min - margin)
        x_max = min(x[-1], x_max + margin)

        mask = (x >= x_min) & (x <= x_max)
        rs = x[mask]
        intensity_original = y[mask]

        if len(rs) == 0:
            print("Nenhum ponto encontrado no intervalo especificado!")
            return None

        print(f"Processando {len(rs)} pontos no intervalo {rs[0]:.1f} - {rs[-1]:.1f} cm⁻¹")

        # Testar múltiplas baselines
        print("\n=== TESTANDO DIFERENTES MÉTODOS DE BASELINE ===")

        # Método 1: Peakutils com deg moderado
        try:
            baseline1 = pk.baseline(intensity_original, deg=3, max_it=1000, tol=1e-10)
            intensity_corr1 = intensity_original - baseline1
            print("Peakutils - OK")
        except Exception as e:
            baseline1 = np.zeros_like(intensity_original)
            intensity_corr1 = intensity_original
            print(f"Peakutils falhou: {e}")

        # Método 2: Baseline por quantil
        try:
            baseline2 = baseline_quantil_otimizada(intensity_original)
            intensity_corr2 = intensity_original - baseline2
            print("Quantil - OK")
        except Exception as e:
            baseline2 = np.zeros_like(intensity_original)
            intensity_corr2 = intensity_original
            print(f"Quantil falhou: {e}")

        # Método 3: SNIP
        try:
            inten_reshaped = intensity_original.reshape(1, -1)
            baseline3 = bs.generate_baseline(inten_reshaped, mode='SNIP')[0]
            intensity_corr3 = intensity_original - baseline3
            print("SNIP - OK")
        except Exception as e:
            baseline3 = np.zeros_like(intensity_original)
            intensity_corr3 = intensity_original
            print(f"SNIP falhou: {e}")

        # Comparação visual das baselines
        largura, altura = obter_tamanho_tela()
        plt.figure(figsize=(largura * 0.9, altura * 0.85))

        # Gráfico 1: Todas as baselines
        plt.subplot(2, 1, 1)
        plt.plot(rs, intensity_original, 'black', label='Dados Originais', linewidth=1.5, alpha=0.8)
        plt.plot(rs, baseline1, 'blue', label='Peakutils', linewidth=2, alpha=0.8)
        plt.plot(rs, baseline2, 'green', label='Quantil', linewidth=2, alpha=0.8)
        plt.plot(rs, baseline3, 'orange', label='SNIP', linewidth=2, alpha=0.8)
        plt.xlabel('Raman Shift (cm⁻¹)')
        plt.ylabel('Intensity (a.u.)')
        plt.legend()
        plt.title('Comparação de Métodos de Baseline')
        plt.grid(True, alpha=0.3)

        # Gráfico 2: Dados corrigidos
        plt.subplot(2, 1, 2)
        plt.plot(rs, intensity_corr1, 'blue', label='Corrigido - Peakutils', linewidth=1, alpha=0.8)
        plt.plot(rs, intensity_corr2, 'green', label='Corrigido - Quantil', linewidth=1, alpha=0.8)
        plt.plot(rs, intensity_corr3, 'orange', label='Corrigido - SNIP', linewidth=1, alpha=0.8)
        plt.xlabel('Raman Shift (cm⁻¹)')
        plt.ylabel('Intensity Corrigida (a.u.)')
        plt.legend()
        plt.title('Dados Após Correção de Baseline')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Seleção de melhor baseline
        print("\n=== SELECIONE A MELHOR BASELINE ===")
        print("As baselines disponíveis são:")
        print("1 - Peakutils")
        print("2 - Quantil")
        print("3 - SNIP")

        escolha = input("\nDigite o número do método (1-3): ").strip()

        if escolha == "1":
            baseline_final = baseline1
            intensity_corrigida = intensity_corr1
            metodo = "Peakutils"
        elif escolha == "2":
            baseline_final = baseline2
            intensity_corrigida = intensity_corr2
            metodo = "Quantil"
        elif escolha == "3":
            baseline_final = baseline3
            intensity_corrigida = intensity_corr3
            metodo = "SNIP"
        else:
            baseline_final = baseline1
            intensity_corrigida = intensity_corr1
            metodo = "Peakutils - padrão"

    except Exception as e:
        print(f"Erro no carregamento ou baseline: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Verificar se as variáveis foram definidas
    if rs is None or intensity_corrigida is None or baseline_final is None:
        print("Variáveis não foram inicializadas corretamente.")
        return None

    # Sistema iterativo
    # Parâmetros iniciais para detecção de picos
    parametros_picos = {
        'height': 0.0,
        'distance': 4,
        'prominence': 0.3,
        'width': 4,
        'wlen': 20
    }

    result_final = None
    metodo_baseline_final = metodo
    resultado_anterior = None
    picos_manuais_atuais = None

    while True:
        # Decidir qual tipo de processamento fazer
        if picos_manuais_atuais is not None:
            result = processar_com_picos_manuais(rs, intensity_corrigida, intensity_original,
                                                 baseline_final, parametros_picos, metodo, picos_manuais_atuais)
            picos_manuais_atuais = None  # Resetar após processar

        elif resultado_anterior is not None:
            # Usar o resultado anterior como referência
            print("Usando ajuste anterior como referência...")
            picos_anteriores = []
            for param_name in resultado_anterior.params:
                if '_center' in param_name:
                    picos_anteriores.append(resultado_anterior.params[param_name].value)

            result = processar_com_picos_manuais(rs, intensity_corrigida, intensity_original,
                                                 baseline_final, parametros_picos, metodo, picos_anteriores)
        else:
            # Primeira vez - processamento automático
            result = processar_com_parametros(rs, intensity_corrigida, intensity_original,
                                              baseline_final, parametros_picos, metodo)

        # Verificar resultado do ajuste
        if result is None:
            print("Nenhum pico detectado ou falha no ajuste.")
            print("O que você gostaria de fazer?")
            print("1. Selecionar picos manualmente")
            print("2. Alterar método de baseline")
            print("3. Cancelar análise")

            escolha_falha = input("\nDigite sua escolha (1-3): ").strip()

            if escolha_falha == "1":
                picos_base = []
                if resultado_anterior is not None:
                    for param_name in resultado_anterior.params:
                        if '_center' in param_name:
                            picos_base.append(resultado_anterior.params[param_name].value)

                print("Modo SELEÇÃO MANUAL")
                picos_manuais = selecionar_picos_manualmente(rs, intensity_corrigida, picos_base, resultado_anterior)

                if picos_manuais:
                    picos_manuais_atuais = picos_manuais  # Salvar para processar na próxima iteração
                    resultado_anterior = None  # Forçar novo ajuste
                continue
            elif escolha_falha == "2":
                print("Reiniciando com nova baseline...")
                return None
            elif escolha_falha == "3":
                return None
            else:
                print("Opção inválida. Tente novamente.")
                continue
        else:
            # Ajuste bem-sucedido
            resultado_anterior = result

            print("\n===AJUSTE CONCLUÍDO!===")
            print("O que você gostaria de fazer?")
            print("1. Aceitar este ajuste e SALVAR")
            print("2. Refinar picos manualmente")
            print("3. Recomeçar do zero")
            print("4. Cancelar análise")

            escolha = input("\nDigite sua escolha (1-4): ").strip()

            if escolha == "1":
                result_final = result
                break
            elif escolha == "2":
                picos_base = []
                for param_name in result.params:
                    if '_center' in param_name:
                        picos_base.append(result.params[param_name].value)

                picos_manuais = selecionar_picos_manualmente(rs, intensity_corrigida, picos_base, result)

                if picos_manuais:
                    picos_manuais_atuais = picos_manuais  # Salvar para processar na próxima iteração
                    resultado_anterior = None  # Forçar novo ajuste
                continue
            elif escolha == "3":
                resultado_anterior = None
                picos_manuais_atuais = None
                continue
            elif escolha == "4":
                return None
            else:
                print("Opção inválida. Tente novamente.")
                continue

    # Salvar resultados
    if result_final is not None:
        print("\n===PRONTO PARA SALVAR RESULTADOS!===")

        # Criar janela para selecionar pasta
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        pasta_destino = filedialog.askdirectory(title="Selecione pasta para salvar os resultados")
        root.destroy()

        if pasta_destino:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_base = os.path.splitext(os.path.basename(arquivo))[0]

            # Salvar parâmetros dos picos
            df_parametros = extrair_por_pico_linha_formatada(result_final.params, casas_decimais=2, incluir_erros=False)
            caminho_parametros = os.path.join(pasta_destino, f"Parâmetros_{nome_base}_{timestamp}.csv")
            df_parametros.to_csv(caminho_parametros, index=False, sep=';', encoding='utf-8-sig')

            print(f"\n===ANÁLISE INDIVIDUAL CONCLUÍDA!===")
            print(f"Pasta: {pasta_destino}")
            print(f"Parâmetros salvos: {os.path.basename(caminho_parametros)}")
            print(f"Chi-quadrado: {result_final.chisqr:.2f}")

            return result_final
        else:
            print("Nenhuma pasta selecionada. Resultados não salvos.")
            return None
    else:
        print("Nenhum resultado para salvar.")
        return None

# Processa um espectro individual de forma automática
def processar_espectro_automatico(arquivo):
    print(f"Processando {os.path.basename(arquivo)}...")
    try:
        # Carregar dados
        data = np.loadtxt(arquivo)
        x = data[:, 0]
        y = data[:, 1]

        # Detectar automaticamente a região do espectro
        threshold = np.max(y) * 0.01  # 1% da intensidade máxima
        signal_indices = np.where(y > threshold)[0]

        if len(signal_indices) == 0:
            raise ValueError("Nenhum sinal significativo detectado no espectro")

        auto_x_min = x[signal_indices[0]]
        auto_x_max = x[signal_indices[-1]]

        # Aplicar limites com margem de segurança
        margin = (auto_x_max - auto_x_min) * 0.05  # 5% de margem
        auto_x_min = max(x[0], auto_x_min - margin)
        auto_x_max = min(x[-1], auto_x_max + margin)

        mask = (x >= auto_x_min) & (x <= auto_x_max)
        rs = x[mask]
        intensity_original = y[mask]

        print(f"Região automática: {auto_x_min:.1f} - {auto_x_max:.1f} cm⁻¹")

        if len(rs) == 0:
            raise ValueError(f"Nenhum ponto com sinal significativo detectado no espectro")

        print(f"{len(rs)} pontos no intervalo {rs[0]:.1f} - {rs[-1]:.1f} cm⁻¹")

        # Baseline automática
        try:
            baseline_final = pk.baseline(intensity_original, deg=3, max_it=1000, tol=1e-10)
            intensity_corrigida = intensity_original - baseline_final
            print("Baseline: Peakutils")
        except Exception as e:
            # Fallback para baseline por quantil
            baseline_final = baseline_quantil_otimizada(intensity_original)
            intensity_corrigida = intensity_original - baseline_final
            print("Baseline: Quantil")

        # Suavização
        inten = intensity_corrigida.reshape(1, -1)
        intensity = smo.smoothing(inten, mode='sav_gol')[0]

        # Detecção de picos
        peaks, props = find_peaks(
            intensity,
            height=0.0,
            distance=4,
            prominence=0.3,
            width=4,
            wlen=20
        )

        peak_positions = rs[peaks]
        peak_heights = intensity[peaks]

        print(f"{len(peaks)} picos detectados")

        k = min(60, len(peak_positions))  # Número de Lorentzianas

        if k == 0:
            raise ValueError("Nenhum pico detectado para ajuste")

        peak_data = np.column_stack((peak_positions, peak_heights))

        # Clusterização para centros iniciais
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=20, max_iter=500).fit(peak_data)
        centers = sorted(kmeans.cluster_centers_, key=lambda x: x[0])

        model = None
        params = None

        # Configurar modelo
        for i, (center, amp) in enumerate(centers):
            lorentz = LorentzianModel(prefix=f'l{i + 1}_')
            if model is None:
                model = lorentz
                params = lorentz.make_params()
            else:
                model += lorentz
                params.update(lorentz.make_params())

            amp_factor = 5 if k > 40 else 10
            sigma_init = 3 if k > 40 else 5

            params[f'l{i + 1}_center'].set(value=center, min=rs.min(), max=rs.max())
            params[f'l{i + 1}_amplitude'].set(value=amp * amp_factor, min=0, max=amp * 20)
            params[f'l{i + 1}_sigma'].set(value=sigma_init, min=0.5, max=20)

        # Adicionar constante
        constant = ConstantModel(prefix='c_')
        model = constant + model
        params.update(constant.make_params())
        params['c_c'].set(value=np.mean(intensity) * 0.1, min=0)

        print(f"Ajuste com {k} Lorentzianas...")

        # Ajuste
        result = model.fit(data=intensity, params=params, x=rs,
                           method='leastsq',
                           max_nfev=5000,
                           nan_policy='omit')

        return result, rs, intensity_original

    except Exception as e:
        print(f"Erro no processamento: {e}")
        raise e

# Processar um espectro individual com parâmetros de detecção personalizados
def processar_espectro_automatico_com_parametros(arquivo, parametros_picos):

    print(f"Processando {os.path.basename(arquivo)}...")

    try:
        # Carregar dados
        data = np.loadtxt(arquivo)
        x = data[:, 0]
        y = data[:, 1]

        # Detectar automaticamente a região do espectro
        threshold = np.max(y) * 0.01  # 1% da intensidade máxima
        signal_indices = np.where(y > threshold)[0]

        if len(signal_indices) == 0:
            raise ValueError("Nenhum sinal significativo detectado no espectro")

        auto_x_min = x[signal_indices[0]]
        auto_x_max = x[signal_indices[-1]]

        # Aplicar limites com margem de segurança
        margin = (auto_x_max - auto_x_min) * 0.05  # 5% de margem
        auto_x_min = max(x[0], auto_x_min - margin)
        auto_x_max = min(x[-1], auto_x_max + margin)

        mask = (x >= auto_x_min) & (x <= auto_x_max)
        rs = x[mask]
        intensity_original = y[mask]

        print(f"Região automática: {auto_x_min:.1f} - {auto_x_max:.1f} cm⁻¹")

        if len(rs) == 0:
            raise ValueError("Nenhum ponto com sinal significativo detectado no espectro")

        print(f"{len(rs)} pontos no intervalo {rs[0]:.1f} - {rs[-1]:.1f} cm⁻¹")

        # Baseline automática
        try:
            baseline_final = pk.baseline(intensity_original, deg=3, max_it=1000, tol=1e-10)
            intensity_corrigida = intensity_original - baseline_final
            print("Baseline: Peakutils")
        except Exception as e:
            # Fallback para baseline por quantis
            baseline_final = baseline_quantil_otimizada(intensity_original)
            intensity_corrigida = intensity_original - baseline_final
            print("Baseline: Quantil")

        # Suavização
        inten = intensity_corrigida.reshape(1, -1)
        intensity = smo.smoothing(inten, mode='sav_gol')[0]

        peaks, props = find_peaks(
            intensity,
            height=parametros_picos['height'],
            distance=parametros_picos['distance'],
            prominence=parametros_picos['prominence'],
            width=parametros_picos['width'],
            wlen=parametros_picos['wlen']
        )

        peak_positions = rs[peaks]
        peak_heights = intensity[peaks]

        print(f"{len(peaks)} picos detectados")

        k = min(60, len(peak_positions))  # Número de Lorentzianas

        if k == 0:
            raise ValueError("Nenhum pico detectado para ajuste")

        peak_data = np.column_stack((peak_positions, peak_heights))

        # Clusterização para centros iniciais
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=20, max_iter=500).fit(peak_data)
        centers = sorted(kmeans.cluster_centers_, key=lambda x: x[0])

        model = None
        params = None

        # Configurar modelo
        for i, (center, amp) in enumerate(centers):
            lorentz = LorentzianModel(prefix=f'l{i + 1}_')
            if model is None:
                model = lorentz
                params = lorentz.make_params()
            else:
                model += lorentz
                params.update(lorentz.make_params())

            amp_factor = 5 if k > 40 else 10
            sigma_init = 3 if k > 40 else 5

            params[f'l{i + 1}_center'].set(value=center, min=rs.min(), max=rs.max())
            params[f'l{i + 1}_amplitude'].set(value=amp * amp_factor, min=0, max=amp * 20)
            params[f'l{i + 1}_sigma'].set(value=sigma_init, min=0.5, max=20)

        # Adicionar constante
        constant = ConstantModel(prefix='c_')
        model = constant + model
        params.update(constant.make_params())
        params['c_c'].set(value=np.mean(intensity) * 0.1, min=0)

        print(f"Ajuste com {k} Lorentzianas...")

        # Ajuste
        result = model.fit(data=intensity, params=params, x=rs,
                           method='leastsq',
                           max_nfev=5000,
                           nan_policy='omit')

        return result, rs, intensity_original

    except Exception as e:
        print(f"Erro no processamento: {e}")
        raise e

# Processar um espectro individual para análise comparativa (com escolha de baseline)
def processar_espectro_individual_para_comparacao(arquivo):

    print(f"\nPROCESSANDO REFERÊNCIA: {os.path.basename(arquivo)}")

    try:
        # Carregar dados
        data = np.loadtxt(arquivo)
        x = data[:, 0]
        y = data[:, 1]

        # Detectar automaticamente a região do espectro
        print("Detectando automaticamente a região do espectro...")
        threshold = np.max(y) * 0.01  # 1% da intensidade máxima
        signal_indices = np.where(y > threshold)[0]

        if len(signal_indices) == 0:
            print("Nenhum sinal significativo detectado!")
            return None

        x_min = x[signal_indices[0]]
        x_max = x[signal_indices[-1]]

        # Aplicar limites com margem de segurança
        margin = (x_max - x_min) * 0.05  # 5% de margem
        x_min = max(x[0], x_min - margin)
        x_max = min(x[-1], x_max + margin)

        mask = (x >= x_min) & (x <= x_max)
        rs = x[mask]
        intensity_original = y[mask]

        print(f"Região detectada: {x_min:.1f} - {x_max:.1f} cm⁻¹")

        if len(rs) == 0:
            print("Nenhum ponto encontrado no intervalo especificado!")
            return None

        print(f"Processando {len(rs)} pontos no intervalo {rs[0]:.1f} - {rs[-1]:.1f} cm⁻¹")

        # TESTAR MÚLTIPLAS BASELINES (igual à análise individual)
        print("\n===TESTANDO DIFERENTES MÉTODOS DE BASELINE===")

        # Método 1: Peakutils com deg moderado
        try:
            baseline1 = pk.baseline(intensity_original, deg=3, max_it=1000, tol=1e-10)
            intensity_corr1 = intensity_original - baseline1
            print("Peakutils - OK")
        except Exception as e:
            baseline1 = np.zeros_like(intensity_original)
            intensity_corr1 = intensity_original
            print(f"Peakutils falhou: {e}")

        # Método 2: Baseline por quantil
        try:
            baseline2 = baseline_quantil_otimizada(intensity_original)
            intensity_corr2 = intensity_original - baseline2
            print("Quantil - OK")
        except Exception as e:
            baseline2 = np.zeros_like(intensity_original)
            intensity_corr2 = intensity_original
            print(f"Quantil falhou: {e}")

        # Método 3: SNIP
        try:
            inten_reshaped = intensity_original.reshape(1, -1)
            baseline3 = bs.generate_baseline(inten_reshaped, mode='SNIP')[0]
            intensity_corr3 = intensity_original - baseline3
            print("SNIP - OK")
        except Exception as e:
            baseline3 = np.zeros_like(intensity_original)
            intensity_corr3 = intensity_original
            print(f"SNIP falhou: {e}")

        # Comparação visual das baselines
        largura, altura = obter_tamanho_tela()
        plt.figure(figsize=(largura * 0.9, altura * 0.85))

        # Gráfico 1: Todas as baselines
        plt.subplot(2, 1, 1)
        plt.plot(rs, intensity_original, 'black', label='Dados Originais', linewidth=1.5, alpha=0.8)
        plt.plot(rs, baseline1, 'blue', label='Peakutils (deg=3)', linewidth=2, alpha=0.8)
        plt.plot(rs, baseline2, 'green', label='Quantil (window=151)', linewidth=2, alpha=0.8)
        plt.plot(rs, baseline3, 'orange', label='SNIP', linewidth=2, alpha=0.8)
        plt.xlabel('Raman Shift (cm⁻¹)')
        plt.ylabel('Intensity (a.u.)')
        plt.legend()
        plt.title('Comparação de Métodos de Baseline - ESPECTRO DE REFERÊNCIA')
        plt.grid(True, alpha=0.3)

        # Gráfico 2: Dados corrigidos
        plt.subplot(2, 1, 2)
        plt.plot(rs, intensity_corr1, 'blue', label='Corrigido - Peakutils', linewidth=1, alpha=0.8)
        plt.plot(rs, intensity_corr2, 'green', label='Corrigido - Quantis', linewidth=1, alpha=0.8)
        plt.plot(rs, intensity_corr3, 'orange', label='Corrigido - SNIP', linewidth=1, alpha=0.8)
        plt.xlabel('Raman Shift (cm⁻¹)')
        plt.ylabel('Intensity Corrigida (a.u.)')
        plt.legend()
        plt.title('Dados Após Correção de Baseline - ESPECTRO DE REFERÊNCIA')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Seleção da melhor baseline
        print("\n=== SELECIONE A MELHOR BASELINE PARA TODOS OS ESPECTROS ===")
        print("As baselines disponíveis são:")
        print("1 - Peakutils")
        print("2 - Quantil")
        print("3 - SNIP")

        escolha = input("\nDigite o número do método (1-3): ").strip()

        if escolha == "1":
            baseline_final = baseline1
            intensity_corrigida = intensity_corr1
            metodo = "Peakutils"
        elif escolha == "2":
            baseline_final = baseline2
            intensity_corrigida = intensity_corr2
            metodo = "Quantil"
        elif escolha == "3":
            baseline_final = baseline3
            intensity_corrigida = intensity_corr3
            metodo = "SNIP"
        else:
            baseline_final = baseline1
            intensity_corrigida = intensity_corr1
            metodo = "Peakutils - padrão"

        print(f"\nMétodo de baseline selecionado: {metodo}")

        # Retornar informações para processar outros espectros
        return {
            'rs': rs,
            'intensity_original': intensity_original,
            'intensity_corrigida': intensity_corrigida,
            'baseline_final': baseline_final,
            'metodo_baseline': metodo
        }

    except Exception as e:
        print(f"Erro no processamento da referência: {e}")
        import traceback
        traceback.print_exc()
        return None

# Processar um espectro usando uma baseline pré-definida (para espectros não-referência)
def processar_espectro_com_baseline_predefinida(arquivo, metodo_baseline, parametros_picos):
    try:
        # Carregar dados
        data = np.loadtxt(arquivo)
        x = data[:, 0]
        y = data[:, 1]

        # Detectar automaticamente a região do espectro
        threshold = np.max(y) * 0.01
        signal_indices = np.where(y > threshold)[0]

        if len(signal_indices) == 0:
            raise ValueError("Nenhum sinal significativo detectado no espectro")

        auto_x_min = x[signal_indices[0]]
        auto_x_max = x[signal_indices[-1]]

        # Aplicar limites com margem de segurança
        margin = (auto_x_max - auto_x_min) * 0.05
        auto_x_min = max(x[0], auto_x_min - margin)
        auto_x_max = min(x[-1], auto_x_max + margin)

        mask = (x >= auto_x_min) & (x <= auto_x_max)
        rs = x[mask]
        intensity_original = y[mask]

        if len(rs) == 0:
            raise ValueError("Nenhum ponto com sinal significativo detectado no espectro")

        # Aplicar baseline pré-definida
        if metodo_baseline == "Peakutils":
            baseline_final = pk.baseline(intensity_original, deg=3, max_it=1000, tol=1e-10)
        elif metodo_baseline == "Quantil":
            baseline_final = baseline_quantil_otimizada(intensity_original)
        elif metodo_baseline == "SNIP":
            inten_reshaped = intensity_original.reshape(1, -1)
            baseline_final = bs.generate_baseline(inten_reshaped, mode='SNIP')[0]
        else:
            # Fallback para Peakutils
            baseline_final = pk.baseline(intensity_original, deg=3, max_it=1000, tol=1e-10)

        intensity_corrigida = intensity_original - baseline_final

        # Suavização
        inten = intensity_corrigida.reshape(1, -1)
        intensity = smo.smoothing(inten, mode='sav_gol')[0]

        # Detecção de picos COM PARÂMETROS PERSONALIZADOS
        peaks, props = find_peaks(
            intensity,
            height=parametros_picos['height'],
            distance=parametros_picos['distance'],
            prominence=parametros_picos['prominence'],
            width=parametros_picos['width'],
            wlen=parametros_picos['wlen']
        )

        peak_positions = rs[peaks]
        peak_heights = intensity[peaks]

        k = min(60, len(peak_positions))

        if k == 0:
            raise ValueError("Nenhum pico detectado para ajuste")

        peak_data = np.column_stack((peak_positions, peak_heights))

        # Clusterização para centros iniciais
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=20, max_iter=500).fit(peak_data)
        centers = sorted(kmeans.cluster_centers_, key=lambda x: x[0])

        model = None
        params = None

        # Configurar modelo
        for i, (center, amp) in enumerate(centers):
            lorentz = LorentzianModel(prefix=f'l{i + 1}_')
            if model is None:
                model = lorentz
                params = lorentz.make_params()
            else:
                model += lorentz
                params.update(lorentz.make_params())

            amp_factor = 5 if k > 40 else 10
            sigma_init = 3 if k > 40 else 5

            params[f'l{i + 1}_center'].set(value=center, min=rs.min(), max=rs.max())
            params[f'l{i + 1}_amplitude'].set(value=amp * amp_factor, min=0, max=amp * 20)
            params[f'l{i + 1}_sigma'].set(value=sigma_init, min=0.5, max=20)

        # Adicionar constante
        constant = ConstantModel(prefix='c_')
        model = constant + model
        params.update(constant.make_params())
        params['c_c'].set(value=np.mean(intensity) * 0.1, min=0)

        # Ajuste
        result = model.fit(data=intensity, params=params, x=rs,
                           method='leastsq',
                           max_nfev=5000,
                           nan_policy='omit')

        return result, rs, intensity_original

    except Exception as e:
        print(f"Erro no processamento: {e}")
        raise e


# Processar múltiplos arquivos de espectro e fazer análise comparativa
def processar_multiplos_espectros():
    # Criar janela Tkinter específica para esta função
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # Faz a janela ficar sobre outras

    # Selecionar múltiplos arquivos
    arquivos = filedialog.askopenfilenames(
        title="Selecione os arquivos de espectro para análise comparativa",
        filetypes=[
            ("Arquivos de texto", "*.txt"),
            ("Arquivos CSV", "*.csv"),
            ("Todos os arquivos", "*.*")
        ]
    )

    # Fechar a janela imediatamente após a seleção
    root.destroy()

    if not arquivos:
        print("Nenhum arquivo selecionado.")
        return

    nomes_arquivos = [os.path.basename(arquivo) for arquivo in arquivos]

    # MOSTRAR arquivos selecionados
    print("\nArquivos selecionados:")
    for i, nome in enumerate(nomes_arquivos, 1):
        print(f"  {i}: {nome}")

    # Selecionar referência
    try:
        ref_index = int(input(f"\nDigite o número do espectro de referência (1-{len(nomes_arquivos)}): ")) - 1
        referencia_nome = nomes_arquivos[ref_index]
        arquivo_referencia = arquivos[ref_index]
    except (ValueError, IndexError):
        print("Seleção inválida. Usando primeiro arquivo como referência.")
        referencia_nome = nomes_arquivos[0]
        arquivo_referencia = arquivos[0]

    # Processar espectro de referência com sistema iterativo
    print(f"\nPROCESSANDO ESPECTRO DE REFERÊNCIA: {referencia_nome}")

    # Parâmetros iniciais para referência
    parametros_referencia = {
        'height': 0.0,
        'distance': 4,
        'prominence': 0.3,
        'width': 4,
        'wlen': 20
    }

    # Processar referência com sistema iterativo
    referencia_resultado = None

    # Limpar qualquer estado anterior
    if hasattr(processar_espectro_referencia_completo, 'baseline_ja_selecionada'):
        delattr(processar_espectro_referencia_completo, 'baseline_ja_selecionada')

    while referencia_resultado is None:
        # Processar referência
        referencia_resultado = processar_espectro_referencia_completo(arquivo_referencia, parametros_referencia)

        if referencia_resultado == 'CANCELAR_TUDO':
            return  # Sai completamente da função

        if referencia_resultado is None:
            # Usuário escolheu alterar baseline ou cancelou individualmente
            print("Reiniciando processamento da referência...")
            continue

    # Extrair informações da referência (quando finalmente obtivermos sucesso)
    metodo_baseline = referencia_resultado['metodo_baseline']
    rs_referencia = referencia_resultado['rs']
    result_referencia = referencia_resultado['result']
    parametros_referencia = referencia_resultado['parametros_finais']

    # Inicializar as variáveis de resultados
    rs_por_espectro = {}
    espectros_dados = {}
    todos_results = {}
    parametros_por_espectro = {}

    # Salvar resultados da referência
    espectros_dados[referencia_nome] = extrair_dados_picos(result_referencia.params)
    todos_results[referencia_nome] = result_referencia
    rs_por_espectro[referencia_nome] = rs_referencia
    parametros_por_espectro[referencia_nome] = parametros_referencia.copy()

    # Processamento individual de cada espectro com sistema iterativo
    print("\n===INICIANDO PROCESSAMENTO DOS ESPECTROS RESTANTES===")

    # Processar cada espectro com sistema iterativo individual
    for i, arquivo in enumerate(arquivos):
        nome = os.path.basename(arquivo)

        # Pular a referência (já processada)
        if nome == referencia_nome:
            continue

        print(f"PRÓXIMO ESPECTRO {i + 1}/{len(arquivos)}: {nome}")

        # Usar parâmetros da referência como iniciais
        parametros_picos = parametros_referencia.copy()
        try:
            result, rs, intensity_original = processar_espectro_comparativo_iterativo(
                arquivo, metodo_baseline, parametros_picos
            )

            if result is not None:
                # Aceitar ajuste e salvar resultados
                dados_picos = extrair_dados_picos(result.params)
                espectros_dados[nome] = dados_picos
                todos_results[nome] = result
                rs_por_espectro[nome] = rs
                parametros_por_espectro[nome] = parametros_picos.copy()
            else:
                print(f"Espectro {nome} não foi processado")

        except Exception as e:
            print(f"Erro no processamento de {nome}: {e}")
            print(f"Pulando espectro: {nome}")

    # Fazer análise comparativa apenas se temos pelo menos 2 espectros
    if len(espectros_dados) >= 2:
        print("\n===INICIANDO ANÁLISE COMPARATIVA===")
        print(f"Espectros para comparação: {len(espectros_dados)}")

        # Identificar picos comuns em TODOS os espectros
        picos_comuns_todos = identificar_picos_comuns_todos(espectros_dados)
        print(f"{len(picos_comuns_todos)} picos comuns encontrados em todos os espectros")

        # Salvar resultados
        root = tk.Tk()
        root.withdraw()
        pasta_destino = filedialog.askdirectory(title="Selecione pasta para salvar análise comparativa")
        root.destroy()

        if pasta_destino:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Salvar análise comparativa
            df_comparacao = analisar_comparacao_espectros(espectros_dados, referencia_nome)
            caminho_comparacao = os.path.join(pasta_destino, f"analise_comparativa_{timestamp}.csv")
            df_comparacao.to_csv(caminho_comparacao, index=False, sep=';', encoding='utf-8-sig')

            # Salvar resumo
            resumo = gerar_resumo_analise(df_comparacao)
            caminho_resumo = os.path.join(pasta_destino, f"resumo_analise_{timestamp}.csv")
            resumo.to_csv(caminho_resumo, index=False, sep=';', encoding='utf-8-sig')

            # Salvar parâmetros individuais
            for nome, result in todos_results.items():
                df_individual = extrair_por_pico_linha_formatada(result.params, casas_decimais=2, incluir_erros=True)
                nome_limpo = os.path.splitext(nome)[0]
                caminho_individual = os.path.join(pasta_destino, f"parametros_{nome_limpo}_{timestamp}.csv")
                df_individual.to_csv(caminho_individual, index=False, sep=';', encoding='utf-8-sig')

            # Salvar parâmetros de detecção por espectro
            with open(os.path.join(pasta_destino, f"parametros_por_espectro_{timestamp}.txt"), 'w',
                      encoding='utf-8') as f:
                f.write("PARÂMETROS DE DETECÇÃO POR ESPECTRO:\n")
                f.write("=" * 50 + "\n")
                f.write(f"Baseline utilizada: {metodo_baseline}\n")
                f.write(f"Espectro de referência: {referencia_nome}\n")
                f.write(f"Total de espectros processados: {len(todos_results)}\n")
                f.write(f"Picos comuns encontrados: {len(picos_comuns_todos)}\n\n")

                for nome, parametros in parametros_por_espectro.items():
                    f.write(f"ESPECTRO: {nome}\n")
                    f.write("-" * 30 + "\n")
                    for param, valor in parametros.items():
                        f.write(f"  {param}: {valor}\n")
                    if nome in todos_results:
                        dados_picos = extrair_dados_picos(todos_results[nome].params)
                        f.write(f"  Picos detectados: {len(dados_picos)}\n")
                        f.write(f"  Chi-quadrado: {todos_results[nome].chisqr:.2f}\n")
                    f.write("\n")

            # Criar gráfico dos picos comuns
            if picos_comuns_todos:
                df_picos_comuns = criar_grafico_picos_comuns(espectros_dados, picos_comuns_todos, pasta_destino,
                                                             timestamp, todos_results=todos_results,
                                                             rs_data=rs_por_espectro)
            else:
                print("Não há picos comuns para gerar gráfico")
    else:
        print(f"\nNão há espectros suficientes para análise comparativa")
        print(f"   Espectros processados: {len(espectros_dados)}")
        print(f"   Mínimo necessário: 2 espectros")

# Processar um espectro individual com sistema iterativo e mostrar gráfico
def processar_espectro_individual_iterativo(arquivo, metodo_baseline, parametros_picos):
    try:
        # Carregar dados
        data = np.loadtxt(arquivo)
        x = data[:, 0]
        y = data[:, 1]

        # Detectar automaticamente a região do espectro
        threshold = np.max(y) * 0.01
        signal_indices = np.where(y > threshold)[0]

        if len(signal_indices) == 0:
            raise ValueError("Nenhum sinal significativo detectado no espectro")

        auto_x_min = x[signal_indices[0]]
        auto_x_max = x[signal_indices[-1]]

        # Aplicar limites com margem de segurança
        margin = (auto_x_max - auto_x_min) * 0.05
        auto_x_min = max(x[0], auto_x_min - margin)
        auto_x_max = min(x[-1], auto_x_max + margin)

        mask = (x >= auto_x_min) & (x <= auto_x_max)
        rs = x[mask]
        intensity_original = y[mask]

        if len(rs) == 0:
            raise ValueError("Nenhum ponto com sinal significativo detectado no espectro")

        print(f"{len(rs)} pontos no intervalo {rs[0]:.1f} - {rs[-1]:.1f} cm⁻¹")

        # Aplicar baseline pré-definida
        if metodo_baseline == "Peakutils":
            baseline_final = pk.baseline(intensity_original, deg=3, max_it=1000, tol=1e-10)
        elif metodo_baseline == "Quantil":
            baseline_final = baseline_quantil_otimizada(intensity_original)
        elif metodo_baseline == "SNIP":
            inten_reshaped = intensity_original.reshape(1, -1)
            baseline_final = bs.generate_baseline(inten_reshaped, mode='SNIP')[0]
        else:
            baseline_final = pk.baseline(intensity_original, deg=3, max_it=1000, tol=1e-10)

        intensity_corrigida = intensity_original - baseline_final

        # Suavização
        inten = intensity_corrigida.reshape(1, -1)
        intensity = smo.smoothing(inten, mode='sav_gol')[0]

        peaks, props = find_peaks(
            intensity,
            height=parametros_picos['height'],
            distance=parametros_picos['distance'],
            prominence=parametros_picos['prominence'],
            width=parametros_picos['width'],
            wlen=parametros_picos['wlen']
        )

        peak_positions = rs[peaks]
        peak_heights = intensity[peaks]

        print(f"{len(peaks)} picos detectados")

        k = min(60, len(peak_positions))

        if k == 0:
            raise ValueError("Nenhum pico detectado para ajuste")

        if len(peak_positions) < k:
            print(f"Apenas {len(peak_positions)} picos detectados, reduzindo k para este valor")
            k = len(peak_positions)

        peak_data = np.column_stack((peak_positions, peak_heights))

        # Clusterização para centros iniciais
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=20, max_iter=500).fit(peak_data)
        centers = sorted(kmeans.cluster_centers_, key=lambda x: x[0])

        model = None
        params = None

        # Configurar modelo
        for i, (center, amp) in enumerate(centers):
            lorentz = LorentzianModel(prefix=f'l{i + 1}_')
            if model is None:
                model = lorentz
                params = lorentz.make_params()
            else:
                model += lorentz
                params.update(lorentz.make_params())

            amp_factor = 5 if k > 40 else 10
            sigma_init = 3 if k > 40 else 5

            params[f'l{i + 1}_center'].set(value=center, min=rs.min(), max=rs.max())
            params[f'l{i + 1}_amplitude'].set(value=amp * amp_factor, min=0, max=amp * 20)
            params[f'l{i + 1}_sigma'].set(value=sigma_init, min=0.5, max=20)

        # Adicionar constante
        constant = ConstantModel(prefix='c_')
        model = constant + model
        params.update(constant.make_params())
        params['c_c'].set(value=np.mean(intensity) * 0.1, min=0)

        print(f"Ajuste com {k} Lorentzianas...")

        # Ajuste
        result = model.fit(data=intensity, params=params, x=rs,
                           method='leastsq',
                           max_nfev=5000,
                           nan_policy='omit')

        # Mostrar gráfico do ajuste
        comps = result.eval_components()
        largura, altura = obter_tamanho_tela()
        plt.figure(figsize=(largura * 0.9, altura * 0.85))

        # Subplot 1: Baseline e dados corrigidos
        plt.subplot(2, 1, 1)
        plt.plot(rs, intensity_original, 'gray', label='Dados Originais', linewidth=1, alpha=0.6)
        plt.plot(rs, baseline_final, 'orange', label='Baseline', linewidth=2, linestyle='--')
        plt.plot(rs, intensity_corrigida, 'blue', label='Dados Corrigidos', linewidth=1, alpha=0.8)
        plt.xlabel(r'Raman shift (cm$^{-1}$)')
        plt.ylabel(r'Intensity (a.u.)')
        plt.legend()
        plt.title(f'Correção de Baseline - {os.path.basename(arquivo)}')
        plt.grid(True, alpha=0.3)

        # Subplot 2: Resultado do ajuste
        plt.subplot(2, 1, 2)
        plt.plot(rs, intensity, 'k-', label='Dados Suavizados', linewidth=1, alpha=0.7)
        plt.plot(rs, result.best_fit, 'r-', label='Ajuste', linewidth=2)

        # Marcar picos detectados
        plt.scatter(peak_positions, peak_heights, color='green', s=50,
                    label=f'Picos Detectados ({len(peaks)})', zorder=5)

        # Plot componentes apenas se k for razoável
        if k <= 30:
            for name, comp in comps.items():
                if name != 'c_':
                    plt.plot(rs, comp, '--', alpha=0.3, linewidth=1)

        plt.xlabel(r'Raman shift (cm$^{-1}$)')
        plt.ylabel(r'Intensity (a.u.)')
        plt.legend()
        plt.title(f'Ajuste com {k} Lorentzianas - {os.path.basename(arquivo)}\nχ²: {result.chisqr:.2f}')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return result, rs, intensity_original

    except Exception as e:
        print(f"Erro no processamento: {e}")
        raise e

# Processar o espectro de referência com sistema iterativo completo
def processar_espectro_referencia_completo(arquivo, parametros_picos):
    try:
        # Carregar dados
        data = np.loadtxt(arquivo)
        x = data[:, 0]
        y = data[:, 1]

        # Detectar automaticamente a região do espectro
        threshold = np.max(y) * 0.01
        signal_indices = np.where(y > threshold)[0]

        if len(signal_indices) == 0:
            print("Nenhum sinal significativo detectado!")
            return None

        x_min = x[signal_indices[0]]
        x_max = x[signal_indices[-1]]

        # Aplicar limites com margem de segurança
        margin = (x_max - x_min) * 0.05
        x_min = max(x[0], x_min - margin)
        x_max = min(x[-1], x_max + margin)

        mask = (x >= x_min) & (x <= x_max)
        rs = x[mask]
        intensity_original = y[mask]

        if len(rs) == 0:
            print("Nenhum ponto encontrado no intervalo especificado!")
            return None

        print(f"Processando {len(rs)} pontos no intervalo {rs[0]:.1f} - {rs[-1]:.1f} cm⁻¹")

        # Testar múltiplas baselines
        if not hasattr(processar_espectro_referencia_completo, 'baseline_ja_selecionada'):
            print("\n===TESTANDO DIFERENTES MÉTODOS DE BASELINE===")

            # Método 1: Peakutils com deg moderado
            try:
                baseline1 = pk.baseline(intensity_original, deg=3, max_it=1000, tol=1e-10)
                intensity_corr1 = intensity_original - baseline1
                print("Peakutils - OK")
            except Exception as e:
                baseline1 = np.zeros_like(intensity_original)
                intensity_corr1 = intensity_original
                print(f"Peakutils falhou: {e}")

            # Método 2: Baseline por quantil
            try:
                baseline2 = baseline_quantil_otimizada(intensity_original)
                intensity_corr2 = intensity_original - baseline2
                print("Quantil - OK")
            except Exception as e:
                baseline2 = np.zeros_like(intensity_original)
                intensity_corr2 = intensity_original
                print(f"Quantil falhou: {e}")

            # Método 3: SNIP
            try:
                inten_reshaped = intensity_original.reshape(1, -1)
                baseline3 = bs.generate_baseline(inten_reshaped, mode='SNIP')[0]
                intensity_corr3 = intensity_original - baseline3
                print("SNIP - OK")
            except Exception as e:
                baseline3 = np.zeros_like(intensity_original)
                intensity_corr3 = intensity_original
                print(f"SNIP falhou: {e}")

            # Comparação visual das baselines
            largura, altura = obter_tamanho_tela()
            plt.figure(figsize=(largura * 0.9, altura * 0.85))

            # Gráfico 1: Todas as baselines
            plt.subplot(2, 1, 1)
            plt.plot(rs, intensity_original, 'black', label='Dados Originais', linewidth=1.5, alpha=0.8)
            plt.plot(rs, baseline1, 'blue', label='Peakutils', linewidth=2, alpha=0.8)
            plt.plot(rs, baseline2, 'green', label='Quantil', linewidth=2, alpha=0.8)
            plt.plot(rs, baseline3, 'orange', label='SNIP', linewidth=2, alpha=0.8)
            plt.xlabel('Raman Shift (cm⁻¹)')
            plt.ylabel('Intensity (a.u.)')
            plt.legend()
            plt.title('Comparação de Métodos de Baseline - ESPECTRO DE REFERÊNCIA')
            plt.grid(True, alpha=0.3)

            # Gráfico 2: Dados corrigidos
            plt.subplot(2, 1, 2)
            plt.plot(rs, intensity_corr1, 'blue', label='Corrigido - Peakutils', linewidth=1, alpha=0.8)
            plt.plot(rs, intensity_corr2, 'green', label='Corrigido - Quantil', linewidth=1, alpha=0.8)
            plt.plot(rs, intensity_corr3, 'orange', label='Corrigido - SNIP', linewidth=1, alpha=0.8)
            plt.xlabel('Raman Shift (cm⁻¹)')
            plt.ylabel('Intensity Corrigida (a.u.)')
            plt.legend()
            plt.title('Dados Após Correção de Baseline - ESPECTRO DE REFERÊNCIA')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            # Seleção da melhor baselineE
            print("\n===SELECIONE A MELHOR BASELINE PARA TODOS OS ESPECTROS===")
            print("As baselines disponíveis são:")
            print("1 - Peakutils")
            print("2 - Quantil")
            print("3 - SNIP")

            escolha = input("\nDigite o número do método (1-3): ").strip()

            if escolha == "1":
                baseline_final = baseline1
                intensity_corrigida = intensity_corr1
                metodo = "Peakutils"
            elif escolha == "2":
                baseline_final = baseline2
                intensity_corrigida = intensity_corr2
                metodo = "Quantil"
            elif escolha == "3":
                baseline_final = baseline3
                intensity_corrigida = intensity_corr3
                metodo = "SNIP"
            else:
                baseline_final = baseline1
                intensity_corrigida = intensity_corr1
                metodo = "Peakutils - padrão"

            # Marcar que a baseline já foi selecionada
            processar_espectro_referencia_completo.baseline_ja_selecionada = True
            processar_espectro_referencia_completo.metodo_baseline = metodo
            processar_espectro_referencia_completo.baseline_final = baseline_final
            processar_espectro_referencia_completo.intensity_corrigida = intensity_corrigida
            processar_espectro_referencia_completo.rs = rs
            processar_espectro_referencia_completo.intensity_original = intensity_original

        else:
            # Já selecionou baseline antes, usar os valores salvos
            metodo = processar_espectro_referencia_completo.metodo_baseline
            baseline_final = processar_espectro_referencia_completo.baseline_final
            intensity_corrigida = processar_espectro_referencia_completo.intensity_corrigida
            rs = processar_espectro_referencia_completo.rs
            intensity_original = processar_espectro_referencia_completo.intensity_original
            print(f"Usando baseline pré-selecionada: {metodo}")

        # Sistema iterativo completo para referência
        result_final = None
        resultado_anterior = None
        picos_manuais_atuais = None

        while True:
            # Decidir qual tipo de processamento fazer
            if picos_manuais_atuais is not None:
                # Tem picos manuais para processar
                result = processar_com_picos_manuais(rs, intensity_corrigida, intensity_original,
                                                     baseline_final, parametros_picos, metodo, picos_manuais_atuais)
                picos_manuais_atuais = None  # Resetar após processar

            elif resultado_anterior is not None:
                # Usar o resultado anterior como referência
                print("Usando ajuste anterior como referência...")
                picos_anteriores = []
                for param_name in resultado_anterior.params:
                    if '_center' in param_name:
                        picos_anteriores.append(resultado_anterior.params[param_name].value)

                result = processar_com_picos_manuais(rs, intensity_corrigida, intensity_original,
                                                     baseline_final, parametros_picos, metodo, picos_anteriores)
            else:
                # Primeira vez - processamento automático
                result = processar_com_parametros(rs, intensity_corrigida, intensity_original,
                                                  baseline_final, parametros_picos, metodo)

            # Verificar resultado do ajuste
            if result is None:
                print("Nenhum pico detectado ou falha no ajuste.")
                print("O que você gostaria de fazer?")
                print("1. Selecionar picos manualmente")
                print("2. Alterar método de baseline")
                print("3. Cancelar análise")

                escolha_falha = input("\nDigite sua escolha (1-3): ").strip()

                if escolha_falha == "1":
                    picos_base = []
                    if resultado_anterior is not None:
                        for param_name in resultado_anterior.params:
                            if '_center' in param_name:
                                picos_base.append(resultado_anterior.params[param_name].value)

                    print("Modo SELEÇÃO MANUAL")
                    picos_manuais = selecionar_picos_manualmente(rs, intensity_corrigida, picos_base, resultado_anterior)

                    if picos_manuais:
                        picos_manuais_atuais = picos_manuais  # Salvar para processar na próxima iteração
                        resultado_anterior = None  # Forçar novo ajuste
                    continue
                elif escolha_falha == "2":
                    print("Reiniciando com nova baseline...")
                    # Limpar a flag para forçar nova seleção de baseline
                    if hasattr(processar_espectro_referencia_completo, 'baseline_ja_selecionada'):
                        delattr(processar_espectro_referencia_completo, 'baseline_ja_selecionada')
                    return None
                elif escolha_falha == "3":
                    print("Análise cancelada.")
                    # Retornar um valor especial para indicar cancelamento completo
                    return 'CANCELAR_TUDO'
                else:
                    print("Opção inválida. Tente novamente.")
                    continue
            else:
                # Ajuste bem-sucedido
                resultado_anterior = result

                # Perguntar se o usuário está satisfeito
                print(f"\n===AJUSTE CONCLUÍDO PARA REFERÊNCIA===")
                print("O que você gostaria de fazer?")
                print("1. Aceitar este ajuste como referência")
                print("2. Refinar picos manualmente")
                print("3. Recomeçar do zero")
                print("4. Alterar método de baseline")
                print("5. Cancelar análise")

                escolha = input("\nDigite sua escolha (1-5): ").strip()

                if escolha == "1":
                    result_final = result
                    break
                elif escolha == "2":
                    picos_base = []
                    for param_name in result.params:
                        if '_center' in param_name:
                            picos_base.append(result.params[param_name].value)

                    picos_manuais = selecionar_picos_manualmente(rs, intensity_corrigida, picos_base, result)

                    if picos_manuais:
                        picos_manuais_atuais = picos_manuais  # Salvar para processar na próxima iteração
                        resultado_anterior = None  # Forçar novo ajuste
                    continue
                elif escolha == "3":
                    resultado_anterior = None
                    picos_manuais_atuais = None
                    continue
                elif escolha == "4":
                    # Limpar a flag para forçar nova seleção de baseline
                    if hasattr(processar_espectro_referencia_completo, 'baseline_ja_selecionada'):
                        delattr(processar_espectro_referencia_completo, 'baseline_ja_selecionada')
                    return None
                elif escolha == "5":
                    print("Análise cancelada.")
                    # Retornar um valor especial para indicar cancelamento completo
                    return 'CANCELAR_TUDO'
                else:
                    print("Opção inválida. Tente novamente.")
                    continue

        return {
            'metodo_baseline': metodo,
            'rs': rs,
            'result': result_final,
            'parametros_finais': parametros_picos.copy()
        }

    except Exception as e:
        print(f"Erro no processamento da referência: {e}")
        import traceback
        traceback.print_exc()
        # Limpar flags em caso de erro
        if hasattr(processar_espectro_referencia_completo, 'baseline_ja_selecionada'):
            delattr(processar_espectro_referencia_completo, 'baseline_ja_selecionada')
        return None

# Baseline quantil modificada
def baseline_quantil_otimizada(intensity, window=151, quantile=0.15):
    series = pd.Series(intensity)
    return series.rolling(window=window, center=True, min_periods=1).quantile(quantile).values

# Extrair os picos
def extrair_por_pico_linha_formatada(params, casas_decimais=2, incluir_erros=True):
    picos = set()
    for param_name in params:
        if '_' in param_name and param_name != 'c_c':
            pico = param_name.split('_')[0]
            picos.add(pico)

    picos_ordenados = sorted(picos, key=lambda x: int(x[1:]) if x[1:].isdigit() else float('inf'))
    dados_picos = []
    for pico in picos_ordenados:
        linha = {'Pico': pico}
        for tipo in ['center','amplitude','fwhm']:
            param_name = f"{pico}_{tipo}"
            if param_name in params:
                param = params[param_name]
                valor = round(param.value, casas_decimais)
                linha[tipo.capitalize()] = valor
        dados_picos.append(linha)
    return pd.DataFrame(dados_picos)

# Menu principal
def main():
    print("===SEJA BEM-VINDO(A) AO PySAR - ANÁLISE DE ESPECTROS RAMAN===")
    # Criar uma janela raiz principal e escondê-la
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # Esta linha faz a janela ficar sobre outras

    while True:
        print("\nOpções disponíveis:")
        print("1. Análise Individual (1 espectro)")
        print("2. Análise Comparativa (Múltiplos espectros)")
        print("3. Sair")

        escolha = input("\nDigite sua escolha (1-3): ").strip()

        if escolha == "1":
            file_path = filedialog.askopenfilename(
                title="Selecione o arquivo de espectro Raman",
                filetypes=[("Arquivos de texto", "*.txt"), ("Arquivos CSV", "*.csv"), ("Todos os arquivos", "*.*")]
            )

            if file_path:
                processar_espectro_individual_completo(file_path)
            else:
                print("Nenhum arquivo selecionado.")

        elif escolha == "2":
            processar_multiplos_espectros()

        elif escolha == "3":
            print("Saindo do programa...")
            break

        else:
            print("Opção inválida. Por favor, digite 1, 2 ou 3.")

        # Perguntar se quer continuar
        continuar = input("\nDeseja voltar ao menu principal? (s/n): ").strip().lower()
        if continuar not in ['s', 'sim', 'y', 'yes', '']:
            print("Saindo do programa...")
            break
        else:
            # Continuar no menu principal
            continue

    # Destruir a janela raiz ao sair
    root.destroy()

# Execução do programa
if __name__ == "__main__":
    main()
