# Bibliotecas padrão
import os
os.environ['TQDM_DISABLE'] = '1'
import re
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
# Bibliotecas numéricas e científicas
import numpy as np
import pandas as pd
# Bibliotecas de visualização
import matplotlib
import matplotlib.pyplot as plt
# Bibliotecas para análise de espectros
import peakutils as pk
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from lmfit.models import LorentzianModel, ConstantModel
from pyPreprocessing import baseline_correction as bs, smoothing as smo

# =========== CONFIGURAÇÕES INICIAIS ===========
matplotlib.use('TkAgg')
plt.rcParams['figure.max_open_warning'] = 0
parametros_padrao = {'height': 0.00001, 'distance': 4, 'prominence': 0.0001, 'width': 4, 'wlen': 30}
tamanho_tela = (15, 8)
# Variável global para pasta de saída atual
pasta_saida_atual = None

# =========== FUNÇÕES AUXILIARES BÁSICAS ===========
def criar_pasta_analise(pasta_base=None):
    global pasta_saida_atual

    if pasta_base is None:
        # Criar pasta na área de trabalho ou em local padrão
        desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
        pasta_base = os.path.join(desktop, 'RAMAN ANALYSES')

    # Criar pasta base se não existir
    if not os.path.exists(pasta_base):
        os.makedirs(pasta_base)

    # Criar subpasta com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pasta_analise = os.path.join(pasta_base, f"Analise_{timestamp}")

    # Criar subpastas organizadas
    os.makedirs(pasta_analise)
    os.makedirs(os.path.join(pasta_analise, "GRAPHICS"))
    os.makedirs(os.path.join(pasta_analise, "RESULTS"))

    pasta_saida_atual = pasta_analise
    print(f"\nPasta de análise criada: {pasta_analise}")
    return pasta_analise

def salvar_grafico(fig, nome_arquivo, pasta=None, dpi=300, mostrar=True):
    global pasta_saida_atual

    # Determinar pasta de destino de forma mais concisa
    pasta_destino = pasta or pasta_saida_atual or criar_pasta_analise()

    # Garantir extensão .png
    nome_base = os.path.splitext(nome_arquivo)[0]
    nome_final = f"{nome_base}.png"

    # Criar caminho completo
    pasta_graficos = os.path.join(pasta_destino, "GRAPHICS")
    os.makedirs(pasta_graficos, exist_ok=True)
    caminho_completo = os.path.join(pasta_graficos, nome_final)

    # Salvar figura
    fig.savefig(caminho_completo, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Gráfico salvo: {nome_final}")

    # Gerenciar exibição
    if mostrar:
        plt.show(block=True)
    else:
        plt.close(fig)

    return caminho_completo

def ajustar_parametros_para_normalizacao(parametros):
    # Criar cópia segura dos parâmetros padrão
    resultado = parametros_padrao.copy()

    if isinstance(parametros, dict):
        # Validar e converter tipos dos valores do dicionário
        for key, value in parametros.items():
            if key in resultado:
                try:
                    # Converter para o tipo apropriado
                    if key in ['height', 'prominence']:
                        resultado[key] = float(value)
                    elif key in ['distance', 'width', 'wlen']:
                        resultado[key] = int(float(value))  # Parâmetros inteiros
                except (ValueError, TypeError):
                    print(f"Aviso: Valor inválido para parâmetro '{key}': {value}. Usando valor padrão.")

    return resultado

def calcular_area_espectro(x, y):
    # Verificar se os arrays têm o mesmo comprimento
    if len(x) != len(y):
        raise ValueError(f"Comprimentos diferentes: x({len(x)}) != y({len(y)})")

    if len(x) < 2:
        return 0.0

    return np.trapezoid(y, x)

def normalizar_por_area(x, y, salvar_area=False, epsilon=1e-12):
    # Validar inputs
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=np.float64)
    if not isinstance(y, np.ndarray):
        y = np.array(y, dtype=np.float64)

    if len(x) != len(y):
        raise ValueError(f"Arrays devem ter mesmo comprimento: x({len(x)}) != y({len(y)})")

    # Calcular área
    area = calcular_area_espectro(x, y)

    # Verificar se a área é válida para normalização
    if abs(area) < epsilon:
        print(f"Aviso: Área muito pequena ({area:.2e}). Usando normalização pelo máximo.")

        # Alternativa: normalizar pelo valor máximo
        y_max = np.max(np.abs(y))
        if y_max > epsilon:
            y_norm = y / y_max
            area = y_max  # Para fins de registro
        else:
            print("Erro: Espectro com valores todos próximos de zero.")
            y_norm = y.copy()  # Retorna cópia do original
            area = 0.0
    else:
        # Normalização pela área
        y_norm = y / area

    # Garantir que não há valores NaN ou inf
    if np.any(np.isnan(y_norm)) or np.any(np.isinf(y_norm)):
        print("Aviso: Valores NaN ou infinitos na normalização. Substituindo por zeros.")
        y_norm = np.nan_to_num(y_norm, nan=0.0, posinf=0.0, neginf=0.0)

    if salvar_area:
        return y_norm, float(area)
    return y_norm

# ========== FUNÇÕES DE EXTRAÇÃO E ANÁLISE DE DADOS ==========
def extrair_variavel_fisica(nome_arquivo):
    nome = os.path.splitext(os.path.basename(nome_arquivo))[0].upper()

    # Remover sufixos comuns
    padroes_remocao = [
        r'_[A-Z]$',  # _a, _b, etc
        r'_LADO[A-Z]$',  # _LADOA, etc
        r'_PARTE[A-Z]$',  # _PARTEA, etc
        r'_REP[0-9]*$',  # _REP1, _REP01, etc
        r'_SAMPLE[0-9]*$',  # _SAMPLE1, etc
        r'_\d{8}$',  # datas como _20231231
    ]

    for padrao in padroes_remocao:
        nome = re.sub(padrao, '', nome)

    # 1. Pressão - padrões relaxados
    padroes_pressao = [
        r'(\d+[.,]?\d*)\s*GPA?',  # 300GPa, 300G
        r'(\d+[.,]?\d*)\s*G\b',  # 300G (fim de palavra)
        r'(\d+[.,]?\d*)\s*KBAR',  # 10kbar
        r'P\s*(\d+[.,]?\d*)',  # P300
        r'PRESSAO\s*(\d+[.,]?\d*)',  # PRESSÃO 300
        r'PRESS\s*(\d+[.,]?\d*)',  # PRESS 300
        r'(\d+[.,]?\d*)GPA',  # 300GPA (sem espaço)
    ]

    for padrao in padroes_pressao:
        match = re.search(padrao, nome, re.IGNORECASE)
        if match:
            try:
                valor_str = match.group(1).replace(',', '.')
                valor = float(valor_str)

                # Se for kbar, converter para GPa (1 kbar = 0.1 GPa)
                if 'KBAR' in nome.upper():
                    valor = valor * 0.1

                return "pressure", round(valor, 3), "GPa"
            except (ValueError, AttributeError) as e:
                print(f"Aviso: Erro ao converter valor de pressão: {e}")
                continue

    # 2. Temperatura - padrões relaxados
    padroes_temperatura = [
        r'(\d+[.,]?\d*)\s*K\b',  # 300K
        r'(\d+[.,]?\d*)\s*C\b',  # 25C
        r'(\d+[.,]?\d*)\s*°C',  # 25°C
        r'T\s*(\d+[.,]?\d*)',  # T300
        r'TEMP\s*(\d+[.,]?\d*)',  # TEMP 300
        r'TEMPERATURE\s*(\d+[.,]?\d*)',  # TEMPERATURE 300
        r'(\d+[.,]?\d*)[CK]$',  # 300K ou 25C no final
    ]

    for padrao in padroes_temperatura:
        match = re.search(padrao, nome, re.IGNORECASE)
        if match:
            try:
                valor_str = match.group(1).replace(',', '.')
                valor = float(valor_str)

                # Converter C para K
                nome_upper = nome.upper()
                if ('C' in nome_upper or '°C' in nome_upper) and 'K' not in nome_upper:
                    valor = valor + 273.15

                return "temperature", round(valor, 2), "K"
            except (ValueError, AttributeError) as e:
                print(f"Aviso: Erro ao converter valor de temperatura: {e}")
                continue

    # 3. Tentar extrair números simples (último recurso)
    padroes_numeros = [
        r'(\d+[.,]?\d+)$',  # Números no final
        r'(\d+[.,]?\d+)G',  # Números seguidos de G
        r'(\d+[.,]?\d+)K',  # Números seguidos de K
    ]

    for padrao in padroes_numeros:
        match = re.search(padrao, nome)
        if match:
            try:
                valor_str = match.group(1).replace(',', '.')
                valor = float(valor_str)

                # Tentar inferir tipo pelo contexto
                if 'G' in nome or 'P' in nome:
                    return "pressure", round(valor, 3), "GPa"
                elif 'K' in nome or 'C' in nome or 'T' in nome:
                    # Se tem K, assume já está em Kelvin
                    if 'K' in nome.upper():
                        return "temperature", round(valor, 2), "K"
                    # Se tem C, converte
                    elif 'C' in nome.upper():
                        return "temperature", round(valor + 273.15, 2), "K"
                    else:
                        return "temperature", round(valor, 2), "K"
                else:
                    # Sem contexto claro, retorna como índice
                    return "indice", round(valor, 0), ""
            except (ValueError, AttributeError):
                continue

    return None, None, None

def extrair_dados_picos(params):
    dados = {}

    # Verificar tipo de params e obter itens
    if hasattr(params, 'items'):  # Dicionário
        items = params.items()
    elif hasattr(params, '__dict__'):  # Objeto com atributos
        items = vars(params).items()
    elif hasattr(params, 'keys'):  # Objeto similar a dicionário
        items = [(k, params[k]) for k in params.keys()]
    else:
        # Tentar iterar diretamente
        try:
            items = [(k, v) for k, v in params]
        except (TypeError, ValueError):
            print(f"Erro: Tipo não suportado para params: {type(params)}")
            return {}

    # Identificar picos únicos
    picos = set()
    for param_name, param_value in items:
        param_name_str = str(param_name)

        if '_' in param_name_str and param_name_str != 'c_c':
            partes = param_name_str.split('_')
            if len(partes) >= 2:
                pico = partes[0]
                picos.add(pico)

    # Extrair dados para cada pico
    for pico in sorted(picos, key=lambda x: int(x[1:]) if x[1:].isdigit() else float('inf')):
        dados[pico] = {
            'amplitude': 0.0,
            'center': 0.0,
            'sigma': 0.0,
            'fwhm': 0.0,
            'height': 0.0
        }

        # Extrair valores
        for param_name, param_value in items:
            param_name_str = str(param_name)

            if param_name_str.startswith(f'{pico}_'):
                atributo = param_name_str.split('_', 1)[1]

                # Obter valor
                if hasattr(param_value, 'value'):
                    valor = param_value.value
                elif isinstance(param_value, (int, float)):
                    valor = param_value
                else:
                    try:
                        valor = float(param_value)
                    except (ValueError, TypeError):
                        valor = 0.0
                        print(f"Aviso: Não foi possível converter valor para {param_name_str}")

                # Atribuir ao dicionário
                if atributo in dados[pico]:
                    dados[pico][atributo] = float(valor)

                # Calcular FWHM se temos sigma (FWHM = 2 * sigma)
                if atributo == 'sigma' and valor > 0:
                    dados[pico]['fwhm'] = 2.0 * abs(float(valor))

    return dados

def extrair_por_pico_linha_formatada(params, casas_decimais=2):
    # Primeiro extrair dados estruturados
    picos_data = extrair_dados_picos(params)

    if not picos_data:
        return "Nenhum pico encontrado.\n"

    # Ordenar picos numericamente
    def extrair_numero(pico_str):
        match = re.search(r'\d+', pico_str)
        return int(match.group()) if match else float('inf')

    picos_ordenados = sorted(picos_data.keys(), key=extrair_numero)

    # Construir texto formatado
    texto = [
        "PEAKS PARAMETERS",
        "=" * 50,
        ""
    ]

    # 1. Centers (cm⁻¹)
    centers = []
    for pico in picos_ordenados:
        if 'center' in picos_data[pico] and picos_data[pico]['center'] != 0:
            centers.append(f"{picos_data[pico]['center']:.{casas_decimais}f}")

    if centers:
        texto.append(f"Centers (cm⁻¹): {' '.join(centers)}")
        texto.append("")

    # 2. Amplitudes
    amplitudes = []
    for pico in picos_ordenados:
        if 'amplitude' in picos_data[pico] and picos_data[pico]['amplitude'] != 0:
            amplitudes.append(f"{picos_data[pico]['amplitude']:.{casas_decimais}f}")

    if amplitudes:
        texto.append(f"Amplitudes: {' '.join(amplitudes)}")
        texto.append("")

    # 3. FWHMs (cm⁻¹)
    fwhms = []
    for pico in picos_ordenados:
        fwhm_valor = picos_data[pico].get('fwhm', 0.0)
        if fwhm_valor > 0:
            fwhms.append(f"{fwhm_valor:.{casas_decimais}f}")
        elif 'sigma' in picos_data[pico] and picos_data[pico]['sigma'] > 0:
            # Calcular FWHM a partir do sigma se disponível
            fwhm_calculado = 2.0 * picos_data[pico]['sigma']
            fwhms.append(f"{fwhm_calculado:.{casas_decimais}f}")

    if fwhms:
        texto.append(f"FWHMs (cm⁻¹): {' '.join(fwhms)}")
        texto.append("")

    # 5. Chi-quadrado (se disponível)
    try:
        if hasattr(params, 'parent') and hasattr(params.parent, 'chisqr'):
            chisqr = params.parent.chisqr
            texto.append(f"Chi-square: {chisqr:.{casas_decimais}f}")
            texto.append("")
        elif hasattr(params, 'chisqr'):
            texto.append(f"Chi-square: {params.chisqr:.{casas_decimais}f}")
            texto.append("")
    except AttributeError:
        pass  # Chi-quadrado não disponível

    return "\n".join(texto)

# ========== FUNÇÕES AUXILIARES PARA AJUSTE LORENTZIANO ==========
def _preparar_dados_para_ajuste(rs, intensity_corrigida, aplicar_normalizacao=True):
    # Suavização
    inten = intensity_corrigida.reshape(1, -1)
    intensity = smo.smoothing(inten, mode='sav_gol')[0]

    # Normalização
    if aplicar_normalizacao:
        intensity = normalizar_por_area(rs, intensity)

    return intensity

def _detectar_picos_automaticamente(intensity, parametros_deteccao):
    parametros_validos = {
        'height': float(parametros_deteccao.get('height', 0.00001)),
        'distance': int(parametros_deteccao.get('distance', 4)),
        'prominence': float(parametros_deteccao.get('prominence', 0.0001)),
        'width': int(parametros_deteccao.get('width', 4)),
        'wlen': int(parametros_deteccao.get('wlen', 30))
    }

    peaks, props = find_peaks(intensity, **parametros_validos)
    return peaks, props

def _processar_posicoes_picos(rs, intensity, peaks):
    if len(peaks) == 0:
        return [], []

    peak_positions = rs[peaks]
    peak_heights = intensity[peaks]

    print(f"{len(peaks)} picos detectados automaticamente")
    return peak_positions, peak_heights

def _processar_picos_manuais(rs, intensity, picos_manuais):
    if not picos_manuais:
        return [], []

    peak_data = []
    for pico in picos_manuais:
        idx = np.argmin(np.abs(rs - pico))
        altura = intensity[idx]
        peak_data.append([pico, altura])

    print(f"{len(picos_manuais)} picos fornecidos manualmente")
    return np.array(peak_data)

def _agrupar_picos_kmeans(peak_data, k_max=60):
    if len(peak_data) == 0:
        return []

    k = min(k_max, len(peak_data))

    if len(peak_data) < k:
        print(f"Apenas {len(peak_data)} picos disponíveis, reduzindo k para este valor")
        k = len(peak_data)

    if k == 0:
        print("Nenhum pico para agrupar!")
        return []

    if k > k_max:
        print(f"k={k} é muito alto. Limitando a {k_max} Lorentzianas")
        k = k_max

    if k > 1:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=20, max_iter=500).fit(peak_data)
        centers = sorted(kmeans.cluster_centers_, key=lambda x: x[0])
    else:
        centers = peak_data.reshape(-1, 2) if len(peak_data.shape) == 1 else peak_data

    print(f"Agrupados em {k} centros para ajuste")
    return centers

def _criar_modelo_lorentziano(centers, rs, intensity):
    model = None
    params = None

    for i, (center, amp) in enumerate(centers):
        lorentz = LorentzianModel(prefix=f'l{i + 1}_')

        if model is None:
            model = lorentz
            params = lorentz.make_params()
        else:
            model += lorentz
            params.update(lorentz.make_params())

        # Determinar parâmetros iniciais baseados no número de Lorentzianas
        k = len(centers)
        amp_factor = 5 if k > 40 else 10
        sigma_init = 3 if k > 40 else 5

        params[f'l{i + 1}_center'].set(
            value=center,
            min=rs.min(),
            max=rs.max()
        )
        params[f'l{i + 1}_amplitude'].set(
            value=amp * amp_factor,
            min=0,
            max=amp * 20
        )
        params[f'l{i + 1}_sigma'].set(
            value=sigma_init,
            min=0.5,
            max=20
        )

    # Adicionar componente constante
    constant = ConstantModel(prefix='c_')
    model = constant + model
    params.update(constant.make_params())
    params['c_c'].set(value=np.mean(intensity) * 0.1, min=0)

    return model, params

def _executar_ajuste_lorentziano(model, params, rs, intensity):
    print(f"Ajustando com {len(params) // 4} Lorentzianas...")

    result = model.fit(
        data=intensity,
        params=params,
        x=rs,
        method='leastsq',
        max_nfev=5000,
        nan_policy='omit'
    )

    return result

def _extrair_informacoes_ajuste(result, rs):
    picos_ajuste = []
    alturas_ajuste = []

    for param_name in result.params:
        if '_center' in param_name:
            center = result.params[param_name].value
            idx = np.argmin(np.abs(rs - center))
            altura = result.best_fit[idx]
            picos_ajuste.append(center)
            alturas_ajuste.append(altura)

    return picos_ajuste, alturas_ajuste

def _criar_grafico_ajuste(rs, intensity, intensity_original, baseline_final,
                          intensity_corrigida, result, picos_ajuste, alturas_ajuste,
                          metodo_baseline, aplicar_normalizacao, modo="automático"):
    # Determinar limites do eixo X
    if len(rs) > 0:
        x_min = rs[0]
        x_max = rs[-1]
        margin_x = (x_max - x_min) * 0.01
    else:
        x_min = np.min(rs) if len(rs) > 0 else 0
        x_max = np.max(rs) if len(rs) > 0 else 1000
        margin_x = 10

    # Configurar tamanho da figura baseado no modo
    if modo == "manual":
        fig = plt.figure(figsize=(16, 12))
        plt.subplots_adjust(bottom=0.15)
    else:
        fig = plt.figure(figsize=tamanho_tela)

    # Subplot 1: Correção de Baseline
    plt.subplot(2, 1, 1)
    plt.plot(rs, intensity_original, 'gray', label='Original Data',
             linewidth=1, alpha=0.6)
    plt.plot(rs, baseline_final, 'orange', label='Baseline',
             linewidth=2, linestyle='--')
    plt.plot(rs, intensity_corrigida, 'blue', label='Corrected Data',
             linewidth=1, alpha=0.8)
    plt.xlabel(r'Raman shift (cm$^{-1}$)')
    plt.ylabel(r'Intensity (a.u.)')
    plt.legend()
    plt.title(f'Baseline Correction with {metodo_baseline}')
    plt.grid(True, alpha=0.3)
    plt.xlim(x_min - margin_x, x_max + margin_x)

    # Subplot 2: Ajuste Lorentziano
    plt.subplot(2, 1, 2)
    plt.plot(rs, intensity, 'k-', label='Smoothed Data',
             linewidth=1, alpha=0.7)
    plt.plot(rs, result.best_fit, 'r-', label='Lorentzian fitting',
             linewidth=2)
    plt.xlim(x_min - margin_x, x_max + margin_x)

    # Adicionar marcadores para os picos do ajuste
    if picos_ajuste:
        plt.scatter(picos_ajuste, alturas_ajuste, color='green', s=60,
                    label=f'Number of Peaks ({len(picos_ajuste)})', zorder=5,
                    edgecolors='white', linewidth=1.5)

    # Adicionar componentes individuais (Lorentzianas)
    comps = result.eval_components()
    for name, comp in comps.items():
        if name != 'c_':
            plt.plot(rs, comp, '--', alpha=0.3, linewidth=0.8)

    plt.xlabel(r'Raman Shift (cm$^{-1}$)')
    plt.ylabel(r'Intensity Corrected (a.u.)')
    plt.legend()

    # Definir título baseado no modo
    titulo_ajuste = f'Fitting'
    if aplicar_normalizacao:
        titulo_ajuste += ' (Normalized)'

    plt.title(titulo_ajuste)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig

# ========== FUNÇÃO PRINCIPAL DE PROCESSAMENTO ==========
def processar_ajuste_lorentziano(rs, intensity_corrigida, intensity_original,
                                 baseline_final, metodo_baseline,
                                 parametros_picos=None, picos_manuais=None,
                                 aplicar_normalizacao=True, modo="automático",
                                 mostrar_grafico=True):
    try:
        # 1. Preparar dados
        print(f"\n=== INICIANDO AJUSTE LORENTZIANO ({modo.upper()}) ===")
        intensity = _preparar_dados_para_ajuste(rs, intensity_corrigida, aplicar_normalizacao)

        # 2. Obter posições dos picos (automático ou manual)
        if picos_manuais is not None:
            # Modo manual
            peak_data = _processar_picos_manuais(rs, intensity, picos_manuais)
            if len(peak_data) == 0:
                print("Nenhum pico fornecido para ajuste manual!")
                return None, None
        else:
            # Modo automático
            if parametros_picos is None:
                parametros_picos = ajustar_parametros_para_normalizacao(None)

            peaks, _ = _detectar_picos_automaticamente(intensity, parametros_picos)
            if len(peaks) == 0:
                print("NENHUM PICO DETECTADO com os parâmetros atuais!")
                return None, None

            peak_positions, peak_heights = _processar_posicoes_picos(rs, intensity, peaks)
            if len(peak_positions) == 0:
                return None, None

            peak_data = np.column_stack((peak_positions, peak_heights))

        # 3. Agrupar picos usando KMeans
        centers = _agrupar_picos_kmeans(peak_data)
        if len(centers) == 0:
            print("Nenhum centro encontrado para ajuste!")
            return None, None

        # 4. Criar modelo Lorentziano
        model, params = _criar_modelo_lorentziano(centers, rs, intensity)

        # 5. Executar ajuste
        result = _executar_ajuste_lorentziano(model, params, rs, intensity)

        # 6. Extrair informações do ajuste
        picos_ajuste, alturas_ajuste = _extrair_informacoes_ajuste(result, rs)

        # 7. Criar gráfico
        fig = _criar_grafico_ajuste(
            rs, intensity, intensity_original, baseline_final,
            intensity_corrigida, result, picos_ajuste, alturas_ajuste,
            metodo_baseline, aplicar_normalizacao, modo
        )

        # 8. Mostrar gráfico ao usuário se solicitado
        if mostrar_grafico:
            print("\nExibindo gráfico de ajuste...")
            plt.show(block=True)  # Isso bloqueia até o usuário fechar

        # 9. Mostrar estatísticas
        print(f"Ajuste concluído com {len(picos_ajuste)} picos")
        print(f"Chi-quadrado: {result.chisqr:.5f}")
        print(f"Método de baseline: {metodo_baseline}")

        return result, fig

    except Exception as e:
        print(f"Erro no processamento do ajuste Lorentziano: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ========== FUNÇÕES DE ALTO NÍVEL (BACKWARD COMPATIBILITY) ==========
def processar_com_parametros(rs, intensity_corrigida, intensity_original,
                             baseline_final, parametros_picos, metodo_baseline,
                             aplicar_normalizacao=True, salvar_graficos=False):
    # Garantir que parametros_picos é um dicionário
    if not isinstance(parametros_picos, dict):
        print("Aviso: 'parametros_picos' não é um dicionário. Usando parâmetros padrão.")
        parametros_picos = ajustar_parametros_para_normalizacao(None)

    result, fig = processar_ajuste_lorentziano(
        rs=rs,
        intensity_corrigida=intensity_corrigida,
        intensity_original=intensity_original,
        baseline_final=baseline_final,
        metodo_baseline=metodo_baseline,
        parametros_picos=parametros_picos,
        picos_manuais=None,  # Modo automático
        aplicar_normalizacao=aplicar_normalizacao,
        modo="automático",
        mostrar_grafico=True
    )

    # Salvar gráfico se solicitado
    if salvar_graficos and fig is not None:
        try:
            nome_arquivo = "ajuste.png"
            salvar_grafico(fig, nome_arquivo, mostrar=False)
            print(f"Gráfico salvo: {nome_arquivo}")
        except Exception as e:
            print(f"Aviso: Não foi possível salvar o gráfico: {e}")

    return result, fig

def processar_com_picos_manuais(rs, intensity_corrigida, intensity_original,
                                baseline_final, metodo_baseline, picos_manuais=None,
                                aplicar_normalizacao=True, salvar_graficos=False):
    result, fig = processar_ajuste_lorentziano(
        rs=rs,
        intensity_corrigida=intensity_corrigida,
        intensity_original=intensity_original,
        baseline_final=baseline_final,
        metodo_baseline=metodo_baseline,
        parametros_picos=None,  # Não usa parâmetros automáticos
        picos_manuais=picos_manuais,
        aplicar_normalizacao=aplicar_normalizacao,
        modo="manual",
        mostrar_grafico=True
    )

    # Salvar gráfico se solicitado
    if salvar_graficos and fig is not None:
        try:
            nome_arquivo = "ajuste.png"
            salvar_grafico(fig, nome_arquivo, mostrar=False)
            print(f"Gráfico salvo: {nome_arquivo}")
        except Exception as e:
            print(f"Aviso: Não foi possível salvar o gráfico: {e}")

    return result, fig

# ========== FUNÇÕES DE SELEÇÃO INTERATIVA ==========
def selecionar_picos_manualmente(rs, intensity_corrigida, picos_atuais=None):
    # Configuração inicial
    if picos_atuais is None:
        picos_atuais = []

    # Preparar dados
    try:
        # Garantir que intensity_corrigida seja 2D para smoothing
        if intensity_corrigida.ndim == 1:
            inten = intensity_corrigida.reshape(1, -1)
        else:
            inten = intensity_corrigida

        # Suavização com fallback
        try:
            intensity_suavizada = smo.smoothing(inten, mode='sav_gol')[0]
        except (ImportError, AttributeError):
            # Fallback: usar scipy se pyPreprocessing não disponível
            from scipy.signal import savgol_filter
            intensity_suavizada = savgol_filter(inten[0], window_length=11, polyorder=3)
    except Exception as e:
        print(f"Erro ao preparar dados: {e}")
        # Usar dados originais como fallback
        intensity_suavizada = intensity_corrigida if intensity_corrigida.ndim == 1 else intensity_corrigida[0]

    # Criar figura
    fig, ax = plt.subplots(figsize=tamanho_tela)
    plt.subplots_adjust(bottom=0.15, top=0.92, left=0.1, right=0.95)

    # Determinar limites do eixo X
    if len(rs) > 0:
        x_min = rs[0]
        x_max = rs[-1]
        margin_x = max((x_max - x_min) * 0.01, 5)  # Mínimo 5 unidades
    else:
        x_min = np.min(rs) if len(rs) > 0 else 0
        x_max = np.max(rs) if len(rs) > 0 else 1000
        margin_x = 10

    # Plotar o espectro
    ax.plot(rs, intensity_suavizada, 'b-', linewidth=1.5, alpha=0.8, label='Espectro')
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Intensity', fontsize=12, fontweight='bold')
    ax.set_title('Manual Selection\n(Click: ADD | Shift+Click: REMOVE)',
                 fontsize=13, pad=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min - margin_x, x_max + margin_x)

    # Estado inicial
    picos_selecionados = picos_atuais.copy()
    elementos_graficos = {'marcadores': [], 'lorentzianas': [], 'linhas_centro': []}
    tolerancia_remocao = 10  # cm⁻¹
    distancia_limite = 4  # cm⁻¹

    # Função para criar uma lorentziana (simplificada para visualização)
    def lorentzian(x, center, amplitude, sigma=5):
        return amplitude * (sigma ** 2) / ((x - center) ** 2 + sigma ** 2)

    def atualizar_visualizacao():
        # Limpar elementos antigos
        for elemento in sum(elementos_graficos.values(), []):
            try:
                elemento.remove()
            except:
                pass

        # Limpar listas
        for key in elementos_graficos:
            elementos_graficos[key].clear()

        # Se não há picos selecionados, apenas atualizar legenda
        if not picos_selecionados:
            fig.canvas.draw()
            return

        # Adicionar marcadores e lorentzianas para cada pico
        for pico in picos_selecionados:
            # Altura no pico
            altura = np.interp(pico, rs, intensity_suavizada)

            # Marcador vermelho
            marcador = ax.plot(pico, altura, 'ro', markersize=10,
                               markeredgecolor='black', markerfacecolor='red',
                               markeredgewidth=2, zorder=5)[0]
            elementos_graficos['marcadores'].append(marcador)

            # Lorentziana visual (tracejada)
            amplitude_estimada = altura * 0.8
            y_lorentz = lorentzian(rs, pico, amplitude_estimada)
            lorentz_line = ax.plot(rs, y_lorentz, '--', color='purple',
                                   alpha=0.4, linewidth=1, zorder=1)[0]
            elementos_graficos['lorentzianas'].append(lorentz_line)

            # Linha vertical no centro
            altura_max = np.max(y_lorentz)
            ylim_max = np.max(ax.get_ylim())
            ymax_normalized = altura_max / ylim_max if ylim_max > 0 else 0.5

            vline = ax.axvline(x=pico, ymin=0, ymax=ymax_normalized,
                               color='purple', linestyle=':', alpha=0.3,
                               linewidth=0.8, zorder=0)
            elementos_graficos['linhas_centro'].append(vline)

        fig.canvas.draw()

    # Elementos interativos
    vline_interativa = ax.axvline(x=0, color='red', linestyle='--', alpha=0)
    texto_anotacao = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                             verticalalignment='top', fontsize=10,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Atualizar visualização inicial
    atualizar_visualizacao()

    # Callbacks
    def on_move(event):
        if event.inaxes == ax:
            vline_interativa.set_xdata([event.xdata, event.xdata])
            vline_interativa.set_alpha(0.7)

            if event.xdata is not None and event.ydata is not None:
                texto_anotacao.set_text(f'x = {event.xdata:.2f} cm⁻¹\ny = {event.ydata:.4f}')
            fig.canvas.draw_idle()
        else:
            vline_interativa.set_alpha(0)
            texto_anotacao.set_text('')
            fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return

        x_pos = event.xdata

        if event.key == 'shift':  # Modo REMOÇÃO
            if picos_selecionados:
                distancias = [abs(x_pos - pico) for pico in picos_selecionados]
                idx_proximo = np.argmin(distancias)

                if distancias[idx_proximo] < tolerancia_remocao:
                    pico_removido = picos_selecionados.pop(idx_proximo)
                    print(f"Pico removido: {pico_removido:.2f} cm⁻¹")
                    atualizar_visualizacao()
                else:
                    print(f"Clique mais próximo do pico para remover (> {tolerancia_remocao} cm⁻¹)")

        else:  # Modo ADIÇÃO
            if picos_selecionados:
                distancias = [abs(x_pos - pico) for pico in picos_selecionados]
                distancia_minima = min(distancias)
            else:
                distancia_minima = float('inf')

            if distancia_minima > distancia_limite:
                picos_selecionados.append(x_pos)
                print(f"Pico adicionado: {x_pos:.2f} cm⁻¹")
                atualizar_visualizacao()
            else:
                print(f"Pico muito próximo! Distância mínima: {distancia_limite} cm⁻¹")

    def on_key(event):
        nonlocal resultado_final

        if event.key == 'enter':
            resultado_final = sorted(picos_selecionados)
            plt.close()
        elif event.key == 'escape':
            resultado_final = picos_atuais.copy() if picos_atuais else []
            print("Seleção cancelada.")
            plt.close()
        elif event.key in ['r', 'R']:
            picos_selecionados.clear()
            print("Todos os picos removidos!")
            atualizar_visualizacao()

    # Botões
    from matplotlib.widgets import Button

    ax_finalizar = plt.axes((0.70, 0.02, 0.12, 0.04))
    ax_limpar = plt.axes((0.40, 0.02, 0.12, 0.04))
    ax_cancelar = plt.axes((0.10, 0.02, 0.12, 0.04))

    btn_finalizar = Button(ax_finalizar, 'Finalize (Enter)', color='lightgreen', hovercolor='green')
    btn_limpar = Button(ax_limpar, 'Clean everything (R)', color='orange', hovercolor='red')
    btn_cancelar = Button(ax_cancelar, 'Cancel (Esc)', color='lightcoral', hovercolor='red')

    resultado_final = None

    def finalizar_selecao(event):
        nonlocal resultado_final
        resultado_final = sorted(picos_selecionados)
        plt.close()

    def limpar_selecao(event):
        nonlocal picos_selecionados
        picos_selecionados.clear()
        print("Todos os picos removidos!")
        atualizar_visualizacao()

    def cancelar_selecao(event):
        nonlocal resultado_final
        resultado_final = picos_atuais.copy() if picos_atuais else []
        print("Seleção cancelada.")
        plt.close()

    # Conectar eventos
    btn_finalizar.on_clicked(finalizar_selecao)
    btn_limpar.on_clicked(limpar_selecao)
    btn_cancelar.on_clicked(cancelar_selecao)

    cid_move = fig.canvas.mpl_connect('motion_notify_event', on_move)
    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

    try:
        plt.show(block=True)
    finally:
        # Garantir desconexão dos eventos
        fig.canvas.mpl_disconnect(cid_move)
        fig.canvas.mpl_disconnect(cid_click)
        fig.canvas.mpl_disconnect(cid_key)

    # Retornar resultado
    if resultado_final is not None:
        return resultado_final
    else:
        # Fallback: retornar lista original
        return picos_atuais.copy() if picos_atuais else []

# ========== FUNÇÕES AUXILIARES PARA PROCESSAMENTO ==========
def _carregar_dados_espectro(arquivo):
    data = np.loadtxt(arquivo)
    x = data[:, 0]
    y = data[:, 1]

    # Detectar intervalo com sinal significativo
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
        raise ValueError("Nenhum ponto com sinal significativo detectado")

    return rs, intensity_original, x, y

def _calcular_baseline(intensity_original, metodo):
    try:
        if metodo == "Peakutils":
            baseline = pk.baseline(intensity_original, deg=3, max_it=1000, tol=1e-10)
        elif metodo == "Quantil":
            baseline = pd.Series(intensity_original).rolling(
                window=151, center=True, min_periods=1
            ).quantile(0.15).values
        elif metodo == "SNIP":
            inten_reshaped = intensity_original.reshape(1, -1)
            baseline = bs.generate_baseline(inten_reshaped, mode='SNIP')[0]
        else:
            baseline = pk.baseline(intensity_original, deg=3, max_it=1000, tol=1e-10)

        return baseline, True
    except Exception as e:
        print(f"Método {metodo} falhou: {e}")
        return np.zeros_like(intensity_original), False

def _testar_e_selecionar_baseline(rs, intensity_original, arquivo_temp,
                                  nome_original=None, salvar_comparativo=True):
    print("\n=== TESTANDO DIFERENTES MÉTODOS DE BASELINE ===")
    if nome_original:
        nome_base = os.path.splitext(nome_original)[0]
    else:
        nome_base = os.path.basename(arquivo_temp)[0]
    metodos = ["Peakutils", "Quantil", "SNIP"]
    baselines = {}
    intensidades_corrigidas = {}

    # Calcular baselines
    for metodo in metodos:
        baseline, sucesso = _calcular_baseline(intensity_original, metodo)
        baselines[metodo] = baseline
        intensidades_corrigidas[metodo] = intensity_original - baseline

        status = "OK" if sucesso else "FALHOU"
        print(f"{metodo} - {status}")

    # Criar gráfico de comparação
    if salvar_comparativo:
        fig = plt.figure(figsize=tamanho_tela)

        # Subplot 1: Baselines
        plt.subplot(2, 1, 1)
        plt.plot(rs, intensity_original, 'black', label='Original Data',
                 linewidth=1.5, alpha=0.8)
        cores = {'Peakutils': 'blue', 'Quantil': 'green', 'SNIP': 'orange'}

        for metodo, baseline in baselines.items():
            plt.plot(rs, baseline, color=cores[metodo], label=metodo,
                     linewidth=2, alpha=0.8)

        plt.xlabel('Raman Shift (cm⁻¹)')
        plt.ylabel('Intensity (a.u.)')
        plt.legend()
        plt.title(f'Comparison of Baselines for the Reference Spectra')
        plt.grid(True, alpha=0.3)

        # Subplot 2: Dados corrigidos
        plt.subplot(2, 1, 2)
        for metodo, intensity_corr in intensidades_corrigidas.items():
            plt.plot(rs, intensity_corr, color=cores[metodo],
                     label=f'Corrected - {metodo}', linewidth=1, alpha=0.8)

        plt.xlabel('Raman Shift (cm⁻¹)')
        plt.ylabel('Intensity Corrected (a.u.)')
        plt.legend()
        plt.title('Data After Baseline Correction')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Salvar gráfico
        nome_grafico = f"baseline_comparison_{nome_base}.png"
        salvar_grafico(fig, nome_grafico, mostrar=True)

    # Seleção da melhor baseline
    print("\n=== SELECIONE A MELHOR BASELINE ===")
    print("Métodos disponíveis:")
    for i, metodo in enumerate(metodos, 1):
        print(f"{i} - {metodo}")

    while True:
        escolha = input("\nDigite o número do método (1-3): ").strip()

        if escolha == "1":
            return "Peakutils", baselines["Peakutils"], intensidades_corrigidas["Peakutils"]
        elif escolha == "2":
            return "Quantil", baselines["Quantil"], intensidades_corrigidas["Quantil"]
        elif escolha == "3":
            return "SNIP", baselines["SNIP"], intensidades_corrigidas["SNIP"]
        else:
            print("Opção inválida. Tente novamente.")

def _processamento_iterativo(rs, intensity_corrigida, intensity_original,
                             baseline_final, metodo_baseline, parametros_picos,
                             aplicar_normalizacao=True, modo="comparativo",
                             nome_arquivo=""):
    result_final = None
    picos_manuais = None
    primeiro_ajuste = True
    fig_atual = None

    while True:
        if primeiro_ajuste:
            # Primeiro ajuste automático
            result, fig_atual = processar_com_parametros(
                rs, intensity_corrigida, intensity_original,
                baseline_final, parametros_picos, metodo_baseline,
                aplicar_normalizacao, salvar_graficos=False
            )
            primeiro_ajuste = False

        elif picos_manuais is not None:
            # Ajuste com picos manuais
            print(f"Processando {len(picos_manuais)} picos selecionados manualmente...")
            result, fig_atual = processar_com_picos_manuais(
                rs, intensity_corrigida, intensity_original,
                baseline_final, metodo_baseline, picos_manuais,
                aplicar_normalizacao, salvar_graficos=False
            )
            picos_manuais = None

        else:
            # Re-processamento com parâmetros atuais
            print("Reprocessando com parâmetros atuais...")
            result, fig_atual = processar_com_parametros(
                rs, intensity_corrigida, intensity_original,
                baseline_final, parametros_picos, metodo_baseline,
                aplicar_normalizacao, salvar_graficos=False
            )

        # Verificar se o ajuste foi bem-sucedido
        if result is None or fig_atual is None:
            print("Nenhum pico detectado ou falha no ajuste.")

            if modo == "comparativo" or modo == "individual":
                print("O que você gostaria de fazer?")
                print("1. Selecionar picos manualmente")
                print("2. Ir para o próximo espectro" if modo == "comparativo" else "2. Cancelar análise")

                escolha = input("\nDigite sua escolha (1-2): ").strip()

                if escolha == "1":
                    picos_manuais = selecionar_picos_manualmente(rs, intensity_corrigida)
                    continue
                elif escolha == "2":
                    print(f"Pulando espectro: {nome_arquivo}")
                    return None, None
                else:
                    print("Opção inválida.")
                    continue
            else:  # modo == "referencia"
                print("O que você gostaria de fazer?")
                print("1. Selecionar picos manualmente")
                print("2. Alterar método de baseline")
                print("3. Cancelar análise")

                escolha_falha = input("\nDigite sua escolha (1-3): ").strip()

                if escolha_falha == "1":
                    picos_manuais = selecionar_picos_manualmente(rs, intensity_corrigida)
                    continue
                elif escolha_falha == "2":
                    print("Reiniciando com nova baseline...")
                    return None, "ALTERAR_BASELINE"
                elif escolha_falha == "3":
                    print("Análise cancelada.")
                    return None, "CANCELAR"
                else:
                    print("Opção inválida.")
                    continue
        else:
            # Ajuste bem-sucedido - perguntar próximo passo
            if modo == "referencia":
                opcoes = [
                    "1. Aceitar este ajuste como referência",
                    "2. Refinar picos manualmente",
                    "3. Recomeçar do zero",
                    "4. Alterar método de baseline",
                    "5. Cancelar análise"
                ]
            else:
                opcoes = [
                    "1. Aceitar este ajuste e prosseguir",
                    "2. Refinar picos manualmente",
                    "3. Recomeçar do zero",
                    "4. Ir para o próximo espectro" if modo == "comparativo" else "4. Cancelar análise"
                ]

            print(f"\nAJUSTE CONCLUÍDO PARA: {nome_arquivo}")
            print("O que você gostaria de fazer?")
            for opcao in opcoes:
                print(opcao)

            escolha = input("\nDigite sua escolha: ").strip()

            if escolha == "1":  # Aceitar ajuste
                result_final = result

                # Salvar gráfico final
                if result is not None and fig_atual is not None:
                    try:
                        axs = fig_atual.get_axes()
                        if len(axs) >= 2:
                            titulo_original = axs[1].get_title()
                            sufixo = " "
                            if modo == "referencia":
                                sufixo += "Reference"
                            axs[1].set_title(f"{titulo_original}{sufixo}")

                        prefixo = "fitting_"
                        if modo == "referencia":
                            prefixo = "fitting_reference_"

                        nome_base = os.path.splitext(nome_arquivo)[0]
                        nome_grafico = f"{prefixo}{nome_base}.png"

                        salvar_grafico(fig_atual, nome_grafico, mostrar=False)
                        print(f"Gráfico final salvo: {nome_grafico}")

                    except Exception as e:
                        print(f"Aviso: Não foi possível salvar o gráfico final: {e}")

                # Fechar figura
                if fig_atual is not None:
                    try:
                        plt.close(fig_atual)
                    except:
                        pass

                return result_final, None

            elif escolha == "2":  # Refinar manualmente
                picos_base = []
                for param_name in result.params:
                    if '_center' in param_name:
                        picos_base.append(result.params[param_name].value)

                print("Modo de refinamento - Selecione os picos desejados")
                picos_manuais = selecionar_picos_manualmente(rs, intensity_corrigida, picos_base)

                # Fechar figura atual
                if fig_atual is not None:
                    try:
                        plt.close(fig_atual)
                    except:
                        pass
                continue

            elif escolha == "3":  # Recomeçar do zero
                print("Recomeçando do zero...")
                primeiro_ajuste = True
                picos_manuais = None

                # Fechar figura atual
                if fig_atual is not None:
                    try:
                        plt.close(fig_atual)
                    except:
                        pass
                continue

            elif escolha == "4":  # Próximo espectro ou cancelar
                if modo == "comparativo":
                    print(f"Pulando o espectro: {nome_arquivo}")
                else:
                    print("Análise cancelada.")

                # Fechar figura
                if fig_atual is not None:
                    try:
                        plt.close(fig_atual)
                    except:
                        pass

                return None, "PULAR" if modo == "comparativo" else "CANCELAR"

            elif escolha == "5" and modo == "referencia":  # Alterar baseline (apenas referência)
                print("Alterando método de baseline...")
                return None, "ALTERAR_BASELINE"

            else:
                print("Opção inválida. Tente novamente.")
                continue

def _selecionar_regiao_interativa(rs, intensity, titulo="ESPECTRUM"):
    fig, ax = plt.subplots(figsize=tamanho_tela)

    # Plotar espectro
    ax.plot(rs, intensity, 'b-', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Intensity (a.u.)', fontsize=14, fontweight='bold',
                  rotation=90, labelpad=10, va='center')
    ax.set_title(titulo, fontsize=16, pad=15)
    ax.grid(True, alpha=0.3)

    # Configurar limites
    if len(rs) > 0:
        x_min_plot = rs[0]
        x_max_plot = rs[-1]
        y_max_plot = np.max(intensity)

        margin_x = (x_max_plot - x_min_plot) * 0.01
        ax.set_xlim(x_min_plot - margin_x, x_max_plot + margin_x)
        ax.set_ylim(0, y_max_plot * 1.1)

    # Configurar bordas
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    # Variáveis para seleção
    regiao_selecionada = None
    retangulo_selecao = None
    x_press = None
    retangulo_atual = None

    # Callbacks
    def on_press(event):
        nonlocal x_press, retangulo_atual, retangulo_selecao

        if event.inaxes != ax:
            return

        # Limpar seleção anterior
        if retangulo_selecao:
            retangulo_selecao.remove()
            for line in ax.lines:
                if line.get_linestyle() == '--' and line.get_color() == 'green':
                    line.remove()
            retangulo_selecao = None

        x_press = event.xdata

        # Criar retângulo de seleção
        y_min, y_max = ax.get_ylim()
        retangulo_atual = plt.Rectangle((x_press, y_min), 0, y_max - y_min,
                                        edgecolor='blue', facecolor='blue', alpha=0.2)
        ax.add_patch(retangulo_atual)
        fig.canvas.draw()

    def on_motion(event):
        nonlocal retangulo_atual, x_press

        if event.inaxes != ax or x_press is None:
            return

        x_current = event.xdata
        width = x_current - x_press
        retangulo_atual.set_width(width)
        fig.canvas.draw()

    def on_release(event):
        nonlocal x_press, regiao_selecionada, retangulo_atual, retangulo_selecao

        if event.inaxes != ax or x_press is None:
            return

        x_current = event.xdata
        x_left = min(x_press, x_current)
        x_right = max(x_press, x_current)

        # Verificar se região é muito pequena
        if abs(x_right - x_left) < (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01:
            print("Região muito pequena. Selecione uma área maior.")
            retangulo_atual.remove()
            retangulo_atual = None
            x_press = None
            return

        regiao_selecionada = (x_left, x_right)
        retangulo_selecao = retangulo_atual

        # Destacar região
        retangulo_selecao.set_edgecolor('red')
        retangulo_selecao.set_facecolor('red')
        retangulo_selecao.set_alpha(0.3)

        # Adicionar linhas verticais
        ax.axvline(x=x_left, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(x=x_right, color='green', linestyle='--', linewidth=2, alpha=0.7)

        fig.canvas.draw()
        x_press = None
        retangulo_atual = None

    def on_key(event):
        nonlocal regiao_selecionada, retangulo_selecao

        if event.key == 'enter':
            if regiao_selecionada:
                print(f"\nRegião confirmada: {regiao_selecionada[0]:.1f} - {regiao_selecionada[1]:.1f} cm⁻¹")
                plt.close()
            else:
                print("\nNenhuma região selecionada! Usando espectro completo.")
                regiao_selecionada = None
                plt.close()
        elif event.key == 'escape':
            print("\nUsando espectro completo.")
            regiao_selecionada = None
            plt.close()
        elif event.key in ['r', 'R']:
            # Limpar seleção
            if retangulo_selecao:
                retangulo_selecao.remove()
                retangulo_selecao = None
            for line in ax.lines:
                if line.get_linestyle() == '--' and line.get_color() == 'green':
                    line.remove()
            regiao_selecionada = None
            print("Seleção limpa.")
            fig.canvas.draw()
        elif event.key in ['t', 'T']:
            print("\nUsando espectro completo.")
            regiao_selecionada = None
            plt.close()

    # Conectar eventos
    cid_press = fig.canvas.mpl_connect('button_press_event', on_press)
    cid_motion = fig.canvas.mpl_connect('motion_notify_event', on_motion)
    cid_release = fig.canvas.mpl_connect('button_release_event', on_release)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

    # Botões
    from matplotlib.widgets import Button

    ax_confirmar = plt.axes([0.78, 0.01, 0.15, 0.04])
    ax_todo = plt.axes([0.25, 0.01, 0.15, 0.04])
    ax_limpar = plt.axes([0.62, 0.01, 0.15, 0.04])

    btn_confirmar = Button(ax_confirmar, 'Confirm (Enter)', color='lightgreen')
    btn_todo = Button(ax_todo, 'Use everything (T)', color='lightblue')
    btn_limpar = Button(ax_limpar, 'Clean (R)', color='orange')

    def confirmar_selecao(event):
        nonlocal regiao_selecionada
        if regiao_selecionada:
            print(f"\nRegião confirmada: {regiao_selecionada[0]:.1f} - {regiao_selecionada[1]:.1f} cm⁻¹")
        else:
            print("\nNenhuma região selecionada! Usando espectro completo.")
            regiao_selecionada = None
        plt.close()

    def usar_todo_espectro(event):
        nonlocal regiao_selecionada
        print("\nUsando espectro completo.")
        regiao_selecionada = None
        plt.close()

    def limpar_selecao(event):
        nonlocal regiao_selecionada, retangulo_selecao
        if retangulo_selecao:
            retangulo_selecao.remove()
            retangulo_selecao = None
        for line in ax.lines:
            if line.get_linestyle() == '--' and line.get_color() == 'green':
                line.remove()
        regiao_selecionada = None
        print("Seleção limpa.")
        fig.canvas.draw()

    btn_confirmar.on_clicked(confirmar_selecao)
    btn_todo.on_clicked(usar_todo_espectro)
    btn_limpar.on_clicked(limpar_selecao)

    # Layout
    plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.12)
    plt.show()

    # Desconectar eventos
    fig.canvas.mpl_disconnect(cid_press)
    fig.canvas.mpl_disconnect(cid_motion)
    fig.canvas.mpl_disconnect(cid_release)
    fig.canvas.mpl_disconnect(cid_key)

    return regiao_selecionada

# ========== FUNÇÕES PRINCIPAIS DE PROCESSAMENTO ==========
def processar_espectro_comparativo_iterativo(arquivo, metodo_baseline, parametros_picos,
                                             aplicar_normalizacao=True, nome_original=None):
    try:
        # 1. Carregar dados
        rs, intensity_original, x, y = _carregar_dados_espectro(arquivo)
        print(f"{len(rs)} pontos no intervalo {rs[0]:.1f} - {rs[-1]:.1f} cm⁻¹")

        # 2. Calcular baseline
        baseline, sucesso = _calcular_baseline(intensity_original, metodo_baseline)
        if not sucesso:
            print(f"Aviso: Baseline '{metodo_baseline}' não pôde ser calculada. Usando zero.")

        intensity_corrigida = intensity_original - baseline

        # 3. Processamento iterativo
        if nome_original:
            nome_arquivo = nome_original
        else:
            nome_arquivo = os.path.basename(arquivo)
        result_final, status = _processamento_iterativo(
            rs, intensity_corrigida, intensity_original, baseline,
            metodo_baseline, parametros_picos, aplicar_normalizacao,
            modo="comparativo", nome_arquivo=nome_arquivo)

        if result_final is None:
            if status == "PULAR":
                print(f"Espectro {nome_arquivo} pulado.")
            return None, None, None

        return result_final, rs, intensity_original

    except Exception as e:
        print(f"Erro no processamento comparativo: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def processar_espectro_individual(arquivo, aplicar_normalizacao=True):
    global pasta_saida_atual

    # Criar pasta se necessário
    if pasta_saida_atual is None:
        pasta_saida_atual = criar_pasta_analise()

    try:
        # 1. Carregar dados
        rs_auto, intensity_auto, x, y = _carregar_dados_espectro(arquivo)

        # 2. Seleção interativa de região
        print("\n=== SELEÇÃO DE REGIÃO PARA ANÁLISE ===")
        titulo = f'ESPECTRUM: {os.path.basename(arquivo)}'
        regiao_selecionada = _selecionar_regiao_interativa(rs_auto, intensity_auto, titulo)

        # 3. Aplicar corte na região selecionada
        if regiao_selecionada:
            x_left, x_right = regiao_selecionada
            mask = (x >= x_left) & (x <= x_right)
            rs = x[mask]
            intensity_original = y[mask]
            print(f"Processando {len(rs)} pontos na região {x_left:.1f} - {x_right:.1f} cm⁻¹")
        else:
            rs = rs_auto
            intensity_original = intensity_auto
            print(f"Processando {len(rs)} pontos no espectro completo")

        if len(rs) == 0:
            print("Nenhum ponto na região selecionada!")
            return None

        # 4. Testar e selecionar baseline
        metodo_baseline, baseline_final, intensity_corrigida = _testar_e_selecionar_baseline(
            rs, intensity_original, arquivo, salvar_comparativo=True
        )

        # 5. Definir parâmetros para detecção de picos
        parametros_picos = ajustar_parametros_para_normalizacao(None)

        # 6. Processamento iterativo
        print(f"\n=== INICIANDO ANÁLISE ITERATIVA ===")
        nome_arquivo = os.path.basename(arquivo)
        result_final, status = _processamento_iterativo(
            rs, intensity_corrigida, intensity_original, baseline_final,
            metodo_baseline, parametros_picos, aplicar_normalizacao,
            modo="individual", nome_arquivo=nome_arquivo
        )

        if result_final is None:
            print("Análise não concluída.")
            return None

        # 7. Salvar resultados
        print("\n=== PRONTO PARA SALVAR RESULTADOS! ===")
        if pasta_saida_atual:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Salvar parâmetros
            dados_temp = {nome_arquivo: result_final}
            rs_temp = {nome_arquivo: rs}
            dados_brutos_temp = {nome_arquivo: intensity_corrigida}
            salvar_parametros_consolidados(dados_temp, metodo_baseline,
                                           pasta_saida_atual, timestamp)

            # Criar gráfico combinado
            criar_grafico_combinado_completo(dados_temp, rs_temp,
                                             pasta_saida_atual, timestamp,
                                             dados_brutos_por_espectro=dados_brutos_temp)

            print(f"\n=== ANÁLISE INDIVIDUAL CONCLUÍDA! ===")
            print(f"Chi-quadrado: {result_final.chisqr:.5f}")
            print(f"Método de baseline: {metodo_baseline}")
            print(f"Resultados salvos em: {pasta_saida_atual}")

            return result_final
        else:
            print("Nenhuma pasta selecionada. Resultados não salvos.")
            return None

    except Exception as e:
        print(f"Erro no processamento individual: {e}")
        import traceback
        traceback.print_exc()
        return None

def processar_espectro_referencia_completo(arquivo, parametros_picos, aplicar_normalizacao=True, nome_original=None):
    # Variável de classe para persistir baseline entre chamadas
    if not hasattr(processar_espectro_referencia_completo, '_cache_baseline'):
        processar_espectro_referencia_completo._cache_baseline = None

    try:
        # 1. Carregar dados
        rs, intensity_original, x, y = _carregar_dados_espectro(arquivo)
        print(f"Processando {len(rs)} pontos no intervalo {rs[0]:.1f} - {rs[-1]:.1f} cm⁻¹")

        # 2. Selecionar baseline (apenas na primeira vez)
        if processar_espectro_referencia_completo._cache_baseline is None:
            print("\n=== SELECIONANDO BASELINE PARA TODOS OS ESPECTROS ===")
            metodo, baseline, intensity_corrigida = _testar_e_selecionar_baseline(
                rs, intensity_original, arquivo, nome_original=nome_original, salvar_comparativo=True)

            # Salvar no cache
            processar_espectro_referencia_completo._cache_baseline = {
                'metodo': metodo,
                'baseline': baseline,
                'intensity_corrigida': intensity_corrigida,
                'rs': rs,
                'intensity_original': intensity_original
            }
        else:
            # Usar baseline do cache
            cache = processar_espectro_referencia_completo._cache_baseline
            metodo = cache['metodo']
            baseline = cache['baseline']
            intensity_corrigida = cache['intensity_corrigida']
            print(f"Usando baseline pré-selecionada: {metodo}")

        # 3. Validar parâmetros
        if isinstance(parametros_picos, dict):
            parametros_ajustados = parametros_picos.copy()
        else:
            parametros_ajustados = ajustar_parametros_para_normalizacao(None)

        # 4. Processamento iterativo
        if nome_original:
            nome_arquivo = nome_original
        else:
            nome_arquivo = os.path.basename(arquivo)
        result_final, status = _processamento_iterativo(
            rs, intensity_corrigida, intensity_original, baseline,
            metodo, parametros_ajustados, aplicar_normalizacao,
            modo="referencia", nome_arquivo=nome_arquivo)

        # 5. Tratar resultado
        if status == "ALTERAR_BASELINE":
            # Limpar cache e recomeçar
            processar_espectro_referencia_completo._cache_baseline = None
            return None

        elif status == "CANCELAR":
            print("Análise cancelada.")
            return 'CANCELAR_TUDO'

        elif result_final is None:
            return None

        # 6. Retornar resultado
        return {
            'metodo_baseline': metodo,
            'rs': rs,
            'result': result_final,
            'fig': None,  # Figura já foi fechada
            'parametros_finais': parametros_ajustados.copy()
        }

    except Exception as e:
        print(f"Erro no processamento da referência: {e}")
        import traceback
        traceback.print_exc()

        # Limpar cache em caso de erro
        if hasattr(processar_espectro_referencia_completo, '_cache_baseline'):
            processar_espectro_referencia_completo._cache_baseline = None

        return None

# ========== FUNÇÕES DE VISUALIZAÇÃO E GRÁFICOS ==========
def criar_grafico_espectros_brutos_normalizados(dados_espectros, pasta_destino, timestamp):
    if not dados_espectros:
        print("Nenhum dado de espectro disponível")
        return None

    # Preparar dados
    todos_rs = []
    espectros_info = []

    for nome, dados in dados_espectros.items():
        if 'rs' in dados and 'intensity' in dados:
            rs = dados['rs']
            intensity = dados['intensity']

            # Garantir que são arrays numpy
            if not isinstance(rs, np.ndarray):
                rs = np.array(rs)
            if not isinstance(intensity, np.ndarray):
                intensity = np.array(intensity)

            # Verificar se arrays não estão vazios
            if len(rs) == 0 or len(intensity) == 0:
                print(f"Aviso: Espectro '{nome}' vazio")
                continue

            # Normalizar pelo máximo (MANTIDO COMO ORIGINAL)
            if len(intensity) > 0:
                max_val = np.max(np.abs(intensity))
                if max_val > 1e-12:  # Pequena tolerância para evitar divisão por zero
                    intensity_norm = intensity / max_val
                else:
                    intensity_norm = intensity.copy()  # Cópia para não modificar original
            else:
                intensity_norm = intensity.copy()

            # Nome limpo para legenda
            nome_base = os.path.splitext(nome)[0] if '.' in nome else nome
            nome_limpo = nome_base

            espectros_info.append({
                'nome': nome,
                'rs': rs,
                'intensity': intensity_norm,
                'nome_limpo': nome_limpo
            })

            # Adicionar TODOS os valores de rs, não apenas do último
            todos_rs.extend(list(rs))  # Convertendo para lista para garantir

    if not espectros_info:
        print("Nenhum dado válido para plotar")
        return None

    # Ordenar por nome
    espectros_info.sort(key=lambda x: x['nome'])

    # Criar figura
    fig, ax = plt.subplots(figsize=(16, 10))

    # Colormap rainbow
    colormap = plt.cm.rainbow
    cores = colormap(np.linspace(0, 1, len(espectros_info)))

    # Plotar com offset fixo
    offset = 0
    offset_step = 0.5  # Offset fixo entre espectros

    for i, esp in enumerate(espectros_info):
        # Plotar linha
        ax.plot(esp['rs'], esp['intensity'] + offset,
                color=cores[i],
                linewidth=1.5,
                alpha=0.8,
                label=esp['nome_limpo'])

        offset += offset_step

    # Ajustar limites do eixo X (CORRIGIDO para usar todos_rs)
    if todos_rs:
        # Converter para array numpy para cálculos
        todos_rs_array = np.array(todos_rs)
        x_min = np.min(todos_rs_array)
        x_max = np.max(todos_rs_array)

        # Calcular margem (1% ou mínimo de 10 unidades)
        x_range = x_max - x_min
        margin = max(x_range * 0.01, 10)

        # Aplicar novos limites
        ax.set_xlim(x_min - margin, x_max + margin)

    # Configurar eixos
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
    ax.set_ylabel('Normalized Intensity', fontsize=12)
    ax.set_title(f'Normalized Spectra',
                 fontsize=14, pad=20)

    # Remover ticks do eixo Y
    ax.set_yticks([])

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Linhas horizontais para referência
    for i in range(len(espectros_info) + 1):
        ax.axhline(y=i * offset_step, color='gray', linestyle=':',
                   alpha=0.3, linewidth=0.5)

    # Legendas
    if len(espectros_info) <= 10:
        ax.legend(loc='upper right', fontsize=9)
    else:
        # Criar legenda separada
        fig_legenda, ax_legenda = plt.subplots(figsize=(8, 10))
        ax_legenda.axis('off')

        for i, esp in enumerate(espectros_info):
            y_pos = 0.95 - (i * 0.03)
            ax_legenda.plot(0.1, y_pos, 'o', color=cores[i], markersize=8,
                            transform=ax_legenda.transAxes)
            ax_legenda.text(0.15, y_pos, esp['nome_limpo'],
                            transform=ax_legenda.transAxes, fontsize=8, va='center')

        # Salvar legenda separadamente
        nome_legenda = f"legenda_espectros_brutos_{timestamp}.png"
        caminho_legenda = os.path.join(pasta_destino, "graficos", nome_legenda)

        try:
            fig_legenda.savefig(caminho_legenda, dpi=300, bbox_inches='tight')
            plt.close(fig_legenda)
            print(f"Legenda salva: {nome_legenda}")
        except Exception as e:
            print(f"Erro ao salvar legenda: {e}")
            plt.close(fig_legenda)

    # Ajustar layout
    plt.tight_layout()

    # Salvar
    nome_grafico = f"normalized_spectra.png"

    try:
        # Usar salvar_grafico com mostrar=False
        caminho_completo = salvar_grafico(fig, nome_grafico, pasta=pasta_destino, mostrar=False)
        print(f"Gráfico de espectros normalizados salvo: {nome_grafico}")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Erro ao salvar gráfico: {e}")
        plt.close(fig)
        return False

def criar_grafico_combinado_completo(todos_results, rs_por_espectro, pasta_destino, timestamp,
                                     dados_brutos_por_espectro=None):
    if not todos_results or not rs_por_espectro:
        print("Nenhum resultado disponível para criar gráfico combinado")
        return None

    # Ordenar espectros por nome
    nomes_ordenados = sorted(todos_results.keys())
    num_espectros = len(nomes_ordenados)

    # 1. Extrair variáveis físicas
    dados_variaveis = []

    for idx, nome in enumerate(nomes_ordenados):
        tipo, valor, unidade = extrair_variavel_fisica(nome)

        if tipo and valor is not None:
            dados_variaveis.append({
                'tipo': tipo,
                'valor': valor,
                'unidade': unidade or '',
                'arquivo': nome,
                'indice': idx + 1
            })
        else:
            dados_variaveis.append({
                'tipo': 'indice',
                'valor': idx + 1,
                'unidade': '',
                'arquivo': nome,
                'indice': idx + 1
            })

    # 2. Determinar variável principal
    tipos = [d['tipo'] for d in dados_variaveis if d['tipo'] != 'indice']
    variavel_principal = 'indice'

    if tipos:
        from collections import Counter
        variavel_principal = Counter(tipos).most_common(1)[0][0]

    # 3. Coletar centers
    dados_centers = []

    for idx, nome in enumerate(nomes_ordenados):
        if nome in todos_results:
            result = todos_results[nome]

            var_info = next((d for d in dados_variaveis if d['arquivo'] == nome), None)
            if not var_info:
                continue

            centers = []
            for param_name in result.params:
                if '_center' in param_name:
                    centers.append(result.params[param_name].value)

            centers.sort()

            for center in centers:
                dados_centers.append({
                    'center': center,
                    'variavel_valor': var_info['valor'],
                    'variavel_tipo': var_info['tipo'],
                    'arquivo': nome
                })

    # 4. Criar figura
    fig = plt.figure(figsize=tamanho_tela)

    # Gridspec simplificado
    gs = fig.add_gridspec(1, 2, width_ratios=[0.6, 0.4], wspace=0.05)

    # Subplot 1: Centers vs Variável
    ax1 = fig.add_subplot(gs[0])

    # Colormap rainbow
    colormap = plt.cm.rainbow
    cores = colormap(np.linspace(0, 1, num_espectros))

    # Plotar centers
    for idx, nome in enumerate(nomes_ordenados):
        dados_espectro = [d for d in dados_centers if d['arquivo'] == nome]
        if dados_espectro:
            x_vals = [d['variavel_valor'] for d in dados_espectro]
            y_vals = [d['center'] for d in dados_espectro]

            nome_limpo = os.path.splitext(nome)[0]
            ax1.scatter(x_vals, y_vals, color=cores[idx], s=80,
                        label=nome_limpo, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Configurar eixos
    ax1.set_ylabel('Center (cm⁻¹)', fontsize=11, fontweight='bold')

    if variavel_principal == 'temperature':
        ax1.set_xlabel('Temperature (K)', fontsize=11, fontweight='bold')
        ax1.set_title('Centers vs Temperature', fontsize=12, pad=10)
    elif variavel_principal == 'pressure':
        ax1.set_xlabel('Pressure (GPa)', fontsize=11, fontweight='bold')
        ax1.set_title('Centers vs Pressure', fontsize=12, pad=10)
    else:
        ax1.set_xlabel('Índice', fontsize=11, fontweight='bold')
        ax1.set_title('Centers vs Index', fontsize=12, pad=10)

    ax1.grid(True, alpha=0.2, linestyle='--')

    # Subplot 2: ESPECTROS BRUTOS PÓS-BASELINE
    ax2 = fig.add_subplot(gs[1])

    # Calcular offset fixo
    offset_step = 0.5
    offset = 0

    for i, nome in enumerate(nomes_ordenados):
        if nome in rs_por_espectro:
            rs = rs_por_espectro[nome]

            # Tentar obter dados brutos pós-baseline
            dados_brutos = None

            # 1. Tentar do parâmetro dados_brutos_por_espectro
            if dados_brutos_por_espectro and nome in dados_brutos_por_espectro:
                dados_brutos = dados_brutos_por_espectro[nome]

            # 2. Tentar extrair do objeto result (se disponível)
            elif nome in todos_results:
                result = todos_results[nome]

                # Verificar se temos dados no result
                if hasattr(result, 'data'):
                    # Dados usados no ajuste (já pós-baseline e smoothing)
                    dados_brutos = result.data
                else:
                    # Fallback: usar best_fit (não é ideal, mas melhor que nada)
                    dados_brutos = result.best_fit
                    print(f"Aviso ({nome}): Usando dados ajustados como fallback para espectros brutos")

            if dados_brutos is not None:
                # Verificar dimensões
                if len(dados_brutos) != len(rs):
                    print(f"Erro: Dimensões inconsistentes para {nome}: dados({len(dados_brutos)}) != rs({len(rs)})")
                    continue

                # Normalizar pelo máximo (como na função original)
                y_max = np.max(dados_brutos) if np.max(dados_brutos) > 0 else 1
                y_norm = dados_brutos / y_max

                # Plotar espectros brutos pós-baseline
                ax2.plot(rs, y_norm + offset, color=cores[i], linewidth=1, alpha=0.7)
                offset += offset_step

    ax2.set_xlabel('Raman Shift (cm⁻¹)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Normalized Intensity', fontsize=11, fontweight='bold')
    ax2.set_title('Spectra', fontsize=12, pad=10)
    ax2.set_yticks([])  # Remover ticks do eixo Y
    ax2.grid(True, alpha=0.1)

    # Legenda (se poucos espectros)
    if num_espectros <= 8:
        ax1.legend(loc='best', fontsize=8)

    # Salvar
    nome_grafico = f"combined_graphics.png"

    try:
        salvar_grafico(fig, nome_grafico, pasta=pasta_destino, mostrar=True)
        print(f"Gráfico combinado salvo: {nome_grafico}")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Erro ao salvar gráfico combinado: {e}")
        plt.close(fig)
        return False

# ========== FUNÇÕES DE SALVAMENTO DE RESULTADOS ==========
def salvar_parametros_consolidados(todos_results, metodo_baseline, pasta_destino, timestamp):
    if not todos_results:
        print("Nenhum resultado para salvar!")
        return False

    # Filtrar resultados válidos
    resultados_validos = {}
    for nome, result in todos_results.items():
        if result is not None and hasattr(result, 'params'):
            resultados_validos[nome] = result
        else:
            print(f"Aviso: Resultado inválido para '{nome}'. Ignorando.")

    if not resultados_validos:
        print("Nenhum resultado válido para salvar!")
        return False

    # Dicionário para organizar dados por espectro
    dados_por_espectro = {}
    todas_metricas = []

    for nome_arquivo, result in resultados_validos.items():
        nome_limpo = os.path.splitext(nome_arquivo)[0]

        # Extrair dados dos picos
        dados_picos = extrair_dados_picos(result.params)

        if not dados_picos:
            print(f"Aviso: Nenhum pico encontrado em '{nome_limpo}'")
            continue

        # Calcular métricas de qualidade do ajuste
        metricas = calcular_metricas_ajuste(result, dados_picos)
        todas_metricas.append((nome_limpo, metricas))

        # Organizar dados para formatação
        dados_por_espectro[nome_limpo] = {
            'result': result,
            'dados_picos': dados_picos,
            'metricas': metricas,
            'chi_quadrado': result.chisqr if hasattr(result, 'chisqr') else None,
            'num_picos': len(dados_picos)
        }

    if not dados_por_espectro:
        print("Nenhum dado de parâmetro encontrado!")
        return False

    # Garantir que a pasta de resultados exista
    pasta_resultados = os.path.join(pasta_destino, "RESULTS")
    try:
        os.makedirs(pasta_resultados, exist_ok=True)
    except Exception as e:
        print(f"Erro ao criar pasta de resultados: {e}")
        return False

    base_nome = f"general information"
    caminho_txt = os.path.join(pasta_resultados, f"{base_nome}.txt")

    try:
        with open(caminho_txt, 'w', encoding='utf-8') as f:
            # ========== CABEÇALHO GERAL ==========
            f.write("GENERAL ANALYSIS INFORMATION\n")
            f.write("=" * 50 + "\n")
            f.write(f"Date/Time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"Baseline Method: {metodo_baseline}\n")
            f.write(f"Number of Spectra: {len(resultados_validos)}\n")
            f.write("=" * 50 + "\n\n")

            # ========== MÉTRICAS DE QUALIDADE COMPARATIVAS ==========
            if len(todas_metricas) > 1:
                f.write("ADJUSTMENT QUALITY COMPARISON\n")
                f.write("-" * 40 + "\n")

                # Ordenar por chi-quadrado (melhor primeiro)
                espectros_ordenados = sorted(
                    [(nome, dados['chi_quadrado'])
                     for nome, dados in dados_por_espectro.items()
                     if dados['chi_quadrado'] is not None],
                    key=lambda x: x[1]
                )

                if espectros_ordenados:
                    f.write("Chi-square ranking (lower is better):\n")
                    for i, (nome, chisqr) in enumerate(espectros_ordenados, 1):
                        f.write(f"  {i:2d}. {nome:30s}: {chisqr:.6f}\n")
                    f.write("\n")

            # ========== DADOS POR ESPECTRO ==========
            for nome_limpo, dados_esp in sorted(dados_por_espectro.items()):
                f.write(f"{'=' * 50}\n")
                f.write(f"SPECTRUM: {nome_limpo}\n")
                f.write(f"{'=' * 50}\n")

                # ========== INFORMAÇÕES DO AJUSTE ==========
                f.write("FITTING INFORMATION:\n")
                f.write("-" * 40 + "\n")

                if dados_esp['chi_quadrado'] is not None:
                    f.write(f"Chi-square: {dados_esp['chi_quadrado']:.6f}\n")

                f.write(f"Number of Lorentzian peaks: {dados_esp['num_picos']}\n")

                # Métricas específicas deste ajuste
                metricas = dados_esp['metricas']
                if metricas['r_quadrado'] is not None:
                    f.write(f"R² (coefficient of determination): {metricas['r_quadrado']:.4f}\n")
                if metricas['residuos_media'] is not None:
                    f.write(f"Mean residual: {metricas['residuos_media']:.6f}\n")
                if metricas['residuos_std'] is not None:
                    f.write(f"Residual standard deviation: {metricas['residuos_std']:.6f}\n")
                if metricas['separacao_minima'] is not None:
                    f.write(f"Minimum peak separation: {metricas['separacao_minima']:.2f} cm⁻¹\n")
                if metricas['sobreposicao_maxima'] > 0:
                    f.write(f"Maximum peak overlap: {metricas['sobreposicao_maxima']:.2%}\n")

                f.write("\n")

        print(f"Arquivo de informações gerais salvo: {os.path.basename(caminho_txt)}")
        return True

    except Exception as e:
        print(f"Erro ao salvar arquivo TXT: {e}")
        import traceback
        traceback.print_exc()
        return False

def calcular_metricas_ajuste(result, dados_picos):
    metricas = {
        'r_quadrado': None,
        'rmse': None,
        'residuos_media': None,
        'residuos_std': None,
        'sobreposicao_maxima': 0,
        'separacao_minima': None
    }

    try:
        # 1. R² (Coeficiente de determinação)
        if hasattr(result, 'data') and hasattr(result, 'best_fit'):
            y_real = result.data
            y_pred = result.best_fit

            ss_res = np.sum((y_real - y_pred) ** 2)
            ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)

            if ss_tot > 0:
                r2 = 1 - (ss_res / ss_tot)
                metricas['r_quadrado'] = max(0, min(1, r2))  # Limitar entre 0 e 1

        # 2. Estatísticas dos resíduos
        if hasattr(result, 'residual'):
            residuos = result.residual
            metricas['residuos_media'] = np.mean(residuos)
            metricas['residuos_std'] = np.std(residuos)

        # 3. Análise de sobreposição entre picos (específico para Raman)
        if len(dados_picos) > 1:
            centers = []
            fwhms = []

            for pico_id, params in dados_picos.items():
                if 'center' in params:
                    centers.append(params['center'])

                # Calcular FWHM
                if 'fwhm' in params and params['fwhm'] > 0:
                    fwhm = params['fwhm']
                elif 'sigma' in params and params['sigma'] > 0:
                    fwhm = 2.0 * params['sigma']
                else:
                    fwhm = 10  # Valor padrão se não disponível

                fwhms.append(fwhm)

            # Calcular sobreposição máxima
            max_sobreposicao = 0
            min_separacao = float('inf')

            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    # Distância entre centers
                    distancia = abs(centers[i] - centers[j])

                    # Soma das metades dos FWHMs
                    soma_fwhms = (fwhms[i] + fwhms[j]) / 2

                    # Calcular sobreposição
                    if distancia < soma_fwhms:
                        sobreposicao = 1 - (distancia / soma_fwhms)
                        max_sobreposicao = max(max_sobreposicao, sobreposicao)

                    # Separacao mínima
                    min_separacao = min(min_separacao, distancia)

            metricas['sobreposicao_maxima'] = max_sobreposicao
            metricas['separacao_minima'] = min_separacao if min_separacao != float('inf') else None

    except Exception as e:
        print(f"Aviso: Erro ao calcular métricas: {e}")

    return metricas

def salvar_parametros_picos_simples(todos_results, pasta_destino, timestamp):
    if not todos_results:
        return False

    dados_completos = {}

    for nome, result in todos_results.items():
        if result is None or not hasattr(result, 'params'):
            continue

        nome_limpo = os.path.splitext(nome)[0]

        # Extrair dados estruturados dos picos usando a função existente
        dados_picos = extrair_dados_picos(result.params)

        if dados_picos:
            dados_completos[nome_limpo] = {
                'centers': [],
                'amplitudes': [],
                'fwhms': [],
                'sigma': []
            }

            # Organizar dados por pico
            picos_ordenados = sorted(dados_picos.keys(),
                                     key=lambda x: int(x[1:]) if x[1:].isdigit() else float('inf'))

            for pico in picos_ordenados:
                if 'center' in dados_picos[pico]:
                    dados_completos[nome_limpo]['centers'].append(dados_picos[pico]['center'])
                if 'amplitude' in dados_picos[pico]:
                    dados_completos[nome_limpo]['amplitudes'].append(dados_picos[pico]['amplitude'])
                if 'fwhm' in dados_picos[pico] and dados_picos[pico]['fwhm'] > 0:
                    dados_completos[nome_limpo]['fwhms'].append(dados_picos[pico]['fwhm'])
                elif 'sigma' in dados_picos[pico]:
                    dados_completos[nome_limpo]['sigma'].append(dados_picos[pico]['sigma'])

    if not dados_completos:
        return False

    # Salvar arquivo
    pasta_resultados = os.path.join(pasta_destino, "RESULTS")
    os.makedirs(pasta_resultados, exist_ok=True)

    caminho = os.path.join(pasta_resultados, f"peaks_parameters.txt")

    try:
        with open(caminho, 'w', encoding='utf-8') as f:
            # Cabeçalho
            f.write("PEAKS PARAMETERS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")

            # Para cada espectro
            for nome, dados in sorted(dados_completos.items()):
                # Extrair variável física
                tipo, valor, unidade = extrair_variavel_fisica(nome)

                f.write(f"SPECTRA: {nome}\n")
                if tipo and valor is not None:
                    f.write(f"Variable: {tipo}={valor:.2f}{unidade}\n")

                # Verificar se temos dados
                if not dados['centers']:
                    f.write("No peaks detected\n\n")
                    continue

                num_picos = len(dados['centers'])
                f.write(f"Number of peaks: {num_picos}\n")
                f.write("-" * 50 + "\n")

                if dados['centers']:
                    centers_str = " ".join([f"{c:.2f}" for c in dados['centers']])
                    f.write(f"Centers (cm⁻¹): {centers_str}\n\n")

                if dados['amplitudes']:
                    amps_str = " ".join([f"{a:.6f}" for a in dados['amplitudes']])
                    f.write(f"Amplitudes: {amps_str}\n\n")

                if dados['fwhms']:
                    fwhms_str = " ".join([f"{fwhm:.3f}" for fwhm in dados['fwhms']])
                    f.write(f"FWHMs (cm⁻¹): {fwhms_str}\n\n")
                elif dados['sigma']:
                    fwhms_calc = [2.0 * sigma for sigma in dados['sigma']]
                    fwhms_str = " ".join([f"{f:.3f}" for f in fwhms_calc])
                    f.write(f"FWHMs (cm⁻¹): {fwhms_str} (calculated from σ)\n\n")

                f.write("\n" + "=" * 70 + "\n\n")

        print(f"Parâmetros completos dos picos salvos: {os.path.basename(caminho)}")
        return True

    except Exception as e:
        print(f"Erro ao salvar parâmetros dos picos: {e}")
        import traceback
        traceback.print_exc()
        return False

# ========== FUNÇÃO DE PROCESSAMENTO MÚLTIPLO (CORRIGIDA) ==========
def processar_multiplos_espectros(arquivos, aplicar_normalizacao=True):
    global pasta_saida_atual

    # Criar pasta para esta análise se ainda não existir
    if pasta_saida_atual is None:
        pasta_saida_atual = criar_pasta_analise()

    print(f"\n{'=' * 60}")
    print("PROCESSAMENTO MÚLTIPLO DE ESPECTROS RAMAN")
    print(f"{'=' * 60}")

    # 1. CARREGAR TODOS OS DADOS
    print(f"\n1. Carregando {len(arquivos)} espectros...")

    dados_originais = []
    for arquivo in arquivos:
        try:
            rs, intensity_original, x, y = _carregar_dados_espectro(arquivo)
            nome_base = os.path.splitext(os.path.basename(arquivo))[0]

            dados_originais.append({
                'caminho': arquivo,
                'nome': nome_base,
                'nome_completo': os.path.basename(arquivo),
                'rs': rs,
                'intensity': intensity_original,
                'rs_completo': x,
                'intensity_completo': y
            })

            print(f"   {nome_base}: {len(rs)} pontos")

        except Exception as e:
            print(f"   ✗ Erro ao carregar {os.path.basename(arquivo)}: {e}")

    if len(dados_originais) < 2:
        print(f"\nErro: É necessário pelo menos 2 espectros válidos para análise comparativa.")
        print(f"      Espectros carregados: {len(dados_originais)}")
        return False

    # 2. SELEÇÃO INTERATIVA DE REGIÃO COMUM (COM ESPECTROS VISÍVEIS)
    print(f"\n2. Selecionando região comum para análise...")

    # Primeiro, precisamos preparar dados combinados para visualização
    if len(dados_originais) > 0:
        # Usar o primeiro espectro como base para a função
        rs_base = dados_originais[0]['rs']
        intensity_base = dados_originais[0]['intensity']

        # Mas vamos criar uma figura personalizada que mostra TODOS os espectros
        fig, ax = plt.subplots(figsize=tamanho_tela)

        # Plotar todos os espectros
        colormap = plt.cm.rainbow
        cores = colormap(np.linspace(0, 1, len(dados_originais)))

        for i, dados in enumerate(dados_originais):
            ax.plot(dados['rs'], dados['intensity'],
                    color=cores[i],
                    label=dados['nome'],
                    linewidth=1.5,
                    alpha=0.7)

        # Configurar gráfico
        ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Intensity (a.u.)', fontsize=14, fontweight='bold')
        ax.set_title('Selecione Região Comum para Todos os Espectros', fontsize=16, pad=15)
        ax.grid(True, alpha=0.3)

        # Legenda
        if len(dados_originais) <= 10:
            ax.legend(loc='upper right', fontsize=9)
        elif len(dados_originais) <= 15:
            ax.legend(loc='upper right', fontsize=8, ncol=2, title='Espectros', title_fontsize=9)
        else:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2, title='Espectros',
                      title_fontsize=8)

        # Variáveis para seleção
        regiao_selecionada = None
        retangulo_selecao = None
        x_press = None
        retangulo_atual = None

        # Callbacks (código similar ao da função original)
        def on_press(event):
            nonlocal x_press, retangulo_atual, retangulo_selecao

            if event.inaxes != ax:
                return

            # Limpar seleção anterior
            if retangulo_selecao:
                retangulo_selecao.remove()
                for line in ax.lines:
                    if line.get_linestyle() == '--' and line.get_color() == 'green':
                        line.remove()
                retangulo_selecao = None

            x_press = event.xdata

            # Criar retângulo de seleção
            y_min, y_max = ax.get_ylim()
            retangulo_atual = plt.Rectangle((x_press, y_min), 0, y_max - y_min,
                                            edgecolor='blue', facecolor='blue', alpha=0.2)
            ax.add_patch(retangulo_atual)
            fig.canvas.draw()

        def on_motion(event):
            nonlocal retangulo_atual, x_press

            if event.inaxes != ax or x_press is None:
                return

            x_current = event.xdata
            width = x_current - x_press
            retangulo_atual.set_width(width)
            fig.canvas.draw()

        def on_release(event):
            nonlocal x_press, regiao_selecionada, retangulo_atual, retangulo_selecao

            if event.inaxes != ax or x_press is None:
                return

            x_current = event.xdata
            x_left = min(x_press, x_current)
            x_right = max(x_press, x_current)

            # Verificar se região é muito pequena
            if abs(x_right - x_left) < (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01:
                print("Região muito pequena. Selecione uma área maior.")
                retangulo_atual.remove()
                retangulo_atual = None
                x_press = None
                return

            regiao_selecionada = (x_left, x_right)
            retangulo_selecao = retangulo_atual

            # Destacar região
            retangulo_selecao.set_edgecolor('red')
            retangulo_selecao.set_facecolor('red')
            retangulo_selecao.set_alpha(0.3)

            # Adicionar linhas verticais
            ax.axvline(x=x_left, color='green', linestyle='--', linewidth=2, alpha=0.7)
            ax.axvline(x=x_right, color='green', linestyle='--', linewidth=2, alpha=0.7)

            fig.canvas.draw()
            x_press = None
            retangulo_atual = None

        def on_key(event):
            nonlocal regiao_selecionada, retangulo_selecao

            if event.key == 'enter':
                if regiao_selecionada:
                    print(f"\nRegião confirmada: {regiao_selecionada[0]:.1f} - {regiao_selecionada[1]:.1f} cm⁻¹")
                else:
                    print("\nNenhuma região selecionada! Usando espectro completo.")
                    regiao_selecionada = None
                plt.close()
            elif event.key == 'escape':
                print("\nUsando espectro completo.")
                regiao_selecionada = None
                plt.close()
            elif event.key in ['r', 'R']:
                # Limpar seleção
                if retangulo_selecao:
                    retangulo_selecao.remove()
                    retangulo_selecao = None
                for line in ax.lines:
                    if line.get_linestyle() == '--' and line.get_color() == 'green':
                        line.remove()
                regiao_selecionada = None
                print("Seleção limpa.")
                fig.canvas.draw()
            elif event.key in ['t', 'T']:
                print("\nUsando espectro completo.")
                regiao_selecionada = None
                plt.close()

        # Botões
        from matplotlib.widgets import Button

        ax_confirmar = plt.axes([0.78, 0.01, 0.15, 0.04])
        ax_todo = plt.axes([0.25, 0.01, 0.15, 0.04])
        ax_limpar = plt.axes([0.62, 0.01, 0.15, 0.04])

        btn_confirmar = Button(ax_confirmar, 'Confirmar (Enter)', color='lightgreen')
        btn_todo = Button(ax_todo, 'Usar Todo (T)', color='lightblue')
        btn_limpar = Button(ax_limpar, 'Limpar (R)', color='orange')

        def confirmar_selecao(event):
            nonlocal regiao_selecionada
            if regiao_selecionada:
                print(f"\nRegião confirmada: {regiao_selecionada[0]:.1f} - {regiao_selecionada[1]:.1f} cm⁻¹")
            else:
                print("\nNenhuma região selecionada! Usando espectro completo.")
                regiao_selecionada = None
            plt.close()

        def usar_todo_espectro(event):
            nonlocal regiao_selecionada
            print("\nUsando espectro completo.")
            regiao_selecionada = None
            plt.close()

        def limpar_selecao(event):
            nonlocal regiao_selecionada, retangulo_selecao
            if retangulo_selecao:
                retangulo_selecao.remove()
                retangulo_selecao = None
            for line in ax.lines:
                if line.get_linestyle() == '--' and line.get_color() == 'green':
                    line.remove()
            regiao_selecionada = None
            print("Seleção limpa.")
            fig.canvas.draw()

        # Conectar eventos dos botões
        btn_confirmar.on_clicked(confirmar_selecao)
        btn_todo.on_clicked(usar_todo_espectro)
        btn_limpar.on_clicked(limpar_selecao)

        # Conectar eventos do mouse e teclado
        cid_press = fig.canvas.mpl_connect('button_press_event', on_press)
        cid_motion = fig.canvas.mpl_connect('motion_notify_event', on_motion)
        cid_release = fig.canvas.mpl_connect('button_release_event', on_release)
        cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

        # Ajustar layout e mostrar
        plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.12)
        plt.show(block=True)

        # Desconectar eventos após fechar a janela
        fig.canvas.mpl_disconnect(cid_press)
        fig.canvas.mpl_disconnect(cid_motion)
        fig.canvas.mpl_disconnect(cid_release)
        fig.canvas.mpl_disconnect(cid_key)

    else:
        regiao_selecionada = None

    # 3. APLICAR CORTE NA REGIÃO SELECIONADA
    print(f"\n3. Preparando dados para processamento...")

    dados_processar = []
    arquivos_temp = []

    for dados in dados_originais:
        # Aplicar corte se região foi selecionada
        if regiao_selecionada:
            x_left, x_right = regiao_selecionada
            mask = (dados['rs_completo'] >= x_left) & (dados['rs_completo'] <= x_right)
            rs_cortado = dados['rs_completo'][mask]
            intensity_cortado = dados['intensity_completo'][mask]
        else:
            rs_cortado = dados['rs']
            intensity_cortado = dados['intensity']

        if len(rs_cortado) < 10:
            print(f"   Aviso: {dados['nome']} tem apenas {len(rs_cortado)} pontos na região selecionada")
            continue

        # Criar arquivo temporário
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".txt",
            delete=False
        )
        np.savetxt(temp_file.name, np.column_stack((rs_cortado, intensity_cortado)))

        dados_processar.append({
            'nome': dados['nome_completo'],
            'nome_arquivo_original': dados['nome_completo'],
            'caminho_temp': temp_file.name,
            'rs': rs_cortado,
            'intensity': intensity_cortado
        })

        arquivos_temp.append(temp_file.name)

    if len(dados_processar) < 2:
        print(f"\nErro: Menos de 2 espectros com dados suficientes após corte.")

        # Limpar arquivos temporários
        for temp_file in arquivos_temp:
            try:
                os.unlink(temp_file)
            except:
                pass

        return False

    # 4. CRIAR GRÁFICO DOS ESPECTROS BRUTOS (COM REGIÃO SELECIONADA)
    print(f"\n4. Criando gráfico dos espectros normalizados")

    dados_para_grafico = {}
    for dados in dados_processar:
        dados_para_grafico[dados['nome']] = {
            'rs': dados['rs'],
            'intensity': dados['intensity']
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    criar_grafico_espectros_brutos_normalizados(
        dados_para_grafico, pasta_saida_atual, timestamp
    )

    # 5. PROCESSAR ESPECTRO DE REFERÊNCIA
    print(f"\n5. Processando espectro de referência...")

    referencia = dados_processar[0]
    print(f"   Referência: {referencia['nome']}")

    # Resetar cache de baseline se existir
    if hasattr(processar_espectro_referencia_completo, '_cache_baseline'):
        processar_espectro_referencia_completo._cache_baseline = None

    parametros_padrao = ajustar_parametros_para_normalizacao(None)
    referencia_resultado = None

    while referencia_resultado is None:
        referencia_resultado = processar_espectro_referencia_completo(
            referencia['caminho_temp'], parametros_padrao, aplicar_normalizacao, nome_original=referencia['nome']
        )

        if referencia_resultado == 'CANCELAR_TUDO':
            # Limpar e retornar
            for temp_file in arquivos_temp:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            return False

        if referencia_resultado is None:
            print("   Reiniciando processamento da referência...")
            continue

    # Extrair informações da referência
    metodo_baseline = referencia_resultado['metodo_baseline']
    parametros_referencia = referencia_resultado['parametros_finais']

    # 6. PROCESSAR ESPECTROS RESTANTES
    print(f"\n6. Processando espectros restantes...")

    todos_results = {referencia['nome']: referencia_resultado['result']}
    rs_por_espectro = {referencia['nome']: referencia['rs']}
    dados_brutos_por_espectro = {}

    for i, dados in enumerate(dados_processar[1:], 1):
        print(f"   {i}/{len(dados_processar) - 1}: {dados['nome']}")

        try:
            result, rs, intensity_original = processar_espectro_comparativo_iterativo(
                dados['caminho_temp'], metodo_baseline, parametros_referencia, aplicar_normalizacao,
                nome_original=dados['nome'])

            if result is not None:
                todos_results[dados['nome']] = result
                rs_por_espectro[dados['nome']] = rs

                # Armazenar dados brutos pós-baseline
                if hasattr(result, 'data'):
                    dados_brutos_por_espectro[dados['nome']] = result.data
                else:
                    dados_brutos_por_espectro[dados['nome']] = intensity_original

                print(f"Processado com sucesso")
            else:
                print(f" Não foi possível processar")

        except Exception as e:
            print(f"Erro: {e}")

    # 7. LIMPAR ARQUIVOS TEMPORÁRIOS
    for temp_file in arquivos_temp:
        try:
            os.unlink(temp_file)
        except:
            pass

    # 8. SALVAR RESULTADOS E CRIAR GRÁFICOS FINAIS
    if len(todos_results) >= 2:
        print(f"\n7. Salvando resultados e criando gráficos finais...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Salvar parâmetros consolidados
        salvar_parametros_consolidados(todos_results, metodo_baseline, pasta_saida_atual, timestamp)

        # Salvar centers em formato simples
        salvar_parametros_picos_simples(todos_results, pasta_saida_atual, timestamp)

        # CORREÇÃO: Remover o parâmetro 'mostrar' da chamada
        criar_grafico_combinado_completo(
            todos_results,
            rs_por_espectro,
            pasta_saida_atual,
            timestamp,
            dados_brutos_por_espectro=dados_brutos_por_espectro
        )

        print(f"\n{'=' * 60}")
        print("ANÁLISE COMPARATIVA CONCLUÍDA COM SUCESSO!")
        print(f"{'=' * 60}")
        print(f"Espectros analisados: {len(todos_results)}")
        print(f"Método de baseline: {metodo_baseline}")
        print(f"Pasta de resultados: {pasta_saida_atual}")
        print(f"{'=' * 60}")

        return True

    else:
        print(f"\nErro: Apenas {len(todos_results)} espectro(s) processado(s) com sucesso.")
        print(f"       Mínimo necessário: 2 espectros")

        return False

# ========== FUNÇÃO PRINCIPAL ==========
def main():
    global pasta_saida_atual

    print("=== PYTHON SPECTRUM ANALYSIS RAMAN (PySAR) ===")

    # Criar a janela Tkinter UMA VEZ e reutilizar
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    try:
        while True:
            # Resetar pasta de saída para nova análise
            pasta_saida_atual = None

            # Limpar figuras matplotlib pendentes
            plt.close('all')

            # Selecionar arquivos automaticamente
            arquivos = filedialog.askopenfilenames(
                title="Selecione um ou mais arquivos de espectro Raman",
                filetypes=[("Arquivos de texto", "*.txt"), ("Arquivos CSV", "*.csv"),
                           ("Todos os arquivos", "*.*")],
                parent=root)

            if not arquivos:
                print("Nenhum arquivo selecionado. Programa encerrado.")
                break

            num_arquivos = len(arquivos)

            print(f"\nArquivos selecionados ({num_arquivos}):")
            for i, arquivo in enumerate(arquivos, 1):
                print(f"  {i}: {os.path.basename(arquivo)}")

            # Configurar normalizacao automatica
            aplicar_normalizacao = True  # Sempre normalizar

            try:
                # Criar pasta para esta análise
                pasta_analise = criar_pasta_analise()
                print(f"\nIniciando análise em: {pasta_analise}")

                # Executar analise
                if num_arquivos == 1:
                    print("\n=== INICIANDO ANALISE INDIVIDUAL ===")
                    resultado = processar_espectro_individual(arquivos[0], aplicar_normalizacao)
                    if resultado is None:
                        print("A análise individual não foi concluída com sucesso.")

                elif num_arquivos >= 2:
                    print("\n=== INICIANDO ANALISE COMPARATIVA ===")
                    sucesso = processar_multiplos_espectros(arquivos, aplicar_normalizacao)
                    if not sucesso:
                        print("A análise comparativa não foi concluída com sucesso.")
                else:
                    print("Erro: Número inválido de arquivos.")

            except Exception as e:
                print(f"\nERRO CRÍTICO durante a análise: {e}")
                import traceback
                traceback.print_exc()
                print("\nContinuando para próxima análise...")

            # Perguntar se quer fazer nova análise
            print(f"\n{'=' * 60}")
            nova_analise = input("Deseja fazer uma nova análise? (s/n): ").strip().lower()

            # CORREÇÃO: Condição invertida - se NÃO for uma resposta afirmativa, sair
            if nova_analise not in ['s', 'sim', 'y', 'yes', '']:
                print("\nPrograma encerrado.")
                print("Obrigado por usar PySAR!")
                break
            else:
                print("\n" + "=" * 60)
                print("Iniciando nova análise...\n")

    finally:
        # Garantir que a janela Tkinter seja destruída ao sair
        try:
            root.destroy()
        except:
            pass

        # Limpar todas as figuras matplotlib
        plt.close('all')

# ========== EXECUÇÃO DO PROGRAMA ==========
if __name__ == "__main__":
    # Adicionar tratamento de exceções globais
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrograma interrompido pelo usuário.")
    except Exception as e:
        print(f"\nERRO INESPERADO: {e}")
        import traceback

        traceback.print_exc()
        input("\nPressione Enter para sair...")
