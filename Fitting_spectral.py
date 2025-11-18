# Seja bem vindo(a) ao PySAR (Python Spectrum Analisys Raman)
# Importando bibliotecas necessárias
import matplotlib.pyplot as plt
import peakutils as pk
import numpy as np
import tkinter as tk
import os
from lmfit.models import LorentzianModel, ConstantModel
from sklearn.cluster import KMeans
from scipy.signal import find_peaks, savgol_filter
from pyPreprocessing import baseline_correction as bs, smoothing as smo, transform as tra
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Selecione o arquivo de espectro Raman",
    filetypes=[
        ("Arquivos de texto", "*.txt"),
        ("Arquivos CSV", "*.csv"),
        ("Todos os arquivos", "*.*")
    ]
)

if file_path:
    print(f"Arquivo selecionado: {os.path.basename(file_path)}")

# Função para baseline por quantis (robusta)
def baseline_quantil(intensity, window=101, quantile=0.1):
    """
    Baseline baseada em quantis - não segue picos
    """
    baseline = np.zeros_like(intensity)
    half_window = window // 2

    for i in range(len(intensity)):
        start = max(0, i - half_window)
        end = min(len(intensity), i + half_window + 1)
        baseline[i] = np.quantile(intensity[start:end], quantile)

    return baseline

# Plotagem do gráfico dos dados
data = np.loadtxt(file_path)
x = data[:, 0]
y = data[:, 1]

# Limitando os dados
print("Limitando os dados:")
x_min = float(input("x_mín= "))
x_max = float(input("x_máx= "))
rs = x[abs(x - x_min).argmin():abs(x - x_max).argmin()]
intensity_original = y[abs(x - x_min).argmin():abs(x - x_max).argmin()]

print(f"Processando {len(rs)} pontos no intervalo {rs[0]:.1f} - {rs[-1]:.1f} cm⁻¹")

# TESTAR MÚLTIPLAS BASELINES
print("\n=== TESTANDO DIFERENTES MÉTODOS DE BASELINE ===")

# Método 1: Peakutils com deg moderado
try:
    baseline1 = pk.baseline(intensity_original, deg=3, max_it=1000, tol=1e-10)  # deg=3 é mais balanceado
    intensity_corr1 = intensity_original - baseline1
    print("✓ Peakutils (deg=3) - OK")
except Exception as e:
    baseline1 = np.zeros_like(intensity_original)
    intensity_corr1 = intensity_original
    print(f"✗ Peakutils falhou: {e}")

# Método 2: Baseline por quantis
try:
    baseline2 = baseline_quantil(intensity_original, window=151, quantile=0.15)
    intensity_corr2 = intensity_original - baseline2
    print("✓ Baseline por quantis - OK")
except Exception as e:
    baseline2 = np.zeros_like(intensity_original)
    intensity_corr2 = intensity_original
    print(f"✗ Quantis falhou: {e}")

# Método 3: SNIP (do pyPreprocessing) - geralmente bom para picos
try:
    inten_reshaped = intensity_original.reshape(1, -1)
    baseline3 = bs.generate_baseline(inten_reshaped, mode='SNIP')[0]
    intensity_corr3 = intensity_original - baseline3
    print("✓ SNIP - OK")
except Exception as e:
    baseline3 = np.zeros_like(intensity_original)
    intensity_corr3 = intensity_original
    print(f"✗ SNIP falhou: {e}")

# COMPARAÇÃO VISUAL DAS BASELINES
plt.figure(figsize=(8, 6))

# Gráfico 1: Todas as baselines
plt.subplot(2, 1, 1)
plt.plot(rs, intensity_original, 'black', label='Dados Originais', linewidth=1.5, alpha=0.8)
plt.plot(rs, baseline1, 'blue', label='Peakutils (deg=3)', linewidth=2, alpha=0.8)
plt.plot(rs, baseline2, 'green', label='Quantis (window=151)', linewidth=2, alpha=0.8)
plt.plot(rs, baseline3, 'orange', label='SNIP', linewidth=2, alpha=0.8)
plt.xlabel('Raman Shift (cm⁻¹)')
plt.ylabel('Intensity (a.u.)')
plt.legend()
plt.title('Comparação de Métodos de Baseline')
plt.grid(True, alpha=0.3)

# Gráfico 2: Dados corrigidos
plt.subplot(2, 1, 2)
plt.plot(rs, intensity_corr1, 'blue', label='Corrigido - Peakutils', linewidth=1, alpha=0.8)
plt.plot(rs, intensity_corr2, 'green', label='Corrigido - Quantis', linewidth=1, alpha=0.8)
plt.plot(rs, intensity_corr3, 'orange', label='Corrigido - SNIP', linewidth=1, alpha=0.8)
plt.xlabel('Raman Shift (cm⁻¹)')
plt.ylabel('Intensity Corrigida (a.u.)')
plt.legend()
plt.title('Dados Após Correção de Baseline')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# SELEÇÃO DA MELHOR BASELINE
print("\n=== SELECIONE A MELHOR BASELINE ===")
print("Analise os gráficos e escolha o método que:")
print("- Não sobe nos picos")
print("- Não é muito rígido")
print("- Preserva os vales naturais")
print("- Permite boa detecção de picos")

escolha = input("\nDigite o número do método (1-3): ").strip()

if escolha == "1":
    baseline_final = baseline1
    intensity_corrigida = intensity_corr1
    metodo = "Peakutils (deg=3)"
elif escolha == "2":
    baseline_final = baseline2
    intensity_corrigida = intensity_corr2
    metodo = "Quantis"
elif escolha == "3":
    baseline_final = baseline3
    intensity_corrigida = intensity_corr3
    metodo = "SNIP"
else:
    baseline_final = baseline1  # Padrão
    intensity_corrigida = intensity_corr1
    metodo = "Peakutils (deg=3) - padrão"

print(f"\nMétodo selecionado: {metodo}")

# Suavização
inten = intensity_corrigida.reshape(1, -1)
intensity = smo.smoothing(inten, mode='sav_gol')[0]

# Detecção de picos com parâmetros ajustados
peaks, props = find_peaks(
    intensity,
    height=100,
    distance=1,
    prominence=0.3,
    width=4.5,
    wlen=50
)

peak_positions = rs[peaks]
peak_heights = intensity[peaks]

print(f"\nPicos detectados: {len(peaks)}")
print(f"Método de baseline: {metodo}")

k = 60  # Número de Lorentzianas

# VERIFICAÇÃO CRÍTICA - Se k for muito grande
if len(peak_positions) < k:
    print(f"Aviso: Apenas {len(peak_positions)} picos detectados, reduzindo k para este valor")
    k = len(peak_positions)  # Ajusta k automaticamente

if k > 60:  # Limite prático para ajuste estável
    print(f"Aviso: k={k} é muito alto. Limitando a 60 Lorentzianas para estabilidade")
    k = 60

peak_data = np.column_stack((peak_positions, peak_heights))

# Para k grande, aumentamos n_init e max_iter
kmeans = KMeans(n_clusters=k, random_state=0, n_init=20, max_iter=500).fit(peak_data)
centers = sorted(kmeans.cluster_centers_, key=lambda x: x[0])

model = None
params = None

# MODIFICAÇÃO IMPORTANTE: Limitar a amplitude inicial para k grande
for i, (center, amp) in enumerate(centers):
    lorentz = LorentzianModel(prefix=f'l{i + 1}_')
    if model is None:
        model = lorentz
        params = lorentz.make_params()
    else:
        model += lorentz
        params.update(lorentz.make_params())

    # Ajuste mais conservador para k grande
    amp_factor = 5 if k > 40 else 10  # Reduz factor para k grande
    sigma_init = 3 if k > 40 else 5  # Sigma inicial menor para k grande

    params[f'l{i + 1}_center'].set(value=center, min=rs.min(), max=rs.max())
    params[f'l{i + 1}_amplitude'].set(value=amp * amp_factor, min=0, max=amp * 20)
    params[f'l{i + 1}_sigma'].set(value=sigma_init, min=0.5, max=20)  # Limites mais restritos

# ADICIONAR MODELO CONSTANTE PARA MELHOR ESTABILIDADE
constant = ConstantModel(prefix='c_')
model = constant + model
params.update(constant.make_params())
params['c_c'].set(value=np.mean(intensity) * 0.1, min=0)

print(f"Iniciando ajuste com {k} Lorentzianas...")

# Ajuste com parâmetros otimizados para muitos componentes
try:
    result = model.fit(data=intensity, params=params, x=rs,
                       method='leastsq',  # Método mais robusto
                       max_nfev=5000,  # Máximo de iterações
                       nan_policy='omit')

    comps = result.eval_components()

    # GRÁFICO MELHORADO - MOSTRAR BASELINE
    plt.figure(figsize=(6, 12))

    # Subplot 1: Comparação antes/depois da baseline
    plt.subplot(2, 1, 1)
    plt.plot(rs, intensity_original, 'gray', label='Raw Data', linewidth=1, alpha=0.6)
    plt.plot(rs, baseline_final, 'orange', label='Baseline', linewidth=2, linestyle='--')
    plt.plot(rs, intensity_corrigida, 'blue', label='Corrected Data', linewidth=1, alpha=0.8)
    plt.xlabel(r'Raman shift (cm$^{-1}$)')
    plt.ylabel(r'Intensity (a.u.)')
    plt.legend()
    plt.title('Baseline Correction')
    plt.grid(True, alpha=0.3)

    # Subplot 2: Resultado do ajuste
    plt.subplot(2, 1, 2)
    plt.plot(rs, intensity, 'k-', label='Smoothed Data', linewidth=1, alpha=0.7)
    plt.plot(rs, result.best_fit, 'r-', label='Fitting', linewidth=2)

    # Plot componentes apenas se k for razoável
    if k <= 30:
        for name, comp in comps.items():
            if name != 'c_':  # Ignorar componente constante
                plt.plot(rs, comp, '--', alpha=0.5, linewidth=1)

    plt.xlabel(r'Raman shift (cm$^{-1}$)')
    plt.ylabel(r'Intensity (a.u.)')
    plt.legend()
    plt.title('Fitting After Baseline Correction')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # VERIFICAR SE O AJUSTE FOI BEM-SUCEDIDO
    if hasattr(result, 'chisqr'):
        print(f"Fitting completed with chi-squared: {result.chisqr:.2f}")
        print(f"Chi-squared reduced: {result.redchi:.2f}")
        print(f"Peaks detected: {len(peaks)}")
        print(f"Baseline method: peakutils")
    else:
        print("Ajuste concluído, mas sem métricas de qualidade")

except Exception as e:
    print(f"Erro no ajuste: {e}")
    print("Tentando com método alternativo...")

    # Tentativa com método diferente
    try:
        result = model.fit(data=intensity, params=params, x=rs,
                           method='nelder',  # Método simplex
                           max_nfev=3000)
        comps = result.eval_components()

        plt.figure(figsize=(12, 6))
        plt.plot(rs, intensity, 'k-', label='Dados', linewidth=1, alpha=0.7)
        plt.plot(rs, result.best_fit, 'r-', label='Ajuste', linewidth=2)
        plt.xlabel(r'Raman shift (cm$^{-1}$)')
        plt.ylabel(r'Intensity (a.u.)')
        plt.legend()
        plt.title(f'Ajuste com {k} Lorentzianas (método alternativo)')
        plt.show()

    except Exception as e2:
        print(f"Falha no ajuste: {e2}")
        print("Recomendo reduzir o valor de k ou melhorar a detecção de picos")

try:
    plt.show()
except KeyboardInterrupt:
    print("Gráfico fechado pelo usuário")
    plt.close()
