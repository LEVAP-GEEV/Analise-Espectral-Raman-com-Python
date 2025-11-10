# Importando bibliotecas necessárias
import matplotlib.pyplot as plt
import peakutils as pk
import numpy as np
from lmfit.models import LorentzianModel, ConstantModel
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
from pyPreprocessing import baseline_correction as bs, smoothing as smo, transform as tra

# Plotagem do gráfico dos dados (em data, insira o local onde estão seus dados)
data = np.genfromtxt(r"")
x = data[:, 0]
y = data[:, 1]

# Limitando os dados (fica a seu critério colocar os valores mínimo e máximo)
rs = x[abs(x - mínimo).argmin():abs(x - máximo).argmin()]
intensity_original = y[abs(x - mínimo).argmin():abs(x - máximo).argmin()]

# Calcular baseline
baseline = pk.baseline(intensity_original, deg=10, max_it=1000, tol=1e-10)  # deg=2 é melhor

# Aplicar correção de baseline
intensity_corrigida = intensity_original - baseline

inten = intensity_corrigida.reshape(1,-1)
baseline = bs.generate_baseline(inten, mode='ALSS')[0]

# Plotar baseline (opcional)
plt.plot(rs, intensity_corrigida, 'o', label='Dados', linewidth=0.5, alpha=1)
plt.plot(rs, baseline, 'r-', label='Baseline', linewidth=3, alpha=0.7)
plt.xlabel('Wavenumber (cm^-1)')
plt.ylabel('Intensity (a.u)')
plt.show()

# Suavização
inten = intensity_corrigida.reshape(1, -1)
intensity = smo.smoothing(inten, mode='sav_gol')[0]

# Detecção de picos com parâmetros ajustados
peaks, props = find_peaks(
    intensity,
    height=0.05,
    distance=1,
    prominence=0.02,
    width=1,
    wlen=30
)

peak_positions = rs[peaks]
peak_heights = intensity[peaks]

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
    plt.plot(rs, baseline, 'orange', label='Baseline (peakutils deg=2)', linewidth=2, linestyle='--')
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
        print(f"Baseline method: peakutils (deg=2)")
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
