# Session Ranker — presupuesto de analista, geometría conjunta

**Author:** David Jaimes Olivo ([@forkaizen2023-sys](https://github.com/forkaizen2023-sys))

**What this is:** a lab that ranks unusual *sessions* so a human reviews the top-k first, and that compares Isolation Forest, max|z|, a rate residual, and Mahalanobis on the same holdout.

**What this is not:** an APT detector, a SIEM, ATT&CK coverage, or a 99% false-positive claim.

**What changed in v0.4:** the published 42→99 table was a screenshot. Fit is benign-only when labels exist. `low_and_slow` no longer lives only at night. AUROC is not rounded to 1.000 when scores overlap. `--sweep` and a bimodal manifold exist so the claim can lose.

Open [`planteamiento.html`](planteamiento.html) for the iterations and the matrix.

---

## El problema que sí resuelve

Un lote de 2.000 sesiones no se inspecciona a mano. El analista abre *k* (aquí 30). El objeto matemático es un ranking $\pi$ y

$$\mathrm{Prec}@k,\quad \mathrm{Rec}@k,\quad \mathrm{AP},\quad \mathrm{AUROC},\quad \text{inversiones}.$$

Hay dos geometrías distintas:

1. **Cola 1D** (`brute_force`): extremo en logins fallidos. max|z| basta.
2. **Fuera del manifold** (`rate_exfil`, `low_and_slow`): bytes $\approx$ rate $\times$ duration en tráfico normal. El ataque está *dentro* de los márgenes univariantes y *fuera* de la franja conjunta.

El control de primer principio no es Isolation Forest. Es el residual

$$|\;\log(1+\text{bytes})-\log(1+t)-\mu\;|$$

Si Mahalanobis no gana a eso *y* a max|z|, la historia 5D es teatro. En el generador unimodal gana porque también usa failed-logins y $\Sigma$. En el generador bimodal (dos tasas legítimas) $\Sigma$ deja de ser el modelo y Rec@k de Mahalanobis cae a ~1/3 — solo sobrevive `brute_force`.

## Resultado reproducible

Protocolo: **fit benign**, train seed 42, test seed 99, $n=2000$, $P=24$, $k=30$, `slow_hours=mixed`.

| Método | P@k | R@k | AP | AUROC | inv | worst | brute | low_and_slow | rate_exfil |
|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest | 0.200 | 0.250 | 0.091 | 0.875633 | 5898 | 610 | 0.75 | 0.00 | 0.00 |
| max\|z\| (MAD) | 0.267 | 0.333 | 0.360 | 0.869233 | 6181 | 587 | 1.00 | 0.00 | 0.00 |
| rate residual | 0.500 | 0.625 | 0.544 | 0.732309 | 12695 | 1995 | 0.00 | 1.00 | 0.875 |
| Mahalanobis | **0.800** | **1.000** | **0.987** | **0.999852** | **7** | **25** | 1.00 | 1.00 | 1.00 |

AUROC 0.999852 con 7 inversiones y margin −6.3 no es separación limpia. Es “los 24 ataques caben en k=30”.

Sweep, mismo train 42, 8 test seeds `{1,7,13,21,42,99,123,999}`:

| Método | R median | R min | AUROC median | worst max |
|---|---|---|---|---|
| Isolation Forest | 0.208 | 0.083 | 0.875 | 813 |
| max\|z\| | 0.333 | 0.333 | 0.880 | 783 |
| rate residual | 0.604 | 0.542 | 0.784 | 1995 |
| Mahalanobis | 1.000 | **0.958** | 0.99988 | **31** |

Misma receta, `manifold=bimodal` (dos tasas normales):

| Método | R median | R min | AUROC median | worst max |
|---|---|---|---|---|
| Mahalanobis | **0.333** | 0.333 | 0.974 | 172 |

Eso es el resultado que importa: $\Sigma$ no es un detector de sesiones. Es el estimador correcto *cuando el generador es un solo elipsoide*.

## Uso

```bash
pip install -r requirements.txt
python log_analyzer.py
python log_analyzer.py --sweep
python log_analyzer.py --sweep --manifold bimodal
python log_analyzer.py --input data/sample_sessions.csv --top-k 30
python log_analyzer.py --input data/sample_sessions.csv --save-model models/ranker.pkl
python -m pytest -q
```

`--fit-on auto` (default): benign si hay `is_attack`, si no all.
`--transductive` ajusta y puntúa el mismo lote.
`--cov ledoit` cambia EmpiricalCovariance por Ledoit-Wolf.

CSV mínimo: `timestamp,bytes_sent,duration_seconds,failed_login_attempts`.
Opcional: `src_ip`, `label`, `is_attack`.
Negativos y timestamps rotos fallan en carga.

Features del modelo: `log(bytes)`, `log(duration)`, `failed_login_attempts`, $\sin(2\pi h/24)$, $\cos(2\pi h/24)$.
El residual de tasa **no** entra al bosque ni a $\Sigma$; es un ranker aparte.

## Estructura

```
LICENSE
planteamiento.html   Vista: iteraciones + matriz + v0.4
log_analyzer.py      CLI
simulator.py         Manifold single|bimodal + 3 familias
detector.py          IF, max|z|, residual, Mahalanobis
sweep.py             Semillas, no screenshots
tests/
data/sample_sessions.csv
pyproject.toml
.github/workflows/pytest.yml
```

## License

MIT. See [LICENSE](LICENSE).

## Autor

David Jaimes Olivo — investigación en AppSec, cadena de suministro CI y detección.

- GitHub: [github.com/forkaizen2023-sys](https://github.com/forkaizen2023-sys)
- Relacionado: [agent-oidc-coupling](https://github.com/forkaizen2023-sys/agent-oidc-coupling)
