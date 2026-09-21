# Session Ranker — presupuesto de analista, geometría conjunta

**Author:** David Jaimes Olivo ([@forkaizen2023-sys](https://github.com/forkaizen2023-sys))

**What this is:** a lab that ranks unusual *sessions* so a human reviews the top-k first, and that **compares** Isolation Forest, max|z| and Mahalanobis on the same holdout.

**What this is not:** an APT detector, a SIEM, or a 99% false-positive claim.

Open [`https://forkaizen2023-sys.github.io/session-ranker/`](planteamiento.html) for the three iterations, the mathematical matrix, and the final numbers in one view.



---

## El problema que sí resuelve

Un lote de 2.000 sesiones no se inspecciona a mano. El analista abre *k* (aquí 30). El objeto matemático es un ranking $\pi$ y

$$\mathrm{Prec}@k,\quad \mathrm{Rec}@k,\quad \mathrm{AP},\quad \mathrm{AUROC}.$$

Hay dos geometrías distintas:

1. **Cola 1D** (`brute_force`): extremo en logins fallidos. `max|z|` e Isolation Forest bastan.
2. **Fuera del manifold** (`rate_exfil`, `low_and_slow`): bytes $\approx$ rate $\times$ duration en tráfico normal. El ataque está *dentro* de los márgenes univariantes y *fuera* de la franja conjunta. Ahí Isolation Forest (cortes alineados a los ejes) se distrae con los extremos del manifold; Mahalanobis usa $\Sigma$.

## Resultado reproducible (inductivo)

Fit seed 42, score seed 99, $n=2000$, 24 ataques, $k=30$:

| Método | P@k | R@k | AP | AUROC | brute | low_and_slow | rate_exfil |
|---|---|---|---|---|---|---|---|
| Isolation Forest | 0.267 | 0.333 | 0.326 | 0.904 | 1.00 | 0.00 | 0.00 |
| max\|z\| (mediana/MAD) | 0.267 | 0.333 | 0.380 | 0.912 | 1.00 | 0.00 | 0.00 |
| Mahalanobis | **0.800** | **1.000** | **0.984** | **1.000** | 1.00 | 1.00 | 1.00 |
| Reglas (sin techo) | 0.476 | 0.417 | — | — | — | — | — |

Isolation Forest no “pierde el lab”: mide lo que debe. En outliers de correlación, el estimador alineado a $\Sigma$ es el que corresponde.

## Uso

```bash
pip install -r requirements.txt
python log_analyzer.py
python log_analyzer.py --input data/sample_sessions.csv --top-k 30
python -m pytest -q
```

CI: GitHub Actions corre `pytest` en Python 3.11 y 3.12 en cada push (`.github/workflows/pytest.yml`). Las Actions van ancladas por SHA.

Protocolo por defecto: **inductivo** (entrena en un lote, rankea otro). `--transductive` ajusta y puntúa el mismo lote.

CSV mínimo: `timestamp,bytes_sent,duration_seconds,failed_login_attempts`. Opcional: `src_ip`, `label`, `is_attack`.

Features del modelo: `log(bytes)`, `log(duration)`, `failed_login_attempts`, $\sin(2\pi h/24)$, $\cos(2\pi h/24)$.

## Estructura

```
LICENSE              MIT
planteamiento.html   Vista única: 3 cambios + matriz matemática + estado final
log_analyzer.py      CLI
simulator.py         Manifold correlacionado + 3 familias
detector.py          IF, max|z|, Mahalanobis, mismas k
tests/               Horas, overlap marginal, holdout
data/sample_sessions.csv
.github/workflows/pytest.yml
```

## License

MIT. See [LICENSE](LICENSE).

## Autor

David Jaimes Olivo — investigación en AppSec, cadena de suministro CI y detección.

- GitHub: [github.com/forkaizen2023-sys](https://github.com/forkaizen2023-sys)
- Relacionado: [agent-oidc-coupling](https://github.com/forkaizen2023-sys/agent-oidc-coupling) (modelo de amenaza operador: agente + identidad de carga)
