"""Ajustes visuales compartidos: densidad, pie de pagina y textos de ayuda.

El tema en si vive en ``.streamlit/config.toml`` (color, tipografia, radios,
bordes) y ya cubre casi todo sin CSS. Lo que el archivo de tema no expone
-el padding superior heredado del layout por defecto de Streamlit, y un pie
de pagina propio- se ajusta aqui con el minimo CSS necesario, nunca
reescribiendo selectores internos de Streamlit que cambian entre versiones
salvo el contenedor de bloque, que es estable desde hace varias.

`METRIC_HELP` es el unico sitio donde vive la explicacion de cada metrica:
una pagina que muestra `mase` importa su texto de aqui en vez de escribirlo
inline, para que las cinco paginas digan lo mismo con las mismas palabras.
"""

from __future__ import annotations

import streamlit as st

__all__ = ["METRIC_HELP", "apply_density", "footer"]

_REPO_URL = "https://github.com/MattxPz/chronolab-project"
_METHODOLOGY_URL = f"{_REPO_URL}/blob/main/docs/ARCHITECTURE.md"

METRIC_HELP: dict[str, str] = {
    "mase": (
        "Mean Absolute Scaled Error. Escala el error frente al naive estacional "
        "de la propia serie: MASE < 1 significa que el modelo bate a ese naive; "
        "MASE > 1, que pierde frente a el. Comparable entre series de escalas "
        "distintas, a diferencia de MAE o RMSE."
    ),
    "rmse": "Raiz del error cuadratico medio, en las unidades de la serie. Penaliza mas los errores grandes que el MAE.",
    "mae": "Error absoluto medio, en las unidades de la serie. Mas robusto a atipicos que el RMSE.",
    "smape": "Error porcentual absoluto simetrico, en [0, 200%]. Comparable entre series, pero inestable cerca de cero.",
    "mape": "Error porcentual absoluto medio. Se dispara si la serie pasa cerca de cero; ver sMAPE o MASE para series con ceros.",
    "coverage_95": "Fraccion de veces que el intervalo al 95% contuvo el valor real. Deberia rondar 0.95: por debajo, el modelo es demasiado confiado.",
    "coverage_80": "Fraccion de veces que el intervalo al 80% contuvo el valor real. Deberia rondar 0.80.",
    "coverage_50": "Fraccion de veces que el intervalo al 50% contuvo el valor real. Deberia rondar 0.50.",
    "fit_seconds_total": "Coste total de entrenamiento del modelo en el backtest, en segundos.",
    "dm_stat": (
        "Estadistico de Diebold-Mariano entre dos modelos. Negativo: el primer "
        "modelo (fila) pierde menos que el segundo (columna). El signo importa "
        "mas que la magnitud; el p-valor dice si la diferencia es significativa."
    ),
    "range_recall": "Fraccion del rango temporal de los eventos anomalos que el detector cubrio, sin exigir el instante exacto.",
    "range_precision": "De lo que el detector marco como anomalo, que fraccion cae dentro de un evento real (por rango, no por instante).",
    "auc_pr": "Area bajo la curva precision-recall puntual. Mas alto es mejor; el suelo lo marca la prevalencia de anomalias.",
    "vus_pr": "Volumen bajo la superficie precision-recall, integrado sobre varias tolerancias temporales. Generaliza AUC-PR a deteccion con margen de error en el instante.",
    "trend_strength": "Cuanto domina la tendencia sobre el residuo (Hyndman & Wang, 2015). Cerca de 1: la tendencia explica casi toda la variacion.",
    "spectral_entropy": "Entropia del espectro de potencia, en [0, 1]. Cerca de 0: pocas frecuencias dominan (predecible). Cerca de 1: espectro plano, dificil de predecir.",
    "coverage": "Fraccion de la rejilla temporal esperada que tiene un dato (no es un hueco).",
}
"""Texto de ayuda por metrica, para pasar a `help=` en `st.metric`/`st.dataframe`."""


def apply_density(*, page_title: str) -> None:
    """Configura la pagina y aprieta el espaciado por defecto de Streamlit.

    Debe ser la primera llamada de cada script de pagina, justo despues de
    ``st.set_page_config`` (que cada pagina sigue declarando por separado,
    porque Streamlit lo exige antes de cualquier otro comando).

    Parameters
    ----------
    page_title
        Se usa solo para el `id` del bloque de estilo; no repite
        `st.set_page_config`.
    """
    st.markdown(
        f"""
        <style id="chronolab-density-{page_title.lower().replace(" ", "-")}">
        .block-container {{
            padding-top: 2.25rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }}
        [data-testid="stMetric"] {{
            padding: 0.35rem 0.1rem;
        }}
        [data-testid="stMetricLabel"] {{
            font-size: 0.8rem;
            opacity: 0.75;
        }}
        [data-testid="stSidebar"] .block-container {{
            padding-top: 1.5rem;
        }}
        div[data-testid="stExpander"] details summary p {{
            font-size: 0.95rem;
            font-weight: 600;
        }}
        hr {{
            margin: 1.1rem 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def footer() -> None:
    """Pie de pagina fijo: repositorio, metodologia y aviso de modo demo.

    Se dibuja con `st.container` y markdown, no con CSS de posicion fija:
    un pie pegado al fondo de la ventana taparia contenido en pantallas
    estrechas, que es justo lo que el requisito de "mobile-friendly" pide
    evitar.
    """
    st.divider()
    left, right = st.columns([3, 2])
    with left:
        st.caption(
            f"[chronolab en GitHub]({_REPO_URL}) · "
            f"[metodologia (ARCHITECTURE.md)]({_METHODOLOGY_URL}) · "
            "modo demo: artefactos versionados en `reports/results/`, sin red."
        )
    with right:
        st.caption("La app solo lee artefactos precomputados — nunca entrena ni recalcula (A5).")
