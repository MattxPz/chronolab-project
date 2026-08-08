"""IsolationForest sobre un vector de features de ventana.

Siete features por punto, todas retrospectivas (ninguna mira mas alla de ``t``):
valor, residuo del forecaster (``y - y_hat``), z-score movil, primera y segunda
derivada, energia espectral local y hora del dia. El bosque no necesita las
features estandarizadas -cada arbol parte cada dimension por su cuenta- asi que
no hay escalador que ajustar.

**Score comparable con el resto del arnes, no solo "acotado a un rango".**
`chronolab.anomaly.events.aggregate_events` marca un punto cuando
``score >= -log10(alpha)``, igual para cualquier detector que le pase la
trama: esa formula esta grabada en el motor de eventos y no admite un
detector con una escala distinta. Por eso `score` aqui no es la salida cruda
de PyOD reescalada a ``[0, 1]``: es el mismo p-valor conformal que usa
`chronolab.anomaly.conformal`, aplicado a una magnitud de no conformidad
distinta.

```
raw_t  = IsolationForest.decision_function(x_t)     # mayor = mas anomalo, convencion PyOD
p_t    = (1 + #{i in pool_calib : raw_i >= raw_t}) / (n_calib + 1)
score  = -log10(p_t)
```

Bajo intercambiabilidad entre el pool de calibracion y los puntos normales que
se puntuan, esto es deteccion de anomalias conformal (Laxhammar & Falkman,
2010): la tasa de falsos positivos queda acotada por construccion **para
cualquier** medida de no conformidad, no solo para el residuo CQR del detector
principal. Es la misma garantia, aplicada a otro `raw_t`.

**Por que el pool de calibracion no puede ser el mismo tramo que entrena el
bosque.** Un `IsolationForest` ajustado y evaluado sobre los mismos puntos les
asigna una puntuacion optimista -el bosque los ha visto-, lo que rompe la
intercambiabilidad justo donde importa: los p-valores saldrian sesgados hacia
"normal". `fit` reparte `calib` cronologicamente en dos tramos disjuntos:
el primero entrena el bosque, el segundo -nunca visto por el bosque- construye
el pool. Es el mismo principio que el split conformal clasico, aplicado antes
de la ventana en lugar de dentro de ella.

**Por que un solo pool global y no uno por hora, a diferencia del detector
conformal.** La hora local ya es una de las siete columnas de entrada: el
propio bosque puede aprender que un valor es normal a las 3 de la madrugada y
anomalo a las 6 de la tarde, en lugar de necesitar un grupo de calibracion
separado por franja horaria. Partir ademas el pool por hora, con paneles de
pocos meses, dejaria cada grupo por debajo de `min_calib` sin ganar nada: la
heterocedasticidad horaria ya esta dentro del modelo, no fuera de el.

**`severity` y `side` generalizan el criterio del detector conformal a una
magnitud de un solo lado.** No hay intervalo con dos bordes aqui, asi que
`severity` se mide contra el cuantil de referencia del pool en unidades de
MAD (desviacion absoluta mediana, escalada para estimar sigma bajo
normalidad), en vez de en anchuras de intervalo. `side` toma el signo del
residuo -por encima o por debajo del pronostico-, que es la misma columna que
usa `chronolab.anomaly.events` para separar picos de consumo de caidas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from chronolab.anomaly.conformal import SCORE_COLUMNS
from chronolab.anomaly.protocols import DetectorRequirements, ScoringFrame
from chronolab.data.calendar import local_hour
from chronolab.errors import CutoffViolation
from chronolab.panel import PanelSpec
from chronolab.types import DetectorId, ModelId

__all__ = [
    "FEATURE_COLUMNS",
    "FittedIsolationForestDetector",
    "IsolationForestDetector",
    "pool_score",
    "pool_severity",
]

FEATURE_COLUMNS: tuple[str, ...] = (
    "value",
    "residual",
    "roll_zscore",
    "diff1",
    "diff2",
    "spectral_energy",
    "hour",
)
"""Columnas del vector de features de ventana, en el orden que entra al bosque."""

_EPS = 1e-8
"""Piso de las desviaciones para evitar divisiones por cero en series casi constantes."""


def _require_pyod() -> Any:
    """Importa `IForest` de PyOD bajo demanda.

    Returns
    -------
    Any
        La clase `pyod.models.iforest.IForest`. Tipo `Any` a proposito: `pyod`
        esta en la cuarentena de tipos de docs/ARCHITECTURE.md D16.

    Raises
    ------
    ImportError
        Si `pyod` no esta instalado.
    """
    try:
        from pyod.models.iforest import IForest
    except ImportError as exc:  # pragma: no cover  ejercitado solo sin el extra ml
        raise ImportError(
            "chronolab.anomaly.isolation necesita el extra 'ml': `uv sync --extra ml`."
        ) from exc
    return IForest


def pool_score(raw: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """P-valor conformal de cada magnitud contra un pool, expresado como ``-log10(p)``.

    Formula identica a la del detector conformal principal
    (`chronolab.anomaly.conformal`), generalizada a cualquier magnitud de no
    conformidad de un solo lado, mayor-es-peor: `raw` no tiene por que ser un
    residuo CQR. Es lo que hace que `score` viva en la misma escala en todos
    los detectores del proyecto y sea utilizable por
    `chronolab.anomaly.events.aggregate_events` sin cambios.

    Parameters
    ----------
    raw
        Magnitudes crudas a puntuar, un valor por fila puntuable.
    pool
        Pool de calibracion, ordenado o no.

    Returns
    -------
    numpy.ndarray
        ``-log10(p)`` en ``float32``, saturado en ``log10(n_pool + 1)``: con
        ``n_pool`` puntos de calibracion no hay informacion para distinguir
        colas mas alla de ese punto, igual que en el detector conformal.
    """
    if pool.size == 0:
        return np.full(raw.shape, np.nan, dtype=np.float32)
    sorted_pool = np.sort(pool)
    count_ge = pool.size - np.searchsorted(sorted_pool, raw, side="left")
    p_value = (1.0 + count_ge) / (pool.size + 1.0)
    return (-np.log10(p_value)).astype(np.float32)


def pool_severity(raw: np.ndarray, pool: np.ndarray, *, alpha_ref: float) -> np.ndarray:
    """Cuanto se sale una magnitud del cuantil de referencia, en unidades de MAD.

    Generaliza `severity` a una magnitud de un solo lado: no hay anchura de
    intervalo que normalice, asi que se usa la desviacion absoluta mediana del
    pool, escalada por 1.4826 para que estime la desviacion tipica bajo
    normalidad. Es la misma razon de ser que la `severity` del detector
    conformal -ordenar la cola donde `score` satura- con la escala que le
    corresponde a una magnitud sin par de bordes.

    Parameters
    ----------
    raw
        Magnitudes crudas.
    pool
        Pool de calibracion.
    alpha_ref
        Nivel de referencia: `severity` vale cero en el cuantil
        ``1 - alpha_ref`` del pool.

    Returns
    -------
    numpy.ndarray
        ``float32``. ``NaN`` si el pool esta vacio.
    """
    if pool.size == 0:
        return np.full(raw.shape, np.nan, dtype=np.float32)
    median = float(np.median(pool))
    mad = float(np.median(np.abs(pool - median))) * 1.4826
    scale = max(mad, _EPS)
    reference = float(np.quantile(pool, 1.0 - alpha_ref, method="higher"))
    return ((raw - reference) / scale).astype(np.float32)


@dataclass(frozen=True, slots=True)
class IsolationForestDetector:
    """Configuracion del detector. Inmutable y sin estado calibrado.

    Parameters
    ----------
    base_model_id
        Modelo del que sale ``y_hat`` para el residuo. Forma parte del
        `detector_id` por la misma razon que en el detector conformal:
        "IsolationForest sobre residuos de MSTL" y "...de NHITS" son
        detectores distintos.
    window
        Puntos de contexto retrospectivo para el z-score movil, las derivadas
        y la energia espectral local.
    calib_fraction
        Fraccion, por tiempo, del tramo de calibracion reservada como pool de
        p-valores y **no** vista por el bosque. El resto entrena el bosque.
    n_estimators, contamination
        Hiperparametros de `pyod.models.iforest.IForest`. `contamination` solo
        afecta al umbral interno de PyOD, que este detector no usa -el
        umbralizado del proyecto vive en `chronolab.anomaly.thresholds`- pero
        el constructor de la libreria lo exige.
    alpha_ref
        Nivel de referencia de `severity`.
    min_calib
        Puntos minimos que debe tener el pool de calibracion tras filtrar el
        calentamiento. Por debajo, calibrar es fingir una cola que no se ha
        observado.
    seed
        Semilla de `IForest`.

    Raises
    ------
    ValueError
        Si algun parametro cae fuera de su dominio.
    """

    base_model_id: ModelId | None = None
    window: int = 24
    calib_fraction: float = 0.3
    n_estimators: int = 100
    contamination: float = 0.05
    alpha_ref: float = 0.05
    min_calib: int = 50
    seed: int = 0

    def __post_init__(self) -> None:
        """Valida la configuracion."""
        if self.window < 3:
            raise ValueError(f"window debe ser >= 3 (hace falta para diff2): {self.window}")
        if not 0.0 < self.calib_fraction < 1.0:
            raise ValueError(f"calib_fraction fuera de (0, 1): {self.calib_fraction}")
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators debe ser >= 1: {self.n_estimators}")
        if not 0.0 < self.contamination <= 0.5:
            raise ValueError(f"contamination fuera de (0, 0.5]: {self.contamination}")
        if not 0.0 < self.alpha_ref < 1.0:
            raise ValueError(f"alpha_ref fuera de (0, 1): {self.alpha_ref}")
        if self.min_calib < 1:
            raise ValueError(f"min_calib debe ser >= 1: {self.min_calib}")

    @property
    def detector_id(self) -> DetectorId:
        """Identificador estable. Clave de particion de `anomaly_scores`."""
        base = "none" if self.base_model_id is None else str(self.base_model_id)
        return DetectorId(
            f"isoforest_w{self.window:03d}_c{round(self.calib_fraction * 100):02d}"
            f"_n{self.n_estimators}_m{base}"
        )

    @property
    def requires(self) -> DetectorRequirements:
        """Necesidades declaradas.

        ``window`` cuenta los dos puntos extra que exige `diff2`: la primera
        posicion con features completas es la tercera del tramo de contexto.
        """
        return DetectorRequirements(
            needs_forecast=True,
            needs_quantiles=False,
            window=self.window + 2,
            needs_calibration=True,
            fit_cost="cheap",
        )

    def fit(self, calib: ScoringFrame) -> FittedIsolationForestDetector:
        """Calibra el detector con un tramo anterior al que se puntuara.

        Reparte `calib` cronologicamente: el primer ``1 - calib_fraction`` del
        tramo entrena el bosque, el resto -nunca visto por el bosque- se
        puntua con el bosque ya congelado para construir el pool de
        p-valores. Es el reparto que preserva la intercambiabilidad entre el
        pool y los puntos que se van a puntuar despues del cutoff.

        Parameters
        ----------
        calib
            Tramo de calibracion.

        Returns
        -------
        FittedIsolationForestDetector
            Con ``cutoff = calib.end``.

        Raises
        ------
        ValueError
            Si falta alguna columna obligatoria o si, tras filtrar el
            calentamiento, el tramo de entrenamiento o el pool de calibracion
            no llegan al minimo exigido.
        """
        df = _prepared(calib.df, spec=calib.spec, window=self.window)
        _assert_feature_columns(df)
        usable = df.loc[df["scorable"]]

        split_at = _time_split(usable["ds"], self.calib_fraction)
        train_rows = usable.loc[usable["ds"] <= split_at]
        calib_rows = usable.loc[usable["ds"] > split_at]
        if len(train_rows) < self.min_calib:
            raise ValueError(
                f"el tramo de entrenamiento del bosque tiene {len(train_rows)} filas "
                f"puntuables; hacen falta al menos {self.min_calib}"
            )
        if len(calib_rows) < self.min_calib:
            raise ValueError(
                f"el pool de calibracion tiene {len(calib_rows)} filas puntuables; "
                f"hacen falta al menos {self.min_calib}"
            )

        iforest_cls = _require_pyod()
        model = iforest_cls(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.seed,
        )
        model.fit(train_rows[list(FEATURE_COLUMNS)].to_numpy(dtype=float))
        pool = np.asarray(
            model.decision_function(calib_rows[list(FEATURE_COLUMNS)].to_numpy(dtype=float)),
            dtype=float,
        )

        tails = _tails(calib.df, window=self.window)
        return FittedIsolationForestDetector(
            detector=self,
            spec=calib.spec,
            cutoff=calib.end,
            model=model,
            pool=pool,
            tails=tails,
        )


@dataclass(frozen=True, slots=True)
class FittedIsolationForestDetector:
    """Bosque calibrado hasta un instante concreto.

    Attributes
    ----------
    detector
        Configuracion de la que procede.
    spec
        Especificacion del panel con el que se calibro.
    cutoff
        Ultimo instante usado en calibracion.
    model
        `IForest` ya ajustado. Se trata como opaco: solo se le pide
        `decision_function`.
    pool
        Pool de p-valores, magnitudes crudas del tramo de calibracion nunca
        visto por el bosque.
    tails
        Ultimas `requires.window` filas de `calib.df` por serie, para que la
        primera llamada a `score` no repita el calentamiento que ya se pagaria
        de todos modos si el tramo fuese contiguo con la calibracion.
    """

    detector: IsolationForestDetector
    spec: PanelSpec
    cutoff: pd.Timestamp
    model: Any
    pool: np.ndarray
    tails: dict[str, pd.DataFrame]

    @property
    def detector_id(self) -> DetectorId:
        """Identificador del detector del que procede."""
        return self.detector.detector_id

    def score(self, frame: ScoringFrame) -> pd.DataFrame:
        """Puntua cada marca de tiempo del tramo.

        No hay estado que avanzar entre llamadas: a diferencia del detector
        conformal, este no expone `advance`. Se calibra una vez y se espera
        puntuar el tramo de holdout completo en una sola llamada; llamadas
        repetidas sobre tramos separados repiten el calentamiento en cada una,
        salvo en la serie que ya traia cola de la calibracion.

        Parameters
        ----------
        frame
            Tramo a puntuar, con ``frame.start > cutoff``.

        Returns
        -------
        pandas.DataFrame
            Columnas `chronolab.anomaly.conformal.SCORE_COLUMNS`.

        Raises
        ------
        CutoffViolation
            Si `frame` empieza en un instante que la calibracion ya conocia.
        """
        if frame.start <= self.cutoff:
            raise CutoffViolation(
                f"{self.detector_id} esta calibrado hasta {self.cutoff} y se le pide "
                f"puntuar desde {frame.start}, anterior o igual"
            )

        bridged = _bridge(frame.df, tails=self.tails)
        df = _prepared(bridged, spec=frame.spec, window=self.detector.window)
        _assert_feature_columns(df)

        out = frame.df[["unique_id", "ds"]].copy()
        out["unique_id"] = out["unique_id"].astype(str)
        merge_key = ["unique_id", "ds"]
        merged = out.merge(df, on=merge_key, how="left", validate="one_to_one")

        scorable = merged["scorable"].fillna(False).to_numpy(dtype=bool)
        n_rows = len(merged)
        score = np.full(n_rows, np.nan, dtype=np.float32)
        severity = np.full(n_rows, np.nan, dtype=np.float32)
        calib_n = np.zeros(n_rows, dtype=np.int32)
        side = np.zeros(n_rows, dtype=np.int8)

        if scorable.any():
            features = merged.loc[scorable, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
            raw = np.asarray(self.model.decision_function(features), dtype=float)
            score[scorable] = pool_score(raw, self.pool)
            severity[scorable] = pool_severity(raw, self.pool, alpha_ref=self.detector.alpha_ref)
            calib_n[scorable] = self.pool.size
            residual = merged.loc[scorable, "residual"].to_numpy(dtype=float)
            side[scorable] = np.where(residual > 0, 1, np.where(residual < 0, -1, 0)).astype(
                np.int8
            )

        return pd.DataFrame(
            {
                "unique_id": out["unique_id"].to_numpy(),
                "ds": out["ds"].to_numpy(dtype="datetime64[ns]"),
                "score": score,
                "scorable": scorable,
                "severity": severity,
                "calib_n": calib_n,
                "side": side,
            }
        )[list(SCORE_COLUMNS)]


def _time_split(ds: pd.Series, calib_fraction: float) -> pd.Timestamp:
    """Instante que reparte un tramo cronologicamente segun `calib_fraction`.

    Parameters
    ----------
    ds
        Marcas de tiempo del tramo, no necesariamente ordenadas.
    calib_fraction
        Fraccion final reservada al pool de calibracion.

    Returns
    -------
    pandas.Timestamp
        Filas con ``ds <= resultado`` entrenan el bosque; el resto forma el
        pool.
    """
    ordered = pd.Index(ds.unique()).sort_values()
    if len(ordered) < 2:
        return pd.Timestamp(ordered[0])
    index = int(np.clip(round(len(ordered) * (1.0 - calib_fraction)), 1, len(ordered) - 1))
    return pd.Timestamp(ordered[index - 1])


def _tails(df: pd.DataFrame, *, window: int) -> dict[str, pd.DataFrame]:
    """Ultimas `window + 2` filas de cada serie, para puentear el siguiente `score`.

    Parameters
    ----------
    df
        Trama de la que extraer la cola. Tipicamente `calib.df`.
    window
        Longitud de contexto declarada del detector.

    Returns
    -------
    dict
        ``unique_id -> ultimas filas``, ordenadas por `ds`.
    """
    needed = window + 2
    tails: dict[str, pd.DataFrame] = {}
    for uid, group in df.groupby("unique_id", sort=False):
        ordered = group.sort_values("ds")
        tails[str(uid)] = ordered.tail(needed).reset_index(drop=True)
    return tails


def _bridge(df: pd.DataFrame, *, tails: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Antepone la cola de calibracion de cada serie al tramo a puntuar.

    Las filas prestadas de la cola se marcan con ``_borrowed=True`` para que
    `_prepared` calcule features con ellas pero el llamante nunca las
    devuelva como puntuadas: no pertenecen al tramo pedido.

    Parameters
    ----------
    df
        Tramo a puntuar.
    tails
        Colas por serie, de `FittedIsolationForestDetector.tails`.

    Returns
    -------
    pandas.DataFrame
        `df` con las colas antepuestas donde existan.
    """
    parts: list[pd.DataFrame] = []
    for uid, group in df.groupby("unique_id", sort=False):
        key = str(uid)
        tail = tails.get(key)
        ordered = group.sort_values("ds").copy()
        ordered["_borrowed"] = False
        if tail is not None and not tail.empty:
            borrowed = tail.copy()
            borrowed["_borrowed"] = True
            ordered = pd.concat([borrowed, ordered], ignore_index=True)
        parts.append(ordered)
    if not parts:
        result = df.copy()
        result["_borrowed"] = False
        return result
    return pd.concat(parts, ignore_index=True)


def _prepared(df: pd.DataFrame, *, spec: PanelSpec, window: int) -> pd.DataFrame:
    """Calcula el vector de features de cada fila puntuable.

    Todas las features son estrictamente retrospectivas: cada una solo lee
    ``y``, ``y_hat`` y `ds` en ``<= t``. Las filas cuyo contexto no llega a
    `window + 2` puntos, o cuyo `y`/`y_hat` no es finito, quedan con
    ``scorable=False`` y sus columnas de feature a `NaN`.

    Parameters
    ----------
    df
        Trama con `unique_id`, `ds`, `y`, `y_hat`, opcionalmente `_borrowed`.
    spec
        Especificacion del panel, para la hora local.
    window
        Longitud del contexto retrospectivo del z-score, las derivadas y la
        energia espectral.

    Returns
    -------
    pandas.DataFrame
        `unique_id`, `ds`, `scorable`, `residual` y `FEATURE_COLUMNS`, una
        fila por fila de entrada que **no** venga marcada `_borrowed`
        (las prestadas solo aportan contexto y se descartan al final).
    """
    has_borrowed = "_borrowed" in df.columns
    parts: list[pd.DataFrame] = []
    for uid, group in df.groupby("unique_id", sort=False):
        ordered = group.sort_values("ds").reset_index(drop=True)
        y = ordered["y"].to_numpy(dtype=float)
        y_hat = ordered["y_hat"].to_numpy(dtype=float)
        n = len(ordered)

        value = y
        residual = y - y_hat
        roll_mean = pd.Series(y).rolling(window, min_periods=window).mean().to_numpy()
        roll_std = pd.Series(y).rolling(window, min_periods=window).std(ddof=0).to_numpy()
        roll_zscore = (y - roll_mean) / np.maximum(roll_std, _EPS)
        diff1 = np.concatenate(([np.nan], np.diff(y, n=1)))
        diff2 = np.concatenate(([np.nan, np.nan], np.diff(y, n=2)))
        spectral_energy = _rolling_spectral_energy(y, window=window)
        hour = local_hour(ordered["ds"], tz_display=spec.tz_display).to_numpy(dtype=float)

        context_ok = np.arange(n) >= (window + 1)  # dos puntos extra: diff2 y rolling completo
        finite = np.isfinite(value) & np.isfinite(residual)
        scorable = context_ok & finite & np.isfinite(roll_zscore) & np.isfinite(spectral_energy)

        frame = pd.DataFrame(
            {
                "unique_id": str(uid),
                "ds": ordered["ds"].to_numpy(),
                "scorable": scorable,
                "residual": residual,
                "value": value,
                "roll_zscore": roll_zscore,
                "diff1": diff1,
                "diff2": diff2,
                "spectral_energy": spectral_energy,
                "hour": hour,
            }
        )
        # `_borrowed` viaja alineado fila a fila con `ordered`, del mismo
        # `sort_values("ds")` que produjo `y`/`y_hat`: no depende de reordenar
        # `df` por separado, que rompería la alineación si el orden de
        # aparición de las series en `df` no coincidiera con el alfabetico.
        frame["_borrowed"] = ordered["_borrowed"].to_numpy(dtype=bool) if has_borrowed else False
        parts.append(frame)

    columns = ["unique_id", "ds", "scorable", "residual", *FEATURE_COLUMNS, "_borrowed"]
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)

    result = result.loc[~result["_borrowed"]].drop(columns="_borrowed").reset_index(drop=True)
    # `residual` ya esta en `FEATURE_COLUMNS`: anadirla dos veces a la lista de
    # columnas de esta asignacion duplicaria la columna en el resultado.
    result.loc[~result["scorable"], list(FEATURE_COLUMNS)] = np.nan
    return result


def _rolling_spectral_energy(y: np.ndarray, *, window: int) -> np.ndarray:
    """Energia espectral local: potencia media de la ventana retrospectiva, sin la componente DC.

    Para cada posicion `t` con `window` puntos de contexto completo, centra la
    ventana ``y[t-window+1 .. t]`` en su propia media y calcula
    ``sum(|rfft(ventana)|^2) / window`` (identidad de Parseval): mayor cuanto
    mas energia de alta frecuencia -ruido, escalones abruptos- tiene el tramo
    reciente, en vez de solo su varianza.

    Parameters
    ----------
    y
        Serie completa, en orden temporal.
    window
        Longitud de la ventana.

    Returns
    -------
    numpy.ndarray
        Misma longitud que `y`. `NaN` donde no hay `window` puntos previos
        completos.
    """
    n = len(y)
    energy = np.full(n, np.nan)
    for t in range(window - 1, n):
        segment = y[t - window + 1 : t + 1]
        if not np.isfinite(segment).all():
            continue
        centred = segment - segment.mean()
        spectrum = np.fft.rfft(centred)
        energy[t] = float(np.sum(np.abs(spectrum) ** 2)) / window
    return energy


def _assert_feature_columns(df: pd.DataFrame) -> None:
    """Comprueba que la trama preparada trae las columnas del vector de features.

    Parameters
    ----------
    df
        Salida de `_prepared`.

    Raises
    ------
    ValueError
        Si falta alguna columna. Solo puede ocurrir por un error de
        programacion interno, no por datos de entrada malformados -esos ya
        habrian fallado antes, al construir `ScoringFrame`.
    """
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"faltan columnas de features: {sorted(missing)}")
