"""Dataset de ventanas deslizantes con escalado por serie ajustado solo con train.

Este modulo es **deliberadamente puro numpy**: no importa `torch` ni a nivel de
modulo ni dentro de sus funciones. Dos razones, y ninguna es cosmetica:

1. `torch` vive en el extra `deep` (D20), y el job `quality` de CI hace
   `uv sync` a secas. `tests/unit/test_module_tree.py` importa **todos** los
   modulos del arbol en ese entorno, asi que un `import torch` aqui —a nivel de
   modulo— romperia CI. El tensor solo aparece en `models/torch/trainer.py`,
   que es quien envuelve estos arrays.
2. Mas importante: lo que hay que poder auditar de este modulo es **donde se
   corta el tiempo**, no como se mueve un tensor. Al ser numpy puro, los tests
   de fuga (`tests/leakage/`) lo ejercitan en el entorno por defecto, sin
   instalar medio gigabyte de dependencias para comprobar una aritmetica de
   indices.

Las dos piezas y su invariante:

- `SeriesScaler` estandariza **por serie**, con media y desviacion calculadas
  *solo* con el tramo que se le pasa a `fit`. El motor de backtesting entrega a
  `Forecaster.fit` un `Panel` ya recortado a ``ds <= cutoff``
  (docs/ARCHITECTURE.md, fuga L2), asi que ajustar aqui es ajustar con train y
  nada mas. `inverse_target` deshace la transformacion sobre las predicciones,
  que es la mitad que mas se olvida.
- `build_windows` genera las ventanas ``contexto -> horizonte``. Una ventana
  termina su contexto en `t` y predice ``t+1 .. t+h``: **nunca** incluye `t+1`
  en la entrada. Como el panel esta en rejilla completa (invariante I3), contar
  posiciones y contar pasos de tiempo son la misma operacion, que es lo que
  hace imposible que una ventana cruce un hueco sin enterarse.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from chronolab.panel import Panel

__all__ = ["SeriesScaler", "WindowBatch", "build_windows", "context_matrix"]

_EPS = 1e-8
"""Piso de la desviacion tipica al escalar.

Una serie constante en el tramo de entrenamiento tiene desviacion cero y
dividir por ella produce `inf`. El piso la deja pasar como una serie de
desviacion despreciable —que es lo que es— en vez de envenenar el batch
entero con `NaN`.
"""


@dataclass(frozen=True, slots=True)
class SeriesScaler:
    """Estandarizacion por serie, ajustada **solo** con el tramo de entrenamiento.

    Guarda una media y una desviacion por ``unique_id`` para la objetivo, y un
    unico par global por columna exogena. La asimetria es deliberada: la
    objetivo vive en la escala de cada serie —un cliente de 30 kW y otro de
    3000 kW no son comparables— mientras que las exogenas de calendario y
    temperatura comparten escala entre series por construccion, y darles
    estadisticos por serie solo anadiria ruido de estimacion en las series
    cortas.

    Attributes
    ----------
    target_mean, target_std
        ``unique_id -> estadistico`` de la objetivo.
    exog_mean, exog_std
        ``columna -> estadistico`` de cada exogena, compartido entre series.
    exog_columns
        Columnas exogenas escaladas, en orden estable.
    """

    target_mean: dict[str, float]
    target_std: dict[str, float]
    exog_mean: dict[str, float]
    exog_std: dict[str, float]
    exog_columns: tuple[str, ...]

    @classmethod
    def fit(cls, train: Panel, exog_columns: Sequence[str] = ()) -> SeriesScaler:
        """Ajusta los estadisticos con el panel de entrenamiento y nada mas.

        Parameters
        ----------
        train
            Rebanada de entrenamiento que entrega el motor, ya recortada a
            ``ds <= cutoff``. Este es el unico dato que el escalador llega a
            ver en toda su vida: no existe un metodo que acepte el panel
            completo.
        exog_columns
            Columnas exogenas a escalar. Vacia si el modelo no usa ninguna.

        Returns
        -------
        SeriesScaler

        Raises
        ------
        KeyError
            Si alguna columna de `exog_columns` no esta en el panel.
        """
        target = train.spec.target
        target_mean: dict[str, float] = {}
        target_std: dict[str, float] = {}
        for uid, group in train.df.groupby("unique_id", sort=False):
            values = group[target].to_numpy(dtype=float)
            observed = values[np.isfinite(values)]
            target_mean[str(uid)] = float(observed.mean()) if observed.size else 0.0
            target_std[str(uid)] = float(observed.std()) if observed.size else 1.0

        exog_mean: dict[str, float] = {}
        exog_std: dict[str, float] = {}
        for column in exog_columns:
            values = train.df[column].to_numpy(dtype=float)
            observed = values[np.isfinite(values)]
            exog_mean[column] = float(observed.mean()) if observed.size else 0.0
            exog_std[column] = float(observed.std()) if observed.size else 1.0

        return cls(
            target_mean=target_mean,
            target_std=target_std,
            exog_mean=exog_mean,
            exog_std=exog_std,
            exog_columns=tuple(exog_columns),
        )

    def _target_stats(self, uid: str) -> tuple[float, float]:
        """Media y desviacion (con piso) de una serie, o el par neutro si no se vio en train."""
        mean = self.target_mean.get(uid, 0.0)
        std = self.target_std.get(uid, 1.0)
        return mean, max(std, _EPS)

    def transform_target(self, values: np.ndarray, uid: str) -> np.ndarray:
        """Lleva la objetivo de una serie a escala estandarizada.

        Parameters
        ----------
        values
            Valores en la escala original.
        uid
            Serie a la que pertenecen.

        Returns
        -------
        numpy.ndarray
            ``(values - mean) / std`` con los estadisticos de esa serie.
        """
        mean, std = self._target_stats(uid)
        return (np.asarray(values, dtype=float) - mean) / std

    def inverse_target(self, values: np.ndarray, uid: str) -> np.ndarray:
        """Deshace `transform_target`: de escala estandarizada a la original.

        Es la mitad que mas se olvida de un escalado, y la que hace que un
        modelo aparentemente razonable publique predicciones con un error de
        varios ordenes de magnitud.

        Parameters
        ----------
        values
            Valores en escala estandarizada, tipicamente la salida del modelo.
        uid
            Serie a la que pertenecen.

        Returns
        -------
        numpy.ndarray
            ``values * std + mean``.
        """
        mean, std = self._target_stats(uid)
        return np.asarray(values, dtype=float) * std + mean

    def transform_exog(self, frame: pd.DataFrame) -> np.ndarray:
        """Estandariza las columnas exogenas de una trama, en el orden declarado.

        Parameters
        ----------
        frame
            Trama que contiene al menos `self.exog_columns`.

        Returns
        -------
        numpy.ndarray
            De forma ``(len(frame), len(exog_columns))``. Vacia con la segunda
            dimension a cero si no hay exogenas, para que el codigo que
            concatena no necesite un caso especial.
        """
        if not self.exog_columns:
            return np.zeros((len(frame), 0), dtype=np.float32)
        columns = []
        for column in self.exog_columns:
            values = frame[column].to_numpy(dtype=float)
            std = max(self.exog_std.get(column, 1.0), _EPS)
            columns.append((values - self.exog_mean.get(column, 0.0)) / std)
        return np.column_stack(columns).astype(np.float32)


@dataclass(frozen=True, slots=True)
class WindowBatch:
    """Ventanas deslizantes ya escaladas, listas para envolver en tensores.

    Attributes
    ----------
    context
        ``(n_windows, input_size, 1 + n_exog)``: la objetivo retrasada y, si
        las hay, las exogenas del **mismo** tramo de contexto.
    futr
        ``(n_windows, h, n_exog)``: exogenas del tramo a predecir. Solo pueden
        estar aqui las columnas conocidas a futuro; el llamante lo garantiza
        eligiendo que pasa en `exog_columns`.
    target
        ``(n_windows, h)``: la objetivo del tramo a predecir, escalada.
    series
        ``(n_windows,)``: el `unique_id` de cada ventana, para poder invertir
        el escalado con los estadisticos correctos.
    """

    context: np.ndarray
    futr: np.ndarray
    target: np.ndarray
    series: np.ndarray

    def __len__(self) -> int:
        """Numero de ventanas."""
        return int(self.context.shape[0])


def _series_arrays(panel: Panel, scaler: SeriesScaler) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Valores escalados por serie: objetivo rellenada hacia atras y exogenas.

    El `ffill` es la unica imputacion admitida por la barrera L10 de
    docs/ARCHITECTURE.md: mira solo hacia atras. Rellena los huecos internos
    que el invariante I3 conserva como `NaN` explicito, de modo que el
    contexto de una ventana no lleve agujeros; las ventanas cuyo **objetivo**
    seguia siendo `NaN` antes del relleno se descartan en `build_windows`, que
    es distinto: imputar la entrada es una decision de modelado, inventarse la
    respuesta que se evalua no lo es.

    Parameters
    ----------
    panel
        Panel de entrenamiento.
    scaler
        Escalador ya ajustado.

    Returns
    -------
    list of tuple
        ``(unique_id, objetivo_escalada, exogenas_escaladas)`` por serie, en
        orden cronologico dentro de cada una.
    """
    target = panel.spec.target
    result: list[tuple[str, np.ndarray, np.ndarray]] = []
    for uid, group in panel.df.groupby("unique_id", sort=False):
        ordered = group.sort_values("ds")
        raw = ordered[target].to_numpy(dtype=float)
        filled = pd.Series(raw).ffill().to_numpy()
        scaled = scaler.transform_target(filled, str(uid))
        # Una serie sin ninguna observacion real deja `NaN` tras el ffill; se
        # neutraliza a cero en escala estandarizada (la media de la serie) para
        # que no envenene el batch, y sus ventanas se descartan igualmente
        # porque su objetivo sigue siendo `NaN` en `raw`.
        scaled = np.nan_to_num(scaled, nan=0.0)
        exog = scaler.transform_exog(ordered)
        result.append((str(uid), scaled, exog))
    return result


def build_windows(panel: Panel, scaler: SeriesScaler, *, input_size: int, h: int) -> WindowBatch:
    """Genera todas las ventanas ``contexto -> horizonte`` de un panel de entrenamiento.

    Para cada serie y cada posicion `t` con contexto e horizonte completos, la
    ventana lleva:

    - contexto ``[t - input_size + 1, t]`` de la objetivo y de las exogenas,
    - exogenas del tramo ``[t + 1, t + h]``,
    - objetivo del tramo ``[t + 1, t + h]``.

    El instante ``t + 1`` **nunca** entra en el contexto. Es la version
    aritmetica de la barrera anti-fuga: no hay una segunda ruta que construya
    ventanas, asi que no hay forma de escribir un off-by-one que meta el primer
    punto a predecir en la entrada.

    Parameters
    ----------
    panel
        Panel de entrenamiento, ya recortado por el motor a ``ds <= cutoff``.
    scaler
        Escalador ajustado con ese mismo panel.
    input_size
        Longitud del contexto en pasos, mayor o igual que uno.
    h
        Horizonte en pasos, mayor o igual que uno.

    Returns
    -------
    WindowBatch
        Posiblemente vacio si ninguna serie da para una ventana completa; el
        llamante decide si eso es un error (lo es, para entrenar) o no.

    Raises
    ------
    ValueError
        Si `input_size` o `h` son menores que uno.
    """
    if input_size < 1:
        raise ValueError(f"input_size debe ser >= 1: {input_size}")
    if h < 1:
        raise ValueError(f"h debe ser >= 1: {h}")

    target = panel.spec.target
    n_exog = len(scaler.exog_columns)

    contexts: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    series: list[str] = []

    scaled_by_series = {uid: (y, x) for uid, y, x in _series_arrays(panel, scaler)}

    for uid, group in panel.df.groupby("unique_id", sort=False):
        key = str(uid)
        scaled, exog = scaled_by_series[key]
        raw = group.sort_values("ds")[target].to_numpy(dtype=float)
        n = scaled.size

        for end in range(input_size - 1, n - h):
            start = end - input_size + 1
            horizon = slice(end + 1, end + 1 + h)
            # Una ventana cuyo objetivo tiene huecos no se imputa: se descarta.
            # Entrenar contra un valor inventado ensena justo eso.
            if not np.isfinite(raw[horizon]).all():
                continue

            context = scaled[start : end + 1].reshape(input_size, 1)
            if n_exog:
                context = np.concatenate([context, exog[start : end + 1]], axis=1)
                futures.append(exog[horizon])
            else:
                futures.append(np.zeros((h, 0), dtype=np.float32))
            contexts.append(context.astype(np.float32))
            targets.append(scaled[horizon].astype(np.float32))
            series.append(key)

    if not contexts:
        return WindowBatch(
            context=np.zeros((0, input_size, 1 + n_exog), dtype=np.float32),
            futr=np.zeros((0, h, n_exog), dtype=np.float32),
            target=np.zeros((0, h), dtype=np.float32),
            series=np.empty(0, dtype=object),
        )

    return WindowBatch(
        context=np.stack(contexts),
        futr=np.stack(futures),
        target=np.stack(targets),
        series=np.array(series, dtype=object),
    )


def context_matrix(
    panel: Panel, scaler: SeriesScaler, *, input_size: int
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Ultimo contexto disponible de cada serie, para predecir desde el cutoff.

    Es la contrapartida de `build_windows` en inferencia: alli se necesitan
    todas las ventanas con objetivo conocido; aqui solo la ultima, la que
    termina exactamente en el cutoff.

    Parameters
    ----------
    panel
        Panel de entrenamiento.
    scaler
        Escalador ajustado con ese panel.
    input_size
        Longitud del contexto en pasos.

    Returns
    -------
    tuple
        Matriz ``(n_series, input_size, 1 + n_exog)`` y los `unique_id` en el
        mismo orden que sus filas.

    Raises
    ------
    ValueError
        Si alguna serie tiene menos de `input_size` observaciones: predecir con
        un contexto rellenado a ceros seria pasar por bueno un modelo que en
        realidad no tiene con que.
    """
    rows: list[np.ndarray] = []
    ids: list[str] = []
    for uid, scaled, exog in _series_arrays(panel, scaler):
        if scaled.size < input_size:
            raise ValueError(
                f"la serie '{uid}' tiene {scaled.size} observaciones y el contexto "
                f"exige {input_size}"
            )
        context = scaled[-input_size:].reshape(input_size, 1)
        if scaler.exog_columns:
            context = np.concatenate([context, exog[-input_size:]], axis=1)
        rows.append(context.astype(np.float32))
        ids.append(uid)
    return np.stack(rows), tuple(ids)
