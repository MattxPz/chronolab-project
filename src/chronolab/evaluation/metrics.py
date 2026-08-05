"""Metricas de prediccion: MASE, RMSE, MAE, sMAPE, pinball, cobertura y CRPS discreto.

El denominador de MASE se calcula por serie y por ventana con el train de esa
ventana; usarlo global seria fuga. Se persiste para que sea auditable.

Por que MASE es la metrica principal del proyecto
-------------------------------------------------
Cuatro propiedades, y ninguna la tiene MAPE:

1. **Es adimensional y comparable entre series.** Una serie de 30 kW y otra de
   3 000 kW producen MAE incomparables. MASE divide por un error de referencia
   medido *en la propia serie*, asi que promediar sobre series significa algo.
2. **Esta definida cuando la serie pasa por cero.** La demanda nocturna, la
   generacion solar y cualquier serie con paradas tienen ceros o casi ceros.
   MAPE divide por el valor observado y ahi no existe.
3. **Penaliza igual por arriba y por abajo.** MAPE castiga mas las
   sobreprediccion que la infraprediccion —el error relativo maximo de predecir
   de menos esta acotado por el 100 %, el de predecir de mas no lo esta— y esa
   asimetria sesga la seleccion de modelos hacia los que predicen bajo. Es un
   sesgo del criterio, no del modelo, y no deja ningun rastro en el leaderboard.
4. **Tiene un cero natural interpretable.** ``MASE < 1`` significa "bate al naive
   estacional sobre la escala de su propio entrenamiento". Un numero que se puede
   defender en una frase.

MAPE se calcula igualmente porque es lo que la mayoria de la gente espera ver y
porque su ausencia levantaria mas preguntas que su presencia. Pero es
**informativa**: no se usa para ordenar el leaderboard, no se usa para
seleccionar, y cuando la serie se acerca a cero emite `UnstableMetricWarning`
diciendo cuantas observaciones la estan dominando.

Convenios comunes a todo el modulo
----------------------------------
- Los pares con ``NaN`` en el valor observado o en el predicho **se descartan**,
  nunca se imputan: el panel conserva sus huecos como `NaN` explicito (I3) y una
  metrica no es el sitio donde inventar el dato que falta. El numero de
  observaciones que quedan viaja aparte, en `n_obs`, y es lo que delata a un
  modelo evaluado sobre la mitad de los puntos.
- Sin observaciones validas se devuelve ``NaN``, no cero ni excepcion: un run que
  no pudo evaluarse tiene que verse distinto de uno que acerto perfectamente.
- MAPE y sMAPE se devuelven en **porcentaje**; el resto, en las unidades que les
  corresponden.
"""

import warnings
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from chronolab.errors import UnstableMetricWarning
from chronolab.models.protocols import QUANTILES, quantile_column
from chronolab.panel import Panel

__all__ = [
    "NEAR_ZERO_RATIO",
    "crps_discrete",
    "empirical_coverage",
    "interval_levels",
    "interval_width",
    "mae",
    "mape",
    "mase",
    "mase_denominators",
    "pinball_loss",
    "point_metrics",
    "probabilistic_metrics",
    "rmse",
    "seasonal_naive_mae",
    "smape",
]

NEAR_ZERO_RATIO: float = 1e-3
"""Fraccion de la magnitud media por debajo de la cual un valor se considera "casi cero".

Un umbral absoluto no sirve: ``0.5`` es casi cero en una serie de 3 000 kW y es
la mitad de la serie en una de 1 kW. El umbral se fija en relacion a
``mean(|y|)`` para que el aviso de MAPE signifique lo mismo en las dos.
"""


# --------------------------------------------------------------------------- #
# Utilidades internas
# --------------------------------------------------------------------------- #


def _paired(y: npt.ArrayLike, y_hat: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Alinea observado y predicho descartando los pares incompletos.

    Parameters
    ----------
    y, y_hat
        Secuencias de la misma longitud.

    Returns
    -------
    tuple of numpy.ndarray
        Los pares en los que ambos valores son finitos.

    Raises
    ------
    ValueError
        Si las longitudes no coinciden. Alinearlas por posicion cuando difieren
        es la forma silenciosa de comparar cosas distintas.
    """
    observed = np.asarray(y, dtype=float).ravel()
    predicted = np.asarray(y_hat, dtype=float).ravel()
    if observed.shape != predicted.shape:
        raise ValueError(
            f"y y y_hat deben tener la misma longitud: {observed.shape} vs {predicted.shape}"
        )
    valid = np.isfinite(observed) & np.isfinite(predicted)
    return observed[valid], predicted[valid]


def _quantile_matrix(
    frame: pd.DataFrame, quantiles: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrae observado y matriz de cuantiles de una trama de `forecasts`.

    Parameters
    ----------
    frame
        Trama con ``y`` y las columnas de cuantil.
    quantiles
        Rejilla de cuantiles pedida.

    Returns
    -------
    tuple
        Cuantiles presentes, valor observado y matriz ``(n_obs, n_quantiles)``.
        Solo se conservan las filas en las que el observado y **todos** los
        cuantiles son finitos: un modelo sin soporte probabilistico escribe
        ``NaN`` en esas columnas y no debe contaminar la metrica de los que si.
    """
    available = np.array([q for q in quantiles if quantile_column(q) in frame.columns], dtype=float)
    if available.size == 0 or frame.empty:
        return available, np.empty(0), np.empty((0, available.size))

    columns = [quantile_column(q) for q in available]
    observed = frame["y"].to_numpy(dtype=float)
    predicted = frame[columns].to_numpy(dtype=float)

    valid = np.isfinite(observed) & np.isfinite(predicted).all(axis=1)
    return available, observed[valid], predicted[valid]


# --------------------------------------------------------------------------- #
# Metricas puntuales
# --------------------------------------------------------------------------- #


def mae(y: npt.ArrayLike, y_hat: npt.ArrayLike) -> float:
    r"""Error absoluto medio.

    .. math:: \mathrm{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|

    Parameters
    ----------
    y, y_hat
        Valores observados y predichos.

    Returns
    -------
    float
        En las unidades de la serie. ``NaN`` si no queda ningun par valido.
    """
    observed, predicted = _paired(y, y_hat)
    if observed.size == 0:
        return float("nan")
    return float(np.mean(np.abs(observed - predicted)))


def rmse(y: npt.ArrayLike, y_hat: npt.ArrayLike) -> float:
    r"""Raiz del error cuadratico medio.

    .. math:: \mathrm{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}

    Penaliza los errores grandes mas que MAE, asi que ordena los modelos de otra
    manera cuando hay picos. Se reportan las dos, no una.

    Parameters
    ----------
    y, y_hat
        Valores observados y predichos.

    Returns
    -------
    float
        En las unidades de la serie. ``NaN`` si no queda ningun par valido.
    """
    observed, predicted = _paired(y, y_hat)
    if observed.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def mape(
    y: npt.ArrayLike,
    y_hat: npt.ArrayLike,
    *,
    near_zero_ratio: float = NEAR_ZERO_RATIO,
) -> float:
    r"""Error porcentual absoluto medio: informativa, nunca de seleccion.

    .. math::

        \mathrm{MAPE} = \frac{100}{n} \sum_{i=1}^{n}
        \left| \frac{y_i - \hat{y}_i}{y_i} \right|

    Las observaciones con ``y = 0`` se excluyen porque la expresion no esta
    definida en ellas; las que estan *cerca* de cero si entran, y son las que
    hacen que el numero deje de significar nada: un error de 0.2 sobre un
    observado de 0.1 aporta un 200 % a la media. Por eso, cuando aparecen, se
    emite `UnstableMetricWarning` con el recuento exacto en lugar de devolver un
    numero grande sin explicacion.

    Parameters
    ----------
    y, y_hat
        Valores observados y predichos.
    near_zero_ratio
        Un observado cuenta como "casi cero" si ``|y| < near_zero_ratio *
        mean(|y|)``. Ver `NEAR_ZERO_RATIO`.

    Returns
    -------
    float
        Porcentaje. ``NaN`` si no queda ningun par valido o si todos los
        observados son cero.

    Warns
    -----
    UnstableMetricWarning
        Si hay ceros exactos —que se descartan— o valores casi nulos —que se
        conservan y dominan el resultado—.

    See Also
    --------
    mase : la metrica principal del proyecto, que no tiene este problema.
    """
    observed, predicted = _paired(y, y_hat)
    if observed.size == 0:
        return float("nan")

    magnitude = np.abs(observed)
    scale = float(np.mean(magnitude))
    nonzero = magnitude > 0.0
    n_zero = int(np.count_nonzero(~nonzero))
    n_near_zero = int(np.count_nonzero(nonzero & (magnitude < near_zero_ratio * scale)))

    if n_zero or n_near_zero:
        warnings.warn(
            f"MAPE sobre una serie que pasa por cero: {n_zero} observaciones nulas "
            f"(descartadas, la metrica no esta definida en ellas) y {n_near_zero} por "
            f"debajo del {near_zero_ratio:.1%} de la magnitud media ({scale:.4g}), que "
            "dominan el resultado. Usa MASE para comparar modelos.",
            UnstableMetricWarning,
            stacklevel=2,
        )

    if not nonzero.any():
        return float("nan")
    errors = np.abs(observed[nonzero] - predicted[nonzero]) / magnitude[nonzero]
    return float(100.0 * np.mean(errors))


def smape(y: npt.ArrayLike, y_hat: npt.ArrayLike) -> float:
    r"""Error porcentual absoluto medio simetrico, en la version acotada a 200 %.

    .. math::

        \mathrm{sMAPE} = \frac{100}{n} \sum_{i=1}^{n}
        \frac{2\,|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}

    Circulan al menos tres definiciones de sMAPE que difieren en un factor dos y
    en si el denominador lleva valor absoluto; los numeros publicados con una no
    son comparables con los de otra. Esta es la de la competicion M4: rango
    ``[0, 200]``, cero cuando la prediccion es exacta.

    Sigue sin resolver el problema de MAPE, solo lo suaviza: cuando observado y
    predicho son ambos casi cero el cociente es inestable igual. Las filas en las
    que el denominador es exactamente cero —que solo ocurre si ambos son cero, es
    decir, prediccion perfecta— aportan ``0``.

    Parameters
    ----------
    y, y_hat
        Valores observados y predichos.

    Returns
    -------
    float
        Porcentaje en ``[0, 200]``. ``NaN`` si no queda ningun par valido.
    """
    observed, predicted = _paired(y, y_hat)
    if observed.size == 0:
        return float("nan")

    denominator = np.abs(observed) + np.abs(predicted)
    ratio = np.zeros_like(denominator)
    nonzero = denominator > 0.0
    ratio[nonzero] = 2.0 * np.abs(observed[nonzero] - predicted[nonzero]) / denominator[nonzero]
    return float(100.0 * np.mean(ratio))


def seasonal_naive_mae(y_train: npt.ArrayLike, *, season: int) -> float:
    r"""Denominador de MASE: MAE del naive estacional **sobre el entrenamiento**.

    .. math::

        q = \frac{1}{n_{\text{train}} - m}
        \sum_{t=m+1}^{n_{\text{train}}} \left| Y_t - Y_{t-m} \right|

    donde :math:`Y` es la serie del **tramo de entrenamiento de la ventana en
    curso** y :math:`m` es la longitud estacional (`PanelSpec.mase_season`).

    Aqui es donde se rompen la mayoria de los repositorios de forecasting: el
    denominador se calcula sobre el conjunto de **test**, o sobre la serie
    completa, o una sola vez para todo el run. Las tres variantes son fuga —el
    denominador incorpora informacion posterior al cutoff— y ademas destruyen la
    comparabilidad, porque el mismo modelo cambia de MASE segun que ventana se
    evalue. La definicion de Hyndman y Koehler (2006) es explicita: la escala
    sale del historico disponible en el momento de predecir, y solo de el.

    Los pares en los que alguno de los dos valores es `NaN` se descartan, de modo
    que un hueco en el entrenamiento reduce el numero de diferencias pero no
    contamina la escala.

    Parameters
    ----------
    y_train
        Serie de entrenamiento **de una sola serie**, en orden cronologico y
        sobre rejilla completa (invariante I3). Que la rejilla sea completa es lo
        que hace que ``Y_{t-m}`` sea de verdad el valor de hace `m` pasos y no el
        de hace `m` observaciones.
    season
        Longitud estacional en pasos, mayor o igual que uno.

    Returns
    -------
    float
        El denominador `q`. ``NaN`` si el entrenamiento no llega a `m + 1`
        observaciones utiles.

    Warns
    -----
    UnstableMetricWarning
        Si `q` es cero, es decir, si el naive estacional no comete ningun error
        en el entrenamiento —una serie perfectamente periodica o constante—. En
        ese caso MASE queda indefinida y se devuelve ``NaN``.

    Raises
    ------
    ValueError
        Si `season` es menor que uno.

    Examples
    --------
    Con ``m = 2`` sobre seis observaciones hay cuatro diferencias estacionales:

    >>> seasonal_naive_mae([10, 20, 30, 40, 50, 62], season=2)
    20.5
    """
    if season < 1:
        raise ValueError(f"la longitud estacional debe ser >= 1: {season}")

    values = np.asarray(y_train, dtype=float).ravel()
    if values.size <= season:
        return float("nan")

    current, lagged = values[season:], values[:-season]
    valid = np.isfinite(current) & np.isfinite(lagged)
    if not valid.any():
        return float("nan")

    denominator = float(np.mean(np.abs(current[valid] - lagged[valid])))
    if denominator == 0.0:
        warnings.warn(
            f"el naive estacional (m={season}) no comete ningun error en el "
            "entrenamiento, asi que el denominador de MASE es cero y la metrica "
            "queda indefinida para esta serie y esta ventana",
            UnstableMetricWarning,
            stacklevel=2,
        )
        return float("nan")
    return denominator


def mase(
    y: npt.ArrayLike,
    y_hat: npt.ArrayLike,
    *,
    denominator: float | npt.ArrayLike,
) -> float:
    r"""Error absoluto escalado medio, la metrica principal del proyecto.

    .. math::

        \mathrm{MASE} = \frac{1}{n_{\text{test}}}
        \sum_{i=1}^{n_{\text{test}}} \frac{|y_i - \hat{y}_i|}{q}

    con `q` el MAE del naive estacional calculado **sobre el entrenamiento** de
    la ventana (`seasonal_naive_mae`), nunca sobre el tramo evaluado.

    El denominador se admite como escalar o como vector alineado con las
    observaciones. El vector es el caso normal en este proyecto: cuando se agrega
    sobre varias ventanas —o sobre varias series— cada fila lleva el `q` de *su*
    ventana y *su* serie, se escala primero y se promedia despues. Al reves
    —promediar los errores y dividir por un `q` medio— se mezclan escalas y el
    resultado no es MASE de nada.

    Interpretacion: ``MASE = 1`` empata con el naive estacional; por debajo de
    uno lo bate; por encima, no. Como el naive estacional se mide sobre el
    entrenamiento y el modelo sobre el test, un ``MASE`` ligeramente mayor que
    uno no significa necesariamente que el modelo sea inutil, sino que el tramo
    evaluado era mas dificil que su historico.

    Parameters
    ----------
    y, y_hat
        Valores observados y predichos del tramo evaluado.
    denominator
        `q` escalar, o un vector con el `q` de cada observacion. Los valores no
        positivos o no finitos anulan su fila.

    Returns
    -------
    float
        Adimensional. ``NaN`` si no queda ninguna observacion con denominador
        utilizable.

    Raises
    ------
    ValueError
        Si el vector de denominadores no tiene la longitud de las observaciones.

    See Also
    --------
    seasonal_naive_mae : el denominador, y donde esta el error clasico.

    Examples
    --------
    >>> mase([10, 20], [12, 18], denominator=20.5)
    0.0975609756097561
    """
    observed = np.asarray(y, dtype=float).ravel()
    predicted = np.asarray(y_hat, dtype=float).ravel()
    if observed.shape != predicted.shape:
        raise ValueError(
            f"y y y_hat deben tener la misma longitud: {observed.shape} vs {predicted.shape}"
        )

    scale = np.asarray(denominator, dtype=float).ravel()
    if scale.size == 1:
        scale = np.repeat(scale, observed.size)
    if scale.shape != observed.shape:
        raise ValueError(
            f"el denominador debe ser escalar o tener {observed.size} valores: {scale.size}"
        )

    valid = np.isfinite(observed) & np.isfinite(predicted) & np.isfinite(scale) & (scale > 0.0)
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(observed[valid] - predicted[valid]) / scale[valid]))


# --------------------------------------------------------------------------- #
# Metricas probabilisticas
# --------------------------------------------------------------------------- #


def pinball_loss(y: npt.ArrayLike, q_hat: npt.ArrayLike, *, quantile: float) -> float:
    r"""Perdida pinball media de un cuantil.

    .. math::

        \mathrm{PL}_\tau(y, \hat{q}) =
        \begin{cases}
            \tau\,(y - \hat{q}) & \text{si } y \ge \hat{q} \\
            (1 - \tau)\,(\hat{q} - y) & \text{si } y < \hat{q}
        \end{cases}

    Es la regla de puntuacion propia del cuantil: se minimiza en expectativa
    cuando :math:`\hat{q}` es el cuantil `tau` verdadero de la distribucion
    predictiva. Por eso un modelo no puede mejorarla ensanchando o estrechando
    sus intervalos a conveniencia, que es justo lo que si permite mirar solo la
    cobertura.

    Parameters
    ----------
    y
        Valores observados.
    q_hat
        Cuantil predicho para cada observacion.
    quantile
        Nivel `tau` en ``(0, 1)``.

    Returns
    -------
    float
        En las unidades de la serie. ``NaN`` si no queda ningun par valido.

    Raises
    ------
    ValueError
        Si `quantile` cae fuera de ``(0, 1)``.

    Examples
    --------
    Con ``tau = 0.25`` una infraprediccion pesa un cuarto de la desviacion:

    >>> pinball_loss([11.0], [8.0], quantile=0.25)
    0.75
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"cuantil fuera de (0, 1): {quantile}")

    observed, predicted = _paired(y, q_hat)
    if observed.size == 0:
        return float("nan")

    difference = observed - predicted
    losses = np.where(difference >= 0.0, quantile * difference, (quantile - 1.0) * difference)
    return float(np.mean(losses))


def empirical_coverage(y: npt.ArrayLike, lower: npt.ArrayLike, upper: npt.ArrayLike) -> float:
    """Fraccion de observaciones dentro del intervalo, extremos incluidos.

    Se compara contra la **cobertura nominal** del intervalo: un intervalo al
    95 % que cubre el 78 % de las observaciones esta mal calibrado, y la
    diferencia es exactamente lo que hay que reportar. La cobertura por si sola
    no basta para elegir modelo —un intervalo de anchura infinita cubre el
    100 %—, y por eso siempre viaja junto a `interval_width` y a la pinball.

    Parameters
    ----------
    y
        Valores observados.
    lower, upper
        Extremos del intervalo para cada observacion.

    Returns
    -------
    float
        Fraccion en ``[0, 1]``. ``NaN`` si no queda ninguna terna valida.
    """
    observed = np.asarray(y, dtype=float).ravel()
    low = np.asarray(lower, dtype=float).ravel()
    high = np.asarray(upper, dtype=float).ravel()
    if not observed.shape == low.shape == high.shape:
        raise ValueError("y, lower y upper deben tener la misma longitud")

    valid = np.isfinite(observed) & np.isfinite(low) & np.isfinite(high)
    if not valid.any():
        return float("nan")
    inside = (observed[valid] >= low[valid]) & (observed[valid] <= high[valid])
    return float(np.mean(inside))


def interval_width(lower: npt.ArrayLike, upper: npt.ArrayLike) -> float:
    """Anchura media del intervalo predictivo.

    Es la mitad del par que hace interpretable a la cobertura: entre dos modelos
    igual de calibrados, gana el que lo consigue con intervalos mas estrechos.

    Parameters
    ----------
    lower, upper
        Extremos del intervalo para cada observacion.

    Returns
    -------
    float
        En las unidades de la serie. ``NaN`` si no queda ningun par valido.
    """
    low = np.asarray(lower, dtype=float).ravel()
    high = np.asarray(upper, dtype=float).ravel()
    if low.shape != high.shape:
        raise ValueError("lower y upper deben tener la misma longitud")

    valid = np.isfinite(low) & np.isfinite(high)
    if not valid.any():
        return float("nan")
    return float(np.mean(high[valid] - low[valid]))


def crps_discrete(
    y: npt.ArrayLike,
    quantile_values: npt.ArrayLike,
    *,
    quantiles: Sequence[float],
) -> float:
    r"""CRPS aproximado por integracion de la perdida pinball sobre la rejilla de cuantiles.

    Se apoya en la identidad exacta

    .. math:: \mathrm{CRPS}(F, y) = 2 \int_0^1 \mathrm{PL}_\tau(y, F^{-1}(\tau))\, d\tau

    y la aproxima por regla del trapecio sobre los cuantiles disponibles. El
    nombre lleva el apellido `discrete` a proposito: con siete cuantiles esto
    **no** es el CRPS, es su aproximacion discreta, y llamarlo CRPS a secas seria
    sobrevender. Dos consecuencias que conviene tener presentes:

    - La integral solo cubre ``[min(quantiles), max(quantiles)]``. Con la rejilla
      canonica ``[0.025, 0.975]`` queda fuera un 5 % de masa en las colas, asi
      que el valor es una **cota inferior** del CRPS verdadero.
    - Comparar valores calculados con rejillas distintas no es legitimo. La
      rejilla es del run y por eso se fija en la configuracion.

    Parameters
    ----------
    y
        Valores observados, de longitud ``n``.
    quantile_values
        Matriz ``(n, k)`` con el cuantil predicho de cada observacion, en el
        mismo orden que `quantiles`.
    quantiles
        Rejilla de cuantiles, estrictamente creciente, con al menos dos valores.

    Returns
    -------
    float
        En las unidades de la serie. ``NaN`` si no queda ninguna fila valida.

    Raises
    ------
    ValueError
        Si la rejilla tiene menos de dos cuantiles, no es creciente, o no cuadra
        con las columnas de `quantile_values`.

    Examples
    --------
    Un unico punto observado en 11 con cuantiles 0.25, 0.5 y 0.75 en 8, 10 y 13:

    >>> crps_discrete([11.0], [[8.0, 10.0, 13.0]], quantiles=(0.25, 0.5, 0.75))
    0.5625
    """
    grid = np.asarray(quantiles, dtype=float).ravel()
    if grid.size < 2:
        raise ValueError(f"se necesitan al menos dos cuantiles para integrar: {grid.size}")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError(f"la rejilla de cuantiles debe ser creciente: {quantiles}")

    observed = np.asarray(y, dtype=float).ravel()
    predicted = np.atleast_2d(np.asarray(quantile_values, dtype=float))
    if predicted.shape != (observed.size, grid.size):
        raise ValueError(
            f"quantile_values debe ser ({observed.size}, {grid.size}): {predicted.shape}"
        )

    valid = np.isfinite(observed) & np.isfinite(predicted).all(axis=1)
    if not valid.any():
        return float("nan")

    difference = observed[valid, None] - predicted[valid]
    losses = np.where(difference >= 0.0, grid * difference, (grid - 1.0) * difference)

    widths = np.diff(grid)
    trapezoids = 0.5 * (losses[:, :-1] + losses[:, 1:]) * widths
    return float(2.0 * np.mean(trapezoids.sum(axis=1)))


def interval_levels(quantiles: Sequence[float]) -> tuple[tuple[float, float, float], ...]:
    """Intervalos centrales que una rejilla de cuantiles permite formar.

    Parameters
    ----------
    quantiles
        Rejilla de cuantiles.

    Returns
    -------
    tuple
        Ternas ``(nivel, cuantil_inferior, cuantil_superior)`` para cada par
        simetrico presente, de mayor a menor nivel. La rejilla canonica del
        proyecto produce los niveles 95 %, 80 % y 50 %.

    Examples
    --------
    >>> interval_levels((0.1, 0.5, 0.9))
    ((0.8, 0.1, 0.9),)
    """
    available = {round(float(q), 10) for q in quantiles}
    levels: list[tuple[float, float, float]] = []
    for quantile in sorted(available):
        if quantile >= 0.5:
            continue
        upper = round(1.0 - quantile, 10)
        if upper in available:
            levels.append((round(1.0 - 2.0 * quantile, 10), quantile, upper))
    return tuple(levels)


# --------------------------------------------------------------------------- #
# Nivel de trama: de `forecasts` a numeros
# --------------------------------------------------------------------------- #


def mase_denominators(
    panel: Panel,
    windows: pd.DataFrame,
    *,
    season: int | None = None,
) -> pd.DataFrame:
    """Denominador de MASE por serie y por ventana, calculado con el train de cada una.

    Para cada ventana se recorta el panel a ``[train_start, cutoff]`` —el mismo
    tramo exacto que recibio `Forecaster.fit`— y se calcula `seasonal_naive_mae`
    serie a serie. Nada de lo que hay despues del cutoff interviene, que es la
    propiedad que convierte "no hay fuga en la metrica" en algo verificable: el
    denominador se persiste y un tercero puede recalcularlo.

    Parameters
    ----------
    panel
        Panel canonico completo.
    windows
        Tabla `windows` del run, con al menos ``window_id``, ``train_start`` y
        ``cutoff``.
    season
        Longitud estacional. Por defecto ``panel.spec.mase_season``, que es la
        estacionalidad mas corta declarada para el dataset.

    Returns
    -------
    pandas.DataFrame
        Columnas ``unique_id``, ``window_id`` y ``mase_denominator``.
    """
    period = panel.spec.mase_season if season is None else season
    target = panel.spec.target

    window_ids = windows["window_id"].to_numpy()
    starts = pd.DatetimeIndex(windows["train_start"])
    cutoffs = pd.DatetimeIndex(windows["cutoff"])

    rows: list[dict[str, object]] = []
    for position in range(len(windows)):
        train = panel.slice(starts[position], cutoffs[position])
        for uid, group in train.df.groupby("unique_id", sort=False):
            rows.append(
                {
                    "unique_id": str(uid),
                    "window_id": int(window_ids[position]),
                    "mase_denominator": seasonal_naive_mae(
                        group.sort_values("ds")[target].to_numpy(dtype=float), season=period
                    ),
                }
            )

    frame = pd.DataFrame(rows, columns=["unique_id", "window_id", "mase_denominator"])
    frame["window_id"] = frame["window_id"].astype("int16")
    frame["mase_denominator"] = frame["mase_denominator"].astype("float64")
    return frame


def point_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Metricas puntuales de una trama de `forecasts`.

    Parameters
    ----------
    frame
        Filas de `forecasts` con ``y``, ``y_hat`` y, si se quiere MASE, la
        columna ``mase_denominator`` ya unida por ``(unique_id, window_id)``.

    Returns
    -------
    dict
        ``n_obs``, ``mae``, ``rmse``, ``mape``, ``smape`` y ``mase``. MASE es
        ``NaN`` si la trama no trae denominadores.

    Notes
    -----
    Todo se calcula desde las filas crudas. Ninguna de estas cifras se obtiene
    agregando otra ya agregada: MASE y sMAPE son cocientes, y la media de
    cocientes no es el cociente de medias.
    """
    observed = frame["y"].to_numpy(dtype=float)
    predicted = frame["y_hat"].to_numpy(dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)

    metrics: dict[str, float] = {
        "n_obs": float(np.count_nonzero(valid)),
        "mae": mae(observed, predicted),
        "rmse": rmse(observed, predicted),
        "mape": mape(observed, predicted),
        "smape": smape(observed, predicted),
    }
    if "mase_denominator" in frame.columns:
        scale = frame["mase_denominator"].to_numpy(dtype=float)
        metrics["mase"] = mase(observed, predicted, denominator=scale)
        finite = scale[np.isfinite(scale)]
        metrics["mase_denominator"] = float(np.mean(finite)) if finite.size else float("nan")
    else:
        metrics["mase"] = float("nan")
        metrics["mase_denominator"] = float("nan")
    return metrics


def probabilistic_metrics(
    frame: pd.DataFrame,
    *,
    quantiles: Sequence[float] = QUANTILES,
) -> dict[str, float]:
    """Metricas probabilisticas de una trama de `forecasts`.

    Parameters
    ----------
    frame
        Filas de `forecasts` con ``y`` y las columnas de cuantil.
    quantiles
        Rejilla de cuantiles del run.

    Returns
    -------
    dict
        ``n_obs_prob``, una ``pinball_q<cuantil>`` por cuantil, ``pinball_mean``,
        ``crps_discrete`` y, por cada intervalo central que la rejilla permita
        formar, ``coverage_<nivel>`` y ``width_<nivel>``. Todo ``NaN`` para los
        modelos que no producen cuantiles, que es lo correcto: un punto no tiene
        cobertura, y escribir cero seria decir que la tiene y es pesima.
    """
    grid, observed, predicted = _quantile_matrix(frame, quantiles)
    metrics: dict[str, float] = {"n_obs_prob": float(observed.size)}

    pinballs: list[float] = []
    for position, quantile in enumerate(grid):
        value = pinball_loss(observed, predicted[:, position], quantile=float(quantile))
        metrics[f"pinball_{quantile_column(float(quantile))}"] = value
        pinballs.append(value)
    metrics["pinball_mean"] = float(np.mean(pinballs)) if pinballs else float("nan")

    levels = [float(q) for q in grid]
    metrics["crps_discrete"] = (
        crps_discrete(observed, predicted, quantiles=levels)
        if grid.size >= 2 and observed.size
        else float("nan")
    )

    index = {round(float(q), 10): position for position, q in enumerate(grid)}
    for level, low_q, high_q in interval_levels(levels):
        label = f"{round(level * 100)}"
        if observed.size == 0:
            metrics[f"coverage_{label}"] = float("nan")
            metrics[f"width_{label}"] = float("nan")
            continue
        low = predicted[:, index[low_q]]
        high = predicted[:, index[high_q]]
        metrics[f"coverage_{label}"] = empirical_coverage(observed, low, high)
        metrics[f"width_{label}"] = interval_width(low, high)
    return metrics
