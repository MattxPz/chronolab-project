"""Diebold-Mariano con correccion HAC y HLN, y Model Confidence Set.

El MCS es la respuesta correcta a 'que modelos no puedo descartar' cuando hay
decenas de comparaciones por pares (riesgo R11).

Que un modelo tenga menos MASE que otro no significa que sea mejor: significa
que fue mejor en esta muestra. El contraste de Diebold-Mariano (1995) enfrenta la
hipotesis nula de igualdad de precision predictiva sobre la serie de diferencias
de perdida, y es lo que separa "gano por 0.003" de "gano".

Tres decisiones de implementacion, las tres con consecuencias en el numero:

1. **Varianza HAC con nucleo de Bartlett.** Las diferencias de perdida de
   predicciones a `h` pasos estan autocorrelacionadas hasta el retardo ``h - 1``,
   y con ventanas solapadas (``step_size < h``) tambien entre ventanas. El nucleo
   rectangular del articulo original puede dar una varianza **negativa**; el de
   Bartlett es semidefinido positivo por construccion, asi que el estadistico
   siempre existe. Con ``hac_lag = 0`` los dos coinciden.
2. **Correccion de Harvey, Leybourne y Newbold (1997).** El DM asintotico rechaza
   demasiado con muestras cortas, que es exactamente el regimen de un backtest de
   veinte ventanas. Se corrige el estadistico y se compara contra una `t` de
   Student con ``n - 1`` grados de libertad en lugar de contra una normal.
3. **Sin scipy.** La `t` y la normal se evaluan con funciones propias
   (`_student_t_sf`, apoyada en la beta incompleta regularizada) porque
   `evaluation` tiene que funcionar con las dependencias del nucleo, que es lo
   que instala CI. Los tests las contrastan contra valores tabulados a mano y,
   cuando esta disponible, tambien contra scipy.

Sobre la multiplicidad: con doce modelos hay sesenta y seis pares y a un nivel
del 5 % se esperan tres "significativos" por puro azar. Publicar "bate al
baseline con p < 0.05" despues de haber probado doce es un abuso del contraste,
asi que `pairwise_dm` devuelve siempre el p-valor ajustado y el tamano de la
familia. El Model Confidence Set, que es la respuesta completa, se apoyara en
esta tabla y todavia no esta implementado.
"""

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from chronolab.errors import UnstableMetricWarning

__all__ = [
    "PAIRWISE_COLUMNS",
    "AdjustMethod",
    "DMLoss",
    "DMResult",
    "adjust_p_values",
    "diebold_mariano",
    "dm_from_errors",
    "pairwise_dm",
]

DMLoss = Literal["abs", "sq"]
"""Perdida sobre la que se compara: absoluta o cuadratica.

No es un detalle de implementacion: DM contrasta igualdad de precision **bajo una
perdida**, y dos modelos pueden empatar bajo la absoluta y diferir bajo la
cuadratica si uno falla poco muchas veces y el otro mucho pocas veces. Por eso la
perdida viaja en la tabla de resultados.
"""

AdjustMethod = Literal["holm", "bonferroni", "bh", "none"]
"""Correccion por comparaciones multiples aplicada a la familia de pares."""


# --------------------------------------------------------------------------- #
# Distribuciones, sin scipy
# --------------------------------------------------------------------------- #


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Fraccion continua de la beta incompleta, por el metodo modificado de Lentz.

    Parameters
    ----------
    a, b
        Parametros de la beta.
    x
        Punto de evaluacion en ``(0, 1)``.

    Returns
    -------
    float
        Valor de la fraccion continua.
    """
    tiny = 1e-300
    epsilon = 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d

    for m in range(1, 300):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c

        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        step = d * c
        h *= step
        if abs(step - 1.0) < epsilon:
            break
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Beta incompleta regularizada ``I_x(a, b)``.

    Parameters
    ----------
    a, b
        Parametros de la beta, positivos.
    x
        Punto de evaluacion en ``[0, 1]``.

    Returns
    -------
    float
        Valor en ``[0, 1]``.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    log_prefactor = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    prefactor = math.exp(log_prefactor)
    if x < (a + 1.0) / (a + b + 2.0):
        return prefactor * _beta_continued_fraction(a, b, x) / a
    return 1.0 - prefactor * _beta_continued_fraction(b, a, 1.0 - x) / b


def _student_t_sf(t: float, df: float) -> float:
    """Cola derecha de una `t` de Student: ``P(T > t)``.

    Parameters
    ----------
    t
        Valor del estadistico.
    df
        Grados de libertad, positivos.

    Returns
    -------
    float
        Probabilidad en ``[0, 1]``.
    """
    if not math.isfinite(t):
        return 0.0 if t > 0 else 1.0
    tail = 0.5 * _regularized_incomplete_beta(0.5 * df, 0.5, df / (df + t * t))
    return tail if t > 0 else 1.0 - tail


def _normal_sf(z: float) -> float:
    """Cola derecha de una normal estandar.

    Parameters
    ----------
    z
        Valor del estadistico.

    Returns
    -------
    float
        Probabilidad en ``[0, 1]``.
    """
    if not math.isfinite(z):
        return 0.0 if z > 0 else 1.0
    return 0.5 * math.erfc(z / math.sqrt(2.0))


# --------------------------------------------------------------------------- #
# Diebold-Mariano
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DMResult:
    """Resultado de un contraste de Diebold-Mariano.

    Attributes
    ----------
    stat
        Estadistico, ya corregido si `hln_corrected`. Su **signo lleva la
        direccion**: negativo significa que el primer modelo pierde menos, es
        decir, que es el mejor de los dos.
    p_value
        P-valor bilateral, **sin** ajustar por multiplicidad: eso lo hace
        `pairwise_dm`, que es quien conoce el tamano de la familia.
    n_obs
        Observaciones comparadas.
    hac_lag
        Retardo de truncamiento de la varianza HAC.
    hln_corrected
        Si se aplico la correccion de muestra pequena. Cuando es ``False`` el
        p-valor sale de la normal asintotica y es optimista.
    mean_difference
        Media de la diferencia de perdidas. Es el tamano del efecto, y sin el un
        p-valor no dice si la diferencia importa.
    degenerate
        ``True`` si la varianza estimada es cero: perdidas identicas o diferencia
        constante.
    """

    stat: float
    p_value: float
    n_obs: int
    hac_lag: int
    hln_corrected: bool
    mean_difference: float
    degenerate: bool


def _hac_variance(differences: np.ndarray, *, hac_lag: int) -> float:
    r"""Varianza de la media de una serie autocorrelacionada, con nucleo de Bartlett.

    .. math::

        \widehat{V}(\bar{d}) = \frac{1}{n}\left[\gamma_0 +
        2\sum_{k=1}^{L}\left(1 - \frac{k}{L+1}\right)\gamma_k\right]

    con :math:`\gamma_k = \frac{1}{n}\sum_t (d_t - \bar{d})(d_{t-k} - \bar{d})`.

    Parameters
    ----------
    differences
        Serie de diferencias de perdida.
    hac_lag
        Retardo de truncamiento `L`. Con ``0`` se reduce a ``gamma_0 / n``.

    Returns
    -------
    float
        Estimacion no negativa de la varianza de la media.
    """
    n = differences.size
    centred = differences - differences.mean()
    total = float(np.mean(centred**2))

    for k in range(1, min(hac_lag, n - 1) + 1):
        # Cada autocovarianza se divide entre `n`, no entre los `n - k` productos
        # que la componen. Es lo que exige la garantia de no negatividad del
        # nucleo de Bartlett: con `n - k` se sobrepondera la cola y la suma puede
        # salir negativa, que es justo el defecto del nucleo rectangular que este
        # estimador viene a evitar.
        gamma_k = float(np.sum(centred[k:] * centred[:-k]) / n)
        total += 2.0 * (1.0 - k / (hac_lag + 1.0)) * gamma_k
    return max(total, 0.0) / n


def diebold_mariano(
    loss_a: npt.ArrayLike,
    loss_b: npt.ArrayLike,
    *,
    hac_lag: int = 0,
    hln: bool = True,
) -> DMResult:
    r"""Contrasta igualdad de precision predictiva entre dos modelos.

    Trabaja sobre la serie de diferencias :math:`d_t = L^a_t - L^b_t`. La nula es
    :math:`E[d_t] = 0` y el estadistico es :math:`\bar{d} / \sqrt{\widehat{V}
    (\bar{d})}`, con la varianza HAC de `_hac_variance`.

    Con `hln`, el estadistico se multiplica por

    .. math:: \sqrt{\frac{n + 1 - 2h + h(h-1)/n}{n}}, \qquad h = L + 1

    y se compara contra una `t` de Student con ``n - 1`` grados de libertad
    (Harvey, Leybourne y Newbold, 1997). Sin ella, contra una normal estandar.

    **Perdidas identicas.** Si :math:`d_t = 0` para todo `t` —dos modelos que
    producen la misma prediccion, o un modelo comparado consigo mismo— la
    varianza es cero y el cociente seria ``0/0``. El resultado correcto es el
    trivial: estadistico ``0`` y p-valor ``1``, porque no hay ninguna evidencia
    de diferencia. Se marca ``degenerate=True`` para que quien lea la tabla sepa
    que no es un empate estadistico, es una identidad.

    Si la diferencia es constante y **no** nula, la varianza tambien es cero pero
    la media no: el estadistico diverge. Se devuelve ``inf`` con signo y p-valor
    ``0``, con aviso, porque ese cero es un artefacto de la muestra degenerada y
    no una certeza.

    Parameters
    ----------
    loss_a, loss_b
        Perdidas por observacion de cada modelo, alineadas y **en orden
        temporal**. El orden importa: la correccion HAC supone que las posiciones
        contiguas son instantes contiguos.
    hac_lag
        Retardo de truncamiento. Para predicciones a `h` pasos con separacion
        `gap`, el valor coherente con el motor de backtesting es ``gap + h - 1``.
    hln
        Aplicar la correccion de muestra pequena. Se desactiva sola, con aviso,
        si el factor de ajuste no es positivo.

    Returns
    -------
    DMResult
        Estadistico, p-valor bilateral y contexto.

    Raises
    ------
    ValueError
        Si las series no tienen la misma longitud, si quedan menos de dos
        observaciones o si `hac_lag` es negativo.

    Warns
    -----
    UnstableMetricWarning
        Si la varianza estimada es cero con media no nula, o si la correccion HLN
        no es aplicable.

    Examples
    --------
    >>> resultado = diebold_mariano([2.0, 3.0, 4.0, 5.0], [1.0, 1.0, 1.0, 1.0])
    >>> round(resultado.stat, 6)
    3.872983
    """
    if hac_lag < 0:
        raise ValueError(f"hac_lag debe ser >= 0: {hac_lag}")

    first = np.asarray(loss_a, dtype=float).ravel()
    second = np.asarray(loss_b, dtype=float).ravel()
    if first.shape != second.shape:
        raise ValueError(
            f"las perdidas deben tener la misma longitud: {first.shape} vs {second.shape}"
        )

    differences = first - second
    differences = differences[np.isfinite(differences)]
    n = differences.size
    if n < 2:
        raise ValueError(f"el contraste necesita al menos dos observaciones: {n}")

    mean_difference = float(differences.mean())
    variance = _hac_variance(differences, hac_lag=hac_lag)

    if variance <= 0.0:
        return _degenerate_result(mean_difference, n=n, hac_lag=hac_lag)

    stat = mean_difference / math.sqrt(variance)
    corrected = False
    if hln:
        h = hac_lag + 1
        factor = (n + 1 - 2 * h + h * (h - 1) / n) / n
        if factor > 0.0:
            stat *= math.sqrt(factor)
            corrected = True
        else:
            warnings.warn(
                f"la correccion HLN no es aplicable con n={n} y h={h}: el factor de "
                "ajuste no es positivo, asi que el p-valor sale de la normal asintotica "
                "y sera optimista",
                UnstableMetricWarning,
                stacklevel=2,
            )

    p_value = 2.0 * _student_t_sf(abs(stat), df=n - 1) if corrected else 2.0 * _normal_sf(abs(stat))
    return DMResult(
        stat=stat,
        p_value=min(p_value, 1.0),
        n_obs=n,
        hac_lag=hac_lag,
        hln_corrected=corrected,
        mean_difference=mean_difference,
        degenerate=False,
    )


def _degenerate_result(mean_difference: float, *, n: int, hac_lag: int) -> DMResult:
    """Resultado cuando la varianza estimada de la diferencia es exactamente cero.

    Parameters
    ----------
    mean_difference
        Media de la diferencia de perdidas.
    n
        Observaciones comparadas.
    hac_lag
        Retardo de truncamiento aplicado.

    Returns
    -------
    DMResult
        Con ``degenerate=True``.

    Warns
    -----
    UnstableMetricWarning
        Si la diferencia es constante y no nula.
    """
    if mean_difference == 0.0:
        return DMResult(
            stat=0.0,
            p_value=1.0,
            n_obs=n,
            hac_lag=hac_lag,
            hln_corrected=False,
            mean_difference=0.0,
            degenerate=True,
        )

    warnings.warn(
        "la diferencia de perdidas es constante y no nula: la varianza estimada es cero "
        "y el estadistico de Diebold-Mariano diverge. El p-valor de 0 es un artefacto de "
        "la muestra, no una certeza.",
        UnstableMetricWarning,
        stacklevel=3,
    )
    return DMResult(
        stat=math.copysign(math.inf, mean_difference),
        p_value=0.0,
        n_obs=n,
        hac_lag=hac_lag,
        hln_corrected=False,
        mean_difference=mean_difference,
        degenerate=True,
    )


def dm_from_errors(
    y: npt.ArrayLike,
    y_hat_a: npt.ArrayLike,
    y_hat_b: npt.ArrayLike,
    *,
    loss: DMLoss = "abs",
    hac_lag: int = 0,
    hln: bool = True,
) -> DMResult:
    """Diebold-Mariano a partir de predicciones, con la perdida indicada.

    Parameters
    ----------
    y
        Valores observados.
    y_hat_a, y_hat_b
        Predicciones de cada modelo, alineadas con `y` y en orden temporal.
    loss
        ``"abs"`` para perdida absoluta, ``"sq"`` para cuadratica.
    hac_lag, hln
        Ver `diebold_mariano`.

    Returns
    -------
    DMResult
        Resultado del contraste, calculado sobre las filas en las que los tres
        vectores son finitos.

    Raises
    ------
    ValueError
        Si la perdida no esta admitida o las longitudes no coinciden.
    """
    observed = np.asarray(y, dtype=float).ravel()
    first = np.asarray(y_hat_a, dtype=float).ravel()
    second = np.asarray(y_hat_b, dtype=float).ravel()
    if not observed.shape == first.shape == second.shape:
        raise ValueError("y, y_hat_a y y_hat_b deben tener la misma longitud")

    valid = np.isfinite(observed) & np.isfinite(first) & np.isfinite(second)
    errors_a = observed[valid] - first[valid]
    errors_b = observed[valid] - second[valid]

    if loss == "abs":
        return diebold_mariano(np.abs(errors_a), np.abs(errors_b), hac_lag=hac_lag, hln=hln)
    if loss == "sq":
        return diebold_mariano(errors_a**2, errors_b**2, hac_lag=hac_lag, hln=hln)
    raise ValueError(f"perdida no admitida: {loss}")


# --------------------------------------------------------------------------- #
# Multiplicidad
# --------------------------------------------------------------------------- #


def adjust_p_values(
    p_values: npt.ArrayLike,
    *,
    method: AdjustMethod = "holm",
) -> np.ndarray:
    """Corrige una familia de p-valores por comparaciones multiples.

    Parameters
    ----------
    p_values
        P-valores sin ajustar de la familia **completa**. Ajustar un subconjunto
        elegido despues de ver los resultados no corrige nada.
    method
        - ``"holm"``: Holm-Bonferroni. Controla la tasa de error por familia sin
          suponer independencia y domina uniformemente a Bonferroni, asi que es
          el valor por defecto.
        - ``"bonferroni"``: el mas conservador.
        - ``"bh"``: Benjamini-Hochberg, controla la tasa de falsos
          descubrimientos. Mas potente y mas laxo: sirve para explorar, no para
          afirmar.
        - ``"none"``: los deja como estan. Existe para poder dejar escrito en la
          tabla que no se corrigio, no como atajo.

    Returns
    -------
    numpy.ndarray
        P-valores ajustados, en el orden de entrada y acotados a ``[0, 1]``.

    Raises
    ------
    ValueError
        Si el metodo no esta admitido.

    Examples
    --------
    >>> adjust_p_values([0.01, 0.04], method="bonferroni").tolist()
    [0.02, 0.08]
    >>> adjust_p_values([0.01, 0.04], method="holm").tolist()
    [0.02, 0.04]
    """
    values = np.asarray(p_values, dtype=float).ravel()
    n = values.size
    if n == 0:
        return values
    if method == "none":
        return values.copy()
    if method == "bonferroni":
        bonferroni: np.ndarray = np.minimum(values * n, 1.0)
        return bonferroni
    if method not in ("holm", "bh"):
        raise ValueError(f"metodo de correccion no admitido: {method}")

    order = np.argsort(values, kind="stable")
    ordered = values[order]
    adjusted = np.empty(n, dtype=float)

    if method == "holm":
        # Escalonado descendente: el p-valor mas pequeno se multiplica por los `n`
        # contrastes de la familia, el siguiente por `n - 1`... y se impone
        # monotonia hacia arriba para no rechazar uno grande despues de no haber
        # podido rechazar uno mas pequeno.
        scaled = ordered * (n - np.arange(n))
        adjusted[order] = np.minimum(np.maximum.accumulate(scaled), 1.0)
        return adjusted

    # Benjamini-Hochberg: escalonado ascendente, con monotonia hacia abajo.
    scaled = ordered * n / (np.arange(n) + 1.0)
    adjusted[order] = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
    return adjusted


# --------------------------------------------------------------------------- #
# Matriz de comparaciones
# --------------------------------------------------------------------------- #

PAIRWISE_COLUMNS: tuple[str, ...] = (
    "model_a",
    "model_b",
    "unique_id",
    "loss",
    "stat",
    "p_value",
    "p_value_adjusted",
    "adjust_method",
    "significant",
    "hac_lag",
    "hln_corrected",
    "degenerate",
    "mean_difference",
    "n_obs",
    "n_comparisons",
)
"""Columnas de la tabla de comparaciones por pares (docs/ARCHITECTURE.md §7.4)."""


def pairwise_dm(
    forecasts: pd.DataFrame,
    *,
    loss: DMLoss = "abs",
    hac_lag: int = 0,
    hln: bool = True,
    method: AdjustMethod = "holm",
    alpha: float = 0.05,
    by_series: bool = False,
    models: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Matriz de contrastes de Diebold-Mariano entre todos los modelos.

    Cada par aparece **una sola vez**, en orden lexicografico, y la direccion la
    lleva el signo del estadistico: negativo significa que `model_a` pierde
    menos. Emitir los dos sentidos duplicaria el tamano de la familia y haria mas
    severa la correccion por multiplicidad sin anadir informacion.

    Los dos modelos de un par se comparan sobre las filas que **ambos** tienen:
    si uno fallo en una ventana, esa ventana sale de la comparacion y `n_obs` lo
    refleja. Comparar sobre soportes distintos favorece sistematicamente al que
    se salto los tramos dificiles.

    Parameters
    ----------
    forecasts
        Tabla `forecasts` de un run, ya filtrada al alcance que se quiere
        contrastar. El leaderboard publica ``stage="holdout"`` y filtrar es
        responsabilidad de quien llama, porque esta trama no lleva la etapa;
        `chronolab.evaluation.aggregate.select_stage` hace ese corte.
    loss
        Perdida del contraste.
    hac_lag
        Retardo de truncamiento HAC. Coherente con el motor: ``gap + h - 1``.
    hln
        Correccion de muestra pequena.
    method
        Correccion por comparaciones multiples sobre la familia completa de
        pares. Ver `adjust_p_values`.
    alpha
        Nivel con el que se rellena la columna ``significant``, que compara
        contra el p-valor **ajustado**.
    by_series
        Si es ``True``, un contraste por serie y par, y la familia de correccion
        son todos los pares de todas las series. Si es ``False``, un unico
        contraste por par sobre todas las series.
    models
        Subconjunto de modelos a comparar. Por defecto, todos los de la trama.

    Returns
    -------
    pandas.DataFrame
        Una fila por comparacion, con las columnas de `PAIRWISE_COLUMNS`.

    Raises
    ------
    ValueError
        Si no hay al menos dos modelos que comparar.

    Notes
    -----
    Al agrupar series (``by_series=False``) las filas se ordenan por
    ``(unique_id, window_id, ds)`` y se concatenan. La correccion HAC trata esa
    concatenacion como una sola serie, de modo que en cada frontera entre series
    se estiman `hac_lag` autocovarianzas que mezclan series distintas. Con unas
    pocas series largas el sesgo es despreciable frente al numero de
    observaciones; con muchas series cortas conviene ``by_series=True``.

    El Model Confidence Set —la herramienta correcta para responder "que modelos
    no puedo descartar"— todavia no esta implementado: esta tabla es su insumo,
    no su sustituto.
    """
    available = sorted(set(forecasts["model_id"])) if models is None else sorted(set(models))
    if len(available) < 2:
        raise ValueError(f"se necesitan al menos dos modelos para comparar: {available}")

    ordered = forecasts.sort_values(["unique_id", "window_id", "ds"])
    groups: list[tuple[str | None, pd.DataFrame]] = (
        [(str(uid), part) for uid, part in ordered.groupby("unique_id", sort=True)]
        if by_series
        else [(None, ordered)]
    )

    rows: list[dict[str, object]] = []
    for series_id, part in groups:
        aligned = _align_models(part)
        for model_a, model_b in combinations(available, 2):
            if model_a not in aligned.columns or model_b not in aligned.columns:
                continue
            usable = aligned[["y", model_a, model_b]].dropna()
            if len(usable) < 2:
                continue
            result = dm_from_errors(
                usable["y"].to_numpy(),
                usable[model_a].to_numpy(),
                usable[model_b].to_numpy(),
                loss=loss,
                hac_lag=hac_lag,
                hln=hln,
            )
            rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "unique_id": series_id,
                    "loss": loss,
                    "stat": result.stat,
                    "p_value": result.p_value,
                    "hac_lag": result.hac_lag,
                    "hln_corrected": result.hln_corrected,
                    "degenerate": result.degenerate,
                    "mean_difference": result.mean_difference,
                    "n_obs": result.n_obs,
                }
            )

    if not rows:
        return pd.DataFrame({column: pd.Series(dtype="object") for column in PAIRWISE_COLUMNS})

    frame = pd.DataFrame(rows)
    frame["p_value_adjusted"] = adjust_p_values(frame["p_value"].to_numpy(), method=method)
    frame["adjust_method"] = method
    frame["significant"] = frame["p_value_adjusted"] < alpha
    frame["n_comparisons"] = np.int32(len(frame))
    frame["n_obs"] = frame["n_obs"].astype("int64")
    frame["hac_lag"] = frame["hac_lag"].astype("int16")
    return frame[list(PAIRWISE_COLUMNS)].reset_index(drop=True)


def _align_models(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Pone las predicciones de todos los modelos en columnas, alineadas por instante.

    Parameters
    ----------
    forecasts
        Filas de `forecasts`, ya ordenadas.

    Returns
    -------
    pandas.DataFrame
        Indexada por ``(unique_id, window_id, ds)``, con una columna por modelo y
        la columna ``y``. El orden del indice es el temporal dentro de cada
        serie, que es el que exige la correccion HAC.
    """
    wide = forecasts.pivot_table(
        index=["unique_id", "window_id", "ds"],
        columns="model_id",
        values="y_hat",
        aggfunc="first",
    )
    observed = forecasts.drop_duplicates(subset=["unique_id", "window_id", "ds"]).set_index(
        ["unique_id", "window_id", "ds"]
    )["y"]
    return wide.join(observed.rename("y")).sort_index()
