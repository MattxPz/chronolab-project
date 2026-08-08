"""Metricas de anomalia: por rangos (Tatbul), de afiliacion (Huet), VUS-PR y AUC-PR.

**Nunca point-adjusted.** El *point-adjusted F1* marca como detectado un
segmento entero de verdad si el detector acierta **un solo** punto suyo. La
literatura de benchmarking (Kim et al. 2022, "Towards a Rigorous Evaluation of
Time-series Anomaly Detection"; Paparrizos et al. 2022, TSB-UAD/VUS) demuestra
que con esa regla un detector que emite **ruido aleatorio** obtiene F1
comparable al estado del arte, porque basta con salpicar puntos al azar para
tocar todos los segmentos largos. Una metrica bajo la que el ruido gana no
mide nada. Este modulo no la implementa, y `tests/unit/evaluation/` incluye el
test adversario que reproduce esa inflacion para dejar constancia de por que.

La alternativa no es "una metrica mejor" sino **cinco familias que fallan de
formas distintas**, de modo que un detector solo parece bueno si lo es bajo
criterios que no comparten sesgo:

1. **Por rangos** (`range_precision_recall`). Trata cada anomalia como un
   intervalo, no como puntos sueltos, y separa cuatro preguntas que el F1
   puntual mezcla: si se detecto, cuanto se solapo, en cuantos trozos se
   partio la deteccion y donde cayo dentro del evento.
2. **De afiliacion** (`affiliation_precision_recall`). Mide distancias, no
   solapes, asi que una deteccion que llega tarde por dos pasos no vale lo
   mismo que una que no llega. Calibrada contra el azar: 0.5 es "aleatorio".
3. **VUS-PR** (`vus_pr`). Integra la superficie PR sobre el umbral **y**
   sobre la tolerancia de desalineamiento temporal, con lo que no exige fijar
   ninguno de los dos.
4. **AUC-PR puntual** (`auc_pr`). La referencia minima, sin ninguna nocion de
   rango. Se incluye para poder decir cuanto cambia la conclusion al pasar a
   metricas por rango: si no cambiara, todo lo demas de este modulo sobraria.
5. **Operativas** (`detection_delay`, `false_alarm_rate`). Lo que decide si un
   detector se despliega: cuanto tarda en avisar y cuantas veces avisa en
   falso. Ninguna de las cuatro anteriores lo dice.

Tres reglas de agregacion, y las tres cambian los numeros
----------------------------------------------------------
- **La mascara `scorable` se interseca antes de comparar** (`common_scorable_mask`,
  docs/ARCHITECTURE.md §5.3). Un detector de ventana 512 puntua menos instantes
  que uno de ventana 1; evaluar a cada uno sobre su propio soporte favorece al
  de ventana larga por haberse saltado el arranque de la serie.
- **Lo que es una media sobre rangos se agrega juntando rangos**, no promediando
  medias: `range_precision_recall` y `affiliation_precision_recall` devuelven la
  contribucion de cada rango o zona precisamente para que agregar entre series
  sea concatenar y promediar una vez. Es la misma regla que prohibe promediar
  promedios en `evaluation.aggregate`.
- **AUC-PR y VUS-PR se promedian entre series, no se agrupan.** Agrupar las
  puntuaciones de varias series en un unico ranking exigiria que fuesen
  comparables entre series, y `anomaly.protocols.FittedDetector.score` declara
  exactamente lo contrario: el score es ordinal **dentro** de un par
  (detector, serie). Agruparlas produciria un numero que parece un AUC y no lo
  es.

Por que este modulo no usa `chronolab.anomaly.events`
------------------------------------------------------
`events.py` construye la tabla de eventos que consume la app, y para ordenarlos
necesita `severity`, que un detector sin umbral calibrado —Matrix Profile— no
emite. Este modulo trabaja sobre la mascara binaria directamente, que es lo
unico que los cuatro detectores del proyecto tienen en comun. Comparten en
cambio la **politica de fusion**: `merge_gap` significa aqui lo mismo que alli.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from chronolab.types import DetectorId

__all__ = [
    "METRIC_COLUMNS",
    "AffiliationReport",
    "CardinalityMode",
    "CurveReport",
    "OperationalReport",
    "PositionalBias",
    "RangeReport",
    "affiliation_precision_recall",
    "auc_pr",
    "common_scorable_mask",
    "detection_delay",
    "evaluate_detector",
    "false_alarm_rate",
    "point_precision_recall",
    "pr_curve",
    "range_auc_pr",
    "range_precision_recall",
    "runs_to_ranges",
    "vus_pr",
]

METRIC_COLUMNS: tuple[str, ...] = (
    "detector_id",
    "unique_id",
    "anomaly_type",
    "metric",
    "value",
    "n_obs",
    "n_events",
)
"""Columnas de la tabla larga que devuelve `evaluate_detector`.

Formato largo por el mismo motivo que `metrics.parquet` (docs/ARCHITECTURE.md
§7.4): las metricas son heterogeneas y se anaden con el tiempo, y una tabla
ancha obligaria a migrar el esquema en cada incorporacion. ``unique_id`` a
nulo significa *agregado sobre series*; ``anomaly_type = "all"`` significa
*sobre el tramo completo*, sin restringir a un tipo.
"""

PositionalBias = Literal["flat", "front", "back", "middle"]
"""Donde pesa mas acertar dentro de un rango.

``"flat"`` trata todos los instantes por igual. ``"front"`` premia detectar
pronto, que es lo que importa en operacion. ``"back"`` premia detectar el
final. ``"middle"`` premia el nucleo del evento y descuenta los bordes, donde
la etiqueta de verdad es mas discutible.
"""

CardinalityMode = Literal["one", "reciprocal"]
"""Como se penaliza que una deteccion se parta en trozos.

``"reciprocal"`` divide por el numero de rangos distintos con los que se
solapa: partir un evento real en cinco detecciones vale un quinto. ``"one"``
no penaliza, y entonces fragmentar sale gratis —que es la puerta por la que
vuelve a colarse el vicio del point-adjusted.
"""

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Informes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RangeReport:
    """Precision y recall por rangos, con la contribucion de cada rango.

    Attributes
    ----------
    precision, recall, f1
        Los tres escalares. `NaN` donde el denominador no existe: sin rangos
        reales el recall es indefinido, sin rangos predichos lo es la
        precision. Devolver `NaN` y no cero es deliberado: "no habia nada que
        detectar" y "no detecte nada de lo que habia" son resultados
        distintos y colapsarlos falsea cualquier promedio posterior.
    per_true_range
        Recall aportado por cada rango real, en orden temporal.
    per_pred_range
        Precision aportada por cada rango predicho, en orden temporal. Son
        estos vectores, y no los escalares, los que se agregan entre series.
    n_true_ranges, n_pred_ranges
        Tamanos de los dos conjuntos.
    """

    precision: float
    recall: float
    f1: float
    per_true_range: np.ndarray
    per_pred_range: np.ndarray
    n_true_ranges: int
    n_pred_ranges: int


@dataclass(frozen=True, slots=True)
class AffiliationReport:
    """Precision y recall de afiliacion, con la contribucion de cada zona.

    Attributes
    ----------
    precision, recall, f1
        Los tres escalares, en ``[0, 1]``. **0.5 es el valor esperado del
        azar**, no 0: son probabilidades de que una prediccion uniforme en la
        zona quedase mas lejos que la observada. Leerlos con la intuicion de
        una precision clasica —donde 0.5 suena aceptable— es el unico modo de
        equivocarse con esta metrica, y por eso `evaluate_detector` emite
        siempre las dos familias juntas.
    per_zone_precision
        Precision individual de cada zona **con al menos una prediccion**. Las
        zonas sin prediccion no tienen precision definida y no entran.
    per_zone_recall
        Recall individual de cada zona, todas. Una zona sin ninguna prediccion
        aporta ``0.0``.
    n_zones
        Zonas de afiliacion, una por evento real.
    """

    precision: float
    recall: float
    f1: float
    per_zone_precision: np.ndarray
    per_zone_recall: np.ndarray
    n_zones: int


@dataclass(frozen=True, slots=True)
class CurveReport:
    """Area bajo una curva precision-recall y la linea base que la contextualiza.

    Attributes
    ----------
    area
        Precision media (AUC-PR por integracion escalonada).
    baseline
        Prevalencia de la clase positiva, que es el AUC-PR esperado de un
        detector aleatorio. **Un AUC-PR no significa nada sin ella**: 0.30 es
        excelente con una prevalencia del 1 % y pesimo con una del 29 %. Se
        devuelve pegada al area para que no se pueda citar una sin la otra.
    n_positive, n_obs
        Positivos y observaciones del soporte evaluado.
    """

    area: float
    baseline: float
    n_positive: int
    n_obs: int


@dataclass(frozen=True, slots=True)
class OperationalReport:
    """Lo que decide si un detector se despliega: cuanto tarda y cuanto molesta.

    Attributes
    ----------
    n_true_events, n_detected_events
        Eventos reales y cuantos de ellos recibieron al menos una marca
        dentro de su propia extension.
    detection_rate
        `n_detected_events / n_true_events`. `NaN` sin eventos reales.
    delays
        Retardo en pasos de cada evento **detectado**: instantes desde el
        inicio real hasta la primera marca. Los eventos no detectados no
        aportan.
    mean_delay_steps, median_delay_steps
        Resumenes de `delays`. **Estan condicionados a haber detectado**, y
        eso los hace enganosos por si solos: un detector que solo captura las
        anomalias mas evidentes —las que se ven en el primer paso— exhibe un
        retardo medio excelente precisamente porque se ha perdido las
        dificiles. Solo significan algo leidos junto a `detection_rate`, y
        `evaluate_detector` los emite siempre juntos por ese motivo.
    n_false_alarm_events
        Rangos predichos sin ningun solape con un evento real.
    false_alarms_per_1000
        `n_false_alarm_events / n_obs * 1000`. A nivel de **evento** y no de
        punto: a un operador se le avisa una vez por incidente, no una vez por
        instante, asi que la tasa por puntos sobreestima la molestia real de un
        detector que marca tiradas largas.
    n_obs
        Observaciones del soporte, denominador de la tasa.
    """

    n_true_events: int
    n_detected_events: int
    detection_rate: float
    delays: np.ndarray
    mean_delay_steps: float
    median_delay_steps: float
    n_false_alarm_events: int
    false_alarms_per_1000: float
    n_obs: int


# --------------------------------------------------------------------------- #
# Primitivas de rango
# --------------------------------------------------------------------------- #


def runs_to_ranges(mask: np.ndarray, *, merge_gap: int = 0) -> list[tuple[int, int]]:
    """Tiradas maximas de `True`, fusionando las separadas por huecos cortos.

    Parameters
    ----------
    mask
        Vector booleano en orden temporal.
    merge_gap
        Puntos a `False` que se toleran dentro de una misma tirada. Significa
        lo mismo que en `chronolab.anomaly.events.aggregate_events`: una
        anomalia real casi nunca produce una tirada ininterrumpida, y sin
        tolerancia un solo punto que vuelve a la normalidad parte un evento en
        dos y hunde la precision por rangos.

    Returns
    -------
    list of tuple
        Pares ``(inicio, fin)`` de posiciones, ambos inclusive.

    Raises
    ------
    ValueError
        Si `merge_gap` es negativo.
    """
    if merge_gap < 0:
        raise ValueError(f"merge_gap debe ser >= 0: {merge_gap}")
    if mask.size == 0:
        return []

    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    ranges = [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]
    if merge_gap == 0 or len(ranges) < 2:
        return ranges

    merged: list[tuple[int, int]] = [ranges[0]]
    for start, end in ranges[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end - 1 <= merge_gap:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return merged


def _mask_from_ranges(ranges: Sequence[tuple[int, int]], size: int) -> np.ndarray:
    """Reconstruye la mascara booleana que cubren unos rangos.

    Parameters
    ----------
    ranges
        Pares ``(inicio, fin)`` inclusive.
    size
        Longitud del vector de salida.

    Returns
    -------
    numpy.ndarray
        Booleana. Tras fusionar con `merge_gap`, los huecos tolerados quedan
        cubiertos: un evento detectado abarca su extension completa, igual que
        en `anomaly.events`.
    """
    mask = np.zeros(size, dtype=bool)
    for start, end in ranges:
        mask[start : end + 1] = True
    return mask


def _bias_weights(length: int, bias: PositionalBias) -> np.ndarray:
    """Peso posicional de cada instante dentro de un rango.

    Parameters
    ----------
    length
        Longitud del rango en pasos.
    bias
        Perfil de sesgo posicional.

    Returns
    -------
    numpy.ndarray
        Pesos positivos de longitud `length`.

    Raises
    ------
    ValueError
        Si `bias` no es uno de los perfiles conocidos.
    """
    positions = np.arange(1, length + 1, dtype=float)
    if bias == "flat":
        return np.ones(length, dtype=float)
    if bias == "front":
        return length - positions + 1.0
    if bias == "back":
        return positions
    if bias == "middle":
        middle: np.ndarray = np.minimum(positions, length - positions + 1.0)
        return middle
    raise ValueError(f"bias desconocido: {bias!r}")


def _cardinality_factor(n_overlapping: int, mode: CardinalityMode) -> float:
    """Factor que penaliza solapar con varios rangos a la vez.

    Parameters
    ----------
    n_overlapping
        Rangos distintos del otro conjunto con los que hay solape.
    mode
        Politica de penalizacion.

    Returns
    -------
    float
        ``1.0`` con cero o un rango; segun `mode` con mas de uno.
    """
    if n_overlapping <= 1 or mode == "one":
        return 1.0
    return 1.0 / float(n_overlapping)


def _omega(span: tuple[int, int], other: np.ndarray, bias: PositionalBias) -> float:
    """Solape ponderado de un rango contra una mascara, normalizado a ``[0, 1]``.

    Es la funcion ``omega`` de Tatbul et al. (2018). Como los rangos del otro
    conjunto son disjuntos, la suma de solapes con cada uno equivale al solape
    con su union, que es lo que se calcula aqui.

    Parameters
    ----------
    span
        Rango ``(inicio, fin)`` inclusive.
    other
        Mascara booleana del otro conjunto.
    bias
        Perfil de sesgo posicional.

    Returns
    -------
    float
        Fraccion del peso del rango que queda cubierta.
    """
    start, end = span
    weights = _bias_weights(end - start + 1, bias)
    covered = other[start : end + 1]
    total = float(weights.sum())
    if total <= 0.0:  # pragma: no cover  length >= 1 garantiza peso positivo
        return 0.0
    return float(weights[covered].sum()) / total


def _count_overlapping(span: tuple[int, int], ranges: Sequence[tuple[int, int]]) -> int:
    """Cuantos rangos del otro conjunto solapan con uno dado.

    Parameters
    ----------
    span
        Rango ``(inicio, fin)`` inclusive.
    ranges
        Rangos del otro conjunto.

    Returns
    -------
    int
        Numero de solapes no vacios.
    """
    start, end = span
    return sum(1 for other_start, other_end in ranges if other_start <= end and start <= other_end)


def _f1(precision: float, recall: float) -> float:
    """Media armonica de precision y recall.

    Parameters
    ----------
    precision, recall
        Los dos escalares, posiblemente `NaN`.

    Returns
    -------
    float
        `NaN` si alguno lo es; ``0.0`` si ambos son cero.
    """
    if math.isnan(precision) or math.isnan(recall):
        return math.nan
    if precision + recall <= _EPS:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def range_precision_recall(
    predicted: np.ndarray,
    actual: np.ndarray,
    *,
    alpha: float = 0.0,
    bias: PositionalBias = "flat",
    cardinality: CardinalityMode = "reciprocal",
    merge_gap: int = 0,
) -> RangeReport:
    """Precision y recall por rangos (Tatbul et al., NeurIPS 2018).

    **Que mide.** Trata una anomalia como un intervalo y descompone el acierto
    en cuatro preguntas separables que el F1 puntual funde en un solo numero:

    - *existencia*: se toco el evento;
    - *solape*: que fraccion de el se cubrio;
    - *cardinalidad*: en cuantos trozos se partio la deteccion;
    - *posicion*: donde cayo el acierto dentro del evento.

    Recall se promedia sobre los rangos **reales**, precision sobre los
    **predichos**. La asimetria del parametro `alpha` no es un descuido del
    diseno original: el premio por existencia solo se aplica al recall, porque
    para la precision "he tocado algo" sin solape no merece premio —es
    exactamente un falso positivo.

    **Cuando engana.** Con ``alpha=1`` el recall degenera en "he tocado el
    evento", que es el point-adjusted a nivel de rango y hereda su inflacion.
    Con ``cardinality="one"`` fragmentar sale gratis y un detector intermitente
    puntua como uno solido. Ambos por defecto estan en la posicion conservadora
    (``alpha=0``, ``cardinality="reciprocal"``), y estan expuestos para poder
    **medir** cuanto sube un detector al relajarlos, no para relajarlos.

    Tambien engana en la direccion contraria: con eventos de un solo punto
    —el `spike` de `chronolab.anomaly.injection`— el solape solo puede valer 0
    o 1, asi que esta metrica se vuelve puntual y pierde toda su ventaja. Ahi
    la que informa es la de afiliacion, que si distingue "fallar por un paso"
    de "no detectar".

    **Por que la incluimos.** Es la unica de las cinco que separa fragmentacion
    de cobertura, y la fragmentacion es el modo de fallo caracteristico de los
    detectores puntuales sobre eventos largos.

    Parameters
    ----------
    predicted, actual
        Mascaras booleanas alineadas y del mismo tamano, en orden temporal.
    alpha
        Peso del premio por existencia en el recall, en ``[0, 1]``.
    bias
        Perfil de sesgo posicional.
    cardinality
        Politica de penalizacion de la fragmentacion.
    merge_gap
        Tolerancia de fusion aplicada **solo** a los rangos predichos: los
        reales vienen de la inyeccion y su extension es un dato, no una
        estimacion.

    Returns
    -------
    RangeReport
        Con la contribucion de cada rango, para poder agregar entre series sin
        promediar promedios.

    Raises
    ------
    ValueError
        Si las mascaras no tienen el mismo tamano o si `alpha` sale de
        ``[0, 1]``.
    """
    if predicted.shape != actual.shape:
        raise ValueError(
            f"las mascaras deben tener el mismo tamano: {predicted.shape} y {actual.shape}"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha fuera de [0, 1]: {alpha}")

    size = int(predicted.size)
    true_ranges = runs_to_ranges(actual, merge_gap=0)
    pred_ranges = runs_to_ranges(predicted, merge_gap=merge_gap)
    pred_mask = _mask_from_ranges(pred_ranges, size)
    true_mask = _mask_from_ranges(true_ranges, size)

    per_true = np.array(
        [
            alpha * (1.0 if _count_overlapping(span, pred_ranges) else 0.0)
            + (1.0 - alpha)
            * _cardinality_factor(_count_overlapping(span, pred_ranges), cardinality)
            * _omega(span, pred_mask, bias)
            for span in true_ranges
        ],
        dtype=float,
    )
    per_pred = np.array(
        [
            _cardinality_factor(_count_overlapping(span, true_ranges), cardinality)
            * _omega(span, true_mask, bias)
            for span in pred_ranges
        ],
        dtype=float,
    )

    recall = float(per_true.mean()) if per_true.size else math.nan
    precision = float(per_pred.mean()) if per_pred.size else math.nan
    return RangeReport(
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        per_true_range=per_true,
        per_pred_range=per_pred,
        n_true_ranges=len(true_ranges),
        n_pred_ranges=len(pred_ranges),
    )


# --------------------------------------------------------------------------- #
# Afiliacion
# --------------------------------------------------------------------------- #


def _affiliation_zones(true_ranges: Sequence[tuple[int, int]], size: int) -> list[tuple[int, int]]:
    """Particion del eje temporal en zonas, una por evento real.

    Cada instante se afilia al evento mas cercano; la frontera entre dos
    eventos consecutivos cae en el punto medio del hueco que los separa.

    Parameters
    ----------
    true_ranges
        Eventos reales, ordenados y disjuntos.
    size
        Longitud del eje temporal.

    Returns
    -------
    list of tuple
        Zonas ``(inicio, fin)`` inclusive, contiguas y cubriendo ``[0, size)``.
    """
    zones: list[tuple[int, int]] = []
    for index, (start, end) in enumerate(true_ranges):
        left = 0 if index == 0 else (true_ranges[index - 1][1] + start) // 2 + 1
        right = (
            size - 1 if index == len(true_ranges) - 1 else (end + true_ranges[index + 1][0]) // 2
        )
        zones.append((left, right))
    return zones


def affiliation_precision_recall(predicted: np.ndarray, actual: np.ndarray) -> AffiliationReport:
    """Precision y recall de afiliacion (Huet et al., KDD 2022).

    **Que mide.** Distancias, no solapes. El eje temporal se parte en una zona
    por evento real —cada instante se afilia al evento mas cercano— y dentro de
    cada zona se compara la distancia observada contra la que habria dado una
    prediccion **uniforme al azar en esa misma zona**:

    - precision de la zona: probabilidad de que una prediccion al azar cayese
      *mas lejos* del evento que las que hizo el detector;
    - recall de la zona: probabilidad de que una prediccion al azar quedase
      *mas lejos* de cada punto del evento que la mas cercana del detector.

    Esto resuelve el punto ciego de todas las metricas de solape: una deteccion
    que llega dos pasos tarde tiene solape cero y es indistinguible de no
    detectar nada, cuando operativamente son cosas opuestas. La afiliacion las
    separa porque mide *cuanto* se fallo.

    **Cuando engana.** El valor esperado del azar es **0.5, no 0**. Un 0.55 no
    es "aprobado raspado": es ruido. Ademas la normalizacion es local a la
    zona, asi que el mismo error absoluto puntua distinto segun lo aislado que
    este el evento —en una serie con un unico evento la zona es toda la serie y
    casi cualquier deteccion parece excelente—. Es interpretable *relativa* a
    la densidad de eventos, no en absoluto, y por eso `evaluate_detector` emite
    tambien `n_events`.

    **Por que la incluimos.** Es la unica de las cinco con una linea base
    explicita del azar, y la unica que puntua la cercania cuando el solape es
    exactamente cero.

    Parameters
    ----------
    predicted, actual
        Mascaras booleanas alineadas y del mismo tamano, en orden temporal.

    Returns
    -------
    AffiliationReport
        Con la contribucion de cada zona, para agregar entre series.

    Raises
    ------
    ValueError
        Si las mascaras no tienen el mismo tamano.
    """
    if predicted.shape != actual.shape:
        raise ValueError(
            f"las mascaras deben tener el mismo tamano: {predicted.shape} y {actual.shape}"
        )

    size = int(actual.size)
    true_ranges = runs_to_ranges(actual, merge_gap=0)
    if not true_ranges:
        empty = np.zeros(0, dtype=float)
        return AffiliationReport(math.nan, math.nan, math.nan, empty, empty, 0)

    zones = _affiliation_zones(true_ranges, size)
    precisions: list[float] = []
    recalls: list[float] = []

    for (event_start, event_end), (zone_start, zone_end) in zip(true_ranges, zones, strict=True):
        zone_size = zone_end - zone_start + 1
        predicted_positions = np.flatnonzero(predicted[zone_start : zone_end + 1]) + zone_start

        if predicted_positions.size:
            # dist(t, evento) = 0 dentro; fuera crece linealmente a cada lado.
            # El conjunto {t en zona : dist(t, evento) > d} son las dos colas,
            # asi que su tamano tiene forma cerrada y no hace falta recorrer.
            distances = np.maximum(
                np.maximum(event_start - predicted_positions, predicted_positions - event_end),
                0,
            ).astype(float)
            farther = np.maximum(0.0, (event_start - distances) - zone_start) + np.maximum(
                0.0, zone_end - (event_end + distances)
            )
            precisions.append(float(np.mean(farther / zone_size)))

        event_positions = np.arange(event_start, event_end + 1)
        if predicted_positions.size:
            nearest = np.min(
                np.abs(event_positions[:, None] - predicted_positions[None, :]), axis=1
            ).astype(float)
            # {t en zona : |t - x| <= d} es el intervalo [x-d, x+d] recortado
            # a la zona; lo de fuera es lo que cuenta para el recall.
            covered = (
                np.minimum(zone_end, event_positions + nearest)
                - np.maximum(zone_start, event_positions - nearest)
                + 1.0
            )
            recalls.append(float(np.mean(1.0 - covered / zone_size)))
        else:
            # Sin ninguna prediccion en la zona la distancia es infinita: el
            # evento no se detecto y su recall es cero, no indefinido.
            recalls.append(0.0)

    per_zone_precision = np.asarray(precisions, dtype=float)
    per_zone_recall = np.asarray(recalls, dtype=float)
    precision = float(per_zone_precision.mean()) if per_zone_precision.size else math.nan
    recall = float(per_zone_recall.mean()) if per_zone_recall.size else math.nan
    return AffiliationReport(
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        per_zone_precision=per_zone_precision,
        per_zone_recall=per_zone_recall,
        n_zones=len(zones),
    )


# --------------------------------------------------------------------------- #
# Curvas
# --------------------------------------------------------------------------- #


def _tie_boundaries(ordered_scores: np.ndarray) -> np.ndarray:
    """Ultimo indice de cada bloque de scores empatados, en orden decreciente.

    Se compara con ``!=`` y no con `numpy.diff`: los puntos sin score entran
    como ``-inf``, y ``-inf - (-inf)`` es `NaN`, que `numpy.flatnonzero`
    considera **no nulo**. Con `diff`, dos instantes no puntuables empatados se
    partirian en dos umbrales distintos y la curva pasaria por un punto que
    ningun umbral real alcanza.

    Parameters
    ----------
    ordered_scores
        Scores ya ordenados de mayor a menor.

    Returns
    -------
    numpy.ndarray
        Indices `int`, el ultimo de cada bloque de empates.
    """
    if ordered_scores.size == 0:  # pragma: no cover  los llamantes ya lo descartan
        return np.zeros(0, dtype=int)
    changes = (
        np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1])
        if ordered_scores.size > 1
        else np.zeros(0, dtype=int)
    )
    return np.concatenate([changes, [ordered_scores.size - 1]]).astype(int)


def pr_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Curva precision-recall puntual, con los empates agrupados.

    Agrupar los empates importa: si varios instantes comparten score exacto
    —cosa habitual en un score conformal saturado, donde toda la cola vale
    ``log10(n+1)``— separarlos en umbrales distintos dibujaria un tramo de
    curva que ningun umbral real puede alcanzar.

    Parameters
    ----------
    scores
        Grado de anomalia, mayor es mas anomalo. Los `NaN` cuentan como
        ``-inf``: un punto sin score nunca se marca.
    labels
        Verdad booleana, alineada con `scores`.

    Returns
    -------
    tuple
        ``(precision, recall, thresholds)``, ordenadas por umbral decreciente.
        `precision` y `recall` llevan un punto inicial ``(1.0, 0.0)`` que
        corresponde a no marcar nada.

    Raises
    ------
    ValueError
        Si los tamanos no coinciden.
    """
    if scores.shape != labels.shape:
        raise ValueError(f"tamanos distintos: {scores.shape} y {labels.shape}")

    values = np.nan_to_num(scores.astype(float), nan=-np.inf)
    truth = labels.astype(bool)
    n_positive = int(truth.sum())
    if n_positive == 0 or values.size == 0:
        return np.ones(1), np.zeros(1), np.array([np.inf])

    order = np.argsort(-values, kind="stable")
    ordered_scores = values[order]
    ordered_truth = truth[order]
    cut = _tie_boundaries(ordered_scores)

    true_positive = np.cumsum(ordered_truth)[cut]
    predicted = (cut + 1).astype(float)
    precision = true_positive / predicted
    recall = true_positive / float(n_positive)

    return (
        np.concatenate([[1.0], precision]),
        np.concatenate([[0.0], recall]),
        np.concatenate([[np.inf], ordered_scores[cut]]),
    )


def auc_pr(scores: np.ndarray, labels: np.ndarray) -> CurveReport:
    """AUC-PR puntual por integracion escalonada (precision media).

    **Que mide.** La calidad del *ranking* de instantes, sin fijar umbral y sin
    ninguna nocion de rango. Se integra a escalones —``sum (R_k - R_{k-1}) *
    P_k``— y no por trapecios: la interpolacion lineal en el espacio PR es
    incorrecta (Davis y Goadrich, 2006) y sesga el area al alza.

    **Cuando engana.** Siempre que la unidad que importa sea el evento y no el
    instante, que es el caso normal en series temporales. Un detector que
    acierta 200 puntos de una sola anomalia larga y se pierde otras diez cortas
    saca mejor AUC-PR que uno que detecta las once tarde: la metrica cuenta
    puntos, y las anomalias largas aportan mas puntos. Ademas su suelo es la
    prevalencia, asi que un valor absoluto no se puede leer sin `baseline`.

    **Por que la incluimos.** Como referencia minima y como control: si el
    orden de los detectores fuese el mismo bajo AUC-PR puntual que bajo las
    metricas por rango, todo el aparato de este modulo seria innecesario.
    Publicarla al lado es lo que permite comprobar esa afirmacion en lugar de
    darla por buena.

    Parameters
    ----------
    scores
        Grado de anomalia, mayor es mas anomalo.
    labels
        Verdad booleana, alineada con `scores`.

    Returns
    -------
    CurveReport
        Con la prevalencia pegada al area.
    """
    precision, recall, _ = pr_curve(scores, labels)
    truth = labels.astype(bool)
    n_positive = int(truth.sum())
    n_obs = int(truth.size)
    baseline = n_positive / n_obs if n_obs else math.nan
    if n_positive == 0:
        return CurveReport(math.nan, baseline, n_positive, n_obs)

    area = float(np.sum(np.diff(recall) * precision[1:]))
    return CurveReport(area, baseline, n_positive, n_obs)


def _extend_labels(labels: np.ndarray, buffer: int) -> np.ndarray:
    """Etiquetas con un margen de tolerancia decreciente alrededor de cada evento.

    Reproduce la extension de Paparrizos et al. (2022): a cada lado de un
    evento se anaden ``buffer // 2`` pasos con peso ``sqrt(1 - d / buffer)``,
    recortado a ``1``. Un acierto justo fuera del evento deja de valer cero sin
    llegar a valer lo mismo que uno dentro.

    Parameters
    ----------
    labels
        Verdad booleana.
    buffer
        Longitud del margen. Cero devuelve las etiquetas sin tocar.

    Returns
    -------
    numpy.ndarray
        Pesos en ``[0, 1]``, `float64`.
    """
    extended = labels.astype(float).copy()
    if buffer <= 0:
        return extended

    size = extended.size
    half = buffer // 2
    for start, end in runs_to_ranges(labels, merge_gap=0):
        right = np.arange(end + 1, min(end + 1 + half, size))
        if right.size:
            extended[right] += np.sqrt(1.0 - (right - end) / buffer)
        left = np.arange(max(start - half, 0), start)
        if left.size:
            extended[left] += np.sqrt(1.0 - (start - left) / buffer)
    capped: np.ndarray = np.minimum(extended, 1.0)
    return capped


def range_auc_pr(scores: np.ndarray, labels: np.ndarray, *, buffer: int) -> float:
    """AUC-PR por rangos con una tolerancia de desalineamiento fija.

    Es el corte de VUS-PR a un `buffer` dado. Se calcula sobre todos los
    umbrales distintos a la vez —el conteo acumulado sobre el orden de los
    scores da la curva entera de una pasada— en lugar de recorrer una rejilla
    de umbrales, que ademas de mas lento seria aproximado.

    Parameters
    ----------
    scores
        Grado de anomalia, mayor es mas anomalo.
    labels
        Verdad booleana.
    buffer
        Tolerancia de desalineamiento en pasos.

    Returns
    -------
    float
        Area bajo la curva, integrada por trapecios sobre el recall. `NaN` si
        no hay positivos.

    Notes
    -----
    Se reproducen las convenciones de la implementacion de referencia, y hay
    dos que conviene conocer porque no son neutras: el recall se normaliza por
    ``(buffer + 1) * P`` en lugar de por ``P``, y se multiplica por la
    fraccion de eventos reales tocados. La segunda es la que impide que marcar
    masivamente un unico evento largo sature el recall. Se mantienen tal cual
    para que los numeros sean comparables con los publicados; reformularlas
    daria una cantidad parecida que ya no seria VUS.
    """
    truth = labels.astype(bool)
    n_positive = int(truth.sum())
    if n_positive == 0 or truth.size == 0:
        return math.nan

    values = np.nan_to_num(scores.astype(float), nan=-np.inf)
    extended = _extend_labels(truth, buffer)
    segments = runs_to_ranges(truth, merge_gap=0)

    order = np.argsort(-values, kind="stable")
    ordered_scores = values[order]
    cut = _tie_boundaries(ordered_scores)

    true_positive = np.cumsum(extended[order])[cut]
    n_predicted = (cut + 1).astype(float)
    thresholds = ordered_scores[cut]

    # Un evento queda tocado en cuanto el umbral baja de su score maximo, asi
    # que la fraccion tocada sale de ordenar esos maximos una sola vez.
    segment_max = np.sort(
        np.array([values[start : end + 1].max() for start, end in segments], dtype=float)
    )
    touched = segment_max.size - np.searchsorted(segment_max, thresholds, side="left")
    existence = touched / float(len(segments))

    recall = np.minimum(true_positive / ((buffer + 1.0) * n_positive), 1.0) * existence
    precision = np.where(n_predicted > 0, true_positive / n_predicted, 1.0)

    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum(np.diff(recall) * (precision[1:] + precision[:-1]) / 2.0))


def vus_pr(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    max_buffer: int,
    n_buffers: int | None = None,
) -> float:
    """VUS-PR: volumen bajo la superficie precision-recall (Paparrizos et al., 2022).

    **Que mide.** El area de la curva PR integrada ademas sobre la tolerancia
    de desalineamiento temporal. Elimina las dos elecciones arbitrarias que
    condicionan casi todos los resultados publicados: **el umbral** —igual que
    cualquier AUC— y **cuanto desalineamiento se perdona**, que es la que
    normalmente ni se declara. Un detector que avisa dos pasos tarde puntua
    cero bajo solape estricto y bien bajo tolerancia amplia; VUS-PR no obliga a
    elegir cual de las dos lecturas se publica.

    **Cuando engana.** Es un promedio sobre tolerancias, asi que oculta *donde*
    esta la sensibilidad: dos detectores con el mismo VUS-PR pueden ser uno
    puntualmente exacto y otro sistematicamente tardio. El perfil
    `range_auc_pr` a cada `buffer` es lo que distingue esos dos casos, y es la
    razon de que esa funcion sea publica y no un detalle interno. Ademas hereda
    de AUC-PR la dependencia de la prevalencia.

    **Por que la incluimos.** Es la metrica de referencia del benchmark que
    desacredito el point-adjusted, y la unica de las cinco que no exige fijar
    ningun umbral ni ninguna tolerancia.

    Parameters
    ----------
    scores
        Grado de anomalia, mayor es mas anomalo.
    labels
        Verdad booleana.
    max_buffer
        Tolerancia maxima en pasos. Una eleccion razonable es la duracion
        tipica de un evento: mas alla, el margen abarca eventos vecinos y la
        metrica deja de distinguir cual se detecto.
    n_buffers
        Numero de cortes en ``[0, max_buffer]``. ``None`` usa todos los
        enteros.

    Returns
    -------
    float
        Media de `range_auc_pr` sobre la rejilla de tolerancias. `NaN` sin
        positivos.

    Raises
    ------
    ValueError
        Si `max_buffer` es negativo o `n_buffers` menor que uno.
    """
    if max_buffer < 0:
        raise ValueError(f"max_buffer debe ser >= 0: {max_buffer}")
    if n_buffers is not None and n_buffers < 1:
        raise ValueError(f"n_buffers debe ser >= 1: {n_buffers}")

    if n_buffers is None:
        buffers = np.arange(0, max_buffer + 1)
    else:
        buffers = np.unique(np.linspace(0, max_buffer, n_buffers).round().astype(int))

    areas = [range_auc_pr(scores, labels, buffer=int(buffer)) for buffer in buffers]
    finite = [area for area in areas if not math.isnan(area)]
    return float(np.mean(finite)) if finite else math.nan


# --------------------------------------------------------------------------- #
# Metricas puntuales y operativas
# --------------------------------------------------------------------------- #


def point_precision_recall(predicted: np.ndarray, actual: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall y F1 puntuales clasicos, a umbral fijo.

    **Que mide.** Lo que mediria cualquier clasificador binario, instante a
    instante. **Cuando engana.** Ignora por completo la estructura temporal:
    un acierto a un paso de la anomalia vale exactamente lo mismo que un
    acierto a mil. **Por que la incluimos.** Es el punto de partida sobre el
    que se define el point-adjusted que este modulo rechaza; publicar el F1
    puntual honesto al lado es lo que permite ensenar la diferencia con el
    inflado en lugar de afirmarla.

    Parameters
    ----------
    predicted, actual
        Mascaras booleanas alineadas.

    Returns
    -------
    tuple
        ``(precision, recall, f1)``. `NaN` donde el denominador es cero.
    """
    prediction = predicted.astype(bool)
    truth = actual.astype(bool)
    true_positive = float(np.sum(prediction & truth))
    n_predicted = float(prediction.sum())
    n_true = float(truth.sum())

    precision = true_positive / n_predicted if n_predicted else math.nan
    recall = true_positive / n_true if n_true else math.nan
    return precision, recall, _f1(precision, recall)


def detection_delay(
    predicted: np.ndarray, actual: np.ndarray, *, merge_gap: int = 0
) -> OperationalReport:
    """Retardo de deteccion y tasa de falsas alarmas por cada 1000 observaciones.

    **Que mide.** Las dos cantidades por las que pregunta quien opera el
    sistema: cuantos pasos pasan desde que empieza una anomalia hasta que se
    avisa, y cuantas veces se avisa sin que hubiera nada.

    **Cuando engana.** El retardo medio esta condicionado a haber detectado, y
    esa condicion lo invierte: un detector conservador que solo captura las
    anomalias mas evidentes exhibe el mejor retardo del cuadro precisamente
    porque las dificiles —las que habrian tardado— no entran en la media. Leer
    el retardo sin `detection_rate` al lado es el error clasico, y por eso
    ambos viajan en el mismo informe. La tasa de falsas alarmas, a su vez,
    depende de `merge_gap`: con tolerancia alta, veinte marcas dispersas se
    cuentan como una sola alarma.

    **Por que la incluimos.** Ninguna de las metricas de calidad de ranking
    dice nada sobre latencia, y la latencia es lo que decide si un detector
    sirve para operar o solo para un informe.

    Parameters
    ----------
    predicted, actual
        Mascaras booleanas alineadas.
    merge_gap
        Tolerancia de fusion de los rangos predichos.

    Returns
    -------
    OperationalReport
        Con los retardos individuales, para agregar entre series.

    Raises
    ------
    ValueError
        Si las mascaras no tienen el mismo tamano.
    """
    if predicted.shape != actual.shape:
        raise ValueError(
            f"las mascaras deben tener el mismo tamano: {predicted.shape} y {actual.shape}"
        )

    prediction = predicted.astype(bool)
    true_ranges = runs_to_ranges(actual, merge_gap=0)
    pred_ranges = runs_to_ranges(prediction, merge_gap=merge_gap)
    n_obs = int(prediction.size)

    delays: list[int] = []
    for start, end in true_ranges:
        inside = np.flatnonzero(prediction[start : end + 1])
        if inside.size:
            delays.append(int(inside[0]))

    false_alarms = sum(1 for span in pred_ranges if not _count_overlapping(span, true_ranges))
    delay_array = np.asarray(delays, dtype=float)
    return OperationalReport(
        n_true_events=len(true_ranges),
        n_detected_events=len(delays),
        detection_rate=len(delays) / len(true_ranges) if true_ranges else math.nan,
        delays=delay_array,
        mean_delay_steps=float(delay_array.mean()) if delay_array.size else math.nan,
        median_delay_steps=float(np.median(delay_array)) if delay_array.size else math.nan,
        n_false_alarm_events=false_alarms,
        false_alarms_per_1000=(false_alarms / n_obs * 1000.0) if n_obs else math.nan,
        n_obs=n_obs,
    )


def false_alarm_rate(
    predicted: np.ndarray, actual: np.ndarray, *, merge_gap: int = 0, per: int = 1000
) -> float:
    """Alarmas sin evento real detras, por cada `per` observaciones.

    Atajo sobre `detection_delay` para cuando solo interesa esta cifra. Cuenta
    **eventos** y no puntos: a un operador se le avisa una vez por incidente.

    Parameters
    ----------
    predicted, actual
        Mascaras booleanas alineadas.
    merge_gap
        Tolerancia de fusion de los rangos predichos.
    per
        Base de la tasa.

    Returns
    -------
    float
        Falsas alarmas por cada `per` observaciones.

    Raises
    ------
    ValueError
        Si `per` no es positivo.
    """
    if per < 1:
        raise ValueError(f"per debe ser >= 1: {per}")
    report = detection_delay(predicted, actual, merge_gap=merge_gap)
    if math.isnan(report.false_alarms_per_1000):
        return math.nan
    return report.n_false_alarm_events / report.n_obs * per


# --------------------------------------------------------------------------- #
# Orquestacion
# --------------------------------------------------------------------------- #


def common_scorable_mask(scores_by_detector: Mapping[DetectorId, pd.DataFrame]) -> pd.DataFrame:
    """Interseccion de las mascaras `scorable` de todos los detectores comparados.

    Es el requisito de docs/ARCHITECTURE.md §5.3, y es un detalle pequeno con
    efecto grande: un detector con ventana de 512 puntos puntua menos instantes
    que uno de ventana 1, y evaluar a cada cual sobre su propio soporte premia
    al de ventana larga por haberse saltado el arranque de la serie, que suele
    ser la parte peor condicionada.

    Parameters
    ----------
    scores_by_detector
        Salidas de `FittedDetector.score`, todas sobre la misma rejilla
        ``(unique_id, ds)``.

    Returns
    -------
    pandas.DataFrame
        ``unique_id``, ``ds``, ``scorable``, ordenada por esa clave.

    Raises
    ------
    ValueError
        Si el mapa esta vacio o si dos detectores no comparten rejilla: sin
        la misma rejilla la interseccion seria una union implicita y silenciosa.
    """
    if not scores_by_detector:
        raise ValueError("hacen falta al menos los scores de un detector")

    frames: list[tuple[DetectorId, pd.DataFrame]] = []
    for detector_id, frame in scores_by_detector.items():
        missing = {"unique_id", "ds", "scorable"} - set(frame.columns)
        if missing:
            raise ValueError(f"'{detector_id}' no trae las columnas {sorted(missing)}")
        ordered = frame.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        # Se normalizan clave y resolucion antes de comparar. Lo que aqui se
        # comprueba es que la **rejilla** sea la misma, y dos detectores que
        # emiten los mismos instantes en distinta resolucion de `datetime64`
        # comparten rejilla aunque `DataFrame.equals`, que mira el dtype, diga
        # que no. Una rejilla de verdad distinta sigue saltando.
        ordered["unique_id"] = ordered["unique_id"].astype(str)
        ordered["ds"] = ordered["ds"].astype("datetime64[ns]")
        frames.append((detector_id, ordered))

    _, first = frames[0]
    reference = first[["unique_id", "ds"]]
    mask = first["scorable"].to_numpy(dtype=bool)
    for detector_id, ordered in frames[1:]:
        if not reference.equals(ordered[["unique_id", "ds"]]):
            raise ValueError(
                f"'{detector_id}' no comparte rejilla (unique_id, ds) con los demas detectores; "
                "sin rejilla comun la interseccion de mascaras no esta definida"
            )
        mask = mask & ordered["scorable"].to_numpy(dtype=bool)

    result = reference.copy()
    result["scorable"] = mask
    return result


def evaluate_detector(
    scores: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    detector_id: DetectorId,
    support: pd.DataFrame | None = None,
    alpha: float = 0.05,
    merge_gap: int = 2,
    bias: PositionalBias = "flat",
    cardinality: CardinalityMode = "reciprocal",
    existence_alpha: float = 0.0,
    vus_max_buffer: int = 24,
) -> pd.DataFrame:
    """Evalua un detector contra la verdad inyectada y devuelve la tabla larga.

    Emite dos granos: por ``(serie, tipo)`` y agregados. El agregado sobre
    series **no** promedia los numeros por serie salvo donde no queda mas
    remedio: las metricas que son una media sobre rangos o zonas se recalculan
    juntando los rangos de todas las series, que es lo mismo que exige
    `evaluation.aggregate` al prohibir promediar promedios. Las dos que si se
    promedian —AUC-PR y VUS-PR— lo hacen porque agrupar rankings de series
    distintas violaria el contrato de `score`, que es ordinal solo dentro de un
    par (detector, serie).

    Por que la tabla por tipo **no** trae precision
    -----------------------------------------------
    Una falsa alarma no pertenece a ningun tipo de anomalia: cae donde no habia
    nada. "Precision del detector para escalones" no es una cantidad definida,
    y las tablas publicadas que la traen suelen estar contando como falso
    positivo la deteccion **correcta** de otro tipo. Por eso, restringido a un
    tipo, aqui solo se emite el lado del recall —que si es una media sobre los
    eventos de ese tipo— mas el AUC-PR, que al ser puntual e invariante a
    permutaciones si admite restringir el soporte quitando los instantes de los
    otros tipos. Precision, F1, VUS-PR y falsas alarmas se emiten unicamente en
    ``anomaly_type = "all"``.

    Parameters
    ----------
    scores
        Salida de `FittedDetector.score`: ``unique_id``, ``ds``, ``score`` y
        ``scorable``.
    truth
        Tabla de verdad de `chronolab.anomaly.injection`: ``unique_id``,
        ``ds``, ``is_anomaly``, ``anomaly_type``. Es dispersa; los instantes
        ausentes son normales.
    detector_id
        Identificador que se escribe en la tabla.
    support
        Mascara comun de `common_scorable_mask`. ``None`` usa la del propio
        detector, que solo es legitimo si no se va a comparar con otro.
    alpha
        Nivel con el que se binariza el score: se marca donde
        ``score >= -log10(alpha)``, la misma regla que
        `chronolab.anomaly.events.aggregate_events`.
    merge_gap
        Tolerancia de fusion de los rangos predichos.
    bias, cardinality, existence_alpha
        Parametros de `range_precision_recall`.
    vus_max_buffer
        Tolerancia maxima de `vus_pr`, en pasos.

    Returns
    -------
    pandas.DataFrame
        Columnas `METRIC_COLUMNS`.

    Raises
    ------
    ValueError
        Si faltan columnas obligatorias o si `alpha` sale de ``(0, 1)``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha fuera de (0, 1): {alpha}")
    for name, frame, required in (
        ("scores", scores, {"unique_id", "ds", "score", "scorable"}),
        ("truth", truth, {"unique_id", "ds", "is_anomaly", "anomaly_type"}),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"a '{name}' le faltan las columnas {sorted(missing)}")

    threshold = -math.log10(alpha)
    aligned = _align(scores, truth, support)
    types = sorted({str(value) for value in aligned["anomaly_type"].dropna().unique()})

    rows: list[dict[str, object]] = []
    pooled: dict[str, list[np.ndarray]] = {}
    scalars: dict[str, list[float]] = {}
    counters: dict[str, float] = {}

    for uid, group in aligned.groupby("unique_id", sort=True):
        series = group.sort_values("ds")
        usable = series["scorable"].to_numpy(dtype=bool)
        n_gaps = max(0, len(runs_to_ranges(usable)) - 1)
        values = series.loc[usable, "score"].to_numpy(dtype=float)
        flagged = np.nan_to_num(values, nan=-np.inf) >= threshold
        point_type = series.loc[usable, "anomaly_type"].to_numpy(dtype=object)
        labels = series.loc[usable, "is_anomaly"].to_numpy(dtype=bool)

        rows.extend(
            _series_rows(
                detector_id=detector_id,
                uid=str(uid),
                anomaly_type="all",
                flagged=flagged,
                labels=labels,
                values=values,
                n_gaps=n_gaps,
                merge_gap=merge_gap,
                bias=bias,
                cardinality=cardinality,
                existence_alpha=existence_alpha,
                vus_max_buffer=vus_max_buffer,
                pooled=pooled,
                scalars=scalars,
                counters=counters,
            )
        )

        for anomaly_type in types:
            # Restringir a un tipo exige quitar del soporte los instantes de
            # los demas: dejarlos contaria como fallo la deteccion correcta de
            # otra cosa. Quitarlos es licito aqui porque lo que se emite en
            # este grano no depende de la adyacencia entre instantes lejanos.
            own = point_type == anomaly_type
            other = labels & ~own
            keep = ~other
            rows.extend(
                _series_rows(
                    detector_id=detector_id,
                    uid=str(uid),
                    anomaly_type=anomaly_type,
                    flagged=flagged[keep],
                    labels=own[keep],
                    values=values[keep],
                    n_gaps=n_gaps,
                    merge_gap=merge_gap,
                    bias=bias,
                    cardinality=cardinality,
                    existence_alpha=existence_alpha,
                    vus_max_buffer=vus_max_buffer,
                    pooled=pooled,
                    scalars=scalars,
                    counters=counters,
                    recall_only=True,
                )
            )

    rows.extend(_pooled_rows(detector_id, pooled, scalars, counters, types))
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(METRIC_COLUMNS))
    return frame[list(METRIC_COLUMNS)].reset_index(drop=True)


def _align(scores: pd.DataFrame, truth: pd.DataFrame, support: pd.DataFrame | None) -> pd.DataFrame:
    """Une scores, verdad y mascara comun sobre la rejilla del detector.

    Parameters
    ----------
    scores
        Salida de `FittedDetector.score`.
    truth
        Tabla de verdad, dispersa.
    support
        Mascara comun, o ``None`` para usar la del detector.

    Returns
    -------
    pandas.DataFrame
        ``unique_id``, ``ds``, ``score``, ``scorable``, ``is_anomaly`` y
        ``anomaly_type``, con la verdad completada a `False` donde falta.
    """
    frame = scores[["unique_id", "ds", "score", "scorable"]].copy()
    frame["unique_id"] = frame["unique_id"].astype(str)
    frame["scorable"] = frame["scorable"].astype(bool)

    if support is not None:
        common = support[["unique_id", "ds", "scorable"]].copy()
        common["unique_id"] = common["unique_id"].astype(str)
        common = common.rename(columns={"scorable": "_common"})
        frame = frame.merge(common, on=["unique_id", "ds"], how="left")
        frame["scorable"] = frame["scorable"] & frame["_common"].fillna(False).astype(bool)
        frame = frame.drop(columns="_common")

    labels = truth.loc[truth["is_anomaly"].astype(bool), ["unique_id", "ds", "anomaly_type"]].copy()
    labels["unique_id"] = labels["unique_id"].astype(str)
    labels = labels.drop_duplicates(subset=["unique_id", "ds"], keep="last")
    labels["is_anomaly"] = True

    merged = frame.merge(labels, on=["unique_id", "ds"], how="left")
    merged["is_anomaly"] = merged["is_anomaly"].fillna(False).astype(bool)
    return merged.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def _series_rows(
    *,
    detector_id: DetectorId,
    uid: str,
    anomaly_type: str,
    flagged: np.ndarray,
    labels: np.ndarray,
    values: np.ndarray,
    n_gaps: int,
    merge_gap: int,
    bias: PositionalBias,
    cardinality: CardinalityMode,
    existence_alpha: float,
    vus_max_buffer: int,
    pooled: dict[str, list[np.ndarray]],
    scalars: dict[str, list[float]],
    counters: dict[str, float],
    recall_only: bool = False,
) -> list[dict[str, object]]:
    """Calcula las metricas de una serie y acumula lo necesario para agregarlas.

    Parameters
    ----------
    detector_id, uid, anomaly_type
        Claves de la fila.
    flagged, labels, values
        Mascara marcada, verdad y scores, ya restringidos al soporte.
    n_gaps
        Huecos interiores del soporte. Se emite como metrica porque quitar
        instantes no puntuables junta tramos que en el tiempo no eran
        contiguos, y quien lea una precision por rangos tiene derecho a saber
        cuantas veces paso.
    merge_gap, bias, cardinality, existence_alpha, vus_max_buffer
        Parametros de las metricas.
    pooled, scalars, counters
        Acumuladores de la agregacion, mutados en el sitio.
    recall_only
        `True` en el grano por tipo: omite precision, F1, VUS-PR y falsas
        alarmas, que no estan definidas restringidas a un tipo.

    Returns
    -------
    list of dict
        Filas listas para `METRIC_COLUMNS`.
    """
    ranged = range_precision_recall(
        flagged,
        labels,
        alpha=existence_alpha,
        bias=bias,
        cardinality=cardinality,
        merge_gap=merge_gap,
    )
    affiliation = affiliation_precision_recall(flagged, labels)
    operational = detection_delay(flagged, labels, merge_gap=merge_gap)
    curve = auc_pr(values, labels)

    key = anomaly_type
    _accumulate(pooled, f"{key}|range_recall", ranged.per_true_range)
    _accumulate(pooled, f"{key}|affiliation_recall", affiliation.per_zone_recall)
    _accumulate(pooled, f"{key}|delay", operational.delays)
    _push(scalars, f"{key}|auc_pr", curve.area)
    counters[f"{key}|n_true_events"] = (
        counters.get(f"{key}|n_true_events", 0.0) + operational.n_true_events
    )
    counters[f"{key}|n_detected_events"] = (
        counters.get(f"{key}|n_detected_events", 0.0) + operational.n_detected_events
    )
    counters[f"{key}|n_obs"] = counters.get(f"{key}|n_obs", 0.0) + operational.n_obs
    counters[f"{key}|n_positive"] = counters.get(f"{key}|n_positive", 0.0) + curve.n_positive

    metrics: dict[str, float] = {
        "range_recall": ranged.recall,
        "affiliation_recall": affiliation.recall,
        "detection_rate": operational.detection_rate,
        "detection_delay_mean": operational.mean_delay_steps,
        "detection_delay_median": operational.median_delay_steps,
        "auc_pr": curve.area,
        "auc_pr_baseline": curve.baseline,
    }
    if not recall_only:
        point_precision, point_recall, point_f1 = point_precision_recall(flagged, labels)
        metrics.update(
            {
                "range_precision": ranged.precision,
                "range_f1": ranged.f1,
                "affiliation_precision": affiliation.precision,
                "affiliation_f1": affiliation.f1,
                "point_precision": point_precision,
                "point_recall": point_recall,
                "point_f1": point_f1,
                "vus_pr": vus_pr(values, labels, max_buffer=vus_max_buffer),
                "false_alarms_per_1000": operational.false_alarms_per_1000,
                "n_false_alarm_events": float(operational.n_false_alarm_events),
                "n_support_gaps": float(n_gaps),
            }
        )
        _accumulate(pooled, f"{key}|range_precision", ranged.per_pred_range)
        _accumulate(pooled, f"{key}|affiliation_precision", affiliation.per_zone_precision)
        _push(scalars, f"{key}|vus_pr", metrics["vus_pr"])
        counters[f"{key}|n_false_alarm_events"] = (
            counters.get(f"{key}|n_false_alarm_events", 0.0) + operational.n_false_alarm_events
        )
        counters[f"{key}|n_support_gaps"] = counters.get(f"{key}|n_support_gaps", 0.0) + n_gaps

    return [
        {
            "detector_id": str(detector_id),
            "unique_id": uid,
            "anomaly_type": anomaly_type,
            "metric": name,
            "value": float(value),
            "n_obs": int(operational.n_obs),
            "n_events": int(operational.n_true_events),
        }
        for name, value in metrics.items()
    ]


def _accumulate(pooled: dict[str, list[np.ndarray]], key: str, values: np.ndarray) -> None:
    """Guarda las contribuciones individuales de una serie para agregarlas luego.

    Parameters
    ----------
    pooled
        Acumulador, mutado en el sitio.
    key
        Clave ``tipo|metrica``.
    values
        Contribuciones por rango, zona o evento.
    """
    pooled.setdefault(key, []).append(values)


def _push(scalars: dict[str, list[float]], key: str, value: float) -> None:
    """Guarda un escalar por serie de los que si se promedian.

    Parameters
    ----------
    scalars
        Acumulador, mutado en el sitio.
    key
        Clave ``tipo|metrica``.
    value
        Valor de la serie; los `NaN` no entran en la media.
    """
    if not math.isnan(value):
        scalars.setdefault(key, []).append(value)


def _pooled_rows(
    detector_id: DetectorId,
    pooled: dict[str, list[np.ndarray]],
    scalars: dict[str, list[float]],
    counters: dict[str, float],
    types: Sequence[str],
) -> list[dict[str, object]]:
    """Filas agregadas sobre series, con ``unique_id`` a nulo.

    Parameters
    ----------
    detector_id
        Detector evaluado.
    pooled
        Contribuciones por rango, zona o evento de todas las series.
    scalars
        Valores por serie de las metricas de curva.
    counters
        Conteos acumulados.
    types
        Tipos de anomalia presentes.

    Returns
    -------
    list of dict
        Una fila por ``(tipo, metrica)``.
    """
    rows: list[dict[str, object]] = []
    for key in ("all", *types):
        n_obs = int(counters.get(f"{key}|n_obs", 0.0))
        n_events = int(counters.get(f"{key}|n_true_events", 0.0))
        detected = counters.get(f"{key}|n_detected_events", 0.0)
        values: dict[str, float] = {}

        for name in (
            "range_recall",
            "range_precision",
            "affiliation_recall",
            "affiliation_precision",
        ):
            parts = pooled.get(f"{key}|{name}")
            if parts:
                joined = np.concatenate(parts)
                values[name] = float(joined.mean()) if joined.size else math.nan

        delays = pooled.get(f"{key}|delay")
        if delays:
            joined_delays = np.concatenate(delays)
            if joined_delays.size:
                values["detection_delay_mean"] = float(joined_delays.mean())
                values["detection_delay_median"] = float(np.median(joined_delays))

        if n_events:
            values["detection_rate"] = detected / n_events
        if "range_precision" in values and "range_recall" in values:
            values["range_f1"] = _f1(values["range_precision"], values["range_recall"])
        if "affiliation_precision" in values and "affiliation_recall" in values:
            values["affiliation_f1"] = _f1(
                values["affiliation_precision"], values["affiliation_recall"]
            )

        for name in ("auc_pr", "vus_pr"):
            series_values = scalars.get(f"{key}|{name}")
            if series_values:
                values[name] = float(np.mean(series_values))

        if n_obs:
            values["auc_pr_baseline"] = counters.get(f"{key}|n_positive", 0.0) / n_obs
            if f"{key}|n_false_alarm_events" in counters:
                false_alarms = counters[f"{key}|n_false_alarm_events"]
                values["n_false_alarm_events"] = false_alarms
                values["false_alarms_per_1000"] = false_alarms / n_obs * 1000.0
        if f"{key}|n_support_gaps" in counters:
            values["n_support_gaps"] = counters[f"{key}|n_support_gaps"]

        rows.extend(
            {
                "detector_id": str(detector_id),
                "unique_id": None,
                "anomaly_type": key,
                "metric": name,
                "value": float(value),
                "n_obs": n_obs,
                "n_events": n_events,
            }
            for name, value in values.items()
        )
    return rows
