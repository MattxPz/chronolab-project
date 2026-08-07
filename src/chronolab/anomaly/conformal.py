"""Detector principal: score conformal sobre residuos fuera de muestra.

La tasa de falsos positivos queda controlada por construccion, sin asumir
normalidad de los residuos. El diseno completo, con las alternativas descartadas,
esta en docs/ANOMALY_DESIGN.md; aqui va lo que hay que saber para leer el codigo.

**No conformidad.** Para cada punto se toma el par de cuantiles del modelo al
nivel nominal, ``(l, u)``, y se calcula ``s = max(l - y, y - u)``, negativo dentro
del intervalo y positivo fuera, normalizado por la anchura: ``r = s / (u - l)``.
Se usa CQR y no el residuo absoluto porque hereda la asimetria que el modelo ya
captura —la simetria es falsa en demanda electrica (docs/ARCHITECTURE.md §7.3)— y
porque dividir por la anchura predicha deja ``r`` casi homoscedastico antes de
agrupar nada.

**Dos columnas, y no son redundantes.** ``score = -log10(p)`` con ``p`` el
p-valor conformal del grupo: bajo intercambiabilidad es super-uniforme, luego
``P(p <= alpha) <= alpha``, que es la afirmacion central del detector. Pero
satura en ``log10(n + 1)``: con ``n`` puntos de calibracion no existe informacion
para distinguir severidades mas alla de eso. ``severity`` es la magnitud sin
acotar —cuanto se sale la observacion, en anchuras de intervalo conformalizado— y
es lo que ordena esa cola y lo que agrega `chronolab.anomaly.events`. Las dos son
transformaciones crecientes de ``r`` dentro de un grupo, asi que no pueden
discrepar sobre si un punto esta marcado.

**Un solo mecanismo, tres regimenes.** Con ``pool_size=None`` y ``gamma=0`` esto
es split conformal exacto; con pool finito, conformal de ventana rodante; con
``gamma > 0``, adaptativo en linea (ACI). Que los casos degenerados sean
*exactamente* los metodos clasicos convierte "split frente a adaptativo" en un
experimento con dos escalares en lugar de dos rutas de codigo.

**Cuando se puede actualizar.** El intervalo forma parte de la prediccion, y la
prediccion de un instante se emitio en su origen, cuando ``y`` solo se conocia
hasta ahi. Actualizar con la observacion anterior es incorrecto en cuanto el
adelanto pasa de uno. Como el plan de un run de deteccion es teselado, todos los
puntos de una ventana comparten origen: **el estado avanza en las fronteras de
ventana, no punto a punto**.

**La absorcion de la anomalia.** Mientras una anomalia persiste el error es
constante, ACI baja el nivel efectivo, el intervalo se ensancha y el detector
deja de marcar lo que estaba detectando. La cuarentena de `_ingest` congela la
actualizacion tras unas cuantas marcas seguidas, con tope: pasado `max_freeze` un
desplazamiento permanente **debe** pasar a ser el nuevo normal. Eso es una
politica declarada, no un accidente, y rompe la garantia de tasa del ACI, que
pasa a valer sobre la subsucesion actualizada.
"""

import math
from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Tick

from chronolab.anomaly.protocols import DetectorRequirements, ScoringFrame
from chronolab.data.calendar import local_hour
from chronolab.errors import CutoffViolation
from chronolab.models.protocols import quantile_column
from chronolab.panel import PanelSpec
from chronolab.types import DetectorId, ModelId

__all__ = [
    "ALPHA_GRID",
    "SCORE_COLUMNS",
    "ConformalDetector",
    "FittedConformalDetector",
]

ALPHA_MIN: float = 1e-4
"""Suelo del nivel efectivo de ACI. Por debajo el intervalo es practicamente infinito."""

ALPHA_MAX: float = 0.5
"""Techo del nivel efectivo de ACI. Por encima el detector marcaria casi todo."""

SCORE_COLUMNS: tuple[str, ...] = (
    "unique_id",
    "ds",
    "score",
    "scorable",
    "severity",
    "calib_n",
    "side",
)
"""Columnas que devuelve `FittedConformalDetector.score`, en orden."""


def _default_alpha_grid() -> tuple[float, ...]:
    """Rejilla logaritmica de alfa, con los niveles habituales garantizados.

    Returns
    -------
    tuple of float
        Cuarenta valores geometricos en ``[0.001, 0.2]``, redondeados a cuatro
        decimales, mas ``0.01``, ``0.05`` y ``0.1``, que son los que la gente
        mira y tienen que caer exactamente en un nodo de la rejilla para que la
        interpolacion de `_invert` sea exacta en ellos.
    """
    geometric = {round(float(value), 4) for value in np.geomspace(0.001, 0.2, 40)}
    return tuple(sorted(geometric | {0.01, 0.05, 0.1}))


ALPHA_GRID: tuple[float, ...] = _default_alpha_grid()
"""Rejilla de alfa precomputada del proyecto (docs/ARCHITECTURE.md §7.4)."""

_GroupKey = tuple[int | str, ...]
"""Clave de un grupo de calibracion: nivel de repliegue y partes del grupo."""


@dataclass(slots=True)
class _FreezeState:
    """Cuarentena de una serie: marcas seguidas, pasos congelados y presupuesto."""

    run: int = 0
    frozen_run: int = 0
    frozen_total: int = 0
    excluded: int = 0
    ingested: int = 0
    budget_exhausted: bool = False


@dataclass(slots=True)
class _Pending:
    """Residuo ya puntuado que todavia no se ha realimentado al estado.

    Espera en la cola hasta que llega una ventana cuyo origen es posterior a su
    observacion: antes de eso, esa realimentacion no existia.
    """

    obs: int
    uid: str
    r: float
    chain: tuple[_GroupKey, ...]
    group: _GroupKey | None
    flags: np.ndarray | None
    flagged: bool


@dataclass(slots=True)
class _Scored:
    """Resultado de puntuar un punto contra el estado vigente."""

    group: _GroupKey
    calib_n: int
    score: float
    severity: float
    flags: np.ndarray
    flagged: bool


@dataclass(slots=True)
class _State:
    """Estado en linea del detector. Mutable, interno y siempre copiado antes de usar.

    `score` nunca lo toca: trabaja sobre una copia y la descarta, de modo que
    puntuar dos veces da exactamente lo mismo. Avanzar el estado es explicito y
    devuelve un objeto nuevo, igual que `fit` frente a mutar un modelo.
    """

    alpha_grid: np.ndarray
    ref_index: int
    max_freeze: int
    pools: dict[_GroupKey, list[float]] = field(default_factory=dict)
    order: dict[_GroupKey, deque[float]] = field(default_factory=dict)
    aci: dict[_GroupKey, np.ndarray] = field(default_factory=dict)
    scored: dict[_GroupKey, int] = field(default_factory=dict)
    flagged: dict[_GroupKey, int] = field(default_factory=dict)
    clips: dict[_GroupKey, int] = field(default_factory=dict)
    freeze: dict[str, _FreezeState] = field(default_factory=dict)
    pending: deque[_Pending] = field(default_factory=deque)

    def copy(self) -> "_State":
        """Copia profunda del estado.

        Returns
        -------
        _State
            Sin ninguna estructura compartida con el original. Las entradas de
            `pending` si se comparten, porque son inmutables de hecho: se crean
            al puntuar y no se vuelven a tocar.
        """
        return _State(
            alpha_grid=self.alpha_grid,
            ref_index=self.ref_index,
            max_freeze=self.max_freeze,
            pools={key: list(pool) for key, pool in self.pools.items()},
            order={key: deque(values) for key, values in self.order.items()},
            aci={key: values.copy() for key, values in self.aci.items()},
            scored=dict(self.scored),
            flagged=dict(self.flagged),
            clips=dict(self.clips),
            freeze={
                uid: _FreezeState(
                    run=value.run,
                    frozen_run=value.frozen_run,
                    frozen_total=value.frozen_total,
                    excluded=value.excluded,
                    ingested=value.ingested,
                    budget_exhausted=value.budget_exhausted,
                )
                for uid, value in self.freeze.items()
            },
            pending=deque(self.pending),
        )


@dataclass(frozen=True, slots=True)
class ConformalDetector:
    """Configuracion del detector conformal. Inmutable y sin estado calibrado.

    Parameters
    ----------
    base_model_id
        Modelo del que salen ``y_hat`` y los cuantiles. Forma parte del
        `detector_id` porque "conformal sobre residuos de MSTL" y "sobre residuos
        de NHITS" son detectores distintos.
    alpha_nominal
        Nivel del par de cuantiles del modelo que se conformaliza. ``0.05`` toma
        ``q_0250`` y ``q_9750``.
    alpha_ref
        Nivel al que se mide `severity` y al que se decide la cuarentena. Tiene
        que ser un nodo de `alpha_grid` para que la equivalencia entre
        ``severity > 0`` y ``score >= -log10(alpha_ref)`` sea exacta.
    alpha_grid
        Rejilla de alfa sobre la que corre la recursion de ACI y para la que se
        precomputan umbrales.
    gamma
        Paso de la recursion de ACI. ``0`` la desactiva. Se ajusta sobre ventanas
        de desarrollo, nunca sobre holdout: grande sigue la deriva pero oscila,
        pequeno es estable pero llega tarde.
    pool_size
        Longitud del pool rodante de calibracion por grupo. ``None`` lo deja
        ilimitado, que con ``gamma=0`` es split conformal exacto.
    hour_bins
        Numero de tramos horarios del segundo nivel de repliegue. Tiene que
        dividir a 24.
    lead_bins
        Bordes inferiores de los tramos de adelanto. ``(1, 6, 24)`` produce
        ``{1}``, ``{2..6}``, ``{7..24}`` y ``{>24}``.
    min_calib
        Puntos minimos que ha de tener un grupo para usarse. Un grupo mas fino
        que no llegue se replega al siguiente de la cadena.
    freeze_after
        Marcas consecutivas tras las cuales se congela la actualizacion.
    max_freeze
        Tope de pasos congelados. ``None`` lo fija en dos estacionalidades
        cortas del panel.

    Raises
    ------
    ValueError
        Si algun parametro cae fuera de su dominio, si `alpha_ref` no esta en la
        rejilla o si `hour_bins` no divide a 24.
    """

    base_model_id: ModelId | None = None
    alpha_nominal: float = 0.05
    alpha_ref: float = 0.05
    alpha_grid: tuple[float, ...] = ALPHA_GRID
    gamma: float = 0.01
    pool_size: int | None = 2000
    hour_bins: int = 6
    lead_bins: tuple[int, ...] = (1, 6, 24)
    min_calib: int = 50
    freeze_after: int = 3
    max_freeze: int | None = None

    def __post_init__(self) -> None:
        """Valida la configuracion completa."""
        if not 0.0 < self.alpha_nominal < 1.0:
            raise ValueError(f"alpha_nominal fuera de (0, 1): {self.alpha_nominal}")
        if not self.alpha_grid:
            raise ValueError("alpha_grid no puede estar vacia")
        if list(self.alpha_grid) != sorted(set(self.alpha_grid)):
            raise ValueError(f"alpha_grid debe ser estrictamente creciente: {self.alpha_grid}")
        if not all(0.0 < value < 1.0 for value in self.alpha_grid):
            raise ValueError(f"alpha_grid fuera de (0, 1): {self.alpha_grid}")
        if self.alpha_ref not in self.alpha_grid:
            raise ValueError(
                f"alpha_ref={self.alpha_ref} no esta en alpha_grid; la equivalencia entre "
                "severity y score solo es exacta en los nodos de la rejilla"
            )
        if self.gamma < 0.0:
            raise ValueError(f"gamma debe ser >= 0: {self.gamma}")
        if self.pool_size is not None and self.pool_size < 1:
            raise ValueError(f"pool_size debe ser >= 1 o None: {self.pool_size}")
        if self.hour_bins < 1 or 24 % self.hour_bins:
            raise ValueError(f"hour_bins debe dividir a 24: {self.hour_bins}")
        if not self.lead_bins or list(self.lead_bins) != sorted(set(self.lead_bins)):
            raise ValueError(f"lead_bins debe ser estrictamente creciente: {self.lead_bins}")
        if self.lead_bins[0] < 1:
            raise ValueError(f"lead_bins debe empezar en >= 1: {self.lead_bins}")
        if self.min_calib < 1:
            raise ValueError(f"min_calib debe ser >= 1: {self.min_calib}")
        if self.freeze_after < 1:
            raise ValueError(f"freeze_after debe ser >= 1: {self.freeze_after}")
        if self.max_freeze is not None and self.max_freeze < 0:
            raise ValueError(f"max_freeze debe ser >= 0 o None: {self.max_freeze}")

    @property
    def detector_id(self) -> DetectorId:
        """Identificador estable. Clave de particion de `anomaly_scores`.

        Codifica los tres parametros que cambian el metodo —nivel nominal, paso
        de ACI y longitud del pool— mas el modelo base, porque el mismo detector
        sobre residuos de otro modelo es otro detector.
        """
        pool = "inf" if self.pool_size is None else str(self.pool_size)
        base = "none" if self.base_model_id is None else str(self.base_model_id)
        return DetectorId(
            f"conformal_cqr_a{round(self.alpha_nominal * 10000):04d}"
            f"_g{round(self.gamma * 10000):04d}_k{pool}_h{self.hour_bins}_m{base}"
        )

    @property
    def requires(self) -> DetectorRequirements:
        """Necesidades declaradas.

        ``window=1``: el detector no consume contexto. El calentamiento viene de
        la calibracion, no de una ventana deslizante, asi que no encoge la
        mascara `scorable` comun frente a detectores de ventana larga.
        """
        return DetectorRequirements(
            needs_forecast=True,
            needs_quantiles=True,
            window=1,
            needs_calibration=True,
            fit_cost="cheap",
        )

    @property
    def quantile_columns(self) -> tuple[str, str]:
        """Par de columnas de cuantil que se conformaliza, de menor a mayor."""
        return (
            quantile_column(self.alpha_nominal / 2.0),
            quantile_column(1.0 - self.alpha_nominal / 2.0),
        )

    def fit(self, calib: ScoringFrame) -> "FittedConformalDetector":
        """Calibra el detector con un tramo anterior al que se puntuara.

        Recorre el tramo con exactamente el mismo bucle en linea que `score`, de
        modo que el estado con el que arranca la puntuacion es el que habria
        tenido un detector que llevase corriendo desde el principio de la
        calibracion. Calibrar no es "calcular un cuantil": es dejar el estado
        donde tiene que estar.

        Parameters
        ----------
        calib
            Tramo de calibracion, normalmente el de las ventanas ``dev``.

        Returns
        -------
        FittedConformalDetector
            Con ``cutoff = calib.end``.
        """
        seasonal = calib.spec.seasonalities[0]
        state = _State(
            alpha_grid=np.asarray(self.alpha_grid, dtype=float),
            ref_index=self.alpha_grid.index(self.alpha_ref),
            max_freeze=2 * seasonal if self.max_freeze is None else self.max_freeze,
        )
        _run(self, state, calib)
        return FittedConformalDetector(
            detector=self,
            spec=calib.spec,
            cutoff=calib.end,
            state=state,
        )


@dataclass(frozen=True, slots=True)
class FittedConformalDetector:
    """Detector conformal calibrado hasta un instante concreto.

    Attributes
    ----------
    detector
        Configuracion de la que procede.
    spec
        Especificacion del panel con el que se calibro.
    cutoff
        Ultimo instante usado en calibracion. `score` exige ``ds > cutoff``.
    state
        Estado en linea. Interno: `score` trabaja siempre sobre una copia.
    """

    detector: ConformalDetector
    spec: PanelSpec
    cutoff: pd.Timestamp
    state: _State

    @property
    def detector_id(self) -> DetectorId:
        """Identificador del detector del que procede."""
        return self.detector.detector_id

    def score(self, frame: ScoringFrame) -> pd.DataFrame:
        """Puntua cada marca de tiempo del tramo.

        No muta el estado: corre la recursion sobre una copia y la descarta.
        Puntuar dos veces la misma trama da bit a bit lo mismo, que es lo que
        hace reproducible un detector que por dentro es secuencial. Para llevar
        el estado adelante esta `advance`, que lo dice en el tipo.

        Parameters
        ----------
        frame
            Tramo a puntuar, con ``frame.start > cutoff``.

        Returns
        -------
        pandas.DataFrame
            Columnas `SCORE_COLUMNS`, una fila por ``(unique_id, ds)`` de la
            entrada y en su mismo orden.

        Raises
        ------
        CutoffViolation
            Si `frame` empieza en un instante que la calibracion ya conocia.
        """
        self._assert_after_cutoff(frame)
        return _run(self.detector, self.state.copy(), frame)

    def advance(self, frame: ScoringFrame) -> "FittedConformalDetector":
        """Devuelve un detector nuevo con el estado avanzado sobre `frame`.

        Es lo que hace utilizable el detector en casi tiempo real: la deteccion
        de un tramo no puede depender de haberse acordado de guardar el estado.

        Parameters
        ----------
        frame
            Tramo ya puntuado, con ``frame.start > cutoff``.

        Returns
        -------
        FittedConformalDetector
            Con ``cutoff = frame.end`` y el estado tras recorrer `frame`.

        Raises
        ------
        CutoffViolation
            Si `frame` empieza en un instante que el detector ya conocia.
        """
        self._assert_after_cutoff(frame)
        state = self.state.copy()
        _run(self.detector, state, frame)
        return FittedConformalDetector(
            detector=self.detector,
            spec=self.spec,
            cutoff=frame.end,
            state=state,
        )

    def coverage_report(self) -> pd.DataFrame:
        """Tasa de marcado observada por grupo de calibracion.

        Es la capa de honestidad del detector: si la tasa de un grupo se aleja de
        `alpha_ref`, lo dice el propio detector en lugar de tener que descubrirlo
        alguien. Refleja todo lo recorrido —la calibracion y lo que haya anadido
        `advance`—, porque el estado es acumulativo.

        Returns
        -------
        pandas.DataFrame
            Una fila por grupo efectivamente usado, con ``level`` (posicion en la
            cadena de repliegue, 0 el mas fino), ``n_calib`` (el tamano de pool
            que sostiene la garantia), ``flag_rate`` y el nivel efectivo de ACI.
        """
        columns = [
            "detector_id",
            "level",
            "group",
            "n_calib",
            "n_scored",
            "n_flagged",
            "flag_rate",
            "alpha_ref",
            "alpha_eff",
            "n_aci_clips",
        ]
        rows: list[dict[str, object]] = []
        for key in sorted(self.state.scored, key=str):
            scored = self.state.scored[key]
            flagged = self.state.flagged.get(key, 0)
            effective = self.state.aci.get(key)
            rows.append(
                {
                    "detector_id": str(self.detector_id),
                    "level": int(key[0]),
                    "group": "|".join(str(part) for part in key[1:]),
                    "n_calib": len(self.state.pools.get(key, [])),
                    "n_scored": scored,
                    "n_flagged": flagged,
                    "flag_rate": flagged / scored if scored else math.nan,
                    "alpha_ref": self.detector.alpha_ref,
                    "alpha_eff": (
                        math.nan
                        if effective is None
                        else float(np.maximum.accumulate(effective)[self.state.ref_index])
                    ),
                    "n_aci_clips": self.state.clips.get(key, 0),
                }
            )
        if not rows:
            return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})
        return pd.DataFrame(rows)[columns]

    def freeze_report(self) -> pd.DataFrame:
        """Cuanto se congelo la actualizacion en cada serie, y por que.

        La cuarentena rompe la garantia de tasa del ACI, que asume actualizacion
        en todos los pasos: pasa a valer sobre la subsucesion actualizada. Quien
        lea la tasa tiene que poder ver sobre que se cumple, y `budget_exhausted`
        distingue una anomalia larga de un cambio de regimen que el detector ya
        se ha negado a seguir ignorando.

        Returns
        -------
        pandas.DataFrame
            Una fila por serie realimentada.
        """
        columns = [
            "unique_id",
            "n_ingested",
            "n_frozen_steps",
            "n_excluded",
            "budget_exhausted",
        ]
        rows = [
            {
                "unique_id": uid,
                "n_ingested": value.ingested,
                "n_frozen_steps": value.frozen_total,
                "n_excluded": value.excluded,
                "budget_exhausted": value.budget_exhausted,
            }
            for uid, value in sorted(self.state.freeze.items())
        ]
        if not rows:
            return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})
        return pd.DataFrame(rows)[columns]

    def _assert_after_cutoff(self, frame: ScoringFrame) -> None:
        """Comprueba que el tramo no empieza en el pasado ya calibrado.

        Es la barrera que impide que la calibracion se solape con lo que se
        puntua. Tambien es la que detiene un run cuyo plan no sea teselado: con
        solape, los tramos de desarrollo y de holdout se pisan y esto salta.

        Parameters
        ----------
        frame
            Tramo a puntuar.

        Raises
        ------
        CutoffViolation
            Si ``frame.start <= cutoff``.
        """
        if frame.start <= self.cutoff:
            raise CutoffViolation(
                f"{self.detector_id} esta calibrado hasta {self.cutoff} y se le pide puntuar "
                f"desde {frame.start}, anterior o igual: la calibracion se solaparia con lo "
                "que se puntua"
            )


# --------------------------------------------------------------------------- #
# Bucle en linea
# --------------------------------------------------------------------------- #


def _run(detector: ConformalDetector, state: _State, frame: ScoringFrame) -> pd.DataFrame:
    """Recorre un tramo puntuando y realimentando el estado.

    Parameters
    ----------
    detector
        Configuracion.
    state
        Estado en linea. **Se muta**: el llamante decide si le importa.
    frame
        Tramo a recorrer.

    Returns
    -------
    pandas.DataFrame
        Columnas `SCORE_COLUMNS`, en el orden de filas de la entrada.
    """
    df = frame.df
    n_rows = len(df)
    score = np.full(n_rows, np.nan)
    severity = np.full(n_rows, np.nan)
    calib_n = np.zeros(n_rows, dtype=np.int64)
    side = np.zeros(n_rows, dtype=np.int8)
    scorable = np.zeros(n_rows, dtype=bool)

    if n_rows:
        _process(detector, state, frame, score, severity, calib_n, side, scorable)

    return pd.DataFrame(
        {
            "unique_id": df["unique_id"].astype(str).to_numpy(),
            "ds": df["ds"].to_numpy(dtype="datetime64[ns]"),
            "score": score.astype(np.float32),
            "scorable": scorable,
            "severity": severity.astype(np.float32),
            "calib_n": calib_n.astype(np.int32),
            "side": side,
        }
    )[list(SCORE_COLUMNS)]


def _process(
    detector: ConformalDetector,
    state: _State,
    frame: ScoringFrame,
    score: np.ndarray,
    severity: np.ndarray,
    calib_n: np.ndarray,
    side: np.ndarray,
    scorable: np.ndarray,
) -> None:
    """Nucleo del bucle: bloques por origen, realimentacion y puntuacion.

    El orden de operaciones dentro de cada bloque es el que hace correcta la
    version en linea: primero se realimenta todo lo que ya se podia conocer en el
    origen del bloque, despues se puntua el bloque entero con ese estado
    congelado, y solo al final se encolan sus residuos. Puntuar antes de
    realimentar seria usar informacion vieja; realimentar con el propio bloque
    seria usar informacion que en el origen no existia.

    Parameters
    ----------
    detector
        Configuracion.
    state
        Estado en linea, mutado en el sitio.
    frame
        Tramo a recorrer.
    score, severity, calib_n, side, scorable
        Vectores de salida, rellenados por posicion de fila.
    """
    df = frame.df
    spec = frame.spec
    lo_column, hi_column = detector.quantile_columns
    _assert_columns(df, lo_column, hi_column)

    y = df["y"].to_numpy(dtype=float)
    lower = df[lo_column].to_numpy(dtype=float)
    upper = df[hi_column].to_numpy(dtype=float)
    width = upper - lower
    with np.errstate(invalid="ignore", divide="ignore"):
        residual = np.maximum(lower - y, y - upper) / width

    valid = (
        np.isfinite(y)
        & np.isfinite(lower)
        & np.isfinite(upper)
        & (width > 0)
        & np.isfinite(residual)
        & df["cutoff"].notna().to_numpy()
        & df["h_step"].notna().to_numpy()
    )
    positions = np.flatnonzero(valid)
    if positions.size == 0:
        return

    uid = df["unique_id"].astype(str).to_numpy()
    uid_code = pd.factorize(df["unique_id"].astype(str))[0]
    ds_ns = df["ds"].to_numpy(dtype="datetime64[ns]").astype("int64")
    cutoff_ns = df["cutoff"].to_numpy(dtype="datetime64[ns]").astype("int64")

    hour = local_hour(df["ds"], tz_display=spec.tz_display).to_numpy(dtype=np.int64)
    hour_bin = hour // (24 // detector.hour_bins)
    lead = _lead_steps(df, spec.freq)
    lead_bin = np.searchsorted(np.asarray(detector.lead_bins), lead, side="left")

    ordering = np.lexsort((uid_code[positions], ds_ns[positions], cutoff_ns[positions]))
    positions = positions[ordering]
    block_cutoff = cutoff_ns[positions]
    edges = np.concatenate(
        ([0], np.flatnonzero(np.diff(block_cutoff)) + 1, [positions.size]),
    )

    for block_id in range(len(edges) - 1):
        block = positions[edges[block_id] : edges[block_id + 1]]
        origin = int(block_cutoff[edges[block_id]])
        _drain(detector, state, origin)

        fresh: list[_Pending] = []
        for row in block:
            chain = _chain(uid[row], int(hour[row]), int(hour_bin[row]), int(lead_bin[row]))
            value = float(residual[row])
            scored = _score_point(detector, state, value, chain)
            if scored is not None:
                score[row] = scored.score
                severity[row] = scored.severity
                calib_n[row] = scored.calib_n
                side[row] = 1 if (y[row] - upper[row]) >= (lower[row] - y[row]) else -1
                scorable[row] = True
                state.scored[scored.group] = state.scored.get(scored.group, 0) + 1
                if scored.flagged:
                    state.flagged[scored.group] = state.flagged.get(scored.group, 0) + 1
            fresh.append(
                _Pending(
                    obs=int(ds_ns[row]),
                    uid=uid[row],
                    r=value,
                    chain=chain,
                    group=None if scored is None else scored.group,
                    flags=None if scored is None else scored.flags,
                    flagged=False if scored is None else scored.flagged,
                )
            )
        state.pending.extend(fresh)


def _assert_columns(df: pd.DataFrame, lo_column: str, hi_column: str) -> None:
    """Comprueba que el tramo trae lo que el detector declara necesitar.

    Parameters
    ----------
    df
        Trama del `ScoringFrame`.
    lo_column, hi_column
        Columnas del par de cuantiles conformalizado.

    Raises
    ------
    ValueError
        Si falta alguna columna obligatoria. Un modelo sin cuantiles se envuelve
        antes con `chronolab.models.wrappers.ConformalWrapper`; el detector no
        fabrica un intervalo que el modelo no dio.
    """
    required = {"unique_id", "ds", "y", "cutoff", "h_step", lo_column, hi_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"el tramo no trae las columnas obligatorias {sorted(missing)}; el detector "
            "conformal declara needs_quantiles y no inventa intervalos"
        )


def _chain(uid: str, hour: int, hour_bin: int, lead_bin: int) -> tuple[_GroupKey, ...]:
    """Cadena de repliegue de grupos de calibracion, de mas fina a mas gruesa.

    La heterocedasticidad de la demanda es la razon de que el grupo tenga hora y
    adelanto: un cuantil global daria cobertura marginal correcta y condicional
    desastrosa, es decir un detector que parece calibrado y no lo esta. El
    presupuesto muestral es la razon de que sea una cadena y no un grupo fijo.

    Parameters
    ----------
    uid
        Serie.
    hour
        Hora local, 0-23.
    hour_bin
        Tramo horario.
    lead_bin
        Tramo de adelanto.

    Returns
    -------
    tuple of tuple
        Cinco niveles, del mas fino al global.
    """
    return (
        (0, uid, hour, lead_bin),
        (1, uid, hour_bin, lead_bin),
        (2, uid, hour_bin),
        (3, hour_bin),
        (4,),
    )


def _lead_steps(df: pd.DataFrame, freq: str) -> np.ndarray:
    """Adelanto de cada fila en pasos de rejilla, contado desde su origen.

    Se calcula como ``ds - cutoff`` porque es exacto y no necesita conocer el
    `gap` del plan. En frecuencias que no son de paso fijo —meses, por ejemplo—
    esa resta no es un numero de pasos, y entonces se usa `h_step`, que difiere
    del adelanto real en el `gap`, constante en todo el run.

    Parameters
    ----------
    df
        Trama del `ScoringFrame`.
    freq
        Frecuencia del panel.

    Returns
    -------
    numpy.ndarray
        Adelanto en pasos, ``int64``. Uno donde no se puede determinar.
    """
    steps = df["h_step"].to_numpy(dtype="float64", na_value=np.nan)
    fallback: np.ndarray = np.nan_to_num(steps, nan=1.0).astype(np.int64)
    offset = to_offset(freq)
    if not isinstance(offset, Tick):
        return fallback

    delta = df["ds"].to_numpy(dtype="datetime64[ns]") - df["cutoff"].to_numpy(
        dtype="datetime64[ns]"
    )
    known = ~pd.isna(delta)
    lead = np.ones(len(df), dtype=np.int64)
    step = np.timedelta64(pd.Timedelta(offset).value, "ns")
    lead[known] = (delta[known] // step).astype(np.int64)
    return np.where(lead >= 1, lead, fallback)


def _score_point(
    detector: ConformalDetector,
    state: _State,
    residual: float,
    chain: tuple[_GroupKey, ...],
) -> _Scored | None:
    """Puntua un residuo contra el estado vigente de su grupo.

    Parameters
    ----------
    detector
        Configuracion.
    state
        Estado en linea. Solo se crea el vector de ACI si el grupo es nuevo.
    residual
        No conformidad normalizada del punto.
    chain
        Cadena de repliegue de grupos.

    Returns
    -------
    _Scored or None
        ``None`` si ningun nivel de la cadena llega a `min_calib`, que es el
        calentamiento del detector. Nunca se extrapola una cola no observada.
    """
    pool: list[float] | None = None
    group: _GroupKey | None = None
    for candidate in chain:
        available = state.pools.get(candidate)
        if available is not None and len(available) >= detector.min_calib:
            pool, group = available, candidate
            break
    if pool is None or group is None:
        return None

    size = len(pool)
    count_ge = size - bisect_left(pool, residual)
    p_static = (1.0 + count_ge) / (size + 1.0)

    effective = state.aci.get(group)
    if effective is None:
        effective = state.alpha_grid.copy()
        state.aci[group] = effective
    # ACI no garantiza por si solo que el conjunto marcado crezca con alfa; el
    # maximo acumulado lo impone antes de invertir, que es lo que mantiene el
    # p-valor adaptativo monotono y por tanto el umbral igual a -log10(alfa).
    monotone = np.maximum.accumulate(effective)

    flags = p_static <= monotone
    p_adaptive = _invert(p_static, state.alpha_grid, monotone)
    reference = float(monotone[state.ref_index])

    return _Scored(
        group=group,
        calib_n=size,
        score=-math.log10(p_adaptive),
        severity=_severity(residual, pool, reference),
        flags=flags,
        flagged=bool(flags[state.ref_index]),
    )


def _severity(residual: float, pool: list[float], alpha: float) -> float:
    """Cuanto se sale el punto, en anchuras de intervalo conformalizado.

    Se normaliza por la anchura **conformalizada** y no por la que predijo el
    modelo: la del modelo puede estar descalibrada —bandas absurdamente
    estrechas darian severidades enormes en todas partes— mientras que la
    conformalizada es la escala empiricamente correcta.

    Sale de la identidad ``severity = (r - Q) / (1 + 2Q)``, que es una
    transformacion afin estrictamente creciente de `r` dentro del grupo. De ahi
    que `severity` y `score` no puedan discrepar sobre si un punto esta marcado.

    Parameters
    ----------
    residual
        No conformidad normalizada del punto.
    pool
        Pool de calibracion del grupo, ordenado.
    alpha
        Nivel efectivo de referencia.

    Returns
    -------
    float
        Negativo dentro del intervalo, cero en el borde, ``1.0`` a una anchura
        completa fuera. `NaN` si el pool no llega para ese nivel.
    """
    size = len(pool)
    index = math.ceil((size + 1) * (1.0 - alpha))
    if index > size or index < 1:
        return math.nan
    quantile = pool[index - 1]
    denominator = 1.0 + 2.0 * quantile
    if denominator <= 0.0:
        return math.nan
    return (residual - quantile) / denominator


def _invert(p_static: float, grid: np.ndarray, effective: np.ndarray) -> float:
    """Invierte el nivel efectivo de ACI para obtener el p-valor adaptativo.

    Marcar al nivel ``alpha`` equivale a ``p_static <= effective(alpha)``, asi que
    el p-valor que hace comparable todo es el ``alpha`` que resuelve
    ``effective(alpha) = p_static``. Dentro de la rejilla se interpola en escala
    logaritmica; fuera se extrapola manteniendo la razon del extremo, que es lo
    que conserva la resolucion del score por debajo del alfa mas pequeno de la
    rejilla en lugar de aplastarla a un valor constante.

    Con ``gamma = 0`` se tiene ``effective == grid`` y esto devuelve `p_static`
    exactamente, en toda la recta: split conformal cae por su propio peso.

    Parameters
    ----------
    p_static
        P-valor conformal contra el pool del grupo.
    grid
        Rejilla de alfa, creciente.
    effective
        Nivel efectivo por alfa, ya monotonizado.

    Returns
    -------
    float
        P-valor adaptativo en ``(0, 1]``.
    """
    first, last = float(effective[0]), float(effective[-1])
    if p_static <= first:
        return min(1.0, float(grid[0]) * (p_static / first))
    if p_static >= last:
        return min(1.0, float(grid[-1]) * (p_static / last))

    index = int(np.searchsorted(effective, p_static, side="right")) - 1
    index = min(max(index, 0), len(effective) - 2)
    low, high = float(effective[index]), float(effective[index + 1])
    if high <= low:
        return float(grid[index])
    weight = (math.log(p_static) - math.log(low)) / (math.log(high) - math.log(low))
    logged = math.log(float(grid[index])) + weight * (
        math.log(float(grid[index + 1])) - math.log(float(grid[index]))
    )
    return min(1.0, math.exp(logged))


def _drain(detector: ConformalDetector, state: _State, origin: int) -> None:
    """Realimenta todo lo observado hasta un origen de prediccion.

    Parameters
    ----------
    detector
        Configuracion.
    state
        Estado en linea, mutado en el sitio.
    origin
        Cutoff del bloque que va a puntuarse, en nanosegundos.
    """
    while state.pending and state.pending[0].obs <= origin:
        _ingest(detector, state, state.pending.popleft())


def _ingest(detector: ConformalDetector, state: _State, entry: _Pending) -> None:
    """Incorpora un residuo al estado, salvo que la cuarentena lo impida.

    La cuarentena existe porque ACI absorbe la anomalia que detecta: mientras el
    error persiste el nivel efectivo baja, el intervalo se ensancha y el detector
    se calla. Congelar la actualizacion durante una tirada de marcas lo impide;
    el tope `max_freeze` impide lo contrario, que un desplazamiento permanente
    quede marcado para siempre. El presupuesto acota cuanto se puede excluir:
    excluir mas que eso ya no es robustez, es negarse a ver un cambio de regimen.

    Parameters
    ----------
    detector
        Configuracion.
    state
        Estado en linea, mutado en el sitio.
    entry
        Residuo pendiente de realimentar.
    """
    freeze = state.freeze.setdefault(entry.uid, _FreezeState())
    freeze.ingested += 1
    if entry.flagged:
        freeze.run += 1
    else:
        freeze.run = 0
        freeze.frozen_run = 0

    if entry.flagged and freeze.run >= detector.freeze_after:
        budget = 3.0 * detector.alpha_ref * freeze.ingested
        if freeze.frozen_run < state.max_freeze and freeze.excluded + 1 <= budget:
            freeze.frozen_run += 1
            freeze.frozen_total += 1
            freeze.excluded += 1
            return
        if freeze.excluded + 1 > budget:
            freeze.budget_exhausted = True

    _update_aci(detector, state, entry)
    _push_pools(detector, state, entry)


def _update_aci(detector: ConformalDetector, state: _State, entry: _Pending) -> None:
    """Aplica la recursion de ACI con el error observado en este punto.

    El error tiene que ser el que se cometio con el estado vigente **al
    puntuar**, no con el de ahora: por eso el vector de marcas viaja en la
    entrada pendiente en lugar de recalcularse aqui.

    Parameters
    ----------
    detector
        Configuracion.
    state
        Estado en linea, mutado en el sitio.
    entry
        Residuo pendiente. Si no llego a puntuarse no hay error que realimentar.
    """
    if entry.group is None or entry.flags is None or detector.gamma == 0.0:
        return
    current = state.aci.get(entry.group)
    if current is None:  # pragma: no cover  puntuar siempre crea el vector
        return
    raw = current + detector.gamma * (state.alpha_grid - entry.flags.astype(float))
    clipped = np.clip(raw, ALPHA_MIN, ALPHA_MAX)
    clips = int(np.count_nonzero(clipped != raw))
    if clips:
        state.clips[entry.group] = state.clips.get(entry.group, 0) + clips
    state.aci[entry.group] = clipped


def _push_pools(detector: ConformalDetector, state: _State, entry: _Pending) -> None:
    """Anade el residuo a los pools de todos los niveles de su cadena.

    Se anade a todos y no solo al nivel usado: los niveles gruesos tienen que
    crecer para que el repliegue exista, y representan residuos genuinamente
    agrupados, no un resumen de los finos.

    Parameters
    ----------
    detector
        Configuracion.
    state
        Estado en linea, mutado en el sitio.
    entry
        Residuo pendiente.
    """
    for key in entry.chain:
        pool = state.pools.setdefault(key, [])
        arrival = state.order.setdefault(key, deque())
        insort(pool, entry.r)
        arrival.append(entry.r)
        if detector.pool_size is not None and len(arrival) > detector.pool_size:
            evicted = arrival.popleft()
            del pool[bisect_left(pool, evicted)]
