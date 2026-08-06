"""Adaptador de neuralforecast: NHITS, TFT y PatchTST.

`refit_cost = 'expensive'`: por defecto un unico ajuste por run, y la politica
queda registrada en `model_runs.refit_every`.

Construccion diferida a `fit`
------------------------------
Los tres modelos necesitan `h`, `input_size` y las listas de exogenas para
**construir la red**, no para ajustarla (docs/ARCHITECTURE.md D6, D21). Por eso
las clases de este modulo son configuraciones puras y el objeto de
neuralforecast no existe hasta `fit`, que es donde el motor entrega a la vez el
horizonte del plan y el `PanelSpec` del tramo de entrenamiento. Intentar
construirlo en `__init__` obligaria a que `h` fuese parametro del constructor y
a mantener sincronizados dos horizontes que pueden divergir en silencio.

Presupuesto acotado, early stopping y escalado
------------------------------------------------
Tres parametros gobiernan que el backtest completo sea viable en CPU o GPU
modesta, y los tres son explicitos:

- `max_steps` es el **tope duro** de pasos de optimizacion. No son epocas:
  neuralforecast cuenta pasos de gradiente sobre lotes de ventanas, asi que el
  presupuesto no depende del tamano del panel, que es justo lo que se quiere
  para acotar el coste de un run.
- `early_stop_patience_steps` junto con `val_check_steps` y un `val_size > 0`
  en `fit` activan el early stopping de Lightning. El corte de validacion que
  usa neuralforecast son los **ultimos** `val_size` instantes del tramo de
  entrenamiento: un corte temporal dentro del propio train, nunca sobre el
  tramo evaluado, que el motor ni siquiera le pasa (fuga L2).
- `scaler_type` normaliza **cada ventana de entrada por separado**, con los
  estadisticos de su propio contexto. Es escalado por serie y causal por
  construccion: el contexto de una ventana es, entero, anterior al instante que
  predice. Se prefiere al `local_scaler_type` de `NeuralForecast` —que calcula
  un estadistico por serie sobre todo el `df` de ajuste— porque no depende de
  cuanta historia se le pase.

Cuantiles nativos via `MQLoss`
-------------------------------
Los tres se entrenan con `MQLoss` sobre la rejilla canonica del proyecto, asi
que los cuantiles son estimados, no derivados de una gaussiana. La traduccion
de los nombres de columna que produce neuralforecast (``-lo-95.0``,
``-median``, ``-hi-80.0``...) a la convencion `q_<int>` del proyecto se hace
leyendo `loss.output_names` **de la propia instancia de la perdida**, no
reconstruyendo la formula: es la unica forma de que un cambio de convencion de
nombres en la libreria se note como un cuantil ausente y no como uno mal
asignado.

Interpretabilidad del TFT
--------------------------
`TFTForecaster` expone `variable_importance()` (pesos de seleccion de
variables, pasadas y futuras) y `temporal_attention()` (atencion por instante),
en el formato largo de la tabla `explanations` de docs/ARCHITECTURE.md §7.4
—con `kind` en ``attention_variable`` / ``attention_temporal``— para que la
pagina de explicabilidad de la app las lea sin transformarlas.

Un matiz que conviene tener presente al interpretarlas: neuralforecast guarda
esos pesos en `interpretability_params` **en cada pasada hacia delante**, y los
sobreescribe. Como `fit` termina con una pasada de validacion, los pesos ya
existen nada mas ajustar, pero describen el lote de validacion, no el tramo que
se quiere explicar. Para que digan lo que parece que dicen hay que llamar antes
a `predict` sobre la ventana de interes: es lo que hace el script del hito, y
por eso las dos funciones documentan la precondicion en vez de dejarla
implicita. El guardarrail que si lanza cubre el caso en que no ha habido
ninguna pasada.

Import perezoso
----------------
`neuralforecast` vive en el extra `deep` (D20). El modulo debe poder
**importarse** sin el —lo exige `tests/unit/test_module_tree.py` en el entorno
por defecto de CI— asi que el import real vive en `_require_neuralforecast`,
llamada solo al ajustar.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd

from chronolab.models.protocols import QUANTILES, ModelRequirements, quantile_column
from chronolab.panel import FutrFrame, Panel
from chronolab.types import ModelId

__all__ = [
    "NHITSForecaster",
    "PatchTSTForecaster",
    "TFTForecaster",
    "quiet_lightning",
]

_NHITS_ID = ModelId("nhits")
_TFT_ID = ModelId("tft")
_PATCHTST_ID = ModelId("patchtst")

_ALIAS = "chronolab"
"""Alias con el que se registra la red en `NeuralForecast`.

Fijarlo desacopla los nombres de columna de la salida del nombre de la clase de
la libreria, que cambia entre versiones. **No** puede ser ``"model"``:
`NeuralForecast._get_model_names` arranca su contador de colisiones con esa
clave ya presente, asi que un modelo llamado ``model`` sale renombrado a
``model1``. Aun asi, el codigo de abajo no confia en este valor para localizar
las columnas —las busca por sufijo sobre las que realmente vuelven— y el alias
solo sirve para que los logs sean legibles.
"""

_QUANTILE_TOL = 1e-6
"""Tolerancia al emparejar cuantiles pedidos con los de la perdida.

`MQLoss` guarda su rejilla en un tensor `float32`, asi que ``0.025`` vuelve
como ``0.02500000037252903``. Comparar por igualdad exacta dejaria todas las
columnas de cuantil en `NaN` sin que nada lo dijese: el modelo habria estimado
los cuantiles y el adaptador los estaria tirando a la basura.
"""


def quiet_lightning() -> None:
    """Baja el nivel de log de PyTorch Lightning a errores.

    Un backtest de varias ventanas por varios modelos produce, si no, cientos
    de lineas de "GPU available: False" que entierran la salida util del run.
    Se ofrece como funcion publica y explicita en vez de aplicarse al importar:
    silenciar el logging de una libreria ajena es una decision del programa que
    la usa, no un efecto colateral de un import.
    """
    for name in ("pytorch_lightning", "lightning", "lightning.pytorch"):
        logging.getLogger(name).setLevel(logging.ERROR)


def _require_neuralforecast() -> tuple[Any, Any, Any, Any, Any]:
    """Importa neuralforecast bajo demanda, con un mensaje util si falta el extra.

    Returns
    -------
    tuple
        ``(NeuralForecast, NHITS, TFT, PatchTST, MQLoss)``. Tipado como `Any`
        por la cuarentena D16 (docs/ARCHITECTURE.md).

    Raises
    ------
    ImportError
        Si `neuralforecast` no esta instalado.
    """
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MQLoss
        from neuralforecast.models import NHITS, TFT, PatchTST
    except ImportError as exc:  # pragma: no cover  ejercitado solo sin el extra deep
        raise ImportError(
            "chronolab.models.adapters.neuralforecast necesita el extra 'deep': "
            "`uv sync --extra deep`."
        ) from exc
    return NeuralForecast, NHITS, TFT, PatchTST, MQLoss


def _trainer_kwargs(*, accelerator: str, enable_progress_bar: bool) -> dict[str, Any]:
    """Argumentos que neuralforecast reenvia al `Trainer` de Lightning.

    Se pasan **planos** al constructor del modelo (la firma los recoge con
    ``**trainer_kwargs``), no dentro de un diccionario: pasarlos anidados
    produce un `TypeError` del propio `Trainer`.

    Parameters
    ----------
    accelerator
        ``"cpu"``, ``"gpu"`` o ``"auto"``.
    enable_progress_bar
        Barra de progreso por ventana. Desactivada por defecto en un backtest.

    Returns
    -------
    dict
        Sin checkpointing ni logger: un run de backtesting ajusta una red por
        ventana y no quiere dejar un arbol de checkpoints por cada una.
    """
    return {
        "accelerator": accelerator,
        "enable_progress_bar": enable_progress_bar,
        "enable_checkpointing": False,
        "enable_model_summary": False,
        "logger": False,
    }


def _quantile_suffixes(loss: Any) -> tuple[tuple[float, str], ...]:
    """Pares ``(cuantil, sufijo de columna)`` que emitira neuralforecast.

    Se leen de `loss.quantiles` y `loss.output_names`, emparejados por
    posicion, en lugar de reconstruir la formula de niveles a mano: si la
    libreria cambiase su convencion de nombres, esto sigue emparejando bien, y
    si dejase de emitir un cuantil se nota como ausencia (columna `NaN`) en vez
    de como una asignacion equivocada.

    Parameters
    ----------
    loss
        Instancia de `MQLoss` con la que se construyo el modelo.

    Returns
    -------
    tuple
        Por ejemplo ``((0.025, "-lo-95.0"), ..., (0.5, "-median"), ...)``. Los
        cuantiles vienen de un tensor `float32`, asi que no son exactamente los
        `float` de Python que se pidieron: se comparan con `_QUANTILE_TOL`.
    """
    quantiles = [float(q) for q in loss.quantiles.detach().cpu().numpy()]
    return tuple(zip(quantiles, loss.output_names, strict=True))


def _resolve_column(raw: pd.DataFrame, suffix: str) -> str | None:
    """Localiza la columna de `raw` que termina en `suffix`.

    Buscar por sufijo en vez de componer ``alias + sufijo`` evita depender de
    como `NeuralForecast` haya acabado nombrando al modelo: la libreria
    renombra ante colisiones (ver `_ALIAS`), y con un unico modelo registrado
    hay exactamente una columna por sufijo.

    Parameters
    ----------
    raw
        Salida cruda de `NeuralForecast.predict`.
    suffix
        Sufijo de `loss.output_names`, por ejemplo ``"-median"``.

    Returns
    -------
    str or None
        Nombre de la columna, o ``None`` si no esta.
    """
    matches = [column for column in raw.columns if str(column).endswith(suffix)]
    return matches[0] if len(matches) == 1 else None


def _assign_quantiles(
    raw: pd.DataFrame, *, suffixes: Sequence[tuple[float, str]], quantiles: Sequence[float]
) -> pd.DataFrame:
    """Traduce la salida de `NeuralForecast.predict` a la convencion del proyecto.

    Parameters
    ----------
    raw
        Salida cruda, con `unique_id`, `ds` y una columna por cuantil.
    suffixes
        Pares de `_quantile_suffixes`.
    quantiles
        Cuantiles pedidos por el motor.

    Returns
    -------
    pandas.DataFrame
        ``unique_id``, ``ds``, ``y_hat`` y una columna por cuantil pedido que
        el modelo haya estimado. `y_hat` es la mediana: el minimizador de la
        pinball al 0.5 es la mediana condicional, asi que es el pronostico
        puntual coherente con la perdida con la que se entreno —no una media
        que nadie ha optimizado.

    Raises
    ------
    ValueError
        Si la salida no trae la mediana: sin ella no hay pronostico puntual, y
        rellenarlo con otro cuantil seria publicar un sesgo como si fuese el
        centro de la distribucion.
    """
    result = raw[["unique_id", "ds"]].copy()

    median_suffix = next(
        (suffix for q, suffix in suffixes if math.isclose(q, 0.5, abs_tol=_QUANTILE_TOL)),
        None,
    )
    median_column = _resolve_column(raw, median_suffix) if median_suffix else None
    if median_column is None:
        raise ValueError(
            f"la salida de neuralforecast no trae la mediana (sufijo {median_suffix!r}); "
            f"columnas recibidas: {list(raw.columns)}"
        )
    result["y_hat"] = raw[median_column].to_numpy()

    for quantile in quantiles:
        suffix = next(
            (s for q, s in suffixes if math.isclose(q, quantile, abs_tol=_QUANTILE_TOL)), None
        )
        column = _resolve_column(raw, suffix) if suffix else None
        result[quantile_column(quantile)] = raw[column].to_numpy() if column is not None else np.nan
    return result


@dataclass(frozen=True, slots=True)
class _FittedNeuralForecast:
    """Ajuste comun a los tres adaptadores: un `NeuralForecast` ya entrenado."""

    model_id: ModelId
    cutoff: pd.Timestamp
    h: int
    fit_seconds: float
    freq: str
    input_size: int
    quantiles: tuple[float, ...]
    futr_exog: tuple[str, ...]
    quantile_suffixes: tuple[tuple[float, str], ...]
    nf: Any
    series_ids: tuple[str, ...]

    @property
    def n_params(self) -> int | None:
        """Parametros entrenables de la red, sumados sobre todos sus modulos."""
        net = self.nf.models[0]
        return int(sum(p.numel() for p in net.parameters() if p.requires_grad))

    @property
    def net(self) -> Any:
        """La red de neuralforecast subyacente, ya entrenada."""
        return self.nf.models[0]

    def _futr_frame(self, futr: FutrFrame | None) -> pd.DataFrame | None:
        """Trama futura en el dialecto de neuralforecast, o ``None`` si no hace falta.

        Raises
        ------
        ValueError
            Si el modelo declara exogenas futuras y la trama recibida no las
            trae todas.
        """
        if not self.futr_exog:
            return None
        if futr is None:
            raise ValueError(
                f"{self.model_id}: se declararon las exogenas futuras {list(self.futr_exog)} "
                "y no ha llegado ningun FutrFrame"
            )
        missing = set(self.futr_exog) - set(futr.df.columns)
        if missing:
            raise ValueError(
                f"{self.model_id}: faltan las exogenas futuras {sorted(missing)} en la "
                "trama recibida"
            )
        columns = ["unique_id", "ds", *self.futr_exog]
        return futr.df[columns].sort_values(["unique_id", "ds"]).reset_index(drop=True)

    def predict(
        self,
        futr: FutrFrame | None = None,
        *,
        quantiles: Sequence[float] = QUANTILES,
    ) -> pd.DataFrame:
        """Predice `h` pasos para todas las series del entrenamiento.

        Parameters
        ----------
        futr
            Exogenas futuras de la ventana. Obligatorio si el modelo se
            construyo con `futr_exog_list` no vacia.
        quantiles
            Cuantiles a devolver; los que el modelo no estimo salen `NaN`.

        Returns
        -------
        pandas.DataFrame
            ``unique_id``, ``ds``, ``y_hat`` y una columna por cuantil pedido.

        Raises
        ------
        ValueError
            Si faltan exogenas futuras declaradas, o si los instantes pedidos
            no son los `h` inmediatamente posteriores al cutoff del ajuste: la
            red emite exactamente ese tramo y alinearla con otro seria publicar
            un desfase como si fuese un resultado.
        """
        futr_df = self._futr_frame(futr)
        if futr is not None and not futr.df.empty:
            furthest = pd.Timestamp(futr.df["ds"].max())
            nearest = pd.Timestamp(futr.df["ds"].min())
            steps_far = len(pd.date_range(self.cutoff, furthest, freq=self.freq)) - 1
            steps_near = len(pd.date_range(self.cutoff, nearest, freq=self.freq)) - 1
            if steps_far > self.h or steps_near != 1:
                raise ValueError(
                    f"{self.model_id}: la red emite exactamente los pasos 1..{self.h} desde "
                    f"su cutoff ({self.cutoff}), y se le piden los pasos "
                    f"{steps_near}..{steps_far}. Fija refit_every=1 en el BacktestPlan para "
                    "que cada ventana reajuste con su propio cutoff, y usa gap=0."
                )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = self.nf.predict(futr_df=futr_df)
        return _assign_quantiles(raw, suffixes=self.quantile_suffixes, quantiles=quantiles)

    # ------------------------------------------------------------------ #
    # Interpretabilidad (solo TFT)
    # ------------------------------------------------------------------ #

    def variable_importance(self) -> pd.DataFrame:
        """Pesos de seleccion de variables del TFT, en formato largo.

        El TFT lleva una *variable selection network* por bloque —pasado y
        futuro— que emite un peso por variable y por instante, normalizado a
        uno sobre las variables. Aqui se promedia sobre el tiempo y sobre el
        lote, que es la lectura global "cuanto pesa cada variable" que quiere
        el dashboard; la version por instante vive en la propia matriz que
        devuelve neuralforecast y no se pierde, solo no se publica.

        Los pesos describen **la ultima pasada hacia delante** de la red: si se
        quiere que describan el tramo predicho, hay que llamar a `predict`
        antes (ver el docstring del modulo).

        Returns
        -------
        pandas.DataFrame
            ``kind`` (siempre ``"attention_variable"``), ``feature``, ``block``
            (``past`` / ``future``) y ``value``, ordenada de mayor a menor
            peso dentro de cada bloque. Columnas alineadas con la tabla
            `explanations` de docs/ARCHITECTURE.md §7.4.

        Raises
        ------
        NotImplementedError
            Si el modelo no es un TFT: NHITS y PatchTST no tienen seleccion de
            variables, y devolver una tabla vacia haria pasar por "sin senal"
            lo que en realidad es "sin mecanismo".
        ValueError
            Si la red no ha hecho ninguna pasada hacia delante todavia y por
            tanto no hay pesos que leer.
        """
        net = self.net
        if not hasattr(net, "feature_importances"):
            raise NotImplementedError(
                f"{self.model_id} no tiene red de seleccion de variables; solo el TFT la tiene"
            )
        if not getattr(net, "interpretability_params", None):
            raise ValueError(
                f"{self.model_id}: la red no ha hecho ninguna pasada hacia delante, asi "
                "que no hay pesos de interpretabilidad que leer; llama antes a predict()"
            )

        blocks = {
            "Past variable importance over time": "past",
            "Future variable importance over time": "future",
        }
        rows: list[dict[str, object]] = []
        for title, frame in net.feature_importances().items():
            block = blocks.get(title)
            if block is None:  # pragma: no cover  solo con exogenas estaticas
                continue
            for feature, value in frame.mean(axis=0).items():
                rows.append(
                    {
                        "kind": "attention_variable",
                        "feature": str(feature),
                        "block": block,
                        "value": float(value),
                    }
                )
        return pd.DataFrame(rows, columns=["kind", "feature", "block", "value"]).sort_values(
            ["block", "value"], ascending=[True, False], ignore_index=True
        )

    def temporal_attention(self) -> pd.DataFrame:
        """Atencion temporal media del TFT, por instante y con su `ds` absoluto.

        La matriz que expone neuralforecast es ``(L, L)`` con
        ``L = input_size + h``: cuanto atiende cada posicion a cada otra, ya
        promediada sobre el lote y las cabezas. Aqui se resume a lo que
        interesa en un dashboard de forecasting: **cuanta atencion reciben los
        instantes del contexto por parte de los pasos que se estan
        prediciendo**, promediando las filas del horizonte. Las posiciones se
        traducen a marcas de tiempo absolutas usando el cutoff del ajuste y la
        frecuencia del panel, de modo que la serie se puede dibujar contra el
        eje temporal sin que la app tenga que recalcular nada (A5).

        Returns
        -------
        pandas.DataFrame
            ``kind`` (siempre ``"attention_temporal"``), ``ds``, ``offset``
            (pasos relativos al cutoff: negativos en el contexto, positivos en
            el horizonte) y ``value``.

        Raises
        ------
        NotImplementedError
            Si el modelo no es un TFT.
        ValueError
            Si la red no ha hecho ninguna pasada hacia delante todavia.
        """
        net = self.net
        if not hasattr(net, "attention_weights"):
            raise NotImplementedError(
                f"{self.model_id} no expone atencion temporal; solo el TFT la tiene"
            )
        if not getattr(net, "interpretability_params", None):
            raise ValueError(
                f"{self.model_id}: la red no ha hecho ninguna pasada hacia delante, asi "
                "que no hay pesos de interpretabilidad que leer; llama antes a predict()"
            )

        attention = np.asarray(net.attention_weights(), dtype=float)
        total = attention.shape[-1]
        # Las ultimas `h` filas son los pasos predichos: su media dice a que
        # instantes mira el modelo para producir el horizonte.
        received = attention[-self.h :, :].mean(axis=0)

        context_length = total - self.h
        offsets = np.arange(total) - context_length + 1
        grid = pd.date_range(
            self.cutoff - (context_length - 1) * pd.tseries.frequencies.to_offset(self.freq),
            periods=total,
            freq=self.freq,
        )
        return pd.DataFrame(
            {
                "kind": "attention_temporal",
                "ds": grid,
                "offset": offsets.astype(int),
                "value": received.astype(float),
            }
        )


def _fit_neuralforecast(
    train: Panel,
    h: int,
    *,
    model_builder: Any,
    model_id: ModelId,
    input_size: int,
    quantiles: tuple[float, ...],
    use_futr_exog: bool,
    val_size: int | None,
) -> _FittedNeuralForecast:
    """Rutina de ajuste comun a los tres adaptadores.

    Parameters
    ----------
    train
        Panel de entrenamiento, ya recortado por el motor a ``ds <= cutoff``.
    h
        Horizonte del plan.
    model_builder
        Funcion ``(h, input_size, futr_exog_list, loss) -> modelo`` que
        construye la red concreta. Es lo unico que distingue a NHITS de TFT y
        de PatchTST en este camino.
    model_id
        Identificador de chronolab.
    input_size
        Contexto en pasos.
    quantiles
        Rejilla del run.
    use_futr_exog
        Si se pasan las columnas `futr_exog` del panel a la red.
    val_size
        Instantes finales del train reservados a validacion. ``None`` usa `h`,
        el minimo con sentido: una ventana de validacion del tamano del
        horizonte que se va a predecir.

    Returns
    -------
    _FittedNeuralForecast
    """
    neural_forecast_cls, _, _, _, mq_loss_cls = _require_neuralforecast()

    futr_exog = tuple(train.spec.futr_exog) if use_futr_exog else ()
    loss = mq_loss_cls(quantiles=list(quantiles))
    model = model_builder(h, input_size, list(futr_exog), loss)

    frame = train.df[["unique_id", "ds", train.spec.target, *futr_exog]].copy()
    if train.spec.target != "y":
        frame = frame.rename(columns={train.spec.target: "y"})
    # neuralforecast no admite `NaN` en la objetivo; los huecos que el
    # invariante I3 conserva se rellenan hacia atras (`ffill`), la unica
    # imputacion que admite la barrera L10 porque solo mira al pasado.
    frame["y"] = frame.groupby("unique_id", sort=False)["y"].ffill().bfill()

    started = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nf = neural_forecast_cls(models=[model], freq=train.spec.freq)
        nf.fit(frame, val_size=h if val_size is None else val_size)
    fit_seconds = perf_counter() - started

    return _FittedNeuralForecast(
        model_id=model_id,
        cutoff=train.last_ds,
        h=h,
        fit_seconds=fit_seconds,
        freq=train.spec.freq,
        input_size=input_size,
        quantiles=quantiles,
        futr_exog=futr_exog,
        quantile_suffixes=_quantile_suffixes(loss),
        nf=nf,
        series_ids=train.ids(),
    )


@dataclass(frozen=True)
class _NeuralForecastBase:
    """Configuracion comun a los tres adaptadores de neuralforecast.

    Parameters
    ----------
    input_size
        Contexto en pasos. 168 por defecto: una semana horaria.
    max_steps
        Tope duro de pasos de optimizacion. Es el presupuesto que hace viable
        el backtest completo; subirlo es legitimo y queda medido en
        `fit_seconds`.
    val_check_steps
        Cada cuantos pasos se evalua la validacion (y por tanto, cada cuanto
        puede dispararse el early stopping).
    early_stop_patience_steps
        Evaluaciones de validacion sin mejora antes de parar. ``-1`` desactiva
        el early stopping.
    learning_rate
        Tasa inicial del optimizador.
    batch_size, windows_batch_size
        Series por lote y ventanas por lote de entrenamiento. El valor por
        defecto de `windows_batch_size` en neuralforecast es ``1024``; aqui es
        ``128`` porque es el parametro que mas manda en el coste por paso sobre
        CPU —cada paso propaga ese numero de ventanas de `input_size` puntos— y
        con el se pasa de decenas de minutos por ajuste a decenas de segundos
        sobre el panel curado del proyecto.
    scaler_type
        Normalizacion por ventana de entrada (``"standard"``, ``"robust"``,
        ``"identity"``...). Ver el docstring del modulo.
    quantiles
        Rejilla de cuantiles con la que se entrena `MQLoss`.
    use_futr_exog
        Pasar las columnas `futr_exog` del panel a la red.
    val_size
        Instantes finales del train para validacion; ``None`` usa `h`.
    accelerator
        ``"cpu"``, ``"gpu"`` o ``"auto"``.
    enable_progress_bar
        Barra de progreso de Lightning.
    seed
        Semilla (`random_seed` de neuralforecast).

    Raises
    ------
    ValueError
        Si el presupuesto es incoherente o la rejilla de cuantiles no es
        estrictamente creciente y no incluye la mediana.
    """

    input_size: int = 168
    max_steps: int = 200
    val_check_steps: int = 50
    early_stop_patience_steps: int = 3
    learning_rate: float = 1e-3
    batch_size: int = 32
    windows_batch_size: int = 128
    scaler_type: str = "standard"
    quantiles: tuple[float, ...] = QUANTILES
    use_futr_exog: bool = True
    val_size: int | None = None
    accelerator: Literal["cpu", "gpu", "auto"] = "cpu"
    enable_progress_bar: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        """Valida contexto, presupuesto y rejilla de cuantiles."""
        if self.input_size < 1:
            raise ValueError(f"input_size debe ser >= 1: {self.input_size}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps debe ser >= 1: {self.max_steps}")
        if self.val_check_steps < 1:
            raise ValueError(f"val_check_steps debe ser >= 1: {self.val_check_steps}")
        if list(self.quantiles) != sorted(set(self.quantiles)):
            raise ValueError(f"quantiles debe ser estrictamente creciente: {self.quantiles}")
        if not any(math.isclose(q, 0.5, abs_tol=1e-9) for q in self.quantiles):
            raise ValueError(
                f"la rejilla debe incluir la mediana (0.5) como pronostico puntual: "
                f"{self.quantiles}"
            )
        for quantile in self.quantiles:
            quantile_column(quantile)

    @property
    def requires(self) -> ModelRequirements:
        """Contexto minimo, cuantiles nativos y un unico ajuste por run.

        `min_context` es ``input_size + h`` en la practica; como `h` no se
        conoce hasta `fit`, se declara el contexto mas un paso y la propia
        libreria falla ruidosamente si el tramo no da para ninguna ventana.
        """
        return ModelRequirements(
            needs_futr_exog=self.use_futr_exog,
            supports_quantiles=True,
            min_context=self.input_size + 1,
            refit_cost="expensive",
        )

    def _common_kwargs(self) -> dict[str, Any]:
        """Argumentos que comparten los constructores de los tres modelos."""
        return {
            "max_steps": self.max_steps,
            "val_check_steps": self.val_check_steps,
            "early_stop_patience_steps": self.early_stop_patience_steps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "windows_batch_size": self.windows_batch_size,
            "scaler_type": self.scaler_type,
            "random_seed": self.seed,
            "alias": _ALIAS,
            **_trainer_kwargs(
                accelerator=self.accelerator, enable_progress_bar=self.enable_progress_bar
            ),
        }


@dataclass(frozen=True)
class NHITSForecaster(_NeuralForecastBase):
    """N-HiTS: interpolacion jerarquica multi-tasa sobre bloques MLP.

    Parametros comunes en `_NeuralForecastBase`. Los propios reducen el tamano
    por defecto de la libreria (tres bloques de ``[512, 512]``, ~2.5 M
    parametros) a algo proporcionado al panel curado del proyecto: con
    `mlp_units` de ``[128, 128]`` la red baja a decenas de miles de parametros
    y el ajuste por ventana cae de minutos a segundos en CPU, que es lo que
    hace viable el backtest completo.

    Parameters
    ----------
    mlp_units
        Unidades de cada bloque MLP.
    n_pool_kernel_size, n_freq_downsample
        Tasas de submuestreo de cada pila, el mecanismo por el que N-HiTS
        separa frecuencias.
    dropout_prob_theta
        Dropout de los coeficientes de interpolacion.
    model_id
        Identificador del modelo.
    """

    mlp_units: tuple[tuple[int, int], ...] = ((128, 128), (128, 128), (128, 128))
    n_pool_kernel_size: tuple[int, ...] = (8, 4, 1)
    n_freq_downsample: tuple[int, ...] = (24, 12, 1)
    dropout_prob_theta: float = 0.0
    model_id: ModelId = _NHITS_ID

    def fit(self, train: Panel, *, h: int) -> _FittedNeuralForecast:
        """Construye la red con `h` y las exogenas del panel, y la ajusta."""
        _, nhits_cls, _, _, _ = _require_neuralforecast()

        def build(horizon: int, context: int, futr: list[str], loss: Any) -> Any:
            return nhits_cls(
                h=horizon,
                input_size=context,
                futr_exog_list=futr,
                loss=loss,
                mlp_units=[list(units) for units in self.mlp_units],
                n_pool_kernel_size=list(self.n_pool_kernel_size),
                n_freq_downsample=list(self.n_freq_downsample),
                dropout_prob_theta=self.dropout_prob_theta,
                **self._common_kwargs(),
            )

        return _fit_neuralforecast(
            train,
            h,
            model_builder=build,
            model_id=self.model_id,
            input_size=self.input_size,
            quantiles=self.quantiles,
            use_futr_exog=self.use_futr_exog,
            val_size=self.val_size,
        )


@dataclass(frozen=True)
class TFTForecaster(_NeuralForecastBase):
    """Temporal Fusion Transformer, el unico de los tres con interpretabilidad nativa.

    Sobre el ajuste devuelto (`_FittedNeuralForecast`) quedan disponibles
    `variable_importance()` y `temporal_attention()`, ambas despues de llamar a
    `predict`. Son la fuente de la pagina de explicabilidad de la app.

    Parameters
    ----------
    hidden_size
        Anchura de los bloques GRN y del estado del LSTM interno. 64 por
        defecto frente a los 128 de la libreria: el TFT es, con diferencia, el
        mas caro de los tres, y el coste crece cuadraticamente con esta cifra.
    n_head
        Cabezas de la atencion multi-cabeza.
    attn_dropout, dropout
        Dropout de la atencion y del resto de la red.
    model_id
        Identificador del modelo.
    """

    hidden_size: int = 64
    n_head: int = 4
    attn_dropout: float = 0.0
    dropout: float = 0.1
    model_id: ModelId = _TFT_ID

    def fit(self, train: Panel, *, h: int) -> _FittedNeuralForecast:
        """Construye el TFT con `h` y las exogenas del panel, y lo ajusta."""
        _, _, tft_cls, _, _ = _require_neuralforecast()

        def build(horizon: int, context: int, futr: list[str], loss: Any) -> Any:
            return tft_cls(
                h=horizon,
                input_size=context,
                futr_exog_list=futr,
                loss=loss,
                hidden_size=self.hidden_size,
                n_head=self.n_head,
                attn_dropout=self.attn_dropout,
                dropout=self.dropout,
                **self._common_kwargs(),
            )

        return _fit_neuralforecast(
            train,
            h,
            model_builder=build,
            model_id=self.model_id,
            input_size=self.input_size,
            quantiles=self.quantiles,
            use_futr_exog=self.use_futr_exog,
            val_size=self.val_size,
        )


@dataclass(frozen=True)
class PatchTSTForecaster(_NeuralForecastBase):
    """PatchTST: transformer sobre parches de la serie, con normalizacion reversible.

    PatchTST **no admite exogenas futuras** en neuralforecast: es un modelo
    puramente univariado sobre parches de la propia serie. `use_futr_exog` se
    fuerza a `False` en la construccion y `requires.needs_futr_exog` queda en
    `False`, en lugar de aceptar la configuracion y descartarla en silencio.

    Parameters
    ----------
    patch_len, stride
        Longitud del parche y salto entre parches consecutivos.
    encoder_layers, n_heads, hidden_size, linear_hidden_size
        Tamano del transformer.
    dropout
        Dropout comun de la red.
    model_id
        Identificador del modelo.

    Raises
    ------
    ValueError
        Si se construye con ``use_futr_exog=True``.
    """

    patch_len: int = 24
    stride: int = 12
    encoder_layers: int = 2
    n_heads: int = 4
    hidden_size: int = 64
    linear_hidden_size: int = 128
    dropout: float = 0.1
    model_id: ModelId = _PATCHTST_ID

    def __post_init__(self) -> None:
        """Valida lo comun y rechaza una configuracion con exogenas futuras."""
        super().__post_init__()
        if self.use_futr_exog:
            raise ValueError(
                "PatchTST es univariado y no admite exogenas futuras: construyelo con "
                "use_futr_exog=False en lugar de dejar que se descarten en silencio"
            )

    def fit(self, train: Panel, *, h: int) -> _FittedNeuralForecast:
        """Construye el PatchTST con `h` y lo ajusta; nunca recibe exogenas."""
        _, _, _, patchtst_cls, _ = _require_neuralforecast()

        def build(horizon: int, context: int, futr: list[str], loss: Any) -> Any:
            return patchtst_cls(
                h=horizon,
                input_size=context,
                loss=loss,
                patch_len=self.patch_len,
                stride=self.stride,
                encoder_layers=self.encoder_layers,
                n_heads=self.n_heads,
                hidden_size=self.hidden_size,
                linear_hidden_size=self.linear_hidden_size,
                dropout=self.dropout,
                **self._common_kwargs(),
            )

        return _fit_neuralforecast(
            train,
            h,
            model_builder=build,
            model_id=self.model_id,
            input_size=self.input_size,
            quantiles=self.quantiles,
            use_futr_exog=False,
            val_size=self.val_size,
        )
