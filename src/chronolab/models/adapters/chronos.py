"""Adaptador zero-shot de Chronos-2 / Chronos-Bolt.

No entrena: `fit` captura el contexto y fija el cutoff. Es el caso degenerado
que valida el diseno del protocolo, no un parche (docs/ARCHITECTURE.md §5.2,
tabla "como encaja cada backend").

Solo contexto, nunca exogenas
-------------------------------
Chronos-Bolt y Chronos-2 son modelos fundacionales univariados: se les da la
historia de la serie y devuelven cuantiles, sin covariables. Por eso este
adaptador declara ``needs_futr_exog=False``, ``uses_hist_exog=False`` y
``uses_static_exog=False`` sin condicional: no hay forma de que el modelo
aproveche una exogena aunque el panel la declare, y aceptarla en silencio
seria hacer pasar por soportado algo que el backend no puede usar.

Cuantiles reales, no una rejilla fija — con un matiz en las colas
----------------------------------------------------------------------
A diferencia de `neuralforecast` (rejilla de cuantiles fijada en el
entrenamiento; lo que no se entrena sale `NaN`), `pipeline.predict_quantiles`
acepta **cualquier** nivel en ``(0, 1)`` y nunca devuelve `NaN` por "cuantil no
entrenado". Dentro del rango nativo del modelo (``[0.1, 0.9]`` en Chronos-Bolt)
interpola de verdad con `torch.quantile`. **Fuera** de ese rango —los dos
extremos de la rejilla canonica del proyecto, ``0.025`` y ``0.975``— no
extrapola: clampa al limite entrenado mas cercano (``0.1`` y ``0.9``), con un
aviso de la propia libreria la primera vez que ocurre. El numero sigue siendo
real y monotono, pero la cola queda mas estrecha de lo que seria con un modelo
calibrado a esos niveles; es una limitacion del modelo pre-entrenado, no del
adaptador, y se documenta aqui en vez de dejar que se descubra leyendo
`coverage_025`/`coverage_975` en el leaderboard sin saber por que. Se pide
siempre la mediana ademas de lo solicitado —aunque `predict(quantiles=...)` no
la incluya— porque `y_hat` tiene que ser un pronostico puntual coherente con lo
que devuelve el modelo, no un cuantil que nadie pidio.

Presupuesto de horizonte: el mismo contrato que el LSTM y neuralforecast
----------------------------------------------------------------------------
Chronos, al no tener una cabeza de tamano fijo, podria en principio predecir
cualquier horizonte desde su propio cutoff. Aun asi este adaptador exige que lo
pedido sean exactamente los pasos ``1..h`` desde el cutoff del ajuste, con el
mismo mensaje de error que `models/adapters/torch_lstm.py` y
`models/adapters/neuralforecast.py`: no por una limitacion arquitectonica, sino
para que los tres adaptadores fallen igual ante la misma situacion (reutilizar
un ajuste con ``refit_every > 1``, o un ``gap > 0``) y un run que mezcla
modelos no tenga que recordar cual tolera que caso y cual no.

Descarga de pesos: cacheada, con fallback claro y sin arrastrar al resto
----------------------------------------------------------------------------
`BaseChronosPipeline.from_pretrained` usa la cache de disco de
`huggingface_hub` (``~/.cache/huggingface``) de forma transparente: la segunda
vez que se pide el mismo `pretrained_model_name_or_path` no hay descarga. Ese
pipeline, ademas, se cachea **en memoria** por `(pretrained, device)` con
`functools.lru_cache`, para que un backtest de muchas ventanas no vuelva a
cargar los pesos en cada `fit` — coherente con que `fit_seconds` deba salir
cerca de cero (tabla del §5.2): la carga del modelo no es parte del ajuste, es
un coste de arranque que se paga una vez por proceso.

Si no hay cache local ni red, `from_pretrained` lanza una subclase de
`OSError` (`huggingface_hub.errors.LocalEntryNotFoundError` o un error de
conexion de `requests`, ambas `OSError` en Python 3). Se traduce aqui a
`FoundationModelUnavailable`, con instrucciones de que hacer, en lugar de
dejar escapar una excepcion de una libreria de terceros que no dice nada
sobre la causa. El resto del proyecto (`evaluation`, `artifacts`, `app`) no
importa este modulo ni depende de que la descarga funcione: si Chronos no
esta disponible, ese modelo falla como cualquier otro (`status="failed"` en
`model_runs`, A6), y el resto del run sigue.

Import perezoso
----------------
`chronos-forecasting` y `torch` viven en el extra `deep` (D20). El modulo debe
poder **importarse** sin ellos —lo exige `tests/unit/test_module_tree.py` en
el entorno por defecto de CI, que hace `uv sync` a secas— asi que el import
real vive en `_require_chronos`, llamada solo al ajustar o predecir.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd

from chronolab.errors import FoundationModelUnavailable
from chronolab.models.protocols import QUANTILES, ModelRequirements, quantile_column
from chronolab.panel import FutrFrame, Panel
from chronolab.types import ModelId

__all__ = ["ChronosForecaster"]

_CHRONOS_ID = ModelId("chronos-bolt-small")
"""Identificador por defecto. Coincide con el `pretrained_model_name_or_path`
por defecto; un run que compare varios tamanos debe pasar `model_id` explicito,
igual que hace `NHITSForecaster` para dos configuraciones de la misma clase."""

_DEFAULT_PRETRAINED = "amazon/chronos-bolt-small"
"""Variante pequena de Chronos-Bolt: corre en CPU en segundos, no en minutos.
`pretrained_model_name_or_path` es justo el parametro que ajusta el tamano
—``chronos-bolt-tiny``/``mini``/``small``/``base``, o un repo de Chronos-2 si el
tamano lo permite— sin que el adaptador necesite saber de tamanos."""


def _require_chronos() -> tuple[Any, Any]:
    """Importa `chronos` y `torch` bajo demanda, con un mensaje util si faltan.

    Returns
    -------
    tuple
        ``(BaseChronosPipeline, torch)``. Tipados como `Any` por la cuarentena
        D16 (docs/ARCHITECTURE.md): lo que sale de este modulo hacia el resto
        del proyecto es siempre un `pandas.DataFrame` con el esquema del
        protocolo, nunca un objeto de `chronos` o de `torch`.

    Raises
    ------
    ImportError
        Si `chronos-forecasting` o `torch` no estan instalados.
    """
    try:
        import torch
        from chronos import BaseChronosPipeline
    except ImportError as exc:  # pragma: no cover  ejercitado solo sin el extra deep
        raise ImportError(
            "chronolab.models.adapters.chronos necesita el extra 'deep': `uv sync --extra deep`."
        ) from exc
    return BaseChronosPipeline, torch


@lru_cache(maxsize=8)
def _load_pipeline(pretrained: str, device: str) -> Any:
    """Carga un pipeline de Chronos, cacheado en memoria por `(pretrained, device)`.

    La cache de disco de los pesos la gestiona `huggingface_hub`; esta cache en
    memoria evita releer y reconstruir el modulo de `torch` en cada ventana de
    un backtest, que es lo que mantiene `fit_seconds` cerca de cero (ver el
    docstring del modulo).

    Parameters
    ----------
    pretrained
        Identificador del repositorio de Hugging Face.
    device
        ``"cpu"`` o ``"cuda"``.

    Returns
    -------
    Any
        Pipeline ya cargado, movido a `device` y en modo evaluacion.

    Raises
    ------
    FoundationModelUnavailable
        Si no hay una copia en la cache local de Hugging Face y tampoco se
        puede alcanzar el Hub para descargarla.
    """
    pipeline_cls, _torch = _require_chronos()
    try:
        pipeline = pipeline_cls.from_pretrained(pretrained, dtype="float32")
    except OSError as exc:
        raise FoundationModelUnavailable(
            f"no se han podido obtener los pesos de Chronos '{pretrained}': sin red y "
            "sin cache local de Hugging Face (~/.cache/huggingface). Descarga el modelo "
            f"una vez con conexion (`huggingface-cli download {pretrained}`) o comparte "
            "la cache del equipo; el resto del proyecto no depende de esta descarga."
        ) from exc
    pipeline.inner_model.to(device)
    pipeline.inner_model.eval()
    return pipeline


@dataclass(frozen=True, slots=True)
class _FittedChronos:
    """Contexto capturado hasta un cutoff, listo para que el pipeline lo continue."""

    model_id: ModelId
    cutoff: pd.Timestamp
    h: int
    fit_seconds: float
    freq: str
    context: dict[str, np.ndarray]
    pretrained: str
    device: str

    @property
    def n_params(self) -> int | None:
        """Parametros del pipeline pre-entrenado, sumados sobre todo el modelo.

        No son parametros ajustados en este `fit` —Chronos no entrena— pero
        siguen siendo la medida de tamano que el eje precision-coste del
        leaderboard necesita: sin ella, un Chronos-Bolt-tiny de unos pocos
        millones de parametros y un Chronos-Bolt-base de cientos de millones se
        leerian igual en una tabla ordenada por MASE.
        """
        pipeline = _load_pipeline(self.pretrained, self.device)
        return int(sum(p.numel() for p in pipeline.inner_model.parameters()))

    def _horizon(self, futr: FutrFrame | None) -> dict[str, list[pd.Timestamp]]:
        """Instantes a predecir por serie: los del `FutrFrame` si lo hay, o `cutoff + h*freq`."""
        if futr is not None and not futr.df.empty:
            return {
                str(uid): group.sort_values("ds")["ds"].tolist()
                for uid, group in futr.df.groupby("unique_id", sort=False)
            }
        grid = pd.date_range(self.cutoff, periods=self.h + 1, freq=self.freq)[1:].tolist()
        return dict.fromkeys(self.context, grid)

    def _assert_within_horizon(self, instants: dict[str, list[pd.Timestamp]]) -> None:
        """Comprueba que lo pedido son exactamente los pasos ``1..h`` desde el cutoff.

        Chronos no tiene una cabeza de tamano fijo que imponga este limite; se
        exige de todos modos para que este adaptador falle en el mismo caso, y
        con el mismo mensaje, que `models/adapters/torch_lstm.py` y
        `models/adapters/neuralforecast.py` (ver el docstring del modulo).

        Raises
        ------
        ValueError
            Si el instante mas lejano cae a mas de `h` pasos del cutoff, o si
            el mas cercano no es el paso inmediatamente siguiente.
        """
        furthest = max(ds for series in instants.values() for ds in series)
        nearest = min(ds for series in instants.values() for ds in series)
        steps_to_furthest = len(pd.date_range(self.cutoff, furthest, freq=self.freq)) - 1
        steps_to_nearest = len(pd.date_range(self.cutoff, nearest, freq=self.freq)) - 1

        if steps_to_furthest > self.h or steps_to_nearest != 1:
            raise ValueError(
                f"{self.model_id}: se le piden los pasos {steps_to_nearest}..{steps_to_furthest} "
                f"desde su cutoff ({self.cutoff}), y solo se admiten los pasos 1..{self.h} de este "
                "ajuste. Fija refit_every=1 en el BacktestPlan para que cada ventana reajuste con "
                "su propio cutoff, y usa gap=0 con este modelo."
            )

    def predict(
        self,
        futr: FutrFrame | None = None,
        *,
        quantiles: Sequence[float] = QUANTILES,
    ) -> pd.DataFrame:
        """Continua el contexto capturado en `fit` los `h` pasos siguientes.

        Parameters
        ----------
        futr
            Ignorado como fuente de exogenas —Chronos es univariado— pero se
            usa, si llega, para saber a que instantes exactos predecir: el
            motor lo pasa aunque este modelo no lo necesite (`requires` no
            declara `needs_futr_exog`), porque puede haber un `FutrProvider`
            en el run para otros modelos.
        quantiles
            Cuantiles a devolver. A diferencia de los adaptadores con una
            rejilla entrenada, nunca salen como `NaN` por no estar en un
            conjunto fijo; los que caen fuera del rango nativo del modelo
            quedan clampados al limite mas cercano, no extrapolados (ver el
            docstring del modulo).

        Returns
        -------
        pandas.DataFrame
            ``unique_id``, ``ds``, ``y_hat`` (la mediana) y una columna por
            cuantil pedido.

        Raises
        ------
        ValueError
            Si los instantes pedidos no son exactamente los pasos ``1..h``
            desde el cutoff de este ajuste (ver `_assert_within_horizon`).
        FoundationModelUnavailable
            Si no hay red ni cache local de los pesos.
        """
        instants = self._horizon(futr)
        for uid in self.context:
            found = len(instants.get(uid, []))
            if found != self.h:
                raise ValueError(
                    f"{self.model_id}: se esperaban {self.h} instantes a predecir para "
                    f"'{uid}' y se han encontrado {found}"
                )
        self._assert_within_horizon(instants)

        _pipeline_cls, torch = _require_chronos()
        pipeline = _load_pipeline(self.pretrained, self.device)

        ids = list(self.context)
        levels = sorted({0.5, *quantiles})
        inputs = [torch.tensor(self.context[uid], dtype=torch.float32) for uid in ids]
        quantile_tensor, _mean = pipeline.predict_quantiles(
            inputs, prediction_length=self.h, quantile_levels=levels
        )
        values = quantile_tensor.numpy()  # (n_series, h, n_levels)
        median_index = levels.index(0.5)

        parts: list[pd.DataFrame] = []
        for position, uid in enumerate(ids):
            frame = pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": instants[uid],
                    "y_hat": values[position, :, median_index],
                }
            )
            for quantile in quantiles:
                frame[quantile_column(quantile)] = values[position, :, levels.index(quantile)]
            parts.append(frame)
        return pd.concat(parts, ignore_index=True)


@dataclass(frozen=True)
class ChronosForecaster:
    """Envoltorio zero-shot de Chronos-Bolt / Chronos-2, sin ningun ajuste local.

    `fit` no optimiza nada: recorta el contexto de cada serie a
    `context_length` puntos y guarda el cutoff. Toda la capacidad predictiva
    viene de los pesos pre-entrenados que descarga (y cachea) Hugging Face. Es
    el caso degenerado que valida `models.protocols.Forecaster`: "ajustar" es
    "fijar la frontera de informacion", nada mas, y ese es el unico requisito
    que comparten los seis backends del proyecto (docs/ARCHITECTURE.md §5.2).

    Parameters
    ----------
    pretrained_model_name_or_path
        Repositorio de Hugging Face. Es el parametro que fija el tamano del
        modelo: ``"amazon/chronos-bolt-tiny"`` (~9M parametros) hasta
        ``"amazon/chronos-bolt-base"`` (~205M), o un repo de Chronos-2 si el
        tamano y el hardware lo permiten. El valor por defecto es la variante
        pequena que corre en CPU en segundos.
    context_length
        Puntos finales de cada serie que se conservan como contexto.
        ``None`` deja que el propio pipeline recorte al maximo que soporte su
        arquitectura (2048 pasos en Chronos-Bolt). Un valor explicito acota la
        memoria y el tiempo de inferencia en CPU sobre paneles largos.
    device
        ``"cpu"`` o ``"cuda"``. Por defecto CPU: es el requisito del hito, no
        una limitacion del adaptador.
    min_context
        Pasos minimos de entrenamiento para que una ventana no se salte. Un
        modelo zero-shot no necesita mucho contexto para producir *algo*, asi
        que el valor por defecto es deliberadamente bajo.
    model_id
        Identificador del modelo. Un run que compare varios tamanos de Chronos
        a la vez debe pasar uno distinto por instancia, igual que cualquier
        otro adaptador con varias configuraciones de la misma clase.

    Raises
    ------
    ValueError
        Si `context_length` o `min_context` no son positivos.
    """

    pretrained_model_name_or_path: str = _DEFAULT_PRETRAINED
    context_length: int | None = 512
    device: Literal["cpu", "cuda"] = "cpu"
    min_context: int = 1
    model_id: ModelId = _CHRONOS_ID

    def __post_init__(self) -> None:
        """Valida el contexto y el minimo de entrenamiento."""
        if self.context_length is not None and self.context_length < 1:
            raise ValueError(f"context_length debe ser >= 1 o None: {self.context_length}")
        if self.min_context < 1:
            raise ValueError(f"min_context debe ser >= 1: {self.min_context}")

    @property
    def requires(self) -> ModelRequirements:
        """Zero-shot, sin exogenas, con cuantiles reales y un unico coste de arranque.

        `refit_cost="free"`: reajustar solo vuelve a recortar el contexto, no
        cuesta nada frente a una carga de red o de GPU. Con la politica de
        refit por defecto (`BacktestPlan.refit_every_for`) eso significa un
        ajuste por ventana, que es justo lo que hace falta para que cada
        prediccion use el contexto mas reciente disponible.
        """
        return ModelRequirements(
            supports_quantiles=True,
            min_context=self.min_context,
            is_zero_shot=True,
            refit_cost="free",
        )

    def fit(self, train: Panel, *, h: int) -> _FittedChronos:
        """Recorta el contexto de cada serie y fija el cutoff. No hay optimizacion.

        Parameters
        ----------
        train
            Rebanada de entrenamiento de la ventana.
        h
            Horizonte en pasos, fijo para la vida del objeto ajustado.

        Returns
        -------
        _FittedChronos

        Raises
        ------
        ValueError
            Si alguna serie del entrenamiento no tiene ninguna observacion no
            nula: no hay contexto que darle al modelo.

        Notes
        -----
        `fit_seconds` mide solo este recorte, no la carga de los pesos: esa
        carga esta cacheada en memoria (`_load_pipeline`) y se paga una vez
        por proceso, no una vez por ventana. Es coherente con la fila de
        Chronos en docs/ARCHITECTURE.md §5.2: "`fit_seconds ≈ 0` y se reporta
        como tal".
        """
        started = perf_counter()
        target = train.spec.target
        frame = train.df[["unique_id", "ds", target]].rename(columns={target: "y"})
        # Los huecos del panel (invariante I3) llegan como `NaN` explicito. Se
        # rellenan hacia atras y, si aun quedan al principio de la serie, hacia
        # adelante: ambos rellenos usan solo valores que ya estaban dentro de
        # `train`, nunca informacion posterior al cutoff (barrera L10).
        frame["y"] = frame.groupby("unique_id", sort=False)["y"].ffill().bfill()

        context: dict[str, np.ndarray] = {}
        for uid, group in frame.groupby("unique_id", sort=False):
            values = group.sort_values("ds")["y"].to_numpy(dtype="float32")
            if self.context_length is not None and len(values) > self.context_length:
                values = values[-self.context_length :]
            if not np.isfinite(values).any():
                raise ValueError(
                    f"{self.model_id}: la serie '{uid}' no tiene ninguna observacion valida "
                    "en el entrenamiento; no hay contexto que darle al modelo"
                )
            context[str(uid)] = values

        return _FittedChronos(
            model_id=self.model_id,
            cutoff=train.last_ds,
            h=h,
            fit_seconds=perf_counter() - started,
            freq=train.spec.freq,
            context=context,
            pretrained=self.pretrained_model_name_or_path,
            device=self.device,
        )
