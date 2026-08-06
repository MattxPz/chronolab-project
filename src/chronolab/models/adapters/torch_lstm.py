"""Adaptador del LSTM propio en PyTorch al protocolo `Forecaster`.

Une las tres piezas de `chronolab.models.torch` —escalado y ventanas
(`dataset`), red (`modules`), bucle de entrenamiento (`trainer`)— detras del
mismo contrato que cumplen statsforecast, mlforecast, Prophet y
neuralforecast. Todo el preprocesado dependiente de datos ocurre **dentro de
`fit`**, sobre el `Panel` que el motor ya recorto a ``ds <= cutoff``: no hay
etapa global de preprocesado en el proyecto, y ese hueco es lo que hace
estructuralmente imposible ajustar el escalador con datos futuros
(docs/ARCHITECTURE.md, fuga L2).

Exogenas conocidas a futuro
----------------------------
A diferencia de `models/adapters/mlforecast.py` —que reconstruye el tramo
futuro desde la propia historia y por eso nunca lee el `FutrFrame`—, este
adaptador **si** consume el `FutrFrame`: sus exogenas futuras entran en la
cabeza de la red (ver el docstring de `models/torch/modules.py`), asi que para
predecir necesita sus valores reales en el tramo evaluado. Por eso declara
``needs_futr_exog=True`` cuando el panel trae `futr_exog`, y el motor aborta
el run si no hay `FutrProvider` que se las de. El vintage de esas exogenas lo
decide el proveedor, no este modulo: aqui llegan ya con la semantica temporal
que el run declaro.

Las columnas `hist_exog` **no** se usan. Podrian entrar en el contexto —son
pasado— pero no en el tramo futuro, y sostener dos conjuntos distintos de
canales entre encoder y cabeza complica el modulo sin que el dataset del
proyecto lo justifique. Queda declarado en `ModelRequirements.uses_hist_exog`,
que es `False`, en lugar de silenciado.

Prediccion directa multi-paso
------------------------------
La red proyecta el horizonte completo de una vez, asi que ``supports_recursive``
es `False` y no hay realimentacion de predicciones propias. El coste es que la
cabeza crece con `h`; el beneficio es que el error no se acumula a lo largo
del horizonte y que la comparacion con la estrategia `direct` de mlforecast
mide arquitectura y no estrategia.

Reutilizar un ajuste entre ventanas: por que falla y por que debe fallar
-------------------------------------------------------------------------
`refit_cost="expensive"`, asi que la politica por defecto del motor es un
unico ajuste por run. Pero la cabeza de esta red emite **exactamente** los `h`
pasos siguientes al final de su contexto: no hay forma de estirarla. Al
reutilizar el ajuste en una ventana posterior, los instantes evaluados caen a
mas de `h` pasos del cutoff con el que se entreno, y las `h` salidas se
alinearian con los instantes equivocados —un desfase silencioso que se leeria
como un modelo mediocre en vez de como lo que es, un modelo mal usado.
`predict` lo detecta y lanza; el motor lo registra con ``status="failed"``
(A6) en lugar de publicar el numero desfasado. Es el mismo criterio, y el
mismo mensaje, que `models/adapters/statsforecast.py` aplica a sus intervalos
conformales: un run que quiera este modelo fija ``refit_every=1`` y paga el
ajuste por ventana.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from chronolab.errors import MissingFutrExog
from chronolab.models.protocols import QUANTILES, ModelRequirements, quantile_column
from chronolab.models.torch.dataset import SeriesScaler, build_windows, context_matrix
from chronolab.models.torch.modules import build_lstm_net, count_parameters
from chronolab.models.torch.trainer import (
    TrainConfig,
    TrainReport,
    predict_batch,
    train_lstm,
)
from chronolab.panel import FutrFrame, Panel
from chronolab.types import ModelId

__all__ = ["LSTMForecaster"]

_LSTM_ID = ModelId("lstm")


@dataclass(frozen=True, slots=True)
class _FittedLSTM:
    """Red entrenada hasta un cutoff, con su escalador y su informe de ajuste."""

    model_id: ModelId
    cutoff: pd.Timestamp
    h: int
    fit_seconds: float
    freq: str
    input_size: int
    quantiles: tuple[float, ...]
    exog_columns: tuple[str, ...]
    scaler: SeriesScaler
    device: str
    net: Any
    train_report: TrainReport
    train_panel: Panel

    @property
    def n_params(self) -> int | None:
        """Parametros entrenables de la red. Numero real, no una convencion."""
        return count_parameters(self.net)

    def _horizon(self, futr: FutrFrame | None) -> dict[str, list[pd.Timestamp]]:
        """Instantes a predecir por serie: los del `FutrFrame` si lo hay, o `cutoff + h*freq`."""
        if futr is not None and not futr.df.empty:
            return {
                str(uid): group.sort_values("ds")["ds"].tolist()
                for uid, group in futr.df.groupby("unique_id", sort=False)
            }
        grid = pd.date_range(self.cutoff, periods=self.h + 1, freq=self.freq)[1:].tolist()
        return dict.fromkeys(self.scaler.target_mean, grid)

    def _assert_within_horizon(self, instants: dict[str, list[pd.Timestamp]]) -> None:
        """Comprueba que lo pedido cabe en los `h` pasos que la cabeza sabe emitir.

        Parameters
        ----------
        instants
            Instantes a predecir por serie.

        Raises
        ------
        ValueError
            Si el instante mas lejano esta a mas de `h` pasos del cutoff del
            ajuste, o si el mas cercano no es el paso inmediatamente siguiente.
            Lo primero ocurre al reutilizar el ajuste en una ventana posterior;
            lo segundo, con un ``gap > 0`` en el plan. En ambos casos las `h`
            salidas de la red se alinearian con instantes que no son los suyos
            (ver el docstring del modulo).
        """
        furthest = max(ds for series in instants.values() for ds in series)
        nearest = min(ds for series in instants.values() for ds in series)
        grid = pd.date_range(self.cutoff, furthest, freq=self.freq)
        steps_to_furthest = len(grid) - 1
        steps_to_nearest = len(pd.date_range(self.cutoff, nearest, freq=self.freq)) - 1

        if steps_to_furthest > self.h or steps_to_nearest != 1:
            raise ValueError(
                f"{self.model_id}: la cabeza emite exactamente los pasos 1..{self.h} "
                f"desde su cutoff ({self.cutoff}), y se le piden los pasos "
                f"{steps_to_nearest}..{steps_to_furthest}. Fija refit_every=1 en el "
                "BacktestPlan para que cada ventana reajuste con su propio cutoff, y "
                "usa gap=0 con este modelo."
            )

    def _future_exog(
        self, futr: FutrFrame | None, ids: Sequence[str], instants: dict[str, list[pd.Timestamp]]
    ) -> np.ndarray:
        """Matriz ``(n_series, h, n_exog)`` de exogenas futuras, escalada.

        Raises
        ------
        MissingFutrExog
            Si el modelo usa exogenas futuras y no ha recibido `FutrFrame`.
        ValueError
            Si al `FutrFrame` le faltan columnas declaradas o filas de alguna
            serie.
        """
        if not self.exog_columns:
            return np.zeros((len(ids), self.h, 0), dtype=np.float32)
        if futr is None:
            raise MissingFutrExog(
                f"{self.model_id} necesita exogenas futuras para {list(self.exog_columns)}"
            )
        missing = set(self.exog_columns) - set(futr.df.columns)
        if missing:
            raise ValueError(
                f"{self.model_id}: faltan las exogenas futuras {sorted(missing)} en la "
                "trama recibida"
            )

        rows: list[np.ndarray] = []
        for uid in ids:
            group = futr.df[futr.df["unique_id"].astype(str) == uid].sort_values("ds")
            if len(group) != len(instants[uid]):  # pragma: no cover  el motor ya lo valida
                raise ValueError(
                    f"{self.model_id}: la serie '{uid}' trae {len(group)} instantes futuros "
                    f"y el horizonte exige {len(instants[uid])}"
                )
            rows.append(self.scaler.transform_exog(group))
        return np.stack(rows).astype(np.float32)

    def predict(
        self,
        futr: FutrFrame | None = None,
        *,
        quantiles: Sequence[float] = QUANTILES,
    ) -> pd.DataFrame:
        """Predice el horizonte completo para todas las series del entrenamiento.

        Parameters
        ----------
        futr
            Exogenas futuras de la ventana. Obligatorio si el panel declara
            `futr_exog`: entran en la cabeza de la red.
        quantiles
            Cuantiles a devolver. Los que no esten en la rejilla con la que se
            entreno la red salen como `NaN`: la red tiene un canal por cuantil
            entrenado, y estimar uno que nadie ajusto seria inventarlo.

        Returns
        -------
        pandas.DataFrame
            ``unique_id``, ``ds``, ``y_hat`` y una columna por cuantil pedido,
            ya en la escala original de cada serie.

        Raises
        ------
        MissingFutrExog
            Si hacen falta exogenas futuras y no llegan.
        ValueError
            Si el numero de instantes a predecir no coincide con el horizonte.
        """
        instants = self._horizon(futr)
        context, ids = context_matrix(self.train_panel, self.scaler, input_size=self.input_size)
        for uid in ids:
            found = len(instants.get(uid, []))
            if found != self.h:
                raise ValueError(
                    f"{self.model_id}: se esperaban {self.h} instantes a predecir para "
                    f"'{uid}' y se han encontrado {found}"
                )
        self._assert_within_horizon(instants)

        future_exog = self._future_exog(futr, ids, instants)
        scaled = predict_batch(self.net, context, future_exog, device=self.device)

        # El canal de la mediana es el pronostico puntual: la red se entrena con
        # perdida pinball, y el minimizador del cuantil 0.5 es la mediana
        # condicional. Si la rejilla no la incluye, se toma el canal central.
        trained = list(self.quantiles)
        median_channel = trained.index(0.5) if 0.5 in trained else len(trained) // 2

        parts: list[pd.DataFrame] = []
        for position, uid in enumerate(ids):
            original = self.scaler.inverse_target(scaled[position], uid)
            frame = pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": instants[uid],
                    "y_hat": original[:, median_channel],
                }
            )
            for quantile in quantiles:
                column = quantile_column(quantile)
                frame[column] = (
                    original[:, trained.index(quantile)] if quantile in trained else float("nan")
                )
            parts.append(frame)
        return pd.concat(parts, ignore_index=True)


@dataclass(frozen=True)
class LSTMForecaster:
    """Encoder LSTM propio en PyTorch, con prediccion directa multi-paso.

    No compite por ganar el leaderboard: compite por estar correctamente
    implementado y honestamente evaluado, sobre exactamente las mismas
    ventanas y con exactamente las mismas metricas que el resto. Si pierde
    contra MSTL, el leaderboard lo dice.

    Parameters
    ----------
    input_size
        Longitud del contexto en pasos. 168 por defecto: una semana horaria,
        suficiente para que el encoder vea el ciclo semanal completo.
    hidden_size, num_layers, dropout
        Hiperparametros de la red (`models/torch/modules.build_lstm_net`).
    config
        Presupuesto de entrenamiento (`models/torch/trainer.TrainConfig`):
        epocas acotadas, early stopping, recorte de gradiente y scheduler.
    quantiles
        Rejilla de cuantiles con la que se entrena la cabeza. Debe coincidir
        con la del plan del run para que `predict` no devuelva `NaN`.
    use_futr_exog
        Usar las columnas `futr_exog` del panel. Si es `False`, la red es
        puramente univariada y `requires.needs_futr_exog` pasa a `False`.
    seed
        Semilla del run: fija Python, numpy, torch y el barajado del cargador.
    model_id
        Identificador del modelo.

    Raises
    ------
    ValueError
        Si `input_size` es menor que uno o la rejilla de cuantiles no es
        estrictamente creciente.
    """

    input_size: int = 168
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    config: TrainConfig = field(default_factory=TrainConfig)
    quantiles: tuple[float, ...] = QUANTILES
    use_futr_exog: bool = True
    seed: int = 0
    model_id: ModelId = _LSTM_ID

    def __post_init__(self) -> None:
        """Valida el contexto y la rejilla de cuantiles."""
        if self.input_size < 1:
            raise ValueError(f"input_size debe ser >= 1: {self.input_size}")
        if list(self.quantiles) != sorted(set(self.quantiles)):
            raise ValueError(f"quantiles debe ser estrictamente creciente: {self.quantiles}")
        for quantile in self.quantiles:
            quantile_column(quantile)  # valida el rango (0, 1)

    @property
    def requires(self) -> ModelRequirements:
        """Necesita contexto e horizonte completos, y exogenas futuras si las usa.

        `min_context` es ``input_size + h`` en la practica, pero `h` no se
        conoce hasta `fit`: se declara el contexto mas un paso, y el propio
        `fit` falla ruidosamente si el tramo no da para ninguna ventana. El
        `refit_cost` es ``"expensive"``, asi que la politica por defecto del
        motor es un unico ajuste por run —lo que hace viable el backtest
        completo en CPU modesta— y queda registrada en `model_runs.refit_every`.
        """
        return ModelRequirements(
            needs_futr_exog=self.use_futr_exog,
            supports_quantiles=True,
            min_context=self.input_size + 1,
            refit_cost="expensive",
        )

    def fit(self, train: Panel, *, h: int) -> _FittedLSTM:
        """Ajusta escalador y red con **exclusivamente** los datos de `train`.

        Parameters
        ----------
        train
            Rebanada de entrenamiento de la ventana.
        h
            Horizonte en pasos, fijo para la vida del objeto ajustado: la
            cabeza de la red proyecta exactamente `h` pasos.

        Returns
        -------
        _FittedLSTM

        Raises
        ------
        ImportError
            Si `torch` no esta instalado.
        ValueError
            Si el tramo de entrenamiento no da para ninguna ventana completa.
        """
        started = perf_counter()
        exog_columns = tuple(train.spec.futr_exog) if self.use_futr_exog else ()
        scaler = SeriesScaler.fit(train, exog_columns)
        windows = build_windows(train, scaler, input_size=self.input_size, h=h)
        if len(windows) == 0:
            raise ValueError(
                f"{self.model_id}: el entrenamiento no da para ninguna ventana de "
                f"input_size={self.input_size} mas h={h}"
            )

        net = build_lstm_net(
            n_context_features=1 + len(exog_columns),
            n_futr_features=len(exog_columns),
            h=h,
            n_quantiles=len(self.quantiles),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        report = train_lstm(
            net, windows, quantiles=self.quantiles, config=self.config, seed=self.seed
        )

        return _FittedLSTM(
            model_id=self.model_id,
            cutoff=train.last_ds,
            h=h,
            fit_seconds=perf_counter() - started,
            freq=train.spec.freq,
            input_size=self.input_size,
            quantiles=self.quantiles,
            exog_columns=exog_columns,
            scaler=scaler,
            device=_device_of(net),
            net=net,
            train_report=report,
            train_panel=train,
        )


def _device_of(net: Any) -> str:
    """Dispositivo en el que viven los parametros de una red ya construida."""
    return str(next(net.parameters()).device)
