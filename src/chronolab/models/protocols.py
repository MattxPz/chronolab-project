"""Protocolos `Forecaster` y `FittedForecaster`.

Tres decisiones hacen que este protocolo envuelva por igual statsforecast,
mlforecast, neuralforecast, Prophet, un LSTM propio y Chronos zero-shot sin que
la abstraccion se filtre:

1. `fit` devuelve un objeto nuevo en lugar de mutar `self`. Un `Forecaster` es
   una configuracion; un `FittedForecaster` es configuracion mas informacion
   hasta un cutoff. Chronos, que no entrena, implementa `fit` capturando el
   contexto: es un caso degenerado legitimo, no un parche.
2. `h` es parametro de `fit` y no de `predict`. neuralforecast y PatchTST lo
   necesitan para construir la red, y asi desaparece la posibilidad de que el
   horizonte de entrenamiento y el de prediccion no coincidan.
3. El objeto ajustado conoce su propio cutoff y `predict` lo verifica.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from chronolab.panel import FutrFrame, Panel
from chronolab.types import ModelId, RefitCost

__all__ = ["QUANTILES", "FittedForecaster", "Forecaster", "ModelRequirements", "quantile_column"]

QUANTILES: tuple[float, ...] = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
"""Rejilla canonica de cuantiles del proyecto, fijada en `conf/backtest.yaml`.

Los cuantiles, y no los pares ``lo``/``hi``, son la representacion canonica: la
perdida pinball y el CRPS se definen sobre cuantiles, y guardar intervalos
obligaria a asumir simetria, que es falsa en demanda electrica.
"""


def quantile_column(quantile: float) -> str:
    """Nombre de columna canonico de un cuantil.

    El nombre es ``q_<int>`` con ``int = round(quantile * 10000)`` y relleno a
    cuatro digitos: sin puntos ni signos (compatible con cualquier motor SQL
    sobre parquet), con orden lexicografico igual al numerico y sin la ambiguedad
    de ``lo``/``hi``.

    Parameters
    ----------
    quantile
        Cuantil en el intervalo abierto ``(0, 1)``.

    Returns
    -------
    str
        Nombre de la columna, por ejemplo ``"q_0250"`` para ``0.025``.

    Raises
    ------
    ValueError
        Si el cuantil cae fuera de ``(0, 1)``.

    Examples
    --------
    >>> quantile_column(0.025)
    'q_0250'
    >>> quantile_column(0.5)
    'q_5000'
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"cuantil fuera de (0, 1): {quantile}")
    return f"q_{round(quantile * 10000):04d}"


@dataclass(frozen=True, slots=True)
class ModelRequirements:
    """Capacidades y necesidades declaradas de un modelo.

    El motor de backtesting las lee para decidir que pasarle, que pedirle, cuando
    reajustarlo y que ventanas saltarse. Un modelo que declara mal sus requisitos
    falla rapido y ruidosamente en la primera ventana, no en silencio.

    Parameters
    ----------
    needs_futr_exog
        Si es ``True`` y el run no tiene `FutrProvider`, el run aborta.
    uses_hist_exog, uses_static_exog
        Si el modelo aprovecha esas columnas.
    supports_quantiles
        Si es ``False``, las columnas de cuantil se escriben como ``NaN``. Nunca
        se inventa un intervalo.
    supports_recursive
        Si admite features con ``max_lead`` menor que el adelanto, realimentando
        sus propias predicciones.
    min_context
        Pasos minimos de entrenamiento. Las ventanas mas cortas se saltan con
        aviso, no se recortan.
    handles_nan_target
        Si es ``False``, el adaptador imputa dentro de `fit`, nunca antes.
    is_zero_shot
        Aparece marcado como tal en el leaderboard.
    refit_cost
        Determina la politica de refit por defecto: ``"expensive"`` implica un
        unico ajuste por run, y la politica aplicada queda registrada.
    """

    needs_futr_exog: bool = False
    uses_hist_exog: bool = False
    uses_static_exog: bool = False
    supports_quantiles: bool = False
    supports_recursive: bool = False
    min_context: int = 1
    handles_nan_target: bool = False
    is_zero_shot: bool = False
    refit_cost: RefitCost = "cheap"


@runtime_checkable
class Forecaster(Protocol):
    """Configuracion de un modelo de prediccion. Inmutable y sin estado ajustado."""

    @property
    def model_id(self) -> ModelId:
        """Identificador estable. Clave de particion de los artefactos."""
        ...

    @property
    def requires(self) -> ModelRequirements:
        """Capacidades declaradas. Constante."""
        ...

    def fit(self, train: Panel, *, h: int) -> "FittedForecaster":
        """Ajusta el modelo con **exclusivamente** los datos de `train`.

        Parameters
        ----------
        train
            Rebanada de entrenamiento, ya recortada por el motor a
            ``ds <= window.cutoff``. Es un `Panel` completo: lleva su `spec`, sus
            exogenas historicas y futuras (valores realizados del pasado) y sus
            estaticas. El modelo **no** recibe la ventana ni el panel entero, asi
            que no tiene forma de mirar mas alla del cutoff.
        h
            Horizonte en pasos de ``spec.freq``. Fijo para toda la vida del
            objeto ajustado.

        Returns
        -------
        FittedForecaster
            Objeto nuevo. `self` no se modifica, de modo que el mismo
            `Forecaster` puede ajustarse en muchas ventanas en paralelo sin
            contaminacion cruzada.

        Notes
        -----
        Todo el preprocesado dependiente de datos (escalado, imputacion,
        seleccion de features, calibracion conformal interna) ocurre **aqui
        dentro**, con lo que por construccion se ajusta solo con entrenamiento.
        El proyecto no tiene una etapa global de preprocesado, y ese hueco en la
        arquitectura es intencionado.
        """
        ...


@runtime_checkable
class FittedForecaster(Protocol):
    """Modelo ajustado hasta un instante concreto. Inmutable."""

    @property
    def model_id(self) -> ModelId:
        """Identificador del modelo del que procede."""
        ...

    @property
    def cutoff(self) -> pd.Timestamp:
        """Ultima marca de tiempo **incluida** en el entrenamiento.

        Es la frontera de informacion del objeto. `predict` la usa para verificar
        que nada de lo que se predice cae en el pasado ya conocido.
        """
        ...

    @property
    def h(self) -> int:
        """Horizonte con el que se ajusto."""
        ...

    @property
    def fit_seconds(self) -> float:
        """Coste de ajuste medido. Se persiste para el eje precision/coste."""
        ...

    @property
    def n_params(self) -> int | None:
        """Numero de parametros entrenables, o ``None`` si no aplica."""
        ...

    def predict(
        self,
        futr: FutrFrame | None = None,
        *,
        quantiles: Sequence[float] = QUANTILES,
    ) -> pd.DataFrame:
        """Predice `h` pasos hacia delante para todas las series del entrenamiento.

        Parameters
        ----------
        futr
            Exogenas conocidas a futuro del tramo a predecir, emitidas por un
            `FutrProvider`. Obligatorio si ``requires.needs_futr_exog``. Contiene
            solo columnas `futr_exog`: el modelo no puede acceder a exogenas
            historicas del futuro porque no estan en la estructura.
        quantiles
            Cuantiles a estimar, en ``(0, 1)``. Los modelos con
            ``supports_quantiles=False`` devuelven ``NaN`` en esas columnas.

        Returns
        -------
        pandas.DataFrame
            Exactamente ``n_series * h`` filas. Columnas ``unique_id``, ``ds``,
            ``y_hat`` y una por cuantil segun `quantile_column`. Todas las ``ds``
            cumplen ``ds > cutoff`` y caen en la rejilla de ``spec.freq``.

        Raises
        ------
        CutoffViolation
            Si alguna ``ds`` de `futr` o de la salida es anterior o igual al
            cutoff. Se comprueba siempre, tambien en produccion.
        MissingFutrExog
            Si ``requires.needs_futr_exog`` y `futr` es ``None``.
        """
        ...
