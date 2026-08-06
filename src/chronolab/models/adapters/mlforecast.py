"""Adaptador de mlforecast con LightGBM y XGBoost, en modo recursivo y directo.

Unico adaptador con `supports_recursive = True` (en su variante recursiva): es
quien sabe realimentar la prediccion en los lags sin romper la disponibilidad
temporal. La regla de reparto de responsabilidades entre este modulo y
`chronolab.features.builders` esta explicada alli; el resumen que importa aqui:

- **Lags, ventanas moviles y diferencias de la propia objetivo** (``y``) se
  declaran en `builders.TargetFeatureConfig` y se traducen a `lags=` y
  `lag_transforms=` de mlforecast en `_build_lag_transforms`. La libreria es
  quien las calcula y quien gestiona la recursion; este modulo no reimplementa
  ninguna de las dos cosas.
- **Calendario y termicas** (`builders.calendar_feature_set`,
  `builders.thermal_feature_set`) viajan como columnas regresoras dinamicas:
  se anaden al `DataFrame` de entrenamiento con ``static_features=[]`` —para
  que mlforecast no las trate como constantes por serie— y hay que
  suministrarlas tambien en `X_df` al predecir.

Estrategia recursiva vs directa
--------------------------------
``strategy="recursive"`` deja el `max_horizon` de mlforecast sin fijar: cada
paso del horizonte se predice realimentando la prediccion del paso anterior en
los lags cortos de la objetivo, exactamente lo que persigue el analisis de
docs del proyecto sobre "un modelo por horizonte" frente a recursion.
``strategy="direct"`` fija ``max_horizon=h``: mlforecast ajusta un regresor
independiente por paso 1..h, cada uno mapeando "historia hasta el punto de
anclaje" a "objetivo `k` pasos despues de ese punto", sin recursion. Es
literalmente la implementacion de mlforecast de "un modelo por horizonte", y
por eso construirla a mano —duplicando lo que ya hace `max_horizon`— seria
exactamente el tipo de reimplementacion que el enunciado pide evitar.

Filtrado por adelanto de las features manuales
-------------------------------------------------
Las columnas de `builders.calendar_feature_set` y
`builders.thermal_feature_set` no las genera mlforecast por su cuenta —no son
funcion de la propia objetivo—, y este adaptador **nunca lee el `FutrFrame`**:
reconstruye el tramo futuro de calendario y termicas extendiendo la historia
de entrenamiento (`_FittedMLForecastModel._future_regressors`), no a partir de
una prevision externa. Eso fija el criterio de que features son admisibles, y
es **mas estricto** que el algebra general de
`chronolab.features.roles.select_for_lead`:

- **Calendario**: siempre admisible. Es funcion determinista de ``ds``, asi
  que no depende de ninguna reconstruccion desde historia.
- **Termicas**: solo los retardos ``lag(temp, k)`` con ``k >= h`` (el
  horizonte **maximo** del plan, el caso mas restrictivo: una unica
  configuracion sirve a todos los pasos, tanto en recursiva como en directa).
  La version sin retardar de la temperatura, HDD o CDD se descarta siempre,
  con independencia del rol de la columna —vease
  `_reconstructable_thermal` para el porque: `select_for_lead` daria por
  buenos incluso los retardos cortos de una columna `futr_exog` asumiendo que
  alguien lee su prevision del `FutrFrame`, y aqui nadie lo hace.

Import perezoso
----------------
`mlforecast`, `lightgbm` y `xgboost` viven en el extra `ml` (D20), no en el
nucleo. El modulo tiene que poder **importarse** sin ese extra —lo exige
`tests/unit/test_module_tree.py`—, asi que ningun `import` de esas librerias
aparece a nivel de modulo: viven dentro de `_require_mlforecast`,
`_require_lightgbm` y `_require_xgboost`, llamadas solo al ajustar.

Cuantiles via `PredictionIntervals` conformal de mlforecast
-------------------------------------------------------------
A diferencia de `adapters/statsforecast.py`, aqui el horizonte de calibracion
(`PredictionIntervals(h=...)`) se fija **dentro de `fit`**, con el `h` que el
motor de backtesting pasa en esa llamada, no en el constructor: no hace falta
que `h` sea un parametro de la clase `Forecaster`, porque cada ventana del
motor pide siempre el mismo `h` (`docs/ARCHITECTURE.md` §6.1), y por tanto no
existe la discordancia entre "horizonte de construccion" y "horizonte
pedido" que si sufre `statsforecast.ConformalIntervals`.

Explicabilidad
---------------
`_FittedMLForecastModel.feature_importance` (ganancia nativa de LightGBM o
XGBoost) y `.shap_values` (SHAP sobre una muestra del diseno de
entrenamiento, via `shap.TreeExplainer`) no forman parte del protocolo
`FittedForecaster` —son una capacidad extra de este adaptador, no un
requisito comun a los seis backends— pero viven aqui porque necesitan el
modelo de arbol ajustado y la matriz de diseno exacta que uso mlforecast, que
son detalles internos de este adaptador y no algo que deba filtrarse a
`evaluation/`.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from chronolab.features.builders import (
    DEFAULT_TARGET_FEATURES,
    DEFAULT_THERMAL_FEATURES,
    TargetFeatureConfig,
    ThermalFeatureConfig,
    calendar_feature_set,
    feature_frame,
    select_usable,
    thermal_feature_set,
)
from chronolab.features.ops import Feature, RollStat
from chronolab.models.protocols import QUANTILES, ModelRequirements, quantile_column
from chronolab.panel import FutrFrame, Panel, PanelSpec
from chronolab.types import ModelId

if TYPE_CHECKING:  # pragma: no cover
    from mlforecast import MLForecast

__all__ = ["LightGBMForecaster", "Strategy", "XGBoostForecaster"]

Strategy = Literal["recursive", "direct"]
"""``"recursive"``: mlforecast realimenta sus propias predicciones en los lags
cortos de la objetivo. ``"direct"``: mlforecast ajusta ``h`` regresores
independientes, uno por paso del horizonte (``max_horizon``), sin recursion.
"""

_LIGHTGBM_ID = ModelId("lightgbm")
_XGBOOST_ID = ModelId("xgboost")
_MODEL_ALIAS = "y_hat"
"""Nombre con el que se registra el estimador en `MLForecast`, y por tanto el
nombre de columna que produce `predict`: coincide a proposito con el nombre
canonico del pronostico puntual del proyecto, asi que no hace falta renombrar
nada al traducir la salida de mlforecast al contrato de `FittedForecaster`.
"""


# --------------------------------------------------------------------------- #
# Import perezoso
# --------------------------------------------------------------------------- #


def _require_mlforecast() -> tuple[type, type, type, type, type, type, type, type]:
    """Importa mlforecast bajo demanda, con un mensaje util si falta el extra.

    Returns
    -------
    tuple
        ``(MLForecast, PredictionIntervals, Lag, Combine, RollingMean,
        RollingStd, RollingMin, RollingMax)`` de la libreria instalada.

    Raises
    ------
    ImportError
        Si `mlforecast` no esta instalado.
    """
    try:
        from mlforecast import MLForecast
        from mlforecast.lag_transforms import (
            Combine,
            Lag,
            RollingMax,
            RollingMean,
            RollingMin,
            RollingStd,
        )
        from mlforecast.utils import PredictionIntervals
    except ImportError as exc:  # pragma: no cover  ejercitado solo sin el extra ml
        raise ImportError(
            "chronolab.models.adapters.mlforecast necesita el extra 'ml': `uv sync --extra ml`."
        ) from exc
    return (
        MLForecast,
        PredictionIntervals,
        Lag,
        Combine,
        RollingMean,
        RollingStd,
        RollingMin,
        RollingMax,
    )


def _require_lightgbm() -> Any:
    """Importa `lightgbm.LGBMRegressor` bajo demanda.

    Returns
    -------
    Any
        La clase `LGBMRegressor`. El tipo de retorno es `Any` a proposito:
        `lightgbm` esta en la cuarentena de tipos de D16 (docs/ARCHITECTURE.md),
        y fingir un tipo mas preciso aqui no anadiria seguridad real —ademas de
        que, sin el extra `ml` instalado (el entorno por defecto de CI), mypy
        resuelve el import como `Any` y un retorno anotado `type` dispara
        `no-any-return`.

    Raises
    ------
    ImportError
        Si `lightgbm` no esta instalado.
    """
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover  ejercitado solo sin el extra ml
        raise ImportError(
            "chronolab.models.adapters.mlforecast necesita el extra 'ml' con LightGBM instalado: "
            "`uv sync --extra ml`."
        ) from exc
    return LGBMRegressor


def _require_xgboost() -> Any:
    """Importa `xgboost.XGBRegressor` bajo demanda.

    Returns
    -------
    Any
        La clase `XGBRegressor`. El tipo de retorno es `Any` a proposito, por
        el mismo motivo que `_require_lightgbm`.

    Raises
    ------
    ImportError
        Si `xgboost` no esta instalado.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover  ejercitado solo sin el extra ml
        raise ImportError(
            "chronolab.models.adapters.mlforecast necesita el extra 'ml' con XGBoost instalado: "
            "`uv sync --extra ml`."
        ) from exc
    return XGBRegressor


# --------------------------------------------------------------------------- #
# Validacion
# --------------------------------------------------------------------------- #


def _validate_strategy(strategy: Strategy) -> None:
    """Comprueba que `strategy` es una de las dos admitidas.

    Raises
    ------
    ValueError
        Si no es ``"recursive"`` ni ``"direct"``.
    """
    if strategy not in ("recursive", "direct"):
        raise ValueError(f"strategy debe ser 'recursive' o 'direct': {strategy!r}")


def _validate_levels(levels: tuple[int, ...]) -> None:
    """Comprueba que una rejilla de niveles de intervalo es valida.

    Raises
    ------
    ValueError
        Si algun nivel cae fuera de ``(0, 100)``.
    """
    for level in levels:
        if not 0 < level < 100:
            raise ValueError(f"nivel de intervalo fuera de (0, 100): {level}")


def _validate_calibration_windows(windows: int) -> None:
    """Comprueba que el numero de ventanas de calibracion conformal es admisible.

    Raises
    ------
    ValueError
        Si es menor que dos: `PredictionIntervals` exige al menos dos ventanas
        internas para poder calibrar el intervalo.
    """
    if windows < 2:
        raise ValueError(f"calibration_windows debe ser >= 2: {windows}")


# --------------------------------------------------------------------------- #
# Traduccion de `TargetFeatureConfig` a `lag_transforms=` de mlforecast
# --------------------------------------------------------------------------- #


def _pct_change(current: np.ndarray, past: np.ndarray) -> np.ndarray:
    """Tasa de cambio ``(current - past) / past``, ``NaN`` donde `past` es cero.

    Necesita nombre propio (no una lambda): `mlforecast.lag_transforms.Combine`
    nombra la columna resultante con ``operator.__name__``, y una lambda
    produciria ``<lambda>`` en vez de algo legible en la matriz de diseno.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result: np.ndarray = np.where(past != 0, (current - past) / past, np.nan)
    return result


def _build_lag_transforms(
    config: TargetFeatureConfig,
    *,
    lag_cls: type,
    combine_cls: type,
    roll_classes: Mapping[RollStat, type],
) -> dict[int, list[object]]:
    """Traduce `TargetFeatureConfig` al `lag_transforms=` que espera `MLForecast`.

    mlforecast usa la clave del diccionario como el desplazamiento **de las
    ventanas moviles** (`Rolling*`, que no llevan su propio desplazamiento),
    pero la ignora para `Lag`/`Combine`, que si lo llevan explicito: por eso
    las diferencias y tasas de cambio funcionan igual sin importar bajo que
    clave del diccionario se agrupen.

    Parameters
    ----------
    config
        Numeros declarados en `chronolab.features.builders.TargetFeatureConfig`.
    lag_cls, combine_cls
        `mlforecast.lag_transforms.Lag` y `.Combine`.
    roll_classes
        Estadistico -> clase `mlforecast.lag_transforms.Rolling*` de la libreria.

    Returns
    -------
    dict
        Listo para pasarse como ``lag_transforms=`` a `MLForecast`.
    """
    by_shift: dict[int, list[object]] = {}
    for window in config.roll_windows:
        for stat in config.roll_stats:
            by_shift.setdefault(config.roll_shift, []).append(
                roll_classes[stat](window_size=window)
            )
    for k in config.diff_lags:
        by_shift.setdefault(config.diff_shift, []).append(
            combine_cls(lag_cls(config.diff_shift), lag_cls(config.diff_shift + k), operator.sub)
        )
    for k in config.pct_change_lags:
        by_shift.setdefault(config.diff_shift, []).append(
            combine_cls(lag_cls(config.diff_shift), lag_cls(config.diff_shift + k), _pct_change)
        )
    return by_shift


def _lag_only(features: tuple[Feature, ...]) -> tuple[Feature, ...]:
    """Descarta las termicas sin retardar (temperatura, HDD, CDD "de ahora mismo").

    Parameters
    ----------
    features
        Candidatas termicas.

    Returns
    -------
    tuple of Feature
        Solo las que retardan (``"_lag" in nombre``, la marca que deja
        `chronolab.features.ops.lag`). Ver `_reconstructable_thermal` para el
        motivo: la version sin retardar nunca es reconstruible sin leer el
        `FutrFrame`.
    """
    return tuple(f for f in features if "_lag" in f.name)


def _reconstructable_thermal(
    train: Panel, thermal: ThermalFeatureConfig, h: int
) -> tuple[Feature, ...]:
    """Termicas retardadas que este adaptador puede reconstruir sin leer el `FutrFrame`.

    Aqui se filtra **por el retardo configurado, comparado con `h`
    directamente** —no con `chronolab.features.roles.select_for_lead`, que
    seria el filtro incorrecto para este adaptador en concreto. La razon:
    `select_for_lead` decide si una feature es utilizable segun el `max_lead`
    de su columna de origen, y ``after_lag(UNBOUNDED, k) = UNBOUNDED`` para
    **cualquier** `k` si la temperatura es `futr_exog` (con prevision) —es
    decir, el algebra da por buenos incluso los retardos cortos, asumiendo
    que quien los usa sabe leer esa prevision del `FutrFrame` para rellenar el
    hueco. Este adaptador, a proposito, no lee el `FutrFrame` (docstring del
    modulo): reconstruye el tramo futuro extendiendo la historia de
    entrenamiento con `ops.lag`, y eso **solo** da el valor correcto cuando el
    retardo `k` es al menos tan largo como el horizonte `h` completo —para
    cualquier paso dentro de `[1, h]`, la referencia cae entonces siempre
    dentro de la historia, nunca dentro del propio tramo futuro—. Por eso el
    filtro aqui es ``k >= h`` sin condicionarlo al rol de la columna: es mas
    estricto que el algebra general porque la capacidad real de este
    adaptador (reconstruir solo desde historia) tambien lo es.

    Parameters
    ----------
    train
        Panel de entrenamiento.
    thermal
        Configuracion termica declarada.
    h
        Horizonte del plan para esta ventana.

    Returns
    -------
    tuple of Feature
        Termicas retardadas con ``k >= h``, sin las versiones sin retardar.
        Vacia si ningun retardo configurado alcanza `h`.
    """
    usable_lags = tuple(k for k in thermal.lags if k >= h)
    if not usable_lags:
        return ()
    restricted = ThermalFeatureConfig(
        temp_column=thermal.temp_column,
        heating_base=thermal.heating_base,
        cooling_base=thermal.cooling_base,
        lags=usable_lags,
    )
    return _lag_only(thermal_feature_set(train, restricted))


def _min_context(target: TargetFeatureConfig, thermal: ThermalFeatureConfig | None) -> int:
    """Piso de entrenamiento: la mayor ventana que cualquier feature necesita, mas uno.

    El ``+1`` deja al menos una fila util tras el ``dropna`` que mlforecast
    aplica por defecto al preprocesar: justo en el piso, todas las filas
    tendrian algun `NaN` inicial y no entrenaria nada.
    """
    target_needs = (
        *target.lags,
        *(w + target.roll_shift for w in target.roll_windows),
        *(k + target.diff_shift for k in (*target.diff_lags, *target.pct_change_lags)),
    )
    thermal_needs = thermal.lags if thermal is not None else ()
    longest = max((*target_needs, *thermal_needs), default=0)
    return longest + 1


# --------------------------------------------------------------------------- #
# Cuantiles: traduccion de los niveles de `PredictionIntervals` a la rejilla canonica
# --------------------------------------------------------------------------- #


def _interval_quantiles(level: int) -> tuple[float, float]:
    """Cuantiles inferior y superior de un intervalo central de nivel `level`."""
    lower = (100 - level) / 200.0
    return lower, 1.0 - lower


def _assign_quantiles(
    frame: pd.DataFrame, *, levels: tuple[int, ...], quantiles: Sequence[float]
) -> pd.DataFrame:
    """Traduce las columnas ``y_hat``/``y_hat-lo-L``/``y_hat-hi-L`` a la rejilla canonica.

    Parameters
    ----------
    frame
        Salida cruda de `MLForecast.predict`.
    levels
        Niveles de intervalo presentes en `frame`. Tupla vacia si el modelo se
        ajusto sin `PredictionIntervals`.
    quantiles
        Cuantiles pedidos por el motor.

    Returns
    -------
    pandas.DataFrame
        ``unique_id``, ``ds``, ``y_hat`` y una columna por cuantil pedido que
        se pueda derivar de `levels` o que sea ``0.5``. Los que no
        correspondan a ningun nivel calibrado quedan fuera: el motor los
        rellena con `NaN`, preferible a inventar un intervalo.
    """
    result = frame[["unique_id", "ds"]].copy()
    result["y_hat"] = frame[_MODEL_ALIAS]

    for quantile in quantiles:
        column = quantile_column(quantile)
        if math.isclose(quantile, 0.5, abs_tol=1e-9):
            result[column] = frame[_MODEL_ALIAS]
            continue
        for level in levels:
            lower, upper = _interval_quantiles(level)
            if math.isclose(quantile, lower, abs_tol=1e-9):
                result[column] = frame[f"{_MODEL_ALIAS}-lo-{level}"]
                break
            if math.isclose(quantile, upper, abs_tol=1e-9):
                result[column] = frame[f"{_MODEL_ALIAS}-hi-{level}"]
                break
    return result


# --------------------------------------------------------------------------- #
# Ajuste comun a LightGBM y XGBoost
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _FittedMLForecastModel:
    """Ajuste comun a `LightGBMForecaster` y `XGBoostForecaster`: un `MLForecast` ya ajustado."""

    model_id: ModelId
    cutoff: pd.Timestamp
    h: int
    fit_seconds: float
    freq: str
    strategy: Strategy
    levels: tuple[int, ...]
    spec: PanelSpec
    country: str | None
    subdiv: str | None
    fourier_order: int
    thermal_config: ThermalFeatureConfig | None
    calendar_names: tuple[str, ...]
    thermal_names: tuple[str, ...]
    thermal_history: pd.DataFrame | None
    design_matrix: pd.DataFrame
    series_ids: tuple[str, ...]
    mlf: MLForecast

    @property
    def n_params(self) -> int | None:
        """``None``: un ensamble de arboles no tiene un numero de parametros comparable."""
        return None

    def _horizon(self, futr: FutrFrame | None) -> dict[str, list[pd.Timestamp]]:
        """Instantes a predecir por serie: los del `FutrFrame` si lo hay, o `cutoff + h*freq`."""
        if futr is not None and not futr.df.empty:
            return {
                str(uid): group.sort_values("ds")["ds"].tolist()
                for uid, group in futr.df.groupby("unique_id", sort=False)
            }
        grid = pd.date_range(self.cutoff, periods=self.h + 1, freq=self.freq)[1:].tolist()
        return dict.fromkeys(self.series_ids, grid)

    def _future_regressors(self, future_grid: pd.DataFrame) -> pd.DataFrame | None:
        """Reconstruye calendario y termicas para el tramo futuro, alineadas a `future_grid`.

        Las termicas se recomputan sobre un panel extendido —historia de
        entrenamiento mas el tramo futuro, con la temperatura futura ausente—
        porque los retardos que sobrevivieron a `_reconstructable_thermal` en
        `fit` (``k >= h``) solo leen, por construccion, temperatura anterior
        al cutoff para cualquier paso del horizonte: nunca necesitan el valor
        futuro que falta, asi que recalcularlas aqui reutiliza exactamente
        `features.builders` en lugar de reimplementar el desplazamiento a mano.

        Parameters
        ----------
        future_grid
            ``unique_id``, ``ds`` del tramo a predecir.

        Returns
        -------
        pandas.DataFrame or None
            ``None`` si el ajuste no uso ninguna regresora dinamica.
        """
        if not self.calendar_names and not self.thermal_names:
            return None

        stub = Panel(df=future_grid[["unique_id", "ds"]].copy(), spec=self.spec)
        calendar_candidates = calendar_feature_set(
            stub, country=self.country, subdiv=self.subdiv, fourier_order=self.fourier_order
        )
        calendar = tuple(f for f in calendar_candidates if f.name in self.calendar_names)
        result = feature_frame(stub, calendar)

        if self.thermal_names:
            assert self.thermal_history is not None
            assert self.thermal_config is not None
            temp_column = self.thermal_config.temp_column
            extension = future_grid[["unique_id", "ds"]].assign(**{temp_column: np.nan})
            extended = (
                pd.concat([self.thermal_history, extension], ignore_index=True)
                .sort_values(["unique_id", "ds"])
                .reset_index(drop=True)
            )
            extended_panel = Panel(df=extended, spec=self.spec)
            thermal_candidates = thermal_feature_set(extended_panel, self.thermal_config)
            thermal = tuple(f for f in thermal_candidates if f.name in self.thermal_names)
            thermal_future = feature_frame(extended_panel, thermal).merge(
                future_grid[["unique_id", "ds"]], on=["unique_id", "ds"], how="inner"
            )
            result = result.merge(thermal_future, on=["unique_id", "ds"], how="left")

        return result

    def predict(
        self,
        futr: FutrFrame | None = None,
        *,
        quantiles: Sequence[float] = QUANTILES,
    ) -> pd.DataFrame:
        """Predice `h` pasos para todas las series del ajuste.

        Parameters
        ----------
        futr
            Exogenas futuras de la ventana. Este adaptador nunca lee sus
            columnas de valor —calendario y termicas se reconstruyen enteras
            a partir de `ds`, ver `_future_regressors`—, solo su `ds`, igual
            que `chronolab.models.baselines`.
        quantiles
            Cuantiles a estimar. `NaN` para los que no correspondan a ningun
            nivel calibrado por `PredictionIntervals`.

        Returns
        -------
        pandas.DataFrame
            ``unique_id``, ``ds``, ``y_hat`` y las columnas de cuantil que se
            puedan derivar.

        Raises
        ------
        ValueError
            Si el `FutrFrame`, o la rejilla por defecto, no traen exactamente
            `h` instantes para alguna serie del ajuste.
        """
        horizon = self._horizon(futr)
        for uid, instants in horizon.items():
            if len(instants) != self.h:
                raise ValueError(
                    f"{self.model_id}: se esperaban {self.h} instantes a predecir para "
                    f"'{uid}' y se han encontrado {len(instants)}"
                )

        future_grid = pd.DataFrame(
            {
                "unique_id": np.repeat(list(horizon.keys()), self.h),
                "ds": np.concatenate(
                    [pd.DatetimeIndex(instants).to_numpy() for instants in horizon.values()]
                ),
            }
        )
        x_df = self._future_regressors(future_grid)

        # Sin `ids=`: mlforecast ya predice para todas las series ajustadas
        # por defecto, que es exactamente `self.series_ids`. Pedirlo
        # explicito dispara el camino interno de subseleccion de mlforecast,
        # que con `Combine` (usado por las diferencias y tasas de cambio de
        # `_build_lag_transforms`) revienta con `AttributeError` porque
        # `Combine` no implementa el `take()` que ese camino exige —una
        # limitacion de la libreria, no de este adaptador.
        raw = self.mlf.predict(
            self.h,
            X_df=x_df,
            level=list(self.levels) if self.levels else None,
        )
        return _assign_quantiles(raw, levels=self.levels, quantiles=quantiles)

    def _estimators(self) -> list[Any]:
        """Aplana `mlf.models_` a una lista de estimadores ajustados.

        En estrategia directa cada entrada de `mlf.models_` es una lista con
        un regresor por paso de horizonte (``max_horizon``); en recursiva es
        un unico estimador. Aqui se homogeneiza a una lista en ambos casos.

        Tipada como ``list[Any]`` a proposito (D16): `mlf` es un objeto de
        `mlforecast`, en la cuarentena de tipos del proyecto, y sus
        estimadores subyacentes (`LGBMRegressor`/`XGBRegressor`) tampoco
        tienen stubs. Fingir un tipo mas preciso aqui no anadiria seguridad
        real.
        """
        estimators: list[Any] = []
        for value in self.mlf.models_.values():
            estimators.extend(value if isinstance(value, list) else [value])
        return estimators

    def feature_importance(self) -> pd.DataFrame:
        """Importancia nativa (ganancia media) de cada feature del modelo ajustado.

        En estrategia directa mlforecast ajusta un regresor por paso de
        horizonte: la importancia devuelta es la media sobre esos submodelos,
        no la de uno solo, para que el numero represente al conjunto de
        features en su totalidad y no a un paso arbitrario.

        Returns
        -------
        pandas.DataFrame
            ``feature``, ``importance``, ordenada de mayor a menor.
        """
        names = list(self.mlf.ts.features_order_)
        matrix = np.vstack(
            [
                np.asarray(estimator.feature_importances_, dtype=float)
                for estimator in self._estimators()
            ]
        )
        frame = pd.DataFrame({"feature": names, "importance": matrix.mean(axis=0)})
        return frame.sort_values("importance", ascending=False, ignore_index=True)

    def shap_values(self, *, sample_size: int = 200, seed: int = 0) -> pd.DataFrame:
        """Valores SHAP medios sobre una muestra del diseno de entrenamiento.

        Usa `shap.TreeExplainer`, exacto para modelos de arboles: no hace
        falta la aproximacion por muestreo que exigiria un explicador
        generico. La muestra acota el coste sobre paneles grandes sin cambiar
        la interpretacion, porque el `shap_value` es aditivo fila a fila.

        Parameters
        ----------
        sample_size
            Filas del diseno de entrenamiento a explicar, como maximo.
        seed
            Semilla del muestreo de filas.

        Returns
        -------
        pandas.DataFrame
            ``feature``, ``mean_abs_shap`` (media de ``|shap|`` sobre la
            muestra), ordenada de mayor a menor. Igual que
            `feature_importance`, promedia sobre los submodelos de horizonte
            en estrategia directa.

        Raises
        ------
        ImportError
            Si `shap` no esta instalado.
        """
        try:
            import shap
        except ImportError as exc:  # pragma: no cover  ejercitado solo sin el extra ml
            raise ImportError(
                "shap_values necesita el extra 'ml' con shap instalado: `uv sync --extra ml`."
            ) from exc

        feature_names = list(self.mlf.ts.features_order_)
        design = self.design_matrix[feature_names]
        n = min(sample_size, len(design))
        sample = design.sample(n=n, random_state=seed) if n else design

        per_estimator = np.stack(
            [
                np.abs(np.asarray(shap.TreeExplainer(estimator).shap_values(sample)))
                for estimator in self._estimators()
            ]
        )
        # Media sobre (submodelos de horizonte, filas de la muestra), en ese orden:
        # primero se homogeneiza la estrategia directa a un unico vector por fila,
        # despues se resume la muestra a un unico numero por feature.
        per_row = per_estimator.mean(axis=0)
        frame = pd.DataFrame({"feature": feature_names, "mean_abs_shap": per_row.mean(axis=0)})
        return frame.sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def _fit_mlforecast(
    train: Panel,
    h: int,
    *,
    estimator: object,
    strategy: Strategy,
    target_features: TargetFeatureConfig,
    country: str | None,
    subdiv: str | None,
    fourier_order: int,
    thermal: ThermalFeatureConfig | None,
    use_intervals: bool,
    levels: tuple[int, ...],
    calibration_windows: int,
    num_threads: int,
    model_id: ModelId,
) -> _FittedMLForecastModel:
    """Rutina de ajuste comun a `LightGBMForecaster` y `XGBoostForecaster`.

    Construye la matriz de diseno (calendario filtrado por
    `builders.select_usable` y, si procede, termicas filtradas por
    `_reconstructable_thermal`, ambos al horizonte maximo `h`), configura
    `MLForecast` con los lags/ventanas/diferencias de `target_features` y
    ajusta, en modo directo (``max_horizon=h``) o recursivo segun `strategy`.

    Parameters
    ----------
    train
        Panel de entrenamiento, ya recortado por el motor a ``ds <= cutoff``.
    h
        Horizonte solicitado por el plan.
    estimator
        Instancia sin ajustar de `LGBMRegressor` o `XGBRegressor`.
    strategy, target_features, country, subdiv, fourier_order, thermal,
    use_intervals, levels, calibration_windows, num_threads, model_id
        Ver `LightGBMForecaster`.

    Returns
    -------
    _FittedMLForecastModel
    """
    (
        mlforecast_cls,
        prediction_intervals_cls,
        lag_cls,
        combine_cls,
        rolling_mean_cls,
        rolling_std_cls,
        rolling_min_cls,
        rolling_max_cls,
    ) = _require_mlforecast()
    roll_classes: dict[RollStat, type] = {
        "mean": rolling_mean_cls,
        "std": rolling_std_cls,
        "min": rolling_min_cls,
        "max": rolling_max_cls,
    }

    calendar_candidates = calendar_feature_set(
        train, country=country, subdiv=subdiv, fourier_order=fourier_order
    )
    calendar_retained = select_usable(calendar_candidates, h, supports_recursive=False)

    thermal_retained: tuple[Feature, ...] = ()
    if thermal is not None:
        thermal_retained = _reconstructable_thermal(train, thermal, h)

    design = feature_frame(train, calendar_retained + thermal_retained)
    target = train.spec.target
    design[target] = train.df[target].to_numpy()
    if target != "y":
        design = design.rename(columns={target: "y"})

    lag_transforms = _build_lag_transforms(
        target_features, lag_cls=lag_cls, combine_cls=combine_cls, roll_classes=roll_classes
    )
    mlf = mlforecast_cls(
        models={_MODEL_ALIAS: estimator},
        freq=train.spec.freq,
        lags=list(target_features.lags),
        lag_transforms=lag_transforms,
        num_threads=num_threads,
    )
    intervals = (
        prediction_intervals_cls(n_windows=calibration_windows, h=h) if use_intervals else None
    )
    max_horizon = h if strategy == "direct" else None

    started = perf_counter()
    mlf.fit(design, static_features=[], max_horizon=max_horizon, prediction_intervals=intervals)
    fit_seconds = perf_counter() - started
    design_matrix = mlf.preprocess(design, static_features=[], max_horizon=max_horizon)

    thermal_history = None
    if thermal_retained:
        thermal_history = train.df[["unique_id", "ds", thermal.temp_column]].copy()  # type: ignore[union-attr]

    return _FittedMLForecastModel(
        model_id=model_id,
        cutoff=train.last_ds,
        h=h,
        fit_seconds=fit_seconds,
        freq=train.spec.freq,
        strategy=strategy,
        levels=levels if use_intervals else (),
        spec=train.spec,
        country=country,
        subdiv=subdiv,
        fourier_order=fourier_order,
        thermal_config=thermal,
        calendar_names=tuple(f.name for f in calendar_retained),
        thermal_names=tuple(f.name for f in thermal_retained),
        thermal_history=thermal_history,
        design_matrix=design_matrix,
        series_ids=train.ids(),
        mlf=mlf,
    )


# --------------------------------------------------------------------------- #
# Los dos `Forecaster`
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LightGBMForecaster:
    """LightGBM sobre mlforecast, en estrategia recursiva o directa.

    Parameters
    ----------
    strategy
        ``"recursive"`` (por defecto) o ``"direct"``. Ver el docstring del
        modulo.
    target_features
        Lags, ventanas moviles y diferencias de la propia objetivo que
        mlforecast debe generar. Ver `chronolab.features.builders.TargetFeatureConfig`.
    country, subdiv, fourier_order
        Configuracion de `chronolab.features.builders.calendar_feature_set`.
        ``country="ES"`` por defecto: festivos de España, coherentes con la
        serie de demanda electrica que es el foco del proyecto.
    thermal
        Configuracion de `chronolab.features.builders.thermal_feature_set`, o
        ``None`` para omitir las termicas —paneles sin exogena de
        temperatura.
    use_intervals
        Ajustar con `mlforecast.utils.PredictionIntervals` (conformal). Si es
        `False`, `predict` no produce cuantiles.
    levels
        Niveles de intervalo, en ``(0, 100)``.
    calibration_windows
        Ventanas internas que usa mlforecast para calibrar el intervalo
        conformal.
    params
        Hiperparametros extra de `lightgbm.LGBMRegressor` (``num_leaves``,
        ``learning_rate``, ``n_estimators``...), el punto de entrada que usa
        el tuning de Optuna de `chronolab.evaluation.tuning`.
    seed
        Semilla de `LGBMRegressor`.
    num_threads
        Hilos de mlforecast para el preprocesado. `LGBMRegressor` paraleliza
        el ajuste por su cuenta segun sus propios parametros.
    model_id
        Identificador del modelo. Al comparar las variantes recursiva y
        directa en el mismo run hace falta uno distinto por variante —los
        `model_id` deben ser unicos dentro de un `backtest()`—, por ejemplo
        ``ModelId("lightgbm_recursive")`` y ``ModelId("lightgbm_direct")``.

    Raises
    ------
    ValueError
        Si `strategy` no es admitida, si algun nivel de `levels` cae fuera de
        ``(0, 100)``, o si `calibration_windows` es menor que dos.
    """

    strategy: Strategy = "recursive"
    target_features: TargetFeatureConfig = DEFAULT_TARGET_FEATURES
    country: str | None = "ES"
    subdiv: str | None = None
    fourier_order: int = 2
    thermal: ThermalFeatureConfig | None = DEFAULT_THERMAL_FEATURES
    use_intervals: bool = True
    levels: tuple[int, ...] = (80, 95)
    calibration_windows: int = 2
    params: Mapping[str, object] = field(default_factory=dict)
    seed: int = 0
    num_threads: int = 1
    model_id: ModelId = _LIGHTGBM_ID

    def __post_init__(self) -> None:
        """Valida estrategia, niveles y ventanas de calibracion."""
        _validate_strategy(self.strategy)
        _validate_levels(self.levels)
        _validate_calibration_windows(self.calibration_windows)

    @property
    def requires(self) -> ModelRequirements:
        """`supports_recursive` solo en la variante recursiva; refit barato por defecto."""
        return ModelRequirements(
            min_context=_min_context(self.target_features, self.thermal),
            refit_cost="cheap",
            supports_recursive=self.strategy == "recursive",
        )

    def fit(self, train: Panel, *, h: int) -> _FittedMLForecastModel:
        """Ajusta un `LGBMRegressor` via `MLForecast` sobre el panel de entrenamiento."""
        lgbm_cls = _require_lightgbm()
        estimator = lgbm_cls(random_state=self.seed, verbosity=-1, **dict(self.params))
        return _fit_mlforecast(
            train,
            h,
            estimator=estimator,
            strategy=self.strategy,
            target_features=self.target_features,
            country=self.country,
            subdiv=self.subdiv,
            fourier_order=self.fourier_order,
            thermal=self.thermal,
            use_intervals=self.use_intervals,
            levels=self.levels,
            calibration_windows=self.calibration_windows,
            num_threads=self.num_threads,
            model_id=self.model_id,
        )


@dataclass(frozen=True)
class XGBoostForecaster:
    """XGBoost sobre mlforecast, en estrategia recursiva o directa.

    Parametros identicos a `LightGBMForecaster`; ver su docstring. La unica
    diferencia es el estimador subyacente, `xgboost.XGBRegressor`.

    Raises
    ------
    ValueError
        Si `strategy` no es admitida, si algun nivel de `levels` cae fuera de
        ``(0, 100)``, o si `calibration_windows` es menor que dos.
    """

    strategy: Strategy = "recursive"
    target_features: TargetFeatureConfig = DEFAULT_TARGET_FEATURES
    country: str | None = "ES"
    subdiv: str | None = None
    fourier_order: int = 2
    thermal: ThermalFeatureConfig | None = DEFAULT_THERMAL_FEATURES
    use_intervals: bool = True
    levels: tuple[int, ...] = (80, 95)
    calibration_windows: int = 2
    params: Mapping[str, object] = field(default_factory=dict)
    seed: int = 0
    num_threads: int = 1
    model_id: ModelId = _XGBOOST_ID

    def __post_init__(self) -> None:
        """Valida estrategia, niveles y ventanas de calibracion."""
        _validate_strategy(self.strategy)
        _validate_levels(self.levels)
        _validate_calibration_windows(self.calibration_windows)

    @property
    def requires(self) -> ModelRequirements:
        """`supports_recursive` solo en la variante recursiva; refit barato por defecto."""
        return ModelRequirements(
            min_context=_min_context(self.target_features, self.thermal),
            refit_cost="cheap",
            supports_recursive=self.strategy == "recursive",
        )

    def fit(self, train: Panel, *, h: int) -> _FittedMLForecastModel:
        """Ajusta un `XGBRegressor` via `MLForecast` sobre el panel de entrenamiento."""
        xgb_cls = _require_xgboost()
        estimator = xgb_cls(random_state=self.seed, verbosity=0, **dict(self.params))
        return _fit_mlforecast(
            train,
            h,
            estimator=estimator,
            strategy=self.strategy,
            target_features=self.target_features,
            country=self.country,
            subdiv=self.subdiv,
            fourier_order=self.fourier_order,
            thermal=self.thermal,
            use_intervals=self.use_intervals,
            levels=self.levels,
            calibration_windows=self.calibration_windows,
            num_threads=self.num_threads,
            model_id=self.model_id,
        )
