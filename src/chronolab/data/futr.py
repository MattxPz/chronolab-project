"""`FutrProvider`: exogenas conocidas a futuro con semantica de vintage explicita.

Tres implementaciones por honestidad decreciente: `ArchivedForecastProvider`,
`SimulatedForecastProvider` y `RealizedFutrProvider` (presciencia perfecta).
Barrera contra la fuga L4 de docs/ARCHITECTURE.md §8.

La exogena futura no es una columna: es una funcion ``(as_of, ds) -> valor``. Por
eso el que la entrega es un objeto con vintage declarado y no una proyeccion mas
del panel. De las tres implementaciones, aqui vive `RealizedFutrProvider`, la
unica que no necesita fuentes con historia de previsiones; las otras dos llegan
con las fuentes reales (hito H5) y sus firmas ya estan fijadas por este protocolo.
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from chronolab.errors import CutoffViolation, PanelValidationError, PerfectForesightWarning
from chronolab.panel import FutrFrame, Panel
from chronolab.types import SeriesId, Vintage

if TYPE_CHECKING:  # pragma: no cover
    # Import diferido: `data` no depende de `evaluation` en tiempo de ejecucion.
    # `Window` solo se necesita para tipar, y el protocolo es estructural.
    from chronolab.evaluation.splitters import Window

__all__ = ["FutrProvider", "RealizedFutrProvider"]


@runtime_checkable
class FutrProvider(Protocol):
    """Provee exogenas futuras para una ventana, con semantica de vintage explicita."""

    @property
    def vintage(self) -> Vintage:
        """Semantica temporal de los valores que entrega. Constante."""
        ...

    def futr(self, window: "Window", ids: Sequence[SeriesId]) -> FutrFrame:
        """Exogenas futuras conocidas en ``window.cutoff`` para el tramo de prediccion.

        Parameters
        ----------
        window
            Ventana de backtesting. Define `cutoff` (la informacion disponible) y
            el tramo ``[first_pred, last_pred]`` que hay que cubrir.
        ids
            Series para las que se piden exogenas. Deben ser las del
            entrenamiento de la ventana.

        Returns
        -------
        FutrFrame
            Exactamente ``len(ids) * window.h`` filas, todas con
            ``ds > window.cutoff``.
        """
        ...


@dataclass(frozen=True, slots=True)
class RealizedFutrProvider:
    """Exogenas futuras con el valor **realizado**: presciencia perfecta.

    Toma del panel el valor que la exogena acabo teniendo, que es justo el numero
    que nadie tenia en el `cutoff`. El resultado de un run con este proveedor es
    una **cota superior** de rendimiento, no una estimacion de lo que el sistema
    lograria en produccion, y solo es publicable si se etiqueta como tal. Por eso
    el aviso se emite al construirlo y no al usarlo: quien lo instancia ya ha
    tomado la decision.

    En el tramo de **entrenamiento** el valor realizado si es el correcto —es lo
    que un sistema real tendria del pasado— y por eso vive en el `Panel`. Esta
    asimetria es deliberada: el vintage solo aplica al cruzar el cutoff, que es
    exactamente donde actua este objeto.

    Parameters
    ----------
    panel
        Panel canonico del que se leen los valores realizados.

    Warns
    -----
    PerfectForesightWarning
        Siempre, en construccion.
    """

    panel: Panel

    def __post_init__(self) -> None:
        """Avisa de que este proveedor produce una cota superior, no un resultado."""
        warnings.warn(
            "RealizedFutrProvider usa el valor realizado de las exogenas futuras: "
            "el run mide presciencia perfecta y debe etiquetarse como cota superior",
            PerfectForesightWarning,
            stacklevel=3,
        )

    @property
    def vintage(self) -> Vintage:
        """Vintage declarado: `Vintage.REALIZED`."""
        return Vintage.REALIZED

    def futr(self, window: "Window", ids: Sequence[SeriesId]) -> FutrFrame:
        """Valores realizados de las `futr_exog` en el tramo de prediccion.

        La trama devuelta contiene **solo** las columnas declaradas `futr_exog`.
        Ni la objetivo ni las `hist_exog` estan omitidas por convenio: no existen
        en la estructura, que es la barrera contra la fuga L7.

        Parameters
        ----------
        window
            Ventana de backtesting.
        ids
            Series a cubrir, tipicamente las del entrenamiento de la ventana.

        Returns
        -------
        FutrFrame
            ``len(ids) * window.h`` filas ordenadas por ``(unique_id, ds)``.

        Raises
        ------
        CutoffViolation
            Si algun instante del tramo pedido no es posterior al cutoff. No
            deberia poder ocurrir con una `Window` valida: es la comprobacion que
            hace que tampoco pueda ocurrir con una manipulada.
        PanelValidationError
            Si el panel no cubre por completo el tramo pedido para esas series.
            Entregar menos filas de las debidas dejaria al modelo prediciendo
            sobre un horizonte distinto del que se evalua.
        """
        if window.first_pred <= window.cutoff:  # pragma: no cover  Window ya lo impide
            raise CutoffViolation(
                f"tramo de prediccion que empieza en {window.first_pred}, "
                f"anterior o igual al cutoff {window.cutoff}"
            )

        wanted = [str(uid) for uid in ids]
        df = self.panel.df
        mask = (
            (df["ds"] >= window.first_pred)
            & (df["ds"] <= window.last_pred)
            & df["unique_id"].isin(wanted)
        )
        columns = ["unique_id", "ds", *self.panel.spec.futr_exog]
        frame = df.loc[mask, columns].sort_values(["unique_id", "ds"]).reset_index(drop=True)

        expected = len(wanted) * window.h
        if len(frame) != expected:
            raise PanelValidationError(
                f"el panel cubre {len(frame)} de las {expected} filas de exogenas futuras "
                f"que exige la ventana {window.window_id} "
                f"([{window.first_pred}, {window.last_pred}], {len(wanted)} series)"
            )

        return FutrFrame(df=frame, window=window, vintage=self.vintage)
