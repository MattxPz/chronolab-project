"""`Window` y `RollingOriginSplitter`: unico emisor de particiones del proyecto.

No existe ninguna funcion que acepte mascaras booleanas, indices arbitrarios ni
fechas sueltas para partir un panel. Es deliberado: si la unica forma de obtener
una particion es esta, no hay forma de escribir un split aleatorio por accidente.
"""

from dataclasses import dataclass

import pandas as pd

from chronolab.errors import WindowValidationError
from chronolab.panel import Panel
from chronolab.types import SplitMode, Stage

__all__ = ["RollingOriginSplitter", "Window"]


@dataclass(frozen=True, slots=True)
class Window:
    """Una ventana de origen rodante. Inmutable y autoconsistente.

    Parameters
    ----------
    window_id
        Indice base cero, creciente en el tiempo.
    stage
        ``"dev"`` para ventanas de desarrollo (tuning y seleccion) o
        ``"holdout"`` para las de reporte. Que sean valores distintos y que el
        optimizador solo reciba las de desarrollo es lo que impide ajustar
        hiperparametros sobre las ventanas que luego se publican.
    train_start, cutoff
        Extremos **inclusivos** del tramo de entrenamiento. `cutoff` es la
        frontera de informacion de la ventana.
    first_pred, last_pred
        Extremos **inclusivos** del tramo de evaluacion. Se cumple
        ``first_pred = cutoff + (gap + 1) * freq`` y
        ``last_pred = first_pred + (h - 1) * freq``.
    h
        Horizonte en pasos.
    gap
        Pasos descartados entre `cutoff` y `first_pred`. Emula latencia de datos
        y corta la autocorrelacion de corto alcance.

    Raises
    ------
    WindowValidationError
        Si la ventana es internamente inconsistente.
    """

    window_id: int
    stage: Stage
    train_start: pd.Timestamp
    cutoff: pd.Timestamp
    first_pred: pd.Timestamp
    last_pred: pd.Timestamp
    h: int
    gap: int

    def __post_init__(self) -> None:
        """Verifica el orden temporal y la coherencia de `h` y `gap`."""
        if self.window_id < 0:
            raise WindowValidationError(f"window_id debe ser >= 0: {self.window_id}")
        if self.h < 1:
            raise WindowValidationError(f"h debe ser >= 1: {self.h}")
        if self.gap < 0:
            raise WindowValidationError(f"gap debe ser >= 0: {self.gap}")
        if self.train_start > self.cutoff:
            raise WindowValidationError(
                f"train_start ({self.train_start}) posterior a cutoff ({self.cutoff})"
            )
        if self.first_pred <= self.cutoff:
            raise WindowValidationError(
                f"first_pred ({self.first_pred}) no es posterior a cutoff ({self.cutoff})"
            )
        if self.last_pred < self.first_pred:
            raise WindowValidationError(
                f"last_pred ({self.last_pred}) anterior a first_pred ({self.first_pred})"
            )

    def lead(self, h_step: int) -> int:
        """Adelanto real desde el cutoff de un paso de prediccion.

        Parameters
        ----------
        h_step
            Paso de prediccion, entre ``1`` y ``h``, relativo a `first_pred`.

        Returns
        -------
        int
            ``gap + h_step``, que es la distancia en pasos desde el cutoff.

        Raises
        ------
        ValueError
            Si `h_step` cae fuera de ``[1, h]``.
        """
        if not 1 <= h_step <= self.h:
            raise ValueError(f"h_step fuera de [1, {self.h}]: {h_step}")
        return self.gap + h_step


@dataclass(frozen=True, slots=True)
class RollingOriginSplitter:
    """Genera ventanas de origen rodante por aritmetica sobre la rejilla del panel.

    Parameters
    ----------
    h
        Horizonte de prediccion en pasos.
    n_windows
        Numero de ventanas a generar.
    step_size
        Separacion entre cutoffs consecutivos, en pasos. Si es menor que `h` las
        ventanas de evaluacion se solapan; se permite, pero el solape se registra
        porque afecta a la independencia que asume el test de Diebold-Mariano.
    gap
        Pasos descartados entre el cutoff y la primera prediccion.
    mode
        ``"expanding"`` si el entrenamiento crece, ``"sliding"`` si es de
        longitud fija.
    train_size
        Longitud del entrenamiento en pasos. Obligatorio en ``"sliding"``.
    holdout_windows
        Numero de ventanas finales marcadas ``stage="holdout"``. Las anteriores
        son ``"dev"``.
    min_context
        Ventanas cuyo entrenamiento sea mas corto se descartan con aviso, nunca
        se recortan.

    Raises
    ------
    WindowValidationError
        Si la configuracion es incoherente.
    """

    h: int
    n_windows: int
    step_size: int = 1
    gap: int = 0
    mode: SplitMode = "expanding"
    train_size: int | None = None
    holdout_windows: int = 0
    min_context: int = 1

    def __post_init__(self) -> None:
        """Valida la configuracion del splitter."""
        if self.h < 1:
            raise WindowValidationError(f"h debe ser >= 1: {self.h}")
        if self.n_windows < 1:
            raise WindowValidationError(f"n_windows debe ser >= 1: {self.n_windows}")
        if self.step_size < 1:
            raise WindowValidationError(f"step_size debe ser >= 1: {self.step_size}")
        if self.gap < 0:
            raise WindowValidationError(f"gap debe ser >= 0: {self.gap}")
        if self.min_context < 1:
            raise WindowValidationError(f"min_context debe ser >= 1: {self.min_context}")
        if not 0 <= self.holdout_windows <= self.n_windows:
            raise WindowValidationError(
                f"holdout_windows fuera de [0, {self.n_windows}]: {self.holdout_windows}"
            )
        if self.mode == "sliding" and self.train_size is None:
            raise WindowValidationError("el modo 'sliding' exige train_size")
        if self.train_size is not None and self.train_size < self.min_context:
            raise WindowValidationError(
                f"train_size ({self.train_size}) menor que min_context ({self.min_context})"
            )

    def split(self, panel: Panel) -> tuple[Window, ...]:
        """Genera las ventanas de un panel.

        Las ventanas se construyen por aritmetica sobre la rejilla regular del
        panel, que el invariante I3 garantiza completa.

        Parameters
        ----------
        panel
            Panel canonico a partir.

        Returns
        -------
        tuple of Window
            Ventanas ordenadas por `cutoff` creciente. Las ultimas
            `holdout_windows` llevan ``stage="holdout"``.
        """
        raise NotImplementedError
