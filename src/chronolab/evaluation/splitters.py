"""`Window` y `RollingOriginSplitter`: unico emisor de particiones del proyecto.

No existe ninguna funcion que acepte mascaras booleanas, indices arbitrarios ni
fechas sueltas para partir un panel. Es deliberado: si la unica forma de obtener
una particion es esta, no hay forma de escribir un split aleatorio por accidente.
"""

import warnings
from dataclasses import dataclass

import pandas as pd

from chronolab.errors import ShortTrainWarning, WindowValidationError
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
        panel, que el invariante I3 garantiza completa: nunca por mascaras, ni
        por fechas sueltas, ni por indices. El anclaje es el **final** del panel,
        de modo que la ultima ventana evalua exactamente hasta la ultima marca
        disponible y los cutoffs anteriores se obtienen restando `step_size`.

        Por construccion se cumple, para toda ventana:

        - ``train_start <= cutoff < first_pred <= last_pred``,
        - ``first_pred = cutoff + (gap + 1) * freq``,
        - ``last_pred = first_pred + (h - 1) * freq``,

        y por tanto el tramo de entrenamiento y el de evaluacion son disjuntos
        con al menos `gap` pasos entre medias.

        Parameters
        ----------
        panel
            Panel canonico a partir.

        Returns
        -------
        tuple of Window
            Ventanas ordenadas por `cutoff` creciente y renumeradas desde cero.
            Las que corresponden a los ultimos `holdout_windows` cutoffs del plan
            llevan ``stage="holdout"``.

        Raises
        ------
        WindowValidationError
            Si el panel no da para ninguna ventana del plan.

        Warns
        -----
        ShortTrainWarning
            Si alguna ventana del plan se descarta porque su entrenamiento no
            alcanza `min_context` (o `train_size` en modo deslizante). Se
            descartan las mas antiguas, que son las que tienen menos historia
            detras.
        """
        grid = panel.grid()
        # El ultimo cutoff posible deja sitio para el gap y para los h pasos
        # evaluados: es el ancla desde la que se cuenta hacia atras.
        last_cutoff_idx = len(grid) - 1 - self.gap - self.h
        if last_cutoff_idx < 0:
            raise WindowValidationError(
                f"el panel tiene {len(grid)} pasos y el plan exige al menos "
                f"{self.gap + self.h + 1} (gap={self.gap}, h={self.h})"
            )

        windows: list[Window] = []
        discarded: list[str] = []
        for planned_id in range(self.n_windows):
            cutoff_idx = last_cutoff_idx - (self.n_windows - 1 - planned_id) * self.step_size
            if cutoff_idx < 0:
                discarded.append(f"#{planned_id}: el panel empieza despues de su cutoff")
                continue

            train_start_idx = self._train_start_index(cutoff_idx)
            train_length = cutoff_idx - train_start_idx + 1
            if train_start_idx < 0 or train_length < self.min_context:
                discarded.append(
                    f"#{planned_id}: entrenamiento de {max(train_length, 0)} pasos "
                    f"< min_context={self.min_context}"
                )
                continue

            first_pred_idx = cutoff_idx + self.gap + 1
            windows.append(
                Window(
                    window_id=len(windows),
                    stage=self._stage(planned_id),
                    train_start=grid[train_start_idx],
                    cutoff=grid[cutoff_idx],
                    first_pred=grid[first_pred_idx],
                    last_pred=grid[first_pred_idx + self.h - 1],
                    h=self.h,
                    gap=self.gap,
                )
            )

        if not windows:
            raise WindowValidationError(
                f"ninguna de las {self.n_windows} ventanas del plan cabe en el panel: "
                + "; ".join(discarded)
            )
        if discarded:
            warnings.warn(
                f"{len(discarded)} de {self.n_windows} ventanas descartadas por "
                f"entrenamiento insuficiente: " + "; ".join(discarded),
                ShortTrainWarning,
                stacklevel=2,
            )
        return tuple(windows)

    def _train_start_index(self, cutoff_idx: int) -> int:
        """Calcula el indice de rejilla en el que empieza el entrenamiento de un cutoff.

        Parameters
        ----------
        cutoff_idx
            Posicion del cutoff en la rejilla del panel.

        Returns
        -------
        int
            ``0`` en modo expansivo. En modo deslizante, el indice que deja
            exactamente `train_size` pasos; puede ser negativo, y entonces la
            ventana no cabe y se descarta.
        """
        if self.mode == "expanding":
            return 0
        if self.train_size is None:  # pragma: no cover  garantizado en __post_init__
            raise WindowValidationError("el modo 'sliding' exige train_size")
        return cutoff_idx - self.train_size + 1

    def _stage(self, planned_id: int) -> Stage:
        """Etapa de la ventana numero `planned_id` del plan.

        Se decide sobre la numeracion **del plan** y no sobre la de las ventanas
        supervivientes: asi el conjunto de holdout no cambia porque una ventana
        antigua se haya descartado por historia corta.

        Parameters
        ----------
        planned_id
            Indice de la ventana dentro del plan, base cero.

        Returns
        -------
        Stage
            ``"holdout"`` para los ultimos `holdout_windows` cutoffs del plan.
        """
        return "holdout" if planned_id >= self.n_windows - self.holdout_windows else "dev"
