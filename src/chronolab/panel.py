"""El contrato interno de datos: `PanelSpec`, `Panel` y `FutrFrame`.

El formato largo de Nixtla se adopta como formato de *transporte*, no como
contrato: no declara roles, ni frecuencia, ni huso, ni huecos, que son las cuatro
cosas cuya confusion produce fuga. El contrato es el tipo que lo envuelve, y
ningun `DataFrame` desnudo cruza una frontera de modulo.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from chronolab.errors import PanelValidationError
from chronolab.types import DatasetId, SeriesId, Vintage

if TYPE_CHECKING:  # pragma: no cover
    # Import diferido: `evaluation` depende de `panel`, no al reves.
    from chronolab.evaluation.splitters import Window

__all__ = ["FutrFrame", "Panel", "PanelSpec"]

RESERVED_COLUMNS: frozenset[str] = frozenset({"unique_id", "ds"})
"""Nombres de columna que son claves del panel y no pueden ser columnas de valor."""


@dataclass(frozen=True, slots=True)
class PanelSpec:
    """Declaracion del contenido semantico de un panel.

    Parameters
    ----------
    dataset_id
        Identificador estable del dataset; parte de la clave de los artefactos.
    freq
        Alias de offset de pandas de la rejilla temporal, por ejemplo ``"h"``. Es
        la frecuencia *real* de los datos, no un deseo: se valida contra el panel.
    seasonalities
        Longitudes estacionales en pasos, de la mas corta a la mas larga, por
        ejemplo ``(24, 168, 8766)``. La primera es la que usa MASE.
    target
        Nombre de la columna objetivo. Siempre ``"y"`` por convencion Nixtla.
    futr_exog
        Exogenas conocidas a futuro en el instante de predecir.
    hist_exog
        Exogenas conocidas solo hasta el cutoff.
    static_exog
        Atributos constantes por serie, almacenados en la trama lateral `static`.
    tz_display
        Zona horaria en la que se presentan los datos al usuario. **No** afecta al
        almacenamiento, que siempre es UTC ingenuo (invariante I2).

    Raises
    ------
    PanelValidationError
        Si los roles se solapan, si alguna columna usa un nombre reservado, o si
        `seasonalities` esta vacia o no es estrictamente creciente.
    """

    dataset_id: DatasetId
    freq: str
    seasonalities: tuple[int, ...]
    target: str = "y"
    futr_exog: tuple[str, ...] = ()
    hist_exog: tuple[str, ...] = ()
    static_exog: tuple[str, ...] = ()
    tz_display: str = "UTC"

    def __post_init__(self) -> None:
        """Valida la coherencia de la especificacion."""
        if not self.seasonalities:
            raise PanelValidationError("seasonalities no puede estar vacia")
        if any(s < 2 for s in self.seasonalities):
            raise PanelValidationError(f"seasonalities debe ser >= 2: {self.seasonalities}")
        if list(self.seasonalities) != sorted(set(self.seasonalities)):
            raise PanelValidationError(
                f"seasonalities debe ser estrictamente creciente: {self.seasonalities}"
            )

        groups = {
            "target": (self.target,),
            "futr_exog": self.futr_exog,
            "hist_exog": self.hist_exog,
            "static_exog": self.static_exog,
        }
        seen: dict[str, str] = {}
        for role, columns in groups.items():
            for column in columns:
                if column in RESERVED_COLUMNS:
                    raise PanelValidationError(
                        f"'{column}' es una clave del panel y no puede declararse como {role}"
                    )
                if column in seen:
                    raise PanelValidationError(
                        f"'{column}' declarada dos veces: {seen[column]} y {role}"
                    )
                seen[column] = role

    @property
    def mase_season(self) -> int:
        """Longitud estacional del denominador de MASE."""
        return self.seasonalities[0]

    @property
    def value_columns(self) -> tuple[str, ...]:
        """Columnas de valor del panel, en orden estable: objetivo y luego exogenas."""
        return (self.target, *self.futr_exog, *self.hist_exog)

    @property
    def columns(self) -> tuple[str, ...]:
        """Todas las columnas del panel, claves incluidas."""
        return ("unique_id", "ds", *self.value_columns)


@dataclass(frozen=True, slots=True)
class Panel:
    """Panel canonico validado. Unico portador de datos entre modulos.

    Los invariantes I1-I7 se garantizan en construccion. Ningun consumidor debe
    volver a comprobarlos y ningun productor puede saltarselos, porque el unico
    constructor publico es ``chronolab.data.assemble.build_panel``.

    Deliberadamente **no** existen aqui metodos ``scale``, ``impute`` ni
    ``transform``: el proyecto no tiene etapa global de preprocesado, y ese hueco
    en la arquitectura es lo que impide ajustar un escalador con datos futuros.

    Attributes
    ----------
    df
        Trama larga con columnas ``spec.columns``, ordenada por
        ``(unique_id, ds)``, sin duplicados y con rejilla completa.
    spec
        Especificacion semantica del panel.
    static
        Atributos constantes por serie: ``unique_id`` y ``spec.static_exog``, con
        exactamente una fila por serie del panel. ``None`` si no hay estaticas.
    """

    df: pd.DataFrame
    spec: PanelSpec
    static: pd.DataFrame | None = None

    def ids(self) -> tuple[SeriesId, ...]:
        """Identificadores de las series presentes, en orden de aparicion.

        Returns
        -------
        tuple of SeriesId
            Series del panel, sin repeticiones.
        """
        return tuple(SeriesId(str(value)) for value in self.df["unique_id"].unique())

    @property
    def first_ds(self) -> pd.Timestamp:
        """Primera marca de tiempo del panel, sobre todas las series."""
        return pd.Timestamp(self.df["ds"].min())

    @property
    def last_ds(self) -> pd.Timestamp:
        """Ultima marca de tiempo del panel, sobre todas las series."""
        return pd.Timestamp(self.df["ds"].max())

    def grid(self) -> pd.DatetimeIndex:
        """Rejilla temporal completa que cubre el panel, a ``spec.freq``.

        Las series pueden empezar o terminar en instantes distintos; la rejilla
        es la union, de la primera marca a la ultima. Es la referencia sobre la
        que `RollingOriginSplitter` hace su aritmetica: al ser regular por el
        invariante I3, contar pasos y sumar `freq` son la misma operacion, que es
        lo que hace imposible un off-by-one en el `gap`.

        Returns
        -------
        pandas.DatetimeIndex
            De `first_ds` a `last_ds` con paso ``spec.freq``, en UTC ingenuo.
        """
        return pd.date_range(self.first_ds, self.last_ds, freq=self.spec.freq)

    def slice(self, start: pd.Timestamp, end: pd.Timestamp) -> "Panel":
        """Sub-panel con ``start <= ds <= end``.

        Parameters
        ----------
        start, end
            Extremos **inclusivos**, en UTC ingenuo.

        Returns
        -------
        Panel
            Panel nuevo con la misma `spec`. Las estaticas se recortan a las
            series que sobreviven al corte, de modo que se sigue cumpliendo I7
            (una fila por serie del panel) y un join posterior no duplica filas.

        Raises
        ------
        PanelValidationError
            Si `start` es posterior a `end`.
        """
        if start > end:
            raise PanelValidationError(f"slice con start ({start}) posterior a end ({end})")

        mask = (self.df["ds"] >= start) & (self.df["ds"] <= end)
        df = self.df.loc[mask].reset_index(drop=True)

        static = self.static
        if static is not None:
            kept = df["unique_id"].unique()
            static = static.loc[static["unique_id"].isin(kept)].reset_index(drop=True)

        return Panel(df=df, spec=self.spec, static=static)

    def train(self, window: "Window") -> "Panel":
        """Rebanada de entrenamiento de una ventana.

        Devuelve ``window.train_start <= ds <= window.cutoff``. Es lo unico que
        recibe ``Forecaster.fit``: el modelo nunca ve el panel completo, asi que
        no tiene forma de mirar mas alla del cutoff. Como el corte se hace aqui y
        no en el adaptador, cualquier estadistico que el modelo ajuste (escalado,
        imputacion, calibracion) solo puede haber visto ``ds <= cutoff``.

        Parameters
        ----------
        window
            Ventana de backtesting.

        Returns
        -------
        Panel
            Panel recortado al tramo de entrenamiento.
        """
        return self.slice(window.train_start, window.cutoff)

    def actuals(self, window: "Window") -> pd.DataFrame:
        """Valores observados del tramo de evaluacion de una ventana.

        Parameters
        ----------
        window
            Ventana de backtesting.

        Returns
        -------
        pandas.DataFrame
            Columnas ``unique_id``, ``ds`` y la objetivo, para
            ``[window.first_pred, window.last_pred]``. Los huecos del panel
            viajan como ``NaN`` explicito (invariante I3), nunca como filas
            ausentes: una serie sin observacion en un instante evaluado tiene que
            poder distinguirse de una serie que no se evaluo.
        """
        mask = (self.df["ds"] >= window.first_pred) & (self.df["ds"] <= window.last_pred)
        columns = ["unique_id", "ds", self.spec.target]
        return self.df.loc[mask, columns].reset_index(drop=True)

    def to_nixtla(self) -> pd.DataFrame:
        """Vista en el dialecto que esperan statsforecast, mlforecast y neuralforecast.

        El formato largo canonico ya usa el vocabulario del ecosistema
        (``unique_id``, ``ds``, ``y``), asi que la conversion es una proyeccion de
        columnas en orden estable. Se devuelve una copia: lo que salga de aqui
        entra en libreria ajena y no debe poder modificar el panel.

        Returns
        -------
        pandas.DataFrame
            Columnas ``spec.columns``, en ese orden.
        """
        return self.df[list(self.spec.columns)].copy()


@dataclass(frozen=True, slots=True)
class FutrFrame:
    """Exogenas conocidas a futuro para exactamente una ventana.

    Contiene ``unique_id``, ``ds`` y **solo** las columnas de ``spec.futr_exog``.
    Las columnas ``hist_exog`` y la objetivo no estan ausentes por convenio: estan
    ausentes fisicamente. Un modelo no puede leerlas porque no existen en la
    estructura que recibe, que es la barrera mas fuerte del proyecto.

    Solo lo construye un ``chronolab.data.futr.FutrProvider``. No hay constructor
    publico.

    Attributes
    ----------
    df
        ``unique_id``, ``ds`` y las columnas ``futr_exog``. Todas las ``ds``
        cumplen ``ds > window.cutoff``.
    window
        Ventana a la que corresponde.
    vintage
        Semantica temporal de los valores. Entra en el hash de configuracion del
        run y se persiste en la tabla `runs`.
    """

    df: pd.DataFrame
    window: "Window"
    vintage: Vintage
