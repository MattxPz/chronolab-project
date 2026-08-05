"""Primitivas de feature exclusivamente retrospectivas: lag, roll, expand, ewm, diff.

No existe aqui ninguna operacion con ventana centrada o prospectiva sobre
columnas con `max_lead = 0`. La ausencia de la funcion es la barrera (fuga L3).

Dos decisiones dan forma al modulo:

1. **Cada primitiva devuelve un `Feature`**, es decir valores y adelanto juntos.
   El `max_lead` no se declara: lo calcula la operacion a partir del de su
   entrada, con el algebra de `chronolab.features.roles`. Una feature con un
   adelanto equivocado es indistinguible de una correcta al mirarla, asi que no
   puede depender de que alguien se acuerde.
2. **Las ventanas moviles llevan `shift >= 1` obligatorio.** El desplazamiento no
   tiene valor por defecto ``0`` que alguien pueda dejarse: una media movil que
   incluye el instante `t` usa el valor que se quiere predecir. Aqui la ventana
   siempre termina en ``t - shift``.

Las operaciones son por serie: agrupan por ``unique_id`` y nunca cruzan la
frontera entre dos series, que en formato largo son filas contiguas y por tanto
facilisimas de mezclar sin darse cuenta.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

import pandas as pd

from chronolab.features.roles import (
    FeatureSpec,
    MaxLead,
    after_diff,
    after_lag,
    after_lead,
    after_roll,
    column_max_lead,
)
from chronolab.panel import Panel

__all__ = ["Feature", "RollStat", "diff", "ewm", "expand", "from_column", "lag", "lead", "roll"]

RollStat = Literal["mean", "std", "min", "max", "sum", "median"]
"""Estadisticos admitidos en una ventana movil o expansiva."""


@dataclass(frozen=True, slots=True)
class Feature:
    """Una columna derivada junto con su disponibilidad temporal.

    Attributes
    ----------
    values
        Serie alineada con el indice de ``panel.df``.
    spec
        Nombre y `max_lead` calculado, nunca declarado a mano.
    """

    values: pd.Series
    spec: FeatureSpec

    @property
    def name(self) -> str:
        """Nombre de la feature."""
        return self.spec.name

    @property
    def max_lead(self) -> MaxLead:
        """Adelanto maximo para el que la feature es utilizable."""
        return self.spec.max_lead


Source = str | Feature
"""Entrada de una primitiva: una columna cruda del panel o una feature ya derivada."""


def _resolve(panel: Panel, source: Source) -> tuple[pd.Series, MaxLead, str]:
    """Normaliza la entrada de una primitiva a valores, adelanto y nombre.

    Parameters
    ----------
    panel
        Panel de referencia.
    source
        Nombre de columna del panel o `Feature` ya construida.

    Returns
    -------
    tuple
        Valores, `max_lead` de origen y nombre base para la feature resultante.
    """
    if isinstance(source, Feature):
        return source.values, source.spec.max_lead, source.spec.name
    return panel.df[source], column_max_lead(panel.spec, source), source


def _by_series(panel: Panel, values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Pasa valores y clave de serie a indice posicional.

    Trabajar posicionalmente es lo que hace que el resultado no dependa de si el
    indice del panel es un `RangeIndex`, y por tanto que una feature calculada
    sobre un prefijo coincida exactamente con la calculada sobre el panel entero
    y recortada despues (test T1 de estabilidad por prefijos).

    Parameters
    ----------
    panel
        Panel de referencia, ordenado por ``(unique_id, ds)`` (invariante I4).
    values
        Serie de valores alineada con ``panel.df``.

    Returns
    -------
    tuple of pandas.Series
        Valores y `unique_id`, ambos con indice posicional.
    """
    return values.reset_index(drop=True), panel.df["unique_id"].reset_index(drop=True)


def _restore(panel: Panel, values: pd.Series, name: str) -> pd.Series:
    """Devuelve una serie posicional al indice del panel.

    Parameters
    ----------
    panel
        Panel de referencia.
    values
        Serie con indice posicional.
    name
        Nombre que llevara la serie.

    Returns
    -------
    pandas.Series
        Con el indice de ``panel.df``.
    """
    restored = values.sort_index()
    restored.index = panel.df.index
    return restored.rename(name)


class _Aggregable(Protocol):
    """Lo minimo que este modulo necesita de una ventana movil o expansiva de pandas.

    Se declara como protocolo para no depender de la jerarquia interna de pandas
    (`Rolling`, `Expanding`, `RollingGroupby`...), que cambia entre versiones y
    cuyos tipos exactos no aportan nada aqui.
    """

    def mean(self) -> pd.Series:
        """Media de la ventana."""
        ...

    def std(self) -> pd.Series:
        """Desviacion tipica muestral de la ventana."""
        ...

    def min(self) -> pd.Series:
        """Minimo de la ventana."""
        ...

    def max(self) -> pd.Series:
        """Maximo de la ventana."""
        ...

    def sum(self) -> pd.Series:
        """Suma de la ventana."""
        ...

    def median(self) -> pd.Series:
        """Mediana de la ventana."""
        ...


def _aggregate(window: _Aggregable, stat: RollStat) -> pd.Series:
    """Aplica un estadistico a una ventana movil o expansiva ya construida.

    Se escriben las ramas una a una en lugar de resolver el metodo por nombre:
    `getattr` devolveria `Any` y la cuarentena de tipos dejaria de estar donde
    dice estar.

    Parameters
    ----------
    window
        Ventana movil o expansiva de pandas.
    stat
        Estadistico a aplicar.

    Returns
    -------
    pandas.Series
        Resultado de la agregacion.

    Raises
    ------
    ValueError
        Si el estadistico no esta admitido.
    """
    if stat == "mean":
        return window.mean()
    if stat == "std":
        return window.std()
    if stat == "min":
        return window.min()
    if stat == "max":
        return window.max()
    if stat == "sum":
        return window.sum()
    if stat == "median":
        return window.median()
    raise ValueError(f"estadistico no admitido: {stat}")


def from_column(panel: Panel, column: str) -> Feature:
    """Eleva una columna cruda del panel a `Feature`, con el adelanto de su rol.

    Parameters
    ----------
    panel
        Panel de referencia.
    column
        Nombre de la columna.

    Returns
    -------
    Feature
        Con ``max_lead = 0`` si la columna es la objetivo o una `hist_exog`, e
        infinito si es una `futr_exog` o una estatica.
    """
    values, max_lead, name = _resolve(panel, column)
    return Feature(values=values.rename(name), spec=FeatureSpec(name=name, max_lead=max_lead))


def lag(panel: Panel, source: Source, k: int) -> Feature:
    """Valor de `source` `k` pasos antes, dentro de cada serie.

    Parameters
    ----------
    panel
        Panel de referencia.
    source
        Columna del panel o feature ya derivada.
    k
        Retardo en pasos, mayor o igual que uno.

    Returns
    -------
    Feature
        Con ``max_lead = max_lead(source) + k``: retrasar compra adelanto, que es
        justo lo que permite usar ``lag(y, 24)`` para predecir a 24 pasos y no a
        48. Los primeros `k` puntos de cada serie son `NaN`; no se rellenan,
        porque rellenarlos hacia atras seria mirar adelante.

    Raises
    ------
    ValueError
        Si `k` es menor que uno.
    """
    values, source_lead, name = _resolve(panel, source)
    max_lead = after_lag(source_lead, k)
    positional, ids = _by_series(panel, values)
    shifted = positional.groupby(ids, sort=False).shift(k)
    feature_name = f"{name}_lag{k}"
    return Feature(
        values=_restore(panel, shifted, feature_name),
        spec=FeatureSpec(name=feature_name, max_lead=max_lead),
    )


def lead(panel: Panel, source: Source, k: int) -> Feature:
    """Valor de `source` `k` pasos despues, solo legal sobre columnas conocidas a futuro.

    Sobre una columna con `max_lead` finito —la objetivo o una `hist_exog`— la
    llamada falla al construirla, no al ejecutarla: leer el futuro de una serie
    que en el cutoff no se conocia es fuga por definicion, y el sitio donde eso
    se detecta tiene que ser el sitio donde se escribe.

    Parameters
    ----------
    panel
        Panel de referencia.
    source
        Columna del panel o feature ya derivada, con `max_lead` infinito.
    k
        Adelanto en pasos, mayor o igual que uno.

    Returns
    -------
    Feature
        Con `max_lead` infinito.

    Raises
    ------
    ValueError
        Si `k` es menor que uno, o si `source` tiene `max_lead` finito.
    """
    values, source_lead, name = _resolve(panel, source)
    max_lead = after_lead(source_lead, k)
    positional, ids = _by_series(panel, values)
    shifted = positional.groupby(ids, sort=False).shift(-k)
    feature_name = f"{name}_lead{k}"
    return Feature(
        values=_restore(panel, shifted, feature_name),
        spec=FeatureSpec(name=feature_name, max_lead=max_lead),
    )


def diff(panel: Panel, source: Source, k: int = 1) -> Feature:
    """Diferencia de `source` respecto a `k` pasos antes.

    Parameters
    ----------
    panel
        Panel de referencia.
    source
        Columna del panel o feature ya derivada.
    k
        Orden de la diferencia en pasos.

    Returns
    -------
    Feature
        Con el mismo `max_lead` que ``lag(source, k)``: la diferencia usa el
        valor en `t`, asi que hereda la disponibilidad del mas restrictivo.

    Raises
    ------
    ValueError
        Si `k` es menor que uno.
    """
    values, source_lead, name = _resolve(panel, source)
    max_lead = after_diff(source_lead, k)
    positional, ids = _by_series(panel, values)
    differenced = positional.groupby(ids, sort=False).diff(k)
    feature_name = f"{name}_diff{k}"
    return Feature(
        values=_restore(panel, differenced, feature_name),
        spec=FeatureSpec(name=feature_name, max_lead=max_lead),
    )


def roll(
    panel: Panel,
    source: Source,
    window: int,
    *,
    shift: int = 1,
    stat: RollStat = "mean",
) -> Feature:
    """Estadistico de una ventana movil que **termina en** ``t - shift``.

    No hay parametro `center` ni forma de pedir una ventana que incluya `t`: la
    firma no lo admite y el desplazamiento minimo es uno. Es la barrera de la
    fuga L3 en su forma mas fuerte, la ausencia de la operacion.

    Parameters
    ----------
    panel
        Panel de referencia.
    source
        Columna del panel o feature ya derivada.
    window
        Longitud de la ventana en pasos, mayor o igual que uno. La ventana exige
        estar completa: con menos de `window` observaciones el valor es `NaN`, en
        lugar de un promedio de dos puntos que fingiria ser lo mismo.
    shift
        Pasos entre `t` y el final de la ventana, mayor o igual que uno.
    stat
        Estadistico a aplicar.

    Returns
    -------
    Feature
        Con ``max_lead = max_lead(source) + shift``. El tamano de la ventana no
        interviene: lo que fija la disponibilidad es donde acaba, no cuanto
        abarca.

    Raises
    ------
    ValueError
        Si `window` o `shift` son menores que uno, o si `stat` no esta admitido.
    """
    if window < 1:
        raise ValueError(f"la ventana debe ser >= 1: {window}")
    values, source_lead, name = _resolve(panel, source)
    max_lead = after_roll(source_lead, shift=shift)
    positional, ids = _by_series(panel, values)
    shifted = positional.groupby(ids, sort=False).shift(shift)
    rolled = _aggregate(
        shifted.groupby(ids, sort=False).rolling(window, min_periods=window), stat
    ).reset_index(level=0, drop=True)
    feature_name = f"{name}_roll{stat}{window}_s{shift}"
    return Feature(
        values=_restore(panel, rolled, feature_name),
        spec=FeatureSpec(name=feature_name, max_lead=max_lead),
    )


def expand(
    panel: Panel,
    source: Source,
    *,
    shift: int = 1,
    stat: RollStat = "mean",
) -> Feature:
    """Estadistico acumulado desde el inicio de la serie hasta ``t - shift``.

    Parameters
    ----------
    panel
        Panel de referencia.
    source
        Columna del panel o feature ya derivada.
    shift
        Pasos entre `t` y el ultimo punto incluido, mayor o igual que uno.
    stat
        Estadistico a aplicar.

    Returns
    -------
    Feature
        Con ``max_lead = max_lead(source) + shift``.

    Raises
    ------
    ValueError
        Si `shift` es menor que uno o si `stat` no esta admitido.
    """
    values, source_lead, name = _resolve(panel, source)
    max_lead = after_roll(source_lead, shift=shift)
    positional, ids = _by_series(panel, values)
    shifted = positional.groupby(ids, sort=False).shift(shift)
    expanded = _aggregate(
        shifted.groupby(ids, sort=False).expanding(min_periods=1), stat
    ).reset_index(level=0, drop=True)
    feature_name = f"{name}_expand{stat}_s{shift}"
    return Feature(
        values=_restore(panel, expanded, feature_name),
        spec=FeatureSpec(name=feature_name, max_lead=max_lead),
    )


def ewm(panel: Panel, source: Source, *, halflife: float, shift: int = 1) -> Feature:
    """Media exponencial de `source` calculada hasta ``t - shift``.

    Parameters
    ----------
    panel
        Panel de referencia.
    source
        Columna del panel o feature ya derivada.
    halflife
        Semivida del decaimiento en pasos, estrictamente positiva.
    shift
        Pasos entre `t` y el ultimo punto incluido, mayor o igual que uno.

    Returns
    -------
    Feature
        Con ``max_lead = max_lead(source) + shift``.

    Raises
    ------
    ValueError
        Si `halflife` no es positiva o `shift` es menor que uno.
    """
    if halflife <= 0:
        raise ValueError(f"la semivida debe ser > 0: {halflife}")
    values, source_lead, name = _resolve(panel, source)
    max_lead = after_roll(source_lead, shift=shift)
    positional, ids = _by_series(panel, values)
    shifted = positional.groupby(ids, sort=False).shift(shift)
    smoothed = (
        shifted.groupby(ids, sort=False)
        .ewm(halflife=halflife)
        .mean()
        .reset_index(level=0, drop=True)
    )
    feature_name = f"{name}_ewm{halflife:g}_s{shift}"
    return Feature(
        values=_restore(panel, smoothed, feature_name),
        spec=FeatureSpec(name=feature_name, max_lead=max_lead),
    )
