"""Algebra de `max_lead`: propaga la disponibilidad temporal a traves de las operaciones.

`max_lead(c)` es el mayor adelanto para el que `c` se conoce sin recurrir a
predicciones propias. Nunca se declara a mano: se calcula
(docs/ARCHITECTURE.md §4.4).

La distincion binaria futuro/historico es insuficiente para las features
derivadas, y es ahi donde se rompe la mayoria de los proyectos: ``lag(y, 24)`` no
es "historica" a secas, es utilizable para predecir a 1..24 pasos y no mas alla.
La propiedad correcta es un entero, no un booleano, y por eso este modulo existe.
"""

import math
from dataclasses import dataclass

from chronolab.panel import PanelSpec

__all__ = [
    "UNBOUNDED",
    "FeatureSpec",
    "MaxLead",
    "after_diff",
    "after_lag",
    "after_lead",
    "after_roll",
    "column_max_lead",
    "select_for_lead",
    "usable_for_lead",
]

MaxLead = float
"""Adelanto maximo utilizable de una columna, en pasos. `UNBOUNDED` si no lo tiene.

Es `float` y no `int` para poder representar el infinito con el tipo del lenguaje
en vez de con un centinela (``-1``, ``None``, ``999999``) que alguien tendria que
acordarse de interpretar. Los valores finitos son siempre enteros no negativos.
"""

UNBOUNDED: MaxLead = math.inf
"""Conocida en todo el horizonte: calendario, exogenas futuras y estaticas."""


def _validate(max_lead: MaxLead, *, what: str) -> None:
    """Comprueba que un `max_lead` es infinito o un entero no negativo.

    Parameters
    ----------
    max_lead
        Valor a comprobar.
    what
        Descripcion del origen, para el mensaje de error.

    Raises
    ------
    ValueError
        Si es negativo, o finito con parte decimal.
    """
    if math.isinf(max_lead):
        return
    if max_lead < 0 or max_lead != int(max_lead):
        raise ValueError(f"{what}: max_lead debe ser entero >= 0 o UNBOUNDED, no {max_lead}")


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Una feature y hasta que adelanto es utilizable.

    Parameters
    ----------
    name
        Nombre de la columna generada.
    max_lead
        Mayor adelanto `L` para el que la feature se conoce en ``cutoff + L`` sin
        recurrir a predicciones propias.
    recursive_only
        Marca las features que solo son admisibles en estrategia recursiva, es
        decir, aquellas cuyo valor mas alla de `max_lead` requiere realimentar
        predicciones del propio modelo. El motor solo se las pasa a modelos que
        declaran ``supports_recursive``.

    Raises
    ------
    ValueError
        Si `max_lead` no es un entero no negativo ni `UNBOUNDED`.
    """

    name: str
    max_lead: MaxLead
    recursive_only: bool = False

    def __post_init__(self) -> None:
        """Valida el adelanto declarado."""
        _validate(self.max_lead, what=f"feature '{self.name}'")


def column_max_lead(spec: PanelSpec, column: str) -> MaxLead:
    """Adelanto maximo de una columna cruda del panel, deducido de su rol.

    Parameters
    ----------
    spec
        Especificacion del panel, que es quien declara los roles.
    column
        Nombre de la columna.

    Returns
    -------
    MaxLead
        ``0`` para la objetivo y las `hist_exog` —no se conocen en ningun
        instante posterior al cutoff—; `UNBOUNDED` para `futr_exog` y
        `static_exog`.

    Raises
    ------
    KeyError
        Si la columna no tiene rol declarado. No hay valor por defecto a
        proposito: una columna sin rol es justo la que acaba usada como feature
        sin que nadie haya decidido si podia.
    """
    if column == spec.target or column in spec.hist_exog:
        return 0.0
    if column in spec.futr_exog or column in spec.static_exog:
        return UNBOUNDED
    raise KeyError(f"'{column}' no tiene rol declarado en el panel '{spec.dataset_id}'")


def after_lag(source: MaxLead, k: int) -> MaxLead:
    """Adelanto de ``lag(c, k)``.

    Retrasar `k` pasos compra `k` pasos de adelanto: el valor de `c` en ``t - k``
    ya se conoce cuando se quiere predecir ``t``, y sigue conociendose hasta
    ``cutoff + k``. Sobre una columna ya conocida a futuro no cambia nada.

    Parameters
    ----------
    source
        Adelanto de la columna de origen.
    k
        Retardo en pasos, mayor o igual que uno.

    Returns
    -------
    MaxLead
        ``source + k`` si `source` es finito, `UNBOUNDED` en caso contrario.

    Raises
    ------
    ValueError
        Si `k` es menor que uno.
    """
    if k < 1:
        raise ValueError(f"el retardo debe ser >= 1: {k}")
    _validate(source, what="lag")
    return UNBOUNDED if math.isinf(source) else source + k


def after_diff(source: MaxLead, k: int = 1) -> MaxLead:
    """Adelanto de ``diff(c, k)``, que es el de ``lag(c, k)``.

    La diferencia usa el valor en `t` y el de ``t - k``, asi que hereda la
    disponibilidad del mas restrictivo de los dos, que es el de `t`.

    Parameters
    ----------
    source
        Adelanto de la columna de origen.
    k
        Orden de la diferencia en pasos.

    Returns
    -------
    MaxLead
        El mismo que `after_lag`.
    """
    return after_lag(source, k)


def after_roll(source: MaxLead, *, shift: int) -> MaxLead:
    """Adelanto de una ventana movil retrospectiva que termina en ``t - shift``.

    El tamano de la ventana no interviene: lo que fija la disponibilidad es
    **donde acaba**, no cuanto abarca. Una ventana centrada no tiene entrada en
    esta tabla porque la operacion no existe en `chronolab.features.ops`.

    Parameters
    ----------
    source
        Adelanto de la columna de origen.
    shift
        Pasos de desplazamiento hacia atras del final de la ventana, mayor o
        igual que uno.

    Returns
    -------
    MaxLead
        ``source + shift`` si `source` es finito, `UNBOUNDED` en caso contrario.

    Raises
    ------
    ValueError
        Si `shift` es menor que uno.
    """
    if shift < 1:
        raise ValueError(f"el desplazamiento de una ventana movil debe ser >= 1: {shift}")
    _validate(source, what="roll")
    return UNBOUNDED if math.isinf(source) else source + shift


def after_lead(source: MaxLead, k: int) -> MaxLead:
    """Adelanto de ``lead(c, k)``, que solo existe sobre columnas ya conocidas a futuro.

    Adelantar una columna con `max_lead` finito significa leer un valor que en el
    `cutoff` nadie tenia. No es un error de ejecucion que haya que vigilar: es un
    error de **construccion**, y por eso se lanza aqui, al calcular el adelanto,
    antes de que exista ninguna columna.

    Parameters
    ----------
    source
        Adelanto de la columna de origen.
    k
        Adelanto en pasos, mayor o igual que uno.

    Returns
    -------
    MaxLead
        `UNBOUNDED`.

    Raises
    ------
    ValueError
        Si `k` es menor que uno o si `source` es finito.
    """
    if k < 1:
        raise ValueError(f"el adelanto debe ser >= 1: {k}")
    _validate(source, what="lead")
    if not math.isinf(source):
        raise ValueError(
            f"lead({k}) sobre una columna con max_lead={source:g} leeria un valor "
            "posterior al cutoff que nadie conocia: es fuga por construccion"
        )
    return UNBOUNDED


def usable_for_lead(feature: FeatureSpec, lead: int) -> bool:
    """Si una feature es utilizable para predecir a `lead` pasos del cutoff.

    Parameters
    ----------
    feature
        Feature a comprobar.
    lead
        Adelanto real desde el cutoff, es decir ``gap + h_step``, no el paso de
        prediccion. Confundirlos es el off-by-one que hace pasar por buena una
        feature caducada.

    Returns
    -------
    bool
        ``True`` si ``feature.max_lead >= lead``.
    """
    return feature.max_lead >= lead


def select_for_lead(
    features: tuple[FeatureSpec, ...],
    lead: int,
    *,
    supports_recursive: bool = False,
) -> tuple[FeatureSpec, ...]:
    """Filtra las features admisibles para un adelanto dado.

    Es la regla que impone el motor de backtesting, no el adaptador: si cada
    adaptador decidiese que features usar, la comparacion entre estrategia
    directa y recursiva dejaria de significar lo mismo para cada modelo.

    Parameters
    ----------
    features
        Features candidatas con su adelanto ya calculado.
    lead
        Adelanto real desde el cutoff (``gap + h_step``).
    supports_recursive
        Si el modelo declara `supports_recursive`. Solo entonces se admiten
        features con `max_lead` insuficiente, y unicamente si estan marcadas
        `recursive_only`: el modelo se compromete a realimentar sus propias
        predicciones para calcularlas.

    Returns
    -------
    tuple of FeatureSpec
        Subconjunto admisible, en el orden de entrada.
    """
    return tuple(
        feature
        for feature in features
        if usable_for_lead(feature, lead) or (supports_recursive and feature.recursive_only)
    )
