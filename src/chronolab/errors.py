"""Jerarquia de excepciones y avisos propios.

Toda excepcion del proyecto desciende de `ChronolabError`, de modo que un
llamante pueda distinguir un fallo nuestro de uno de una libreria envuelta.
"""

__all__ = [
    "ArtifactNotFound",
    "ChronolabError",
    "CutoffViolation",
    "LeakageError",
    "MissingFutrExog",
    "PanelValidationError",
    "PerfectForesightWarning",
    "SchemaVersionError",
    "SourceUnavailable",
    "VintageNotSupported",
    "WindowValidationError",
]


class ChronolabError(Exception):
    """Raiz de todas las excepciones del proyecto."""


# --------------------------------------------------------------------------- #
# Fuga de informacion temporal
# --------------------------------------------------------------------------- #


class LeakageError(ChronolabError):
    """Se ha detectado, o se habria producido, fuga de informacion temporal.

    Nunca se captura para continuar: un run que la provoca no produce resultados
    publicables.
    """


class CutoffViolation(LeakageError):
    """Se ha pedido predecir o puntuar un instante que ya era conocido.

    Es la asercion central anti-fuga del proyecto. Se comprueba siempre, tambien
    en produccion: su coste es despreciable frente a su valor.
    """


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #


class PanelValidationError(ChronolabError):
    """El panel o su especificacion incumplen alguno de los invariantes I1-I7."""


class SourceUnavailable(ChronolabError):
    """La fuente remota no ha respondido.

    El decorador de cache la captura y sirve la ultima version valida marcandola
    como obsoleta.
    """


class VintageNotSupported(ChronolabError):
    """Se ha pedido `as_of` a una fuente que no sabe responder por vintage.

    Se lanza en lugar de ignorar el parametro: una fuente que devuelve valores
    revisados cuando le piden los de una fecha concreta produce presciencia
    silenciosa.
    """


# --------------------------------------------------------------------------- #
# Modelos y evaluacion
# --------------------------------------------------------------------------- #


class MissingFutrExog(ChronolabError):
    """El modelo declara `needs_futr_exog` pero no ha recibido `FutrFrame`."""


class WindowValidationError(ChronolabError):
    """Una ventana de backtesting es internamente inconsistente."""


# --------------------------------------------------------------------------- #
# Artefactos
# --------------------------------------------------------------------------- #


class ArtifactNotFound(ChronolabError):
    """No existe el artefacto solicitado, o el run no tiene manifest.

    Un directorio de run sin `manifest.json` esta a medio escribir y se trata
    como inexistente.
    """


class SchemaVersionError(ChronolabError):
    """La version de esquema del artefacto no la entiende este codigo.

    Se rechaza en lugar de interpretar mal columnas ausentes o resignificadas.
    """


# --------------------------------------------------------------------------- #
# Avisos
# --------------------------------------------------------------------------- #


class PerfectForesightWarning(UserWarning):
    """Se estan usando exogenas futuras con vintage `REALIZED`.

    El resultado es una cota superior de rendimiento, no una estimacion de lo que
    el sistema lograria en produccion. Debe etiquetarse como tal en cualquier
    tabla que se publique.
    """
