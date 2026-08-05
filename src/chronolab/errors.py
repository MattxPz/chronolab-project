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
    "PredictionContractError",
    "SchemaVersionError",
    "ShortTrainWarning",
    "SourceUnavailable",
    "StaleCacheWarning",
    "UnstableMetricWarning",
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


class PredictionContractError(ChronolabError):
    """Un modelo ha devuelto una prediccion que incumple el contrato de `predict`.

    Filas de mas o de menos, series que no estaban en el entrenamiento, instantes
    fuera del tramo evaluado o columnas obligatorias ausentes. Es un fallo *del
    modelo*, no del arnes: el motor lo registra con ``status="failed"`` en
    `model_runs` y sigue con el resto, porque un modelo roto que desaparece del
    leaderboard produce una comparacion mentirosa.

    No hereda de `LeakageError` a proposito. Predecir un instante ya conocido si
    es fuga, y eso se senala con `CutoffViolation`, que nunca se captura.
    """


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


class StaleCacheWarning(UserWarning):
    """`CachedSource` esta sirviendo una version obsoleta de la cache.

    Se emite cuando la fuente remota no responde (`SourceUnavailable`) y existe
    una entrada de cache anterior a la ventana de frescura configurada. Servir
    la version obsoleta es preferible a fallar, pero debe quedar visible: el
    llamante decide si eso es aceptable para su caso de uso.
    """


class UnstableMetricWarning(UserWarning):
    """El denominador de una metrica esta cerca de cero y el valor no es fiable.

    Lo emiten MAPE cuando la serie pasa cerca de cero —el error relativo se
    dispara sin que la prediccion sea peor— y MASE cuando el naive estacional del
    entrenamiento no comete ningun error, que deja la metrica indefinida. En
    ambos casos el numero se sigue calculando y se devuelve: ocultarlo obligaria
    a adivinar por que falta, y el aviso dice exactamente cuantas observaciones
    lo provocan.
    """


class ShortTrainWarning(UserWarning):
    """El splitter ha descartado ventanas por falta de entrenamiento.

    Se descartan, nunca se recortan: una ventana con menos historia de la
    declarada no es comparable con las demas, y silenciarla haria que el numero
    de ventanas del run no coincidiese con el del plan sin que nada lo dijese.
    """


class PerfectForesightWarning(UserWarning):
    """Se estan usando exogenas futuras con vintage `REALIZED`.

    El resultado es una cota superior de rendimiento, no una estimacion de lo que
    el sistema lograria en produccion. Debe etiquetarse como tal en cualquier
    tabla que se publique.
    """
