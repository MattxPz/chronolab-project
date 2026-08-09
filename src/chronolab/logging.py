"""Configuracion de logging estructurado y del contexto `run_id`.

Cada registro se emite como una linea JSON en lugar de texto libre. Es lo que
hace que `scripts/refresh_data.py`, que corre desatendido cada seis horas en
`refresh-data.yml`, produzca un log que una plataforma de observabilidad (o un
simple `jq` sobre los logs de Actions) puede filtrar y agregar sin parsear
prosa. Vive fuera de cada script porque el mismo formato lo va a necesitar
`chronolab.api.service`, y duplicarlo en los dos sitios crearia dos formatos
que divergirian en silencio.

El `run_id` de un proceso se guarda en un `contextvars.ContextVar` y no en un
argumento que cada llamada de logging tenga que repetir: eso lo haria opcional
en la practica, y un `run_id` que falta en la mitad de las lineas no sirve para
correlacionar un incidente. Fijarlo una vez con `bind_run_id` al principio del
proceso hace que todo lo que se registre despues lo lleve, incluido el codigo
de libreria que no sabe nada de `chronolab`.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, TextIO

__all__ = ["bind_run_id", "configure_logging", "get_logger"]

_LOGGER_NAME = "chronolab"

run_id_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "chronolab_run_id", default=None
)
"""`run_id` vigente en el hilo o tarea async actual. `None` fuera de un `bind_run_id`."""

_RESERVED_RECORD_KEYS: frozenset[str] = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | frozenset({"message", "asctime"})
"""Atributos que `LogRecord` ya define. Lo que sobre de `record.__dict__` es `extra`."""


class _JsonFormatter(logging.Formatter):
    """Formatea cada `LogRecord` como una unica linea JSON.

    Campos estables en todo registro: ``timestamp`` (UTC, ISO 8601),
    ``level``, ``logger`` y ``message``. ``run_id`` se anade solo si hay uno
    vigente en `run_id_context`, y cualquier campo extra pasado via
    ``logger.info(..., extra={"campo": valor})`` se vuelca tal cual, lo que
    permite que cada sitio de logging declare sus propios campos (por ejemplo
    ``n_rows`` o ``source_id``) sin tocar este formateador.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Serializa `record` a una linea JSON.

        Parameters
        ----------
        record
            Registro emitido por cualquier logger hijo de ``"chronolab"``.

        Returns
        -------
        str
            Documento JSON de una sola linea, sin salto final.
        """
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        run_id = run_id_context.get()
        if run_id is not None:
            payload["run_id"] = run_id

        # Campos declarados con `extra={...}` en la llamada de logging: no viven
        # como atributos propios de `LogRecord`, asi que se identifican por
        # exclusion de los que `logging` ya reserva para su propio uso.
        extra = {
            key: value for key, value in record.__dict__.items() if key not in _RESERVED_RECORD_KEYS
        }
        if extra:
            payload.update(extra)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, sort_keys=True)


def configure_logging(*, level: int = logging.INFO, stream: TextIO | None = None) -> None:
    """Configura el logger raiz ``"chronolab"`` para emitir JSON a `stream`.

    Reemplaza cualquier handler previo del logger, asi que es seguro llamarla
    varias veces (por ejemplo en tests) sin acumular salidas duplicadas.
    `propagate` se desactiva para que el logger raiz de Python (que un import
    de terceros puede haber configurado con su propio formato de texto) no
    vuelva a emitir el mismo registro en texto plano.

    Parameters
    ----------
    level
        Nivel minimo que se emite. `logging.INFO` por defecto.
    stream
        Destino de la salida. `sys.stdout` si no se especifica: en un proceso
        de cron o de contenedor, stdout es lo que el orquestador captura.
    """
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setFormatter(_JsonFormatter())

    logger = logging.getLogger(_LOGGER_NAME)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Logger hijo de ``"chronolab"``, heredando su configuracion.

    Parameters
    ----------
    name
        Nombre del logger, tipicamente ``__name__`` del modulo llamante.
        Si no empieza por ``"chronolab"`` se le antepone, para que siempre
        cuelgue del logger configurado por `configure_logging`.

    Returns
    -------
    logging.Logger
        Logger listo para usar. No requiere que `configure_logging` se haya
        llamado antes: sin ella, hereda el comportamiento por defecto de
        `logging` (texto plano a stderr), lo que mantiene los tests silenciosos
        sin JSON de por medio.
    """
    is_already_qualified = name == _LOGGER_NAME or name.startswith(f"{_LOGGER_NAME}.")
    qualified = name if is_already_qualified else f"{_LOGGER_NAME}.{name}"
    return logging.getLogger(qualified)


class bind_run_id:  # noqa: N801
    """Gestor de contexto que fija `run_id_context` durante su bloque.

    Parameters
    ----------
    run_id
        Identificador que acompanara a cada registro emitido dentro del
        bloque ``with``, en cualquier logger hijo de ``"chronolab"``.

    Examples
    --------
    >>> with bind_run_id("01J9Z..."):
    ...     get_logger(__name__).info("iniciando refresco")
    """

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._token: contextvars.Token[str | None] | None = None

    def __enter__(self) -> None:
        """Activa `run_id` para el bloque."""
        self._token = run_id_context.set(self._run_id)

    def __exit__(self, *exc_info: object) -> None:
        """Restaura el `run_id` (o su ausencia) previo al bloque."""
        if self._token is not None:
            run_id_context.reset(self._token)
