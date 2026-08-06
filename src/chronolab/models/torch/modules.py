"""Definicion del `nn.Module`: encoder LSTM y cabeza de cuantiles.

Arquitectura, y por que cada pieza es la que es
-----------------------------------------------
``contexto -> LSTM -> ultimo estado oculto -> [concat exogenas futuras] -> MLP
-> (h x n_cuantiles)``.

1. **Prediccion directa multi-paso.** La cabeza proyecta de una sola vez los
   `h` pasos del horizonte, en lugar de emitir un paso y realimentarlo. No es
   una simplificacion: es la unica variante que no acumula su propio error a lo
   largo del horizonte, y hace que el modelo se pueda comparar de tu a tu con
   la estrategia `direct` de `models/adapters/mlforecast.py` sin que la
   diferencia de resultados mezcle "arquitectura" con "estrategia de
   despliegue". A cambio, el numero de parametros de la cabeza crece con `h`,
   que es un coste explicito y medido (`n_params` se reporta en el
   leaderboard).
2. **Las exogenas futuras entran en la cabeza, no en el encoder.** El encoder
   resume el pasado; las exogenas del tramo a predecir no son pasado. Meterlas
   en la secuencia del LSTM obligaria a inventar un valor de la objetivo para
   esos instantes —justo lo que se quiere predecir—, asi que se aplanan
   (``h x n_exog``) y se concatenan al estado oculto antes del MLP. Es la
   forma mas simple que respeta la separacion, y la que hace evidente en el
   codigo que ninguna exogena futura toca la recurrencia.
3. **Cabeza de cuantiles, no puntual.** La salida tiene `n_cuantiles` canales
   por paso y se entrena con perdida pinball (`models/torch/trainer.py`). Un
   modelo puntual al que despues se le pega un intervalo gaussiano estaria
   fingiendo una calibracion que nadie ha ajustado.

Import perezoso de `torch`
---------------------------
`torch` vive en el extra `deep` (D20) y el job `quality` de CI hace `uv sync`
a secas, pero `tests/unit/test_module_tree.py` importa **todos** los modulos
del arbol en ese entorno. Por eso aqui no hay ningun `import torch` a nivel de
modulo y la clase no se declara en el cuerpo del fichero: se construye la
primera vez que se pide, dentro de `_net_class()`, y se cachea. El coste es
una indireccion; el beneficio es que el arbol de modulos sigue siendo
importable sin medio gigabyte de dependencias, que es un invariante del
proyecto y no una preferencia.
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_lstm_net", "count_parameters"]

_NET_CLASS: Any = None
"""Cache de la clase construida por `_net_class`. `Any` por la cuarentena D16."""


def _net_class() -> Any:
    """Construye (una sola vez) la clase del encoder LSTM y la devuelve.

    Returns
    -------
    Any
        La clase `nn.Module`. Tipada como `Any` a proposito: `torch` esta en la
        cuarentena de tipos de D16 (docs/ARCHITECTURE.md) y sin el extra `deep`
        instalado mypy no puede resolverla de todos modos.

    Raises
    ------
    ImportError
        Si `torch` no esta instalado. El mensaje dice exactamente que ejecutar.
    """
    global _NET_CLASS
    if _NET_CLASS is not None:
        return _NET_CLASS

    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover  ejercitado solo sin el extra deep
        raise ImportError(
            "chronolab.models.torch.modules necesita el extra 'deep': `uv sync --extra deep`."
        ) from exc

    class LSTMQuantileNet(nn.Module):
        """Encoder LSTM con proyeccion directa a ``h x n_cuantiles``.

        Parameters
        ----------
        n_context_features
            Canales de la secuencia de contexto: la objetivo mas las exogenas
            historicas alineadas con ella.
        n_futr_features
            Exogenas conocidas a futuro, por paso del horizonte. Cero si no hay.
        h
            Horizonte en pasos.
        n_quantiles
            Cuantiles de la rejilla del run.
        hidden_size
            Unidades del estado oculto del LSTM.
        num_layers
            Capas apiladas del LSTM.
        dropout
            Dropout entre capas del LSTM (solo aplica con ``num_layers > 1``) y
            antes de la capa de salida de la cabeza.
        """

        def __init__(
            self,
            *,
            n_context_features: int,
            n_futr_features: int,
            h: int,
            n_quantiles: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.h = h
            self.n_quantiles = n_quantiles
            self.n_futr_features = n_futr_features

            self.encoder = nn.LSTM(
                input_size=n_context_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            head_input = hidden_size + h * n_futr_features
            self.head = nn.Sequential(
                nn.Linear(head_input, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, h * n_quantiles),
            )

        def forward(self, context: Any, futr: Any) -> Any:
            """Predice el horizonte completo para un lote de ventanas.

            Parameters
            ----------
            context
                ``(batch, input_size, n_context_features)``.
            futr
                ``(batch, h, n_futr_features)``. Puede tener la ultima
                dimension a cero si el modelo no usa exogenas futuras.

            Returns
            -------
            torch.Tensor
                ``(batch, h, n_quantiles)``, en escala estandarizada. Los
                cuantiles salen **ordenados**: la capa emite incrementos y se
                acumulan con un `cumsum` sobre valores no negativos, de modo
                que el cruce de cuantiles es imposible por construccion en vez
                de repararse despues. El primer canal es el cuantil mas bajo.
            """
            encoded, _ = self.encoder(context)
            last = encoded[:, -1, :]
            if self.n_futr_features:
                last = torch.cat([last, futr.reshape(futr.shape[0], -1)], dim=1)
            raw = self.head(last).reshape(-1, self.h, self.n_quantiles)
            base = raw[..., :1]
            increments = nn.functional.softplus(raw[..., 1:])
            return torch.cat([base, base + torch.cumsum(increments, dim=-1)], dim=-1)

    _NET_CLASS = LSTMQuantileNet
    return _NET_CLASS


def build_lstm_net(
    *,
    n_context_features: int,
    n_futr_features: int,
    h: int,
    n_quantiles: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> Any:
    """Instancia el encoder LSTM con cabeza de cuantiles.

    Parameters
    ----------
    n_context_features
        Canales de la secuencia de contexto (objetivo mas exogenas historicas).
    n_futr_features
        Exogenas conocidas a futuro por paso del horizonte.
    h
        Horizonte en pasos.
    n_quantiles
        Cuantiles de la rejilla del run.
    hidden_size, num_layers, dropout
        Hiperparametros de la red.

    Returns
    -------
    Any
        Un `nn.Module` listo para entrenar.

    Raises
    ------
    ImportError
        Si `torch` no esta instalado.
    ValueError
        Si alguna dimension declarada es incoherente.
    """
    if n_context_features < 1:
        raise ValueError(f"n_context_features debe ser >= 1: {n_context_features}")
    if n_futr_features < 0:
        raise ValueError(f"n_futr_features debe ser >= 0: {n_futr_features}")
    if h < 1:
        raise ValueError(f"h debe ser >= 1: {h}")
    if n_quantiles < 1:
        raise ValueError(f"n_quantiles debe ser >= 1: {n_quantiles}")
    if hidden_size < 1:
        raise ValueError(f"hidden_size debe ser >= 1: {hidden_size}")
    if num_layers < 1:
        raise ValueError(f"num_layers debe ser >= 1: {num_layers}")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout debe estar en [0, 1): {dropout}")

    cls = _net_class()
    return cls(
        n_context_features=n_context_features,
        n_futr_features=n_futr_features,
        h=h,
        n_quantiles=n_quantiles,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )


def count_parameters(net: Any) -> int:
    """Numero de parametros **entrenables** de una red.

    Es el numero que se persiste en ``model_runs.n_params`` y que el
    leaderboard reporta junto al coste en segundos: la mitad del eje
    precision-coste del proyecto (docs/ARCHITECTURE.md §7.4).

    Parameters
    ----------
    net
        Un `nn.Module`.

    Returns
    -------
    int
        Suma de `numel()` sobre los parametros con `requires_grad`.
    """
    return int(sum(p.numel() for p in net.parameters() if p.requires_grad))
