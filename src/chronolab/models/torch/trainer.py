"""Bucle de entrenamiento con early stopping, semilla fija y registro de coste.

Cinco decisiones, y cada una responde a un modo de fallo concreto:

1. **El corte de validacion es temporal y sale del propio train.** Las ultimas
   `val_fraction` ventanas de `WindowBatch` —que estan ordenadas por serie y,
   dentro de cada serie, por tiempo— se reservan para vigilar el early
   stopping. Nunca se baraja antes de cortar: un corte aleatorio pondria
   ventanas cuyo contexto solapa con las de entrenamiento, y el criterio de
   parada quedaria medido sobre datos que el modelo ya vio. El tramo de
   evaluacion de la ventana del backtest no interviene en ningun momento: el
   motor no se lo pasa a `fit` (docs/ARCHITECTURE.md, fuga L2).
2. **Early stopping sobre esa validacion, con restauracion de los mejores
   pesos.** Parar y quedarse con los pesos de la ultima epoca —que es la peor,
   por eso se paro— es el error clasico que convierte el early stopping en
   ruido. Aqui se guarda el mejor estado y se restaura al terminar.
3. **Gradient clipping por norma.** Un LSTM sobre series con picos produce
   gradientes que explotan en cuanto una ventana cae sobre un atipico; el
   recorte lo convierte en un paso grande en vez de en un `NaN` permanente.
4. **Scheduler `ReduceLROnPlateau` sobre la perdida de validacion.** Reduce la
   tasa de aprendizaje cuando la validacion deja de mejorar, que es
   exactamente la senal que ya se esta calculando para el early stopping: no
   hace falta un segundo criterio ni un calendario fijo de epocas que habria
   que reajustar a mano por dataset.
5. **Semilla fija en las tres fuentes de aleatoriedad** (Python, numpy y
   torch) mas un generador propio para el barajado del `DataLoader`. Sin lo
   ultimo, dos runs con la misma semilla siguen difiriendo, porque el
   `shuffle` del cargador consume el generador global.

Import perezoso de `torch`: igual que en `models/torch/modules.py`, vive dentro
de las funciones para que el arbol de modulos siga siendo importable sin el
extra `deep`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from chronolab.models.torch.dataset import WindowBatch

__all__ = ["TrainConfig", "TrainReport", "seed_everything", "train_lstm"]


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Presupuesto y regularizacion del entrenamiento.

    Los valores por defecto estan elegidos para que un backtest completo sea
    viable en CPU modesta: `max_epochs` acotado, lotes grandes y paciencia
    corta. Subirlos es legitimo, pero es una decision del llamante y queda
    registrada en el coste medido, no escondida en el modulo.

    Parameters
    ----------
    max_epochs
        Tope duro de epocas. Es el presupuesto: el early stopping puede parar
        antes, nunca despues.
    batch_size
        Ventanas por lote.
    learning_rate
        Tasa inicial de Adam.
    weight_decay
        Regularizacion L2 de Adam.
    patience
        Epocas sin mejorar la validacion antes de parar.
    min_delta
        Mejora minima de la perdida de validacion para contar como tal. Sin
        ella, una fluctuacion en el sexto decimal reinicia la paciencia y el
        early stopping no para nunca.
    grad_clip_norm
        Norma maxima del gradiente. ``None`` desactiva el recorte.
    lr_factor, lr_patience
        Factor y paciencia del `ReduceLROnPlateau`.
    val_fraction
        Fraccion final de ventanas reservada para validacion.
    num_workers
        Procesos del `DataLoader`. Cero por defecto: con lotes que caben en
        memoria, arrancar procesos cuesta mas de lo que ahorra, y en Windows
        multiplica el coste de importar torch por trabajador.
    device
        ``"cpu"``, ``"cuda"`` o ``None`` para elegir automaticamente.

    Raises
    ------
    ValueError
        Si algun presupuesto es incoherente.
    """

    max_epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    min_delta: float = 1e-4
    grad_clip_norm: float | None = 1.0
    lr_factor: float = 0.5
    lr_patience: int = 2
    val_fraction: float = 0.2
    num_workers: int = 0
    device: str | None = None

    def __post_init__(self) -> None:
        """Valida el presupuesto declarado."""
        if self.max_epochs < 1:
            raise ValueError(f"max_epochs debe ser >= 1: {self.max_epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size debe ser >= 1: {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate debe ser > 0: {self.learning_rate}")
        if self.patience < 1:
            raise ValueError(f"patience debe ser >= 1: {self.patience}")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"val_fraction debe estar en (0, 1): {self.val_fraction}")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm debe ser > 0 o None: {self.grad_clip_norm}")


@dataclass(frozen=True, slots=True)
class TrainReport:
    """Que ocurrio durante el entrenamiento. Se persiste, no se imprime y se olvida.

    Attributes
    ----------
    epochs_run
        Epocas efectivamente ejecutadas, que con early stopping es menor o
        igual que `TrainConfig.max_epochs`.
    best_epoch
        Epoca cuyos pesos se restauraron (base cero).
    best_val_loss
        Mejor perdida de validacion alcanzada.
    train_losses, val_losses
        Curvas por epoca, para poder dibujar la convergencia sin reentrenar.
    early_stopped
        ``True`` si se paro por paciencia agotada en vez de por agotar el
        presupuesto de epocas.
    n_train_windows, n_val_windows
        Tamano efectivo de cada partición.
    fit_seconds
        Coste medido del bucle completo.
    """

    epochs_run: int
    best_epoch: int
    best_val_loss: float
    train_losses: tuple[float, ...]
    val_losses: tuple[float, ...]
    early_stopped: bool
    n_train_windows: int
    n_val_windows: int
    fit_seconds: float


def seed_everything(seed: int) -> None:
    """Fija la semilla de Python, numpy y torch.

    No cubre el barajado del `DataLoader`, que consume el generador global y
    por tanto depende del orden de las llamadas: para eso `train_lstm` pasa un
    `torch.Generator` propio y explicito.

    Parameters
    ----------
    seed
        Semilla global del run.

    Raises
    ------
    ImportError
        Si `torch` no esta instalado.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover  no hay GPU en CI
        torch.cuda.manual_seed_all(seed)


def _require_torch() -> Any:
    """Importa `torch` bajo demanda, con un mensaje util si falta el extra.

    Returns
    -------
    Any
        El modulo `torch`. Tipado como `Any` por la cuarentena D16.

    Raises
    ------
    ImportError
        Si `torch` no esta instalado.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover  ejercitado solo sin el extra deep
        raise ImportError(
            "chronolab.models.torch.trainer necesita el extra 'deep': `uv sync --extra deep`."
        ) from exc
    return torch


def _resolve_device(requested: str | None) -> str:
    """Elige el dispositivo: el pedido, o CUDA si esta disponible y si no CPU."""
    if requested is not None:
        return requested
    torch = _require_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def pinball_loss(prediction: Any, target: Any, quantiles: Any) -> Any:
    """Perdida pinball media sobre lote, horizonte y cuantiles.

    Es la regla de puntuacion propia del cuantil, la misma que
    `chronolab.evaluation.metrics.pinball_loss` usa para evaluar: entrenar con
    la metrica con la que se reporta evita el desajuste clasico de optimizar
    error cuadratico y publicar cobertura.

    Parameters
    ----------
    prediction
        ``(batch, h, n_quantiles)``.
    target
        ``(batch, h)``.
    quantiles
        Tensor ``(n_quantiles,)`` con los niveles.

    Returns
    -------
    torch.Tensor
        Escalar.
    """
    torch = _require_torch()
    difference = target.unsqueeze(-1) - prediction
    losses = torch.maximum(quantiles * difference, (quantiles - 1.0) * difference)
    return losses.mean()


def _split_windows(batch: WindowBatch, val_fraction: float) -> tuple[slice, slice]:
    """Corte temporal train/validacion sobre las ventanas, sin barajar.

    Parameters
    ----------
    batch
        Ventanas ya construidas, en el orden que produce `build_windows`.
    val_fraction
        Fraccion final reservada a validacion.

    Returns
    -------
    tuple of slice
        Rebanadas de entrenamiento y de validacion. Con muy pocas ventanas la
        de validacion puede quedarse en una sola: es poco, pero es honesto, y
        el `TrainReport` lo dice con `n_val_windows`.
    """
    total = len(batch)
    n_val = max(1, round(total * val_fraction)) if total > 1 else 0
    n_train = total - n_val
    return slice(0, n_train), slice(n_train, total)


def train_lstm(
    net: Any,
    batch: WindowBatch,
    *,
    quantiles: Sequence[float],
    config: TrainConfig,
    seed: int,
) -> TrainReport:
    """Entrena la red sobre las ventanas dadas y devuelve el informe del ajuste.

    La red se modifica **in situ** —queda con los mejores pesos restaurados— y
    lo que se devuelve es el registro de lo que paso, no el modelo: quien
    llama ya tiene la referencia.

    Parameters
    ----------
    net
        Red construida por `chronolab.models.torch.modules.build_lstm_net`.
    batch
        Ventanas de entrenamiento, ya escaladas.
    quantiles
        Rejilla de cuantiles del run, en el mismo orden que los canales de
        salida de la red.
    config
        Presupuesto y regularizacion.
    seed
        Semilla del run.

    Returns
    -------
    TrainReport

    Raises
    ------
    ImportError
        Si `torch` no esta instalado.
    ValueError
        Si no hay ninguna ventana con la que entrenar.
    """
    if len(batch) == 0:
        raise ValueError(
            "no hay ninguna ventana de entrenamiento: el tramo de train es mas corto "
            "que input_size + h, o su objetivo esta enteramente vacia"
        )

    torch = _require_torch()
    from torch.utils.data import DataLoader, TensorDataset

    seed_everything(seed)
    device = _resolve_device(config.device)
    net.to(device)

    train_slice, val_slice = _split_windows(batch, config.val_fraction)
    quantile_tensor = torch.tensor(list(quantiles), dtype=torch.float32, device=device)

    def _tensors(rows: slice) -> tuple[Any, Any, Any]:
        return (
            torch.from_numpy(batch.context[rows]),
            torch.from_numpy(batch.futr[rows]),
            torch.from_numpy(batch.target[rows]),
        )

    train_tensors = _tensors(train_slice)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        drop_last=False,
    )

    has_val = val_slice.stop > val_slice.start
    if has_val:
        val_context, val_futr, val_target = (t.to(device) for t in _tensors(val_slice))

    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience
    )

    best_loss = math.inf
    best_epoch = 0
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    epochs_without_improvement = 0
    early_stopped = False
    train_losses: list[float] = []
    val_losses: list[float] = []

    started = perf_counter()
    for epoch in range(config.max_epochs):
        net.train()
        running = 0.0
        seen = 0
        for context, futr, target in loader:
            context = context.to(device)
            futr = futr.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = pinball_loss(net(context, futr), target, quantile_tensor)
            loss.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
            optimizer.step()

            running += float(loss.item()) * context.shape[0]
            seen += int(context.shape[0])
        train_losses.append(running / max(seen, 1))

        if has_val:
            net.eval()
            with torch.no_grad():
                validation = float(
                    pinball_loss(net(val_context, val_futr), val_target, quantile_tensor).item()
                )
        else:  # pragma: no cover  solo con una unica ventana en todo el train
            validation = train_losses[-1]
        val_losses.append(validation)
        scheduler.step(validation)

        if validation < best_loss - config.min_delta:
            best_loss = validation
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                early_stopped = True
                break

    # Restaurar los mejores pesos, no los ultimos: si se paro por paciencia, los
    # ultimos son precisamente los que dejaron de mejorar.
    net.load_state_dict(best_state)
    fit_seconds = perf_counter() - started

    return TrainReport(
        epochs_run=len(train_losses),
        best_epoch=best_epoch,
        best_val_loss=best_loss,
        train_losses=tuple(train_losses),
        val_losses=tuple(val_losses),
        early_stopped=early_stopped,
        n_train_windows=train_slice.stop - train_slice.start,
        n_val_windows=val_slice.stop - val_slice.start,
        fit_seconds=fit_seconds,
    )


def predict_batch(net: Any, context: np.ndarray, futr: np.ndarray, *, device: str) -> np.ndarray:
    """Pasada de inferencia sobre un lote ya escalado.

    Parameters
    ----------
    net
        Red entrenada.
    context
        ``(n_series, input_size, n_context_features)``.
    futr
        ``(n_series, h, n_futr_features)``.
    device
        Dispositivo en el que evaluar.

    Returns
    -------
    numpy.ndarray
        ``(n_series, h, n_quantiles)`` en escala estandarizada. Invertir el
        escalado es responsabilidad del llamante, que es quien sabe a que serie
        corresponde cada fila.
    """
    torch = _require_torch()
    net.to(device)
    net.eval()
    with torch.no_grad():
        prediction = net(
            torch.from_numpy(np.ascontiguousarray(context, dtype=np.float32)).to(device),
            torch.from_numpy(np.ascontiguousarray(futr, dtype=np.float32)).to(device),
        )
    return np.asarray(prediction.cpu().numpy(), dtype=float)
