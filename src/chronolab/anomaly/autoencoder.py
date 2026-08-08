"""LSTM-Autoencoder: error de reconstruccion como score de anomalia.

Una red pequena -encoder LSTM, cuello de botella lineal, decoder LSTM- se
entrena a reconstruir ventanas deslizantes de la objetivo, escalada por serie.
Un tramo que la red reconstruye mal es un tramo que no se parece a lo que vio
en el entrenamiento, y el error cuadratico de reconstruccion de la ventana es
la magnitud cruda de no conformidad.

**Entrena solo sobre tramos considerados normales.** Antes de construir las
ventanas de entrenamiento se descartan las que contienen algun punto cuyo
z-score robusto (mediana / MAD, no sensible a la propia anomalia que se quiere
excluir) supera `trim_z`. Sin este filtro la red aprenderia a reconstruir
tambien la anomalia -es exactamente lo que un autoencoder hace bien-, y el
error dejaria de distinguir nada.

**Mismo puente a `score` comparable que `chronolab.anomaly.isolation`.** El
error de reconstruccion no vive en una escala interpretable por si solo -
depende de la arquitectura, del escalado, del numero de epocas- asi que se
conformaliza exactamente igual: `-log10(p)` del error contra un pool de
calibracion nunca visto por la red. `pool_score` y `pool_severity` se
reutilizan literalmente de `chronolab.anomaly.isolation`: es la misma
garantia (Laxhammar & Falkman, 2010) aplicada a otra magnitud, y
`chronolab.anomaly.events.aggregate_events` no necesita saber cual.

**El reparto entrenamiento/calibracion es el mismo split conformal que en
IsolationForest**, y por la misma razon: una red evaluada sobre ventanas que
ya vio durante el entrenamiento las reconstruye de forma optimista, lo que
rompe la intercambiabilidad justo donde el p-valor la necesita.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from chronolab.anomaly.conformal import SCORE_COLUMNS
from chronolab.anomaly.isolation import pool_score, pool_severity
from chronolab.anomaly.protocols import DetectorRequirements, ScoringFrame
from chronolab.errors import CutoffViolation
from chronolab.panel import PanelSpec
from chronolab.types import DetectorId

__all__ = ["AutoencoderDetector", "FittedAutoencoderDetector"]

_EPS = 1e-8


def _require_torch() -> Any:
    """Importa `torch` bajo demanda.

    Returns
    -------
    Any
        El modulo `torch`. Tipo `Any` a proposito: `torch` esta en la
        cuarentena de tipos de docs/ARCHITECTURE.md D16.

    Raises
    ------
    ImportError
        Si `torch` no esta instalado.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover  ejercitado solo sin el extra deep
        raise ImportError(
            "chronolab.anomaly.autoencoder necesita el extra 'deep': `uv sync --extra deep`."
        ) from exc
    return torch


def _build_net(*, hidden_size: int, latent_size: int) -> Any:
    """Construye el `nn.Module` del autoencoder.

    Arquitectura "repeat vector": el encoder resume la ventana en su ultimo
    estado oculto, un cuello de botella lineal lo comprime al espacio latente
    y lo expande de vuelta, y el decoder recibe ese contexto **repetido** en
    cada paso -no autorregresivo-, que es la variante mas simple de
    entrenar y evaluar en paralelo sobre todo el batch.

    Importa `torch` por su cuenta -en vez de recibirlo como parametro, como
    hacen el resto de funciones del modulo- para que la clase se declare contra
    el `nn.Module` de verdad donde el extra `deep` este instalado. Es el mismo
    patron que `chronolab.models.torch.modules._net_class`, y como alli, el
    import vive **dentro** de la funcion: a nivel de modulo romperia
    `tests/unit/test_module_tree.py`, que importa todo el arbol en el entorno
    por defecto de CI, donde `torch` no esta.

    Sin el extra, mypy resuelve `nn.Module` como `Any` y `disallow_subclassing_any`
    marcaria esta linea, asi que el modulo esta en la lista de excepciones de
    `pyproject.toml`. La consecuencia practica que conviene recordar: **este
    fallo no lo reproduce un `uv run mypy` local con `torch` instalado**, solo
    el typecheck de CI.

    Parameters
    ----------
    hidden_size
        Anchura del estado oculto de encoder y decoder.
    latent_size
        Dimension del cuello de botella.

    Returns
    -------
    torch.nn.Module
        Con `forward(x)` de forma ``(batch, seq_len, 1) -> (batch, seq_len, 1)``.
    """
    from torch import nn

    class _LSTMAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
            self.to_latent = nn.Linear(hidden_size, latent_size)
            self.from_latent = nn.Linear(latent_size, hidden_size)
            self.decoder = nn.LSTM(
                input_size=hidden_size, hidden_size=hidden_size, batch_first=True
            )
            self.output = nn.Linear(hidden_size, 1)

        def forward(self, x: Any) -> Any:
            _, (h_n, _) = self.encoder(x)
            latent = self.to_latent(h_n[-1])
            context = self.from_latent(latent)
            seq_len = x.shape[1]
            decoder_input = context.unsqueeze(1).repeat(1, seq_len, 1)
            decoded, _ = self.decoder(decoder_input)
            output: Any = self.output(decoded)
            return output

    return _LSTMAutoencoder()


@dataclass(frozen=True, slots=True)
class AutoencoderDetector:
    """Configuracion del detector. Inmutable y sin estado calibrado.

    Parameters
    ----------
    seq_len
        Longitud de la ventana deslizante que se reconstruye.
    hidden_size, latent_size
        Anchura del estado oculto y del cuello de botella.
    calib_fraction
        Fraccion, por tiempo, del tramo de calibracion reservada como pool de
        p-valores y **no** vista por la red durante el entrenamiento.
    trim_z
        Umbral de z-score robusto (mediana / MAD) por encima del cual una
        ventana de entrenamiento se descarta por contener un punto que no
        parece normal.
    epochs, batch_size, lr
        Hiperparametros del bucle de entrenamiento (Adam, MSE).
    alpha_ref
        Nivel de referencia de `severity`.
    min_calib
        Ventanas minimas exigidas en el tramo de entrenamiento y en el pool
        de calibracion.
    seed
        Semilla de `torch` y del barajado de minibatches.
    device
        Dispositivo de `torch` (``"cpu"`` por defecto).

    Raises
    ------
    ValueError
        Si algun parametro cae fuera de su dominio.
    """

    seq_len: int = 24
    hidden_size: int = 16
    latent_size: int = 8
    calib_fraction: float = 0.3
    trim_z: float = 4.0
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    alpha_ref: float = 0.05
    min_calib: int = 50
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Valida la configuracion."""
        if self.seq_len < 2:
            raise ValueError(f"seq_len debe ser >= 2: {self.seq_len}")
        if self.hidden_size < 1 or self.latent_size < 1:
            raise ValueError("hidden_size y latent_size deben ser >= 1")
        if not 0.0 < self.calib_fraction < 1.0:
            raise ValueError(f"calib_fraction fuera de (0, 1): {self.calib_fraction}")
        if self.trim_z <= 0.0:
            raise ValueError(f"trim_z debe ser > 0: {self.trim_z}")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs y batch_size deben ser >= 1")
        if self.lr <= 0.0:
            raise ValueError(f"lr debe ser > 0: {self.lr}")
        if not 0.0 < self.alpha_ref < 1.0:
            raise ValueError(f"alpha_ref fuera de (0, 1): {self.alpha_ref}")
        if self.min_calib < 1:
            raise ValueError(f"min_calib debe ser >= 1: {self.min_calib}")

    @property
    def detector_id(self) -> DetectorId:
        """Identificador estable. Clave de particion de `anomaly_scores`."""
        return DetectorId(
            f"lstm_ae_s{self.seq_len:03d}_h{self.hidden_size}_l{self.latent_size}"
            f"_c{round(self.calib_fraction * 100):02d}"
        )

    @property
    def requires(self) -> DetectorRequirements:
        """Necesidades declaradas. `needs_forecast=False`: reconstruye la objetivo, no un residuo."""
        return DetectorRequirements(
            needs_forecast=False,
            needs_quantiles=False,
            window=self.seq_len,
            needs_calibration=True,
            fit_cost="expensive",
        )

    def fit(self, calib: ScoringFrame) -> FittedAutoencoderDetector:
        """Calibra el detector con un tramo anterior al que se puntuara.

        Reparte `calib` cronologicamente igual que
        `chronolab.anomaly.isolation.IsolationForestDetector.fit`: el primer
        tramo entrena la red -tras descartar sus ventanas no normales-, el
        resto, nunca visto por la red, construye el pool de p-valores.

        Parameters
        ----------
        calib
            Tramo de calibracion.

        Returns
        -------
        FittedAutoencoderDetector
            Con ``cutoff = calib.end``.

        Raises
        ------
        ValueError
            Si falta la columna objetivo o si, tras trocear y filtrar, el
            tramo de entrenamiento o el pool no llegan al minimo exigido.
        """
        if "y" not in calib.df.columns:
            raise ValueError("el tramo de calibracion no trae la columna 'y'")

        split_at = _time_split(calib.df["ds"], self.calib_fraction)
        train_df = calib.df.loc[calib.df["ds"] <= split_at]
        pool_df = calib.df.loc[calib.df["ds"] > split_at]

        scaler = _fit_scaler(train_df)
        train_windows, _ = _windows(train_df, scaler, seq_len=self.seq_len)
        kept = _trim_normal(train_windows, trim_z=self.trim_z)
        if kept.shape[0] < self.min_calib:
            raise ValueError(
                f"el tramo de entrenamiento tiene {kept.shape[0]} ventanas normales "
                f"tras el filtro de {self.trim_z}-sigma; hacen falta al menos {self.min_calib}"
            )

        torch_module = _require_torch()
        torch_module.manual_seed(self.seed)
        net = _build_net(hidden_size=self.hidden_size, latent_size=self.latent_size)
        net.to(self.device)
        _train(
            torch_module,
            net,
            kept,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            device=self.device,
        )
        net.eval()

        pool_windows, _ = _windows(pool_df, scaler, seq_len=self.seq_len)
        if pool_windows.shape[0] < self.min_calib:
            raise ValueError(
                f"el pool de calibracion tiene {pool_windows.shape[0]} ventanas puntuables; "
                f"hacen falta al menos {self.min_calib}"
            )
        pool_error, _ = _reconstruction_error(torch_module, net, pool_windows, device=self.device)

        tails = _tails(calib.df, seq_len=self.seq_len)
        return FittedAutoencoderDetector(
            detector=self,
            spec=calib.spec,
            cutoff=calib.end,
            net=net,
            scaler=scaler,
            pool=pool_error,
            tails=tails,
        )


@dataclass(frozen=True, slots=True)
class FittedAutoencoderDetector:
    """Autoencoder calibrado hasta un instante concreto.

    Attributes
    ----------
    detector
        Configuracion de la que procede.
    spec
        Especificacion del panel con el que se calibro.
    cutoff
        Ultimo instante usado en calibracion.
    net
        Red ya entrenada, en modo evaluacion. Se trata como opaca.
    scaler
        ``unique_id -> (media, desviacion)`` ajustadas solo con el tramo de
        entrenamiento.
    pool
        Errores de reconstruccion del tramo de calibracion, nunca visto por
        la red.
    tails
        Ultimas ``seq_len - 1`` filas de `calib.df` por serie, para puentear
        la primera llamada a `score`.
    """

    detector: AutoencoderDetector
    spec: PanelSpec
    cutoff: pd.Timestamp
    net: Any
    scaler: dict[str, tuple[float, float]]
    pool: np.ndarray
    tails: dict[str, pd.DataFrame]

    @property
    def detector_id(self) -> DetectorId:
        """Identificador del detector del que procede."""
        return self.detector.detector_id

    def score(self, frame: ScoringFrame) -> pd.DataFrame:
        """Puntua cada marca de tiempo del tramo.

        Parameters
        ----------
        frame
            Tramo a puntuar, con ``frame.start > cutoff``.

        Returns
        -------
        pandas.DataFrame
            Columnas `chronolab.anomaly.conformal.SCORE_COLUMNS`.

        Raises
        ------
        CutoffViolation
            Si `frame` empieza en un instante que la calibracion ya conocia.
        """
        if frame.start <= self.cutoff:
            raise CutoffViolation(
                f"{self.detector_id} esta calibrado hasta {self.cutoff} y se le pide "
                f"puntuar desde {frame.start}, anterior o igual"
            )

        bridged = _bridge(frame.df, tails=self.tails)
        windows, positions = _windows(bridged, self.scaler, seq_len=self.detector.seq_len)

        out = frame.df[["unique_id", "ds"]].copy()
        out["unique_id"] = out["unique_id"].astype(str)
        n_rows = len(out)
        score = np.full(n_rows, np.nan, dtype=np.float32)
        severity = np.full(n_rows, np.nan, dtype=np.float32)
        calib_n = np.zeros(n_rows, dtype=np.int32)
        side = np.zeros(n_rows, dtype=np.int8)
        scorable = np.zeros(n_rows, dtype=bool)

        if windows.shape[0]:
            torch_module = _require_torch()
            raw, signed_last = _reconstruction_error(
                torch_module, self.net, windows, device=self.detector.device
            )
            key = pd.MultiIndex.from_arrays([out["unique_id"], out["ds"]])
            lookup = {(uid, ds): i for i, (uid, ds) in enumerate(positions)}
            for row_i, k in enumerate(key):
                pos = lookup.get(k)
                if pos is None:
                    continue
                scorable[row_i] = True
                score[row_i] = pool_score(raw[pos : pos + 1], self.pool)[0]
                severity[row_i] = pool_severity(
                    raw[pos : pos + 1], self.pool, alpha_ref=self.detector.alpha_ref
                )[0]
                calib_n[row_i] = self.pool.size
                value = signed_last[pos]
                side[row_i] = 1 if value > 0 else (-1 if value < 0 else 0)

        return pd.DataFrame(
            {
                "unique_id": out["unique_id"].to_numpy(),
                "ds": out["ds"].to_numpy(dtype="datetime64[ns]"),
                "score": score,
                "scorable": scorable,
                "severity": severity,
                "calib_n": calib_n,
                "side": side,
            }
        )[list(SCORE_COLUMNS)]


def _time_split(ds: pd.Series, calib_fraction: float) -> pd.Timestamp:
    """Instante que reparte un tramo cronologicamente segun `calib_fraction`.

    Identica en espiritu a `chronolab.anomaly.isolation._time_split`; se
    repite aqui porque cada modulo de detector es autosuficiente
    (docs/ARCHITECTURE.md §2) y la funcion es de un solo uso interno.

    Parameters
    ----------
    ds
        Marcas de tiempo del tramo.
    calib_fraction
        Fraccion final reservada al pool de calibracion.

    Returns
    -------
    pandas.Timestamp
        Filas con ``ds <= resultado`` entrenan la red; el resto forma el pool.
    """
    ordered = pd.Index(ds.unique()).sort_values()
    if len(ordered) < 2:
        return pd.Timestamp(ordered[0])
    index = int(np.clip(round(len(ordered) * (1.0 - calib_fraction)), 1, len(ordered) - 1))
    return pd.Timestamp(ordered[index - 1])


def _fit_scaler(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Media y desviacion de `y` por serie, ajustadas solo con `df`.

    Parameters
    ----------
    df
        Tramo de entrenamiento.

    Returns
    -------
    dict
        ``unique_id -> (media, desviacion con piso)``.
    """
    stats: dict[str, tuple[float, float]] = {}
    for uid, group in df.groupby("unique_id", sort=False):
        values = group["y"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        mean = float(finite.mean()) if finite.size else 0.0
        std = float(finite.std()) if finite.size else 1.0
        stats[str(uid)] = (mean, max(std, _EPS))
    return stats


def _windows(
    df: pd.DataFrame, scaler: dict[str, tuple[float, float]], *, seq_len: int
) -> tuple[np.ndarray, list[tuple[str, pd.Timestamp]]]:
    """Ventanas causales escaladas de longitud `seq_len`, terminadas en cada `t`.

    Parameters
    ----------
    df
        Trama con `unique_id`, `ds`, `y`.
    scaler
        Estadisticos de escalado por serie. Una serie ausente usa el par
        neutro ``(0.0, 1.0)``.
    seq_len
        Longitud de la ventana.

    Returns
    -------
    tuple
        Matriz ``(n_windows, seq_len)`` de valores escalados, y la lista de
        ``(unique_id, ds)`` del **ultimo** punto de cada ventana, en el mismo
        orden. Se descartan las ventanas con algun `y` no finito.
    """
    rows: list[np.ndarray] = []
    keys: list[tuple[str, pd.Timestamp]] = []
    for uid, group in df.groupby("unique_id", sort=False):
        key = str(uid)
        mean, std = scaler.get(key, (0.0, 1.0))
        ordered = group.sort_values("ds")
        values = ordered["y"].to_numpy(dtype=float)
        scaled = (values - mean) / std
        timestamps = ordered["ds"].to_numpy()
        for end in range(seq_len - 1, len(scaled)):
            window = scaled[end - seq_len + 1 : end + 1]
            if not np.isfinite(window).all():
                continue
            rows.append(window)
            keys.append((key, pd.Timestamp(timestamps[end])))
    if not rows:
        return np.zeros((0, seq_len), dtype=np.float32), []
    return np.stack(rows).astype(np.float32), keys


def _trim_normal(windows: np.ndarray, *, trim_z: float) -> np.ndarray:
    """Descarta ventanas que contienen algun punto no considerado normal.

    El z-score es robusto (mediana / MAD) y se calcula sobre **todas** las
    ventanas juntas, no por ventana: una anomalia real ocupa pocos puntos
    frente al total, asi que la mediana y la MAD del conjunto apenas se
    mueven por su presencia, que es justo lo que hace que el filtro no sea
    circular.

    Parameters
    ----------
    windows
        Ventanas escaladas, ``(n, seq_len)``.
    trim_z
        Umbral de exclusion.

    Returns
    -------
    numpy.ndarray
        Subconjunto de `windows` sin ninguna que contenga un punto por
        encima del umbral.
    """
    if windows.size == 0:
        return windows
    flat = windows.reshape(-1)
    median = float(np.median(flat))
    mad = float(np.median(np.abs(flat - median))) * 1.4826
    scale = max(mad, _EPS)
    z = np.abs((windows - median) / scale)
    keep = (z <= trim_z).all(axis=1)
    return windows[keep]


def _train(
    torch_module: Any,
    net: Any,
    windows: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> None:
    """Entrena la red a reconstruir `windows` con MSE y Adam.

    Parameters
    ----------
    torch_module
        Modulo `torch`.
    net
        Red a entrenar, mutada en el sitio.
    windows
        Ventanas de entrenamiento, ``(n, seq_len)``, ya filtradas.
    epochs, batch_size, lr
        Hiperparametros del bucle.
    device
        Dispositivo de `torch`.
    """
    tensor = torch_module.tensor(windows, dtype=torch_module.float32, device=device).unsqueeze(-1)
    optimizer = torch_module.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch_module.nn.MSELoss()
    n = tensor.shape[0]

    net.train()
    for _ in range(epochs):
        # Usa el generador global de `torch`, que `fit` ya sembro con `seed`:
        # asi el barajado de minibatches tambien es reproducible, no solo los
        # pesos iniciales de la red.
        permutation = torch_module.randperm(n)
        for start in range(0, n, batch_size):
            batch = tensor[permutation[start : start + batch_size]]
            optimizer.zero_grad()
            reconstructed = net(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()


def _reconstruction_error(
    torch_module: Any, net: Any, windows: np.ndarray, *, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Error cuadratico medio de reconstruccion por ventana, y el residuo con signo del ultimo paso.

    Parameters
    ----------
    torch_module
        Modulo `torch`.
    net
        Red ya entrenada, en modo evaluacion.
    windows
        Ventanas escaladas, ``(n, seq_len)``.
    device
        Dispositivo de `torch`.

    Returns
    -------
    tuple
        ``(error_mse, residuo_ultimo_paso)``, ambos ``(n,)`` en `float64`. El
        segundo es ``entrada[-1] - reconstruccion[-1]`` en escala estandarizada,
        y solo se usa para el signo de `side`.
    """
    tensor = torch_module.tensor(windows, dtype=torch_module.float32, device=device).unsqueeze(-1)
    with torch_module.no_grad():
        reconstructed = net(tensor)
    error = (reconstructed - tensor).pow(2).mean(dim=(1, 2))
    last_residual = tensor[:, -1, 0] - reconstructed[:, -1, 0]
    return (
        error.detach().cpu().numpy().astype(np.float64),
        last_residual.detach().cpu().numpy().astype(np.float64),
    )


def _tails(df: pd.DataFrame, *, seq_len: int) -> dict[str, pd.DataFrame]:
    """Ultimas `seq_len - 1` filas de cada serie, para puentear el siguiente `score`.

    Parameters
    ----------
    df
        Trama de la que extraer la cola. Tipicamente `calib.df`.
    seq_len
        Longitud de la ventana del detector.

    Returns
    -------
    dict
        ``unique_id -> ultimas filas``, ordenadas por `ds`.
    """
    needed = seq_len - 1
    tails: dict[str, pd.DataFrame] = {}
    if needed <= 0:
        return tails
    for uid, group in df.groupby("unique_id", sort=False):
        ordered = group.sort_values("ds")
        tails[str(uid)] = ordered.tail(needed).reset_index(drop=True)
    return tails


def _bridge(df: pd.DataFrame, *, tails: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Antepone la cola de calibracion de cada serie al tramo a puntuar.

    Parameters
    ----------
    df
        Tramo a puntuar.
    tails
        Colas por serie, de `FittedAutoencoderDetector.tails`.

    Returns
    -------
    pandas.DataFrame
        `df` con las colas antepuestas donde existan. Solo lleva las columnas
        `unique_id`, `ds`, `y`, que es todo lo que `_windows` necesita.
    """
    columns = ["unique_id", "ds", "y"]
    parts: list[pd.DataFrame] = []
    for uid, group in df.groupby("unique_id", sort=False):
        key = str(uid)
        tail = tails.get(key)
        ordered = group.sort_values("ds")[columns]
        if tail is not None and not tail.empty:
            ordered = pd.concat([tail[columns], ordered], ignore_index=True)
        parts.append(ordered)
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)
