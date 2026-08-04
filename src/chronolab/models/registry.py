"""Registro `model_id` -> fabrica de `Forecaster`, alimentado por `conf/models.yaml`.

La fabrica recibe un `PanelSpec` porque neuralforecast necesita las listas de
exogenas al construir la red; el objeto subyacente se difiere a `fit`, donde ya
se conoce `h` (docs/ARCHITECTURE.md D21).
"""
