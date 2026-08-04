"""Reglas de rollup de `forecasts` a `metrics`, con marginalizacion explicita.

Dos reglas innegociables: toda fila se calcula desde `forecasts` (nunca se
agrega un agregado, porque MASE y sMAPE son cocientes) y no se comparan filas
con distinto `futr_vintage`.
"""
