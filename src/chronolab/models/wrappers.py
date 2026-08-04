"""`ConformalWrapper`: convierte cualquier `Forecaster` puntual en probabilistico.

Calibra sobre un corte interno del propio train dentro de `fit`, de modo que la
calibracion nunca ve datos posteriores al cutoff.
"""
