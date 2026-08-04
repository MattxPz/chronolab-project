"""Adaptador de Prophet: un ajuste por serie, con festivos y regresores exogenos.

Coste O(n_series) por ventana; es el principal motivo del riesgo R2 y de que el
dataset de evaluacion sea un subconjunto curado.
"""
