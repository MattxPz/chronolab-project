"""Metricas de anomalia: VUS-PR, F1 por rangos y metricas de afiliacion.

Nunca point-adjusted F1: la literatura de benchmarking (TSB-AD) lo describe como
una metrica que infla resultados hasta hacer que el ruido parezca estado del
arte. Antes de comparar, se iguala la mascara `scorable` entre detectores.
"""
