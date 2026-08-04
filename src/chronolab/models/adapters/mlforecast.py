"""Adaptador de mlforecast con LightGBM y XGBoost, en modo recursivo y directo.

Unico adaptador con `supports_recursive = True`: es quien sabe realimentar la
prediccion en los lags sin romper la disponibilidad temporal.
"""
