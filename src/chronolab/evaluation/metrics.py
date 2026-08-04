"""Metricas de prediccion: MASE, RMSE, MAE, sMAPE, pinball, cobertura y CRPS discreto.

El denominador de MASE se calcula por serie y por ventana con el train de esa
ventana; usarlo global seria fuga. Se persiste para que sea auditable.
"""
