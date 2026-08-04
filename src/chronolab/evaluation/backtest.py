"""Motor de backtesting: recorre ventanas por modelo y persiste artefactos.

Aplica la politica de refit, verifica el cutoff en cada prediccion y registra
los fallos con `status` en lugar de silenciarlos.
"""
