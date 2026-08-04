"""Unica ruta de lectura de artefactos. Es la API que consume la app.

Tambien es el unico constructor de `ScoringFrame`, lo que garantiza que un
detector jamas reciba predicciones dentro de muestra (fuga L9).
"""
