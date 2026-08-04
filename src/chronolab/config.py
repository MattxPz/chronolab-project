"""Modelos pydantic de `conf/*.yaml`, resolucion de rutas y hashing de configuracion.

El `config_hash` es el SHA-256 del JSON canonico de la configuracion efectiva y
entra en la identidad de cada run (docs/ARCHITECTURE.md A4).
"""
