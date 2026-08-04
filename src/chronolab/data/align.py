"""Normalizacion temporal: UTC ingenuo, rejilla regular, huecos y duplicados.

Punto unico de conversion horaria del proyecto (`to_utc_naive`). Al vivir toda
la logica de DST aqui, el invariante I2 se audita en un solo sitio.
"""
