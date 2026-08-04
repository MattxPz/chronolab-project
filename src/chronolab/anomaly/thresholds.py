"""`Thresholder`: convierte scores continuos en etiquetas para una rejilla de alfa.

Separar puntuar de umbralizar es obligatorio: VUS-PR necesita el score continuo
y F1 por rangos necesita etiquetas. Si el detector devolviera etiquetas se
perderia irreversiblemente lo que exige la metrica principal.
"""
