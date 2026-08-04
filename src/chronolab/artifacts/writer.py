"""Escritura atomica y particionada de artefactos, con manifest.

Se escribe en `.tmp-<run_id>` y se renombra; `manifest.json` va el ultimo. Hasta
entonces el run es invisible para el lector, lo que hace seguro que el cron
escriba mientras la app lee.
"""
