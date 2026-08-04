"""`CachedSource`: decorador de cache en parquet sobre cualquier `DataSource`.

La cache vive fuera de las fuentes para que cada fuente sea trivialmente
testeable y la politica de invalidacion se cambie en un unico lugar.
"""
