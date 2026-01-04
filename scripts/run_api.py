"""
Script para ejecutar la API simplificada
"""
import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn  # type: ignore


if __name__ == "__main__":
    # Ejecutar la API simplificada (sin base de datos)
    uvicorn.run(
        "src.api.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=['src']
    )



