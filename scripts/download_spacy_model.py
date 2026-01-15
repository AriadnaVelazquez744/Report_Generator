"""
Script para descargar modelos de spaCy
"""
import sys
import subprocess
import os

def download_spacy_model(model_name="es_core_news_sm"):
    """
    Descarga un modelo de spaCy usando el comando oficial
    
    Args:
        model_name: Nombre del modelo a descargar
    """
    print(f"Descargando modelo de spaCy: {model_name}")
    print("Esto puede tardar varios minutos...")
    
    try:
        # Usar subprocess para ejecutar el comando en el entorno correcto
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"✓ Modelo {model_name} descargado correctamente!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error descargando modelo: {e}")
        if e.stdout:
            print(f"  Salida: {e.stdout}")
        if e.stderr:
            print(f"  Error: {e.stderr}")
        
        # Si el error es por wheel inválido, sugerir limpiar caché
        if "invalid" in str(e.stderr).lower() or "wheel" in str(e.stderr).lower():
            print("\n⚠ El archivo descargado parece estar corrupto.")
            print("  Esto puede ser un problema temporal de red o caché.")
            print("\n  Intenta:")
            print("  1. Limpiar caché de pip: pip cache purge")
            print("  2. Intentar de nuevo en unos minutos")
            print("  3. O usar el modelo más pequeño: es_core_news_sm")
        
        # Intentar método alternativo
        print("\nIntentando método alternativo...")
        try:
            import spacy.cli
            spacy.cli.download(model_name)
            print(f"✓ Modelo {model_name} descargado correctamente (método alternativo)!")
            return True
        except Exception as e2:
            print(f"✗ Error con método alternativo: {e2}")
            print("\nPor favor, intenta descargar el modelo manualmente:")
            print(f"  uv run python -m spacy download {model_name}")
            if model_name == "es_core_news_lg":
                print("\n  O usa el modelo más pequeño que funciona mejor:")
                print("  uv run python -m spacy download es_core_news_sm")
            return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Descargar modelo de spaCy")
    parser.add_argument(
        "--model",
        default="es_core_news_sm",
        choices=["es_core_news_sm", "es_core_news_lg","es_dep_news_trf"],
        help="Modelo a descargar (default: es_core_news_sm)"
    )
    
    args = parser.parse_args()
    
    success = download_spacy_model(args.model)
    sys.exit(0 if success else 1)

