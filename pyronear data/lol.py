#!/usr/bin/env python3
import os
import argparse

def remove_predicted_images(root_dir):
    """
    Recorre root_dir en todos sus subdirectorios y elimina
    cualquier archivo .jpg cuyo nombre termine en '_predicted.jpg'.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            # Comprobamos extensión .jpg y sufijo '_predicted'
            if fname.lower().endswith('_predicted.jpg'):
                file_path = os.path.join(dirpath, fname)
                try:
                    os.remove(file_path)
                    print(f"Eliminado: {file_path}")
                except OSError as e:
                    print(f"Error al eliminar {file_path}: {e}")
    print("Proceso completado.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Elimina recursivamente todas las imágenes *_predicted.jpg en un directorio."
    )
    parser.add_argument(
        'root_dir',
        help="Ruta al directorio raíz donde comenzar la búsqueda"
    )
    args = parser.parse_args()
    remove_predicted_images(args.root_dir)
