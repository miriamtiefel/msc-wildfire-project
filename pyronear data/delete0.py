#!/usr/bin/env python3
import os
import re
import argparse

def clean_zero_bboxes(root_dir):
    """
    Recorre recursivamente root_dir, busca todos los archivos .txt (labels)
    y elimina de cada uno las líneas cuyo bounding box esté completamente en 0
    (por ejemplo: "0 0.000000 0.000000 0.000000 0.000000").
    Si, al eliminar esas líneas, el archivo queda sin contenido, se dejará vacío.
    """
    # Patrón que detecta: <clase> 0(.0*) 0(.0*) 0(.0*) 0(.0*)
    pattern = re.compile(r'^\s*\d+\s+0+(?:\.0+)?(?:\s+0+(?:\.0+)?){3}\s*$')

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith('.txt'):
                continue
            file_path = os.path.join(dirpath, fname)
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Filtrar sólo las líneas que **no** son bounding boxes todas en cero
            filtered = [line for line in lines if not pattern.match(line)]
            if len(filtered) != len(lines):
                # Re-escribir el archivo con las líneas válidas (o vacío)
                with open(file_path, 'w') as f:
                    f.writelines(filtered)
                print(f"Procesado: {file_path} (eliminadas {len(lines) - len(filtered)} líneas de bbox=0)")

    print("Proceso completado.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Limpia archivos de labels eliminando líneas de bounding boxes completamente en cero."
    )
    parser.add_argument(
        'root_dir',
        help="Directorio raíz donde buscar recursivamente los archivos .txt de labels"
    )
    args = parser.parse_args()
    clean_zero_bboxes(args.root_dir)
