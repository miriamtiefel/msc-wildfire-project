#!/usr/bin/env python3
import os
import argparse

def clean_fp_dirs(root_dir):
    """
    Recorre root_dir en busca de carpetas que terminen en '_fp'.
    Dentro de cada una:
      1) Vacía todos los .txt en la subcarpeta 'labels'.
      2) Elimina todos los archivos que terminen en '_predicted.jpg'.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Comprueba si la carpeta actual es un directorio *_fp
        if os.path.basename(dirpath).endswith('_fp'):
            # 1) Vaciar .txt en labels/
            labels_dir = os.path.join(dirpath, 'labels')
            if os.path.isdir(labels_dir):
                for fname in os.listdir(labels_dir):
                    if fname.lower().endswith('.txt'):
                        file_path = os.path.join(labels_dir, fname)
                        with open(file_path, 'w'):
                            pass
                        print(f"Vaciado: {file_path}")

            # 2) Eliminar archivos *_predicted.jpg en la raíz de este _fp
            for fname in filenames:
                if fname.endswith('_predicted.jpg'):
                    file_path = os.path.join(dirpath, fname)
                    try:
                        os.remove(file_path)
                        print(f"Eliminado: {file_path}")
                    except OSError as e:
                        print(f"Error al eliminar {file_path}: {e}")

    print("Proceso completado.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Limpia carpetas '*_fp': vacía .txt en labels/ y borra *_predicted.jpg"
    )
    parser.add_argument(
        'root_dir',
        help="Ruta al directorio raíz donde comenzar la búsqueda"
    )
    args = parser.parse_args()
    clean_fp_dirs(args.root_dir)
