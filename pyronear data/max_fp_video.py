#!/usr/bin/env python3
import os
import argparse

def enforce_max_images(root_dir, max_images=30):
    """
    Para cada directorio que termine en '_fp':
      - Busca todos los archivos .jpg.
      - Si hay más de max_images, ordena por nombre y elimina los excedentes.
      - Para cada imagen eliminada, borra también su .txt en labels/.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Solo procesar carpetas *_fp
        if os.path.basename(dirpath).endswith('_fp'):
            # Listado de .jpg en esta carpeta
            jpgs = [f for f in filenames if f.lower().endswith('.jpg')]
            if len(jpgs) > max_images:
                jpgs.sort()  # orden alfabético; puedes cambiar a sorted by mtime si lo prefieres
                to_delete = jpgs[max_images:]
                labels_dir = os.path.join(dirpath, 'labels')

                for img in to_delete:
                    img_path = os.path.join(dirpath, img)
                    try:
                        os.remove(img_path)
                        print(f"Imagen eliminada: {img_path}")
                    except OSError as e:
                        print(f"Error eliminando imagen {img_path}: {e}")

                    # Eliminar el .txt asociado en labels/
                    label_name = os.path.splitext(img)[0] + '.txt'
                    label_path = os.path.join(labels_dir, label_name)
                    if os.path.isfile(label_path):
                        try:
                            os.remove(label_path)
                            print(f"Etiqueta eliminada: {label_path}")
                        except OSError as e:
                            print(f"Error eliminando etiqueta {label_path}: {e}")

    print("Proceso completado.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Limita a 30 las .jpg en carpetas *_fp y borra sus labels correspondientes."
    )
    parser.add_argument(
        'root_dir',
        help="Ruta al directorio raíz donde comenzar la búsqueda"
    )
    parser.add_argument(
        '--max',
        type=int,
        default=30,
        help="Número máximo de imágenes .jpg a conservar (por defecto: 30)"
    )
    args = parser.parse_args()
    enforce_max_images(args.root_dir, args.max)
