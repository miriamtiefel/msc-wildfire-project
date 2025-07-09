#!/usr/bin/env python3
import os
import argparse

def compute_stats_and_avg_area(root_dir):
    """
    Recorre root_dir (que a su vez contiene carpetas labels/) y devuelve:
      videos, sequences, incendios, imgs_with_label, imgs_without_label,
      box_count, total_area
    Nota: NO calcula avg aquí, devolvemos total_area y box_count para poder agregarlos.
    """
    videos = sequences = incendios = imgs_with_label = imgs_without_label = 0
    total_area = 0.0
    box_count = 0

    for current_dir, dirs, files in os.walk(root_dir):
        if os.path.basename(current_dir) == 'labels':
            continue

        if 'labels' in dirs:
            labels_dir = os.path.join(current_dir, 'labels')

            # 1) incendio = carpeta labels NO vacía
            has_nonempty = any(
                os.path.isfile(os.path.join(labels_dir, f)) and
                os.path.getsize(os.path.join(labels_dir, f)) > 0
                for f in os.listdir(labels_dir)
            )
            if has_nonempty:
                incendios += 1

            # 2) video vs secuencia según profundidad relativa
            depth = os.path.relpath(current_dir, root_dir).count(os.sep)
            if depth == 1:
                videos += 1
            else:
                sequences += 1

            # 3) imágenes con/sin label
            for fname in files:
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    base = os.path.splitext(fname)[0]
                    label_file = os.path.join(labels_dir, base + '.txt')
                    if os.path.isfile(label_file) and os.path.getsize(label_file) > 0:
                        imgs_with_label += 1
                        # 4) acumular área de cada bbox
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) != 5:
                                    continue
                                _, _, _, w, h = parts
                                try:
                                    total_area += float(w) * float(h)
                                    box_count += 1
                                except ValueError:
                                    continue
                    else:
                        imgs_without_label += 1

    return videos, sequences, incendios, imgs_with_label, imgs_without_label, box_count, total_area


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Procesa splits train/val/test y calcula estadísticas + avg. área bbox."
    )
    parser.add_argument(
        'base_dir',
        help="Ruta al directorio que contiene las carpetas train, val y test"
    )
    args = parser.parse_args()

    splits = ['train', 'val', 'test']
    # Variables para estadísticas globales
    agg = {
        'videos': 0, 'sequences': 0, 'incendios': 0,
        'imgs_with_label': 0, 'imgs_without_label': 0,
        'box_count': 0, 'total_area': 0.0
    }

    for split in splits:
        split_dir = os.path.join(args.base_dir, split)
        if not os.path.isdir(split_dir):
            print(f"[!] No existe el split '{split}' en {args.base_dir}, se omite.")
            continue

        v, s, i, w, wo, bc, ta = compute_stats_and_avg_area(split_dir)
        avg = (ta / bc) if bc > 0 else 0.0
        total_vs = v + s
        total_img = w + wo

        print(f"\n=== Split: {split} ===")
        print(f"Videos:                       {v}/{total_vs}")
        print(f"Secuencias:                   {s}/{total_vs}")
        print(f"Incendios (labels no vacíos): {i}")
        print(f"Imágenes con label:           {w}/{total_img}")
        print(f"Imágenes sin label:           {wo}/{total_img}")
        print(f"Total bounding boxes:         {bc}")
        print(f"Promedio área bbox:           {avg:.6f} (≈{avg*100:.4f}% área imagen)")

        # Acumular para global
        agg['videos']           += v
        agg['sequences']        += s
        agg['incendios']        += i
        agg['imgs_with_label']  += w
        agg['imgs_without_label'] += wo
        agg['box_count']        += bc
        agg['total_area']       += ta

    # Estadísticas globales
    global_total_vs = agg['videos'] + agg['sequences']
    global_total_img = agg['imgs_with_label'] + agg['imgs_without_label']
    global_avg = (agg['total_area'] / agg['box_count']) if agg['box_count'] > 0 else 0.0

    print(f"\n=== Estadísticas globales ===")
    print(f"Videos:                       {agg['videos']}/{global_total_vs}")
    print(f"Secuencias:                   {agg['sequences']}/{global_total_vs}")
    print(f"Incendios (labels no vacíos): {agg['incendios']}")
    print(f"Imágenes con label:           {agg['imgs_with_label']}/{global_total_img}")
    print(f"Imágenes sin label:           {agg['imgs_without_label']}/{global_total_img}")
    print(f"Total bounding boxes:         {agg['box_count']}")
    print(f"Promedio área bbox global:    {global_avg:.6f} (≈{global_avg*100:.4f}% área imagen)")
