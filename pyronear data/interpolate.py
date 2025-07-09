import os
import cv2

# ----------------------------
# Ajusta aquí tu ruta base
# ----------------------------
root_dir = 'data/train/ptz_tetasur-02'

# ----------------------------
# Variables globales para edición
# ----------------------------
boxes = []           # lista de [cls, x1, y1, x2, y2]
selected = None      # índice de la caja seleccionada
mode = None          # 'move' o 'resize'
handle_idx = None    # esquina a redimensionar (0-3)
start_pt = None      # punto inicial de ratón
orig_box = None      # copia de la caja al iniciar arrastre
H = W = 0            # dimensiones de la imagen
padding = 10         # tolerancia en px para detectar esquinas

# ----------------------------
# Conversiones YOLO ↔ coordenadas
# ----------------------------
def yolo_to_xyxy(line, W, H):
    cls, x_c, y_c, bw, bh = map(float, line.split())
    x1 = int((x_c - bw/2) * W)
    y1 = int((y_c - bh/2) * H)
    x2 = int((x_c + bw/2) * W)
    y2 = int((y_c + bh/2) * H)
    return [cls, x1, y1, x2, y2]

def xyxy_to_yolo(box, W, H):
    cls, x1, y1, x2, y2 = box
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    x_c = (x1 + x2) / 2 / W
    y_c = (y1 + y2) / 2 / H
    return f"{int(cls)} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n"

# ----------------------------
# Helpers para interacción
# ----------------------------
def point_in_box(x, y, box):
    _, x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def corner_hit(x, y, box):
    _, x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    for idx, (cx, cy) in enumerate(corners):
        if abs(x - cx) <= padding and abs(y - cy) <= padding:
            return idx
    return None

# ----------------------------
# Callback de ratón
# ----------------------------
def on_mouse(event, x, y, flags, param):
    global selected, mode, handle_idx, start_pt, orig_box, boxes
    if event == cv2.EVENT_LBUTTONDOWN:
        # primero revisa esquinas
        for i, box in enumerate(boxes):
            h = corner_hit(x, y, box)
            if h is not None:
                selected = i
                mode = 'resize'
                handle_idx = h
                start_pt = (x, y)
                orig_box = box.copy()
                return
        # luego revisa interior de cada caja
        for i, box in enumerate(boxes):
            if point_in_box(x, y, box):
                selected = i
                mode = 'move'
                start_pt = (x, y)
                orig_box = box.copy()
                return
        # clic fuera de todo deselecciona
        selected = None

    elif event == cv2.EVENT_MOUSEMOVE and selected is not None:
        dx = x - start_pt[0]
        dy = y - start_pt[1]
        cls, x1, y1, x2, y2 = orig_box

        if mode == 'move':
            boxes[selected] = [cls, x1 + dx, y1 + dy, x2 + dx, y2 + dy]
        elif mode == 'resize':
            b = boxes[selected]
            # top-left
            if handle_idx == 0:
                b[1], b[2] = x1 + dx, y1 + dy
            # top-right
            elif handle_idx == 1:
                b[3], b[2] = x2 + dx, y1 + dy
            # bottom-left
            elif handle_idx == 2:
                b[1], b[4] = x1 + dx, y2 + dy
            # bottom-right
            elif handle_idx == 3:
                b[3], b[4] = x2 + dx, y2 + dy

    elif event == cv2.EVENT_LBUTTONUP:
        mode = None
        handle_idx = None

# ----------------------------
# Preparar ventana interactiva
# ----------------------------
win = 'Editor de Boxes'
cv2.namedWindow(win)
cv2.setMouseCallback(win, on_mouse)

# ----------------------------
# Recorre carpetas e interpola hacia atrás
# ----------------------------
for dirpath, dirnames, filenames in os.walk(root_dir):
    base = os.path.basename(dirpath).lower()
    if base.endswith('_fp') or base.startswith(('ptz','gupo')):
        continue

    txts = sorted(f for f in filenames if f.lower().endswith('.txt'))
    last_labels = None

    for txt in txts:
        txt_path = os.path.join(dirpath, txt)
        with open(txt_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]

        if lines:
            last_labels = lines
            continue

        if not last_labels:
            continue  # nada que proponer

        # Ruta de la imagen en el nivel padre
        img_name  = os.path.splitext(txt)[0] + '.jpg'
        parent    = os.path.dirname(dirpath)
        img_path  = os.path.join(parent, img_name)
        image     = cv2.imread(img_path)
        if image is None:
            print(f"[ERROR] Imagen no encontrada: {img_path}")
            continue

        H, W = image.shape[:2]
        # Inicializa cajas desde last_labels
        boxes = [yolo_to_xyxy(l, W, H) for l in last_labels]
        selected = None

        # Bucle de edición
        while True:
            vis = image.copy()
            # dibuja todas las cajas
            for i, b in enumerate(boxes):
                color = (0, 255, 0) if i != selected else (0, 165, 255)
                _, x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.imshow(win, vis)

            k = cv2.waitKey(1) & 0xFF
            # Y = guardar, N = descartar, Q = salir
            if k == ord('y'):
                # guarda como YOLO
                with open(txt_path, 'w') as f:
                    for b in boxes:
                        f.write(xyxy_to_yolo(b, W, H))
                print(f"[GUARDADO] {txt_path}")
                break
            elif k == ord('n'):
                print(f"[SKIP] {txt_path} quedó vacío")
                break
            elif k == ord('q'):
                print("Saliendo...")
                cv2.destroyAllWindows()
                exit(0)

cv2.destroyAllWindows()
