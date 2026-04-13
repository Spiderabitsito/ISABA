import cv2
import numpy as np
import requests
import mediapipe as mp
import time

# -------- CONFIG ESP32 ----------
ESP32_IP = "192.168.4.1"
FRAME_URL = f"http://{ESP32_IP}/frame"

# -------- MEDIA PIPE ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------- COLORES POR DEDO ----------
RIGHT_HAND_COLORS = {
    'THUMB': (0, 0, 255),
    'INDEX': (0, 255, 255),
    'MIDDLE': (0, 255, 0),
    'RING': (255, 0, 0),
    'PINKY': (255, 0, 255)
}

LEFT_HAND_COLORS = {
    'THUMB': (0, 128, 255),
    'INDEX': (128, 255, 255),
    'MIDDLE': (128, 255, 128),
    'RING': (255, 128, 128),
    'PINKY': (255, 128, 255)
}

FINGER_LANDMARKS = {
    'THUMB': [1, 2, 3, 4],
    'INDEX': [5, 6, 7, 8],
    'MIDDLE': [9, 10, 11, 12],
    'RING': [13, 14, 15, 16],
    'PINKY': [17, 18, 19, 20]
}

FINGER_CONNECTIONS = {
    'THUMB': [(1, 2), (2, 3), (3, 4)],
    'INDEX': [(5, 6), (6, 7), (7, 8)],
    'MIDDLE': [(9, 10), (10, 11), (11, 12)],
    'RING': [(13, 14), (14, 15), (15, 16)],
    'PINKY': [(17, 18), (18, 19), (19, 20)]
}

PALM_CONNECTIONS = [(0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (0, 17)]

# -------- FUNCIONES ---------
def get_frame(url):
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            return r.content
    except requests.RequestException:
        return None

def draw_colored_landmarks(image, hand_landmarks, hand_label):
    h, w, _ = image.shape
    colors = RIGHT_HAND_COLORS if hand_label == "Right" else LEFT_HAND_COLORS

    # Dibuja palma
    for start_idx, end_idx in PALM_CONNECTIONS:
        start = hand_landmarks.landmark[start_idx]
        end = hand_landmarks.landmark[end_idx]
        cv2.line(image,
                 (int(start.x*w), int(start.y*h)),
                 (int(end.x*w), int(end.y*h)),
                 (180,180,180), 2)

    # Dibuja cada dedo
    for finger_name, color in colors.items():
        for start_idx, end_idx in FINGER_CONNECTIONS[finger_name]:
            start = hand_landmarks.landmark[start_idx]
            end = hand_landmarks.landmark[end_idx]
            cv2.line(image,
                     (int(start.x*w), int(start.y*h)),
                     (int(end.x*w), int(end.y*h)),
                     color, 2)
        for lm_idx in FINGER_LANDMARKS[finger_name]:
            lm = hand_landmarks.landmark[lm_idx]
            cv2.circle(image, (int(lm.x*w), int(lm.y*h)), 5, color, -1)

    # Muñeca
    wrist = hand_landmarks.landmark[0]
    cv2.circle(image, (int(wrist.x*w), int(wrist.y*h)), 7, (220,220,220), -1)

# -------- GESTOS ---------
def dist_points(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def finger_tapping(hand_landmarks):
    return dist_points(hand_landmarks.landmark[4], hand_landmarks.landmark[8]) < 0.05

def hand_open_close(hand_landmarks):
    d = hand_landmarks.landmark[12].y - hand_landmarks.landmark[0].y
    return "abierta" if d > 0.1 else "cerrada"

def pronation_supination(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    index_tip = hand_landmarks.landmark[8]
    v = np.array([index_tip.x - wrist.x, index_tip.y - wrist.y])

    angle_rad = np.arctan2(-v[1], v[0])  # invertimos y porque y+ va hacia abajo en imagen
    angle_deg = (np.degrees(angle_rad) + 360) % 360

    if 60 <= angle_deg <= 120:
        return "supinación (palma arriba)"
    elif 240 <= angle_deg <= 300:
        return "pronación (palma abajo)"
    else:
        return "posición neutra"

# -------- INFO MANO ---------
def draw_hand_info(image, hand_landmarks, handedness_label):
    h, w, _ = image.shape
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    min_x, max_x = int(min(xs)*w), int(max(xs)*w)
    min_y, max_y = int(min(ys)*h), int(max(ys)*h)

    margin = 5
    box_top = max(min_y - 100, 0)
    box_bottom = min(max_y + 100, h-1)
    box_left = max(min_x - 10, 0)
    box_right = min(max_x + 10, w-1)

    colors_bg = (50,50,50) if handedness_label == "Right" else (80,50,80)
    overlay = image.copy()
    cv2.rectangle(overlay, (box_left, box_top), (box_right, box_bottom), colors_bg, -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    y_text = box_top + 25
    cv2.putText(image, handedness_label, (box_left+5, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    y_text += 35

    if finger_tapping(hand_landmarks):
        cv2.putText(image, "Golpeteo de dedos", (box_left+5, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        y_text += 35

    estado_mano = hand_open_close(hand_landmarks)
    cv2.putText(image, f"Mano {estado_mano}", (box_left+5, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,165,0), 2)
    y_text += 35

    rotacion = pronation_supination(hand_landmarks)
    cv2.putText(image, rotacion, (box_left+5, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)

# -------- LOOP PRINCIPAL ---------
while True:
    frame_data = get_frame(FRAME_URL)
    if not frame_data:
        print("❌ Error obteniendo frame")
        time.sleep(0.2)
        continue

    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label
            draw_colored_landmarks(frame, hand_landmarks, label)
            draw_hand_info(frame, hand_landmarks, label)

    # Mostrar sin escalar para mantener resolución original
    cv2.imshow("MediaPipe Hands ESP32", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.03)

cv2.destroyAllWindows()
