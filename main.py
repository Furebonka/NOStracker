import cv2
import mediapipe as mp
from pynput.mouse import Controller as MouseController
import os
import logging
import warnings

# Отключение логов TensorFlow Lite и MediaPipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключаем логи TensorFlow (0 — все логи, 3 — только критические ошибки)

# Отключение предупреждений Python (например, от protobuf)
warnings.filterwarnings("ignore", category=UserWarning)

# Отключение всех логов MediaPipe
logging.getLogger('mediapipe').setLevel(logging.ERROR)

mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)
mouse = MouseController()

# Основной коэффициент смещения для управления мышью носом.
# Чем больше значение, тем медленнее движение мыши в ответ на движение носа.
# Например, 1 — базовая скорость, 0.5 — более быстрое перемещение мыши.
mouse_speed_factor = 0.5  # Регулируйте для изменения общей чувствительности.

# Коэффициент сглаживания для плавного движения мыши.
# Чем ближе значение к 1, тем более плавным будет движение мыши, но с некоторой задержкой.
# Чем ближе к 0 — движение будет резким, но более отзывчивым.
smoothing_factor = 0.1  # Измените для настройки плавности движения.

# Порог чувствительности для того, чтобы игнорировать небольшие движения головы.
# Если изменение позиции носа меньше этого порога, мышь не будет двигаться.
# Например, 0.1 — высокая чувствительность (малейшие движения влияют), 0.5 — низкая чувствительность.
sensitivity_threshold = 0.3  # Регулируйте для изменения чувствительности к мелким движениям.

# Переменные для контроля ускорения движения мыши в разные стороны.
# Эти переменные определяют, насколько быстрее курсор будет двигаться в зависимости от направления.
# Например, увеличение acceleration_down сделает движение вниз более быстрым по сравнению с другими направлениями.
acceleration_up = 1.0      # Ускорение для движения вверх.
acceleration_down = 2.0    # Ускорение для движения вниз.
acceleration_left = 1.0    # Ускорение для движения влево.
acceleration_right = 1.0   # Ускорение для движения вправо.

# Переменные для хранения предыдущих координат носа (нужны для вычисления смещения).
previous_nose_x = None
previous_nose_y = None

# Переменные для хранения сглаженных координат носа (нужны для плавного движения).
smoothed_nose_x = None
smoothed_nose_y = None

# Запускаем процесс отслеживания лица и управления мышью.
with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Преобразуем изображение в формат RGB (нужен для работы MediaPipe).
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Получаем координаты носа (индекс 1 для носа в MediaPipe).
                nose_index = 1  # Индекс для носа.
                nose_x = int(face_landmarks.landmark[nose_index].x * frame.shape[1])
                nose_y = int(face_landmarks.landmark[nose_index].y * frame.shape[0])

                # Если это первый кадр, инициализируем предыдущие и сглаженные координаты носа.
                if previous_nose_x is None or previous_nose_y is None:
                    previous_nose_x = nose_x
                    previous_nose_y = nose_y
                    smoothed_nose_x = nose_x
                    smoothed_nose_y = nose_y

                # Применяем экспоненциальное сглаживание к координатам носа для плавного движения.
                # Чем выше smoothing_factor, тем более плавным будет движение мыши.
                smoothed_nose_x = smoothed_nose_x + smoothing_factor * (nose_x - smoothed_nose_x)
                smoothed_nose_y = smoothed_nose_y + smoothing_factor * (nose_y - smoothed_nose_y)

                # Вычисляем изменение координат носа (смещение).
                delta_x = smoothed_nose_x - previous_nose_x
                delta_y = smoothed_nose_y - previous_nose_y

                # Применяем порог чувствительности: мышь двигается только если смещение превышает порог.
                if abs(delta_x) > sensitivity_threshold or abs(delta_y) > sensitivity_threshold:
                    # Получаем текущие координаты мыши.
                    current_mouse_x, current_mouse_y = mouse.position

                    # Применяем ускорение в зависимости от направления.
                    # Увеличиваем или уменьшаем delta_x и delta_y в зависимости от того, в каком направлении двигается голова.
                    if delta_y < 0:  # Движение головы вверх.
                        delta_y *= acceleration_up  # Применяем ускорение вверх.
                    elif delta_y > 0:  # Движение головы вниз.
                        delta_y *= acceleration_down  # Применяем ускорение вниз.

                    if delta_x < 0:  # Движение головы влево.
                        delta_x *= acceleration_left  # Применяем ускорение влево.
                    elif delta_x > 0:  # Движение головы вправо.
                        delta_x *= acceleration_right  # Применяем ускорение вправо.

                    # Обновляем позицию мыши с учетом смещения и ускорения.
                    # Мышь движется пропорционально смещению и коэффициенту скорости.
                    new_mouse_x = current_mouse_x - (delta_x / mouse_speed_factor)
                    new_mouse_y = current_mouse_y + (delta_y / mouse_speed_factor)

                    # Плавное движение мыши.
                    mouse.position = (new_mouse_x, new_mouse_y)

                # Обновляем предыдущие координаты носа для следующего шага.
                previous_nose_x = smoothed_nose_x
                previous_nose_y = smoothed_nose_y

        # Условие для выхода из цикла по нажатию клавиши Esc.
        if cv2.waitKey(5) & 0xFF == 27:  # Выход по клавише Esc.
            break

# Освобождаем ресурсы после завершения работы.
cap.release()
cv2.destroyAllWindows()
