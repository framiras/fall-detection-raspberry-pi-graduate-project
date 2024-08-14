import cv2
import asyncio
import signal
import time
import numpy as np
import io
from ultralytics import YOLO
from telegram import Bot
from picamera2 import Picamera2
import os

FALLING_INDEX = 1   # индекс падения
NOT_FALLING_INDEX = 0   # индекс не падения
CLASS_LABELS = {FALLING_INDEX: "fall", NOT_FALLING_INDEX: "not_fall"}

# настройки для Telegram
TELEGRAM_BOT_TOKEN = "xxxxxxxxxxxx"
TELEGRAM_CHAT_ID = "xxxxxxxxxxxx"
bot = Bot(token=TELEGRAM_BOT_TOKEN)

LOW_CONFIDENCE_THRESHOLD = 0.40     # низкий попрог уверенности
HIGH_CONFIDENCE_THRESHOLD = 0.50    # высокий попрог уверенности
DRAW_THRESHOLD = 0.5                # чтобы рисовать рамку только с уверенностью 50%
should_continue = True              # флаг для продолжения хода программы
fall_detected = False               
COOLDOWN_PERIOD = 10                # ЛИМИТ времени в секундах после отправки уведомления
last_alert_time = None              # последнее время уведомления


# загрузка модели TFLite  с библиотекой Ultratlytics
current_directory = os.path.dirname(os.path.abspath(__file__))
tflite_model_file = "best_30e_500_float16.tflite"
tflite_model_path = os.path.join(current_directory, tflite_model_file)
tflite_model = YOLO(tflite_model_path, task='detect')

# функция для остановки программы без графического режима
def signal_handler(sig, frame):
    global should_continue
    should_continue = False
    asyncio.create_task(send_message("System is down"))

# функция для отправки текстового уведомления в Telegram
async def send_message(text):
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
    except Exception as e:
        print(f"Failed to send message: {e}")
        
# функция для отправки фото уведомления в Telegram
async def send_photo(frame):
    is_success, buffer = cv2.imencode(".jpg", frame)
    if is_success:
        bio = io.BytesIO(buffer)
        bio.name = 'fall_detected.jpg'
        try:
            await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=bio)
        except Exception as e:
            print(f"Failed to send photo: {e}")
    else:
        await send_message("Failed to send fall photo due to encoding issues.")

# функция обработки результатов
async def process_results(frame, results):
    fall_score = 0
    not_fall_score = 0

    for result in results:
        #print(f"the result is {result}")
        for obj in result.boxes:
            x1, y1, x2, y2 = obj.xyxy[0]
            #print(f"Bounding Box Coordinates: ({x1}, {y1}, {x2}, {y2})")
            print(f"the object is: {obj}")
            print(f"obj.xyxy = {obj.xyxy[0]}")
            score = obj.conf[0].item()
            class_id = obj.cls[0].item()
            #print(f"Bounding Box Coordinates: ({x1}, {y1}, {x2}, {y2}), Score: {score}, Class ID: {class_id}")
            
            if score > DRAW_THRESHOLD:
                draw_detection(frame, x1, y1, x2, y2, score, class_id)

            if class_id == FALLING_INDEX:
                fall_score = max(fall_score, score)
            elif class_id == NOT_FALLING_INDEX:
                not_fall_score = max(not_fall_score, score)

    await detect_fall(fall_score, not_fall_score, frame)

# функция обнаружения падения
async def detect_fall(fall_score, not_fall_score, frame):
    global last_alert_time
    if fall_score > HIGH_CONFIDENCE_THRESHOLD and not_fall_score < LOW_CONFIDENCE_THRESHOLD:
        if not last_alert_time:
            last_alert_time = time.time()
        elif time.time() - last_alert_time > COOLDOWN_PERIOD:
            cv2.putText(frame, "FALL DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            await send_photo(frame)
            last_alert_time = time.time()
    else:
        last_alert_time = None
        
# функция для рисования рамки
def draw_detection(frame, x1, y1, x2, y2, score, class_id):
    color = (0, 0, 255) if class_id == FALLING_INDEX else (0, 255, 0)
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
    label = f'{CLASS_LABELS.get(class_id, "unknown")}: {score:.2f}'
    cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)


async def main(use_gui, use_camera=True):
    global should_continue
    await send_message("System is up")

    if use_camera:
        picam2 = Picamera2()
        picam2.preview_configuration.main.size = (800, 600)
        picam2.preview_configuration.main.format = "RGB888"
        picam2.start()

    try:
        while should_continue:
            if use_camera:
                frame = picam2.capture_array()
                frame = cv2.flip(frame, 0)  # переворот кадра на 180 вертикально 
            else:
                print("Camera is not available.")
                break

            if frame is None:
                #print("Invalid captured frame")
                await send_message("Invalid captured frame")
                continue
                
            results = tflite_model(frame, verbose=False)
            await process_results(frame, results)

            if use_gui:
                cv2.imshow("Fall Detector", frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
    finally:
        if use_camera:
            picam2.stop()
        if use_gui:
            cv2.destroyAllWindows()
        await send_message("System is down")

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    use_camera = True   # камера от Raspberry Pi
    use_gui = True      # графический режим для отлаживания
    asyncio.run(main(use_gui, use_camera))
