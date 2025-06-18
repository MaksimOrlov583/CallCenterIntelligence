import cv2
import time
from emotion_analyzer import EmotionAnalyzer

def process_video(video_path):
    """
    Обрабатывает видеофайл и анализирует эмоции
    
    Args:
        video_path: Путь к видеофайлу
    """
    # Инициализируем анализатор эмоций
    emotion_analyzer = EmotionAnalyzer()
    
    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Ошибка при открытии видеофайла")
        return
        
    # Получаем FPS видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1/fps
    
    while True:
        # Читаем кадр
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Анализируем эмоции
        emotion_results = emotion_analyzer.analyze_emotion(frame)
        
        if emotion_results:
            # Получаем доминирующую эмоцию
            dominant_emotion, probability = emotion_analyzer.get_dominant_emotion(frame)
            
            # Выводим результаты
            print(f"Доминирующая эмоция: {dominant_emotion} (вероятность: {probability:.2f})")
            
            # Отображаем результаты на кадре
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        # Показываем кадр
        cv2.imshow('Video Analysis', frame)
        
        # Ждем нажатия клавиши 'q' для выхода
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        # Задержка для соответствия FPS видео
        time.sleep(frame_delay)
        
    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Путь к видеофайлу
    video_path = "input_video.mp4"  # Замените на путь к вашему видео
    
    # Обрабатываем видео
    process_video(video_path) 