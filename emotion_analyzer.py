import cv2
import numpy as np
from emotion_recognition import EmotionRecognition

class EmotionAnalyzer:
    def __init__(self):
        self.emotion_recognizer = EmotionRecognition()
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
    def analyze_emotion(self, frame):
        """
        Анализирует эмоции на кадре видео
        
        Args:
            frame: Кадр видео в формате numpy array
            
        Returns:
            dict: Словарь с эмоциями и их вероятностями
        """
        try:
            # Конвертируем кадр в формат, подходящий для анализа
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Получаем предсказания эмоций
            predictions = self.emotion_recognizer.predict_emotion(gray)
            
            # Формируем словарь с результатами
            results = {}
            for emotion, probability in zip(self.emotions, predictions):
                results[emotion] = float(probability)
                
            return results
            
        except Exception as e:
            print(f"Ошибка при анализе эмоций: {str(e)}")
            return None
            
    def get_dominant_emotion(self, frame):
        """
        Определяет доминирующую эмоцию на кадре
        
        Args:
            frame: Кадр видео в формате numpy array
            
        Returns:
            tuple: (эмоция, вероятность)
        """
        results = self.analyze_emotion(frame)
        if results:
            dominant_emotion = max(results.items(), key=lambda x: x[1])
            return dominant_emotion
        return None 