import numpy as np
from typing import List, Dict
import librosa
import soundfile as sf
import tempfile
import os
from aniemore.recognizers.voice import VoiceRecognizer

# Примечание: В реальной реализации здесь мы бы импортировали Anyamore
# Поскольку Anyamore не является стандартной библиотекой, мы реализуем имитационную версию
# для демонстрационных целей, которая симулирует функциональность определения эмоций

# Категории эмоций
КАТЕГОРИИ_ЭМОЦИЙ = ['негативные', 'нейтрально', 'радость']

class EmotionAnalyzer:
    def __init__(self):
        try:
            # Инициализируем с явным указанием модели
            self.voice_recognizer = VoiceRecognizer(model_name="voice-emotion-recognition")
        except Exception as e:
            print(f"Ошибка при инициализации VoiceRecognizer: {str(e)}")
            self.voice_recognizer = None
        
    def analyze_emotions(self, audio, sample_rate):
        """
        Анализирует эмоции в аудио с использованием aniemore.
        
        Args:
            audio (numpy.ndarray): Аудио сигнал
            sample_rate (int): Частота дискретизации
            
        Returns:
            list: Список эмоций для каждого сегмента аудио
        """
        if self.voice_recognizer is None:
            print("VoiceRecognizer не инициализирован")
            return ['нейтрально'] * (len(audio) // sample_rate)
            
        try:
            # Сохраняем аудио во временный файл
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, audio, sample_rate)
                
                # Получаем эмоции из голоса
                voice_emotions = self.voice_recognizer.recognize(temp_file.name)
                
                # Преобразуем эмоции в наши категории
                emotions = self._map_emotions(voice_emotions)
                
                # Удаляем временный файл
                os.unlink(temp_file.name)
                
                return emotions
                
        except Exception as e:
            print(f"Ошибка при анализе эмоций: {str(e)}")
            return ['нейтрально'] * (len(audio) // sample_rate)
            
    def _map_emotions(self, emotions):
        """
        Преобразует эмоции aniemore в наши категории.
        
        Args:
            emotions (dict): Словарь эмоций от aniemore
            
        Returns:
            list: Список эмоций в наших категориях
        """
        if not emotions:
            return ['нейтрально']
            
        # Преобразуем эмоции aniemore в наши категории
        emotion_mapping = {
            'anger': 'негативные',
            'disgust': 'негативные',
            'fear': 'негативные',
            'happiness': 'радость',
            'sadness': 'негативные',
            'surprise': 'нейтрально',
            'neutral': 'нейтрально'
        }
        
        # Получаем доминирующую эмоцию
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        
        # Преобразуем в нашу категорию
        mapped_emotion = emotion_mapping.get(dominant_emotion, 'нейтрально')
        
        return [mapped_emotion]

def get_emotion_statistics(emotions):
    """
    Вычисляет статистику по эмоциям.
    
    Args:
        emotions (list): Список эмоций
        
    Returns:
        dict: Статистика по эмоциям
    """
    emotion_counts = {}
    for emotion in emotions:
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1
        else:
            emotion_counts[emotion] = 1
    
    total = len(emotions)
    return {emotion: count/total for emotion, count in emotion_counts.items()}

def get_dominant_emotion(emotions):
    """
    Определяет преобладающую эмоцию.
    
    Args:
        emotions (list): Список эмоций
        
    Returns:
        str: Преобладающая эмоция
    """
    emotion_counts = {}
    for emotion in emotions:
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1
        else:
            emotion_counts[emotion] = 1
    
    return max(emotion_counts.items(), key=lambda x: x[1])[0]

def extract_features(audio, sample_rate):
    """
    Извлекает аудио-характеристики для анализа эмоций.
    
    Args:
        audio (numpy.ndarray): Аудио сигнал
        sample_rate (int): Частота дискретизации
        
    Returns:
        list: Список словарей с характеристиками
    """
    # Расчет размера окна для сегментов ~1 секунда
    window_size = sample_rate
    hop_length = window_size // 2
    
    # Разделение аудио на сегменты
    num_segments = max(1, int((len(audio) - window_size) / hop_length) + 1)
    features = []
    
    for i in range(num_segments):
        start = i * hop_length
        end = min(start + window_size, len(audio))
        
        if end - start < window_size // 2:
            continue
            
        segment = audio[start:end]
        
        # Извлечение характеристик
        feature_dict = {}
        
        # Энергия (громкость)
        feature_dict['energy'] = np.mean(librosa.feature.rms(y=segment))
        
        # Частота пересечения нуля
        feature_dict['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(segment))
        
        # Спектральный центроид
        feature_dict['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(
            y=segment, 
            sr=sample_rate,
            n_fft=min(1024, len(segment))
        ))
        
        # Спектральный спад
        feature_dict['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(
            y=segment, 
            sr=sample_rate,
            n_fft=min(1024, len(segment))
        ))
        
        # Спектральная полоса пропускания
        feature_dict['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(
            y=segment,
            sr=sample_rate,
            n_fft=min(1024, len(segment))
        ))
        
        # Спектральная плостность
        feature_dict['spectral_flatness'] = np.mean(librosa.feature.spectral_flatness(
            y=segment,
            n_fft=min(1024, len(segment))
        ))
        
        # MFCC (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(
            y=segment,
            sr=sample_rate,
            n_mfcc=13,
            n_fft=min(1024, len(segment))
        )
        feature_dict['mfcc_mean'] = np.mean(mfcc, axis=1)
        feature_dict['mfcc_std'] = np.std(mfcc, axis=1)
        
        features.append(feature_dict)
    
    return features

def map_features_to_emotions(features):
    """
    Преобразует аудио-характеристики в эмоции.
    
    Args:
        features (list): Список словарей с характеристиками
        
    Returns:
        list: Список эмоций
    """
    emotions = []
    
    # Вычисляем средние значения для нормализации
    all_energies = [f['energy'] for f in features]
    all_zcr = [f['zero_crossing_rate'] for f in features]
    all_centroids = [f['spectral_centroid'] for f in features]
    
    mean_energy = np.mean(all_energies)
    mean_zcr = np.mean(all_zcr)
    mean_centroid = np.mean(all_centroids)
    
    std_energy = np.std(all_energies)
    std_zcr = np.std(all_zcr)
    std_centroid = np.std(all_centroids)
    
    for i, feature in enumerate(features):
        # Получаем характеристики
        energy = feature['energy']
        zcr = feature['zero_crossing_rate']
        centroid = feature['spectral_centroid']
        rolloff = feature['spectral_rolloff']
        bandwidth = feature['spectral_bandwidth']
        flatness = feature['spectral_flatness']
        mfcc_mean = feature['mfcc_mean']
        mfcc_std = feature['mfcc_std']
        
        # Нормализуем значения относительно среднего
        norm_energy = (energy - mean_energy) / (std_energy + 1e-6)
        norm_zcr = (zcr - mean_zcr) / (std_zcr + 1e-6)
        norm_centroid = (centroid - mean_centroid) / (std_centroid + 1e-6)
        
        # Вычисляем дополнительные метрики
        energy_variance = np.var(mfcc_mean)
        pitch_stability = np.mean(np.abs(np.diff(mfcc_mean)))
        
        # Определяем эмоцию на основе комбинации характеристик
        negative_score = 0
        joy_score = 0
        neutral_score = 0
        
        # Оценка негативных эмоций (более строгие критерии)
        if norm_energy > 1.0:  # Значительно повышенный порог для энергии
            negative_score += 1.5  # Уменьшенный вес
        if norm_zcr > 0.8:  # Повышенный порог для ZCR
            negative_score += 1.5
        if bandwidth > 4000:  # Повышенный порог для полосы пропускания
            negative_score += 1
        if energy_variance > 0.9:  # Повышенный порог для вариации энергии
            negative_score += 1
        if pitch_stability > 0.7:  # Повышенный порог для стабильности тона
            negative_score += 1
            
        # Дополнительные критерии для негативных эмоций
        if flatness > 0.7:  # Повышенный порог для спектральной плостности
            negative_score += 0.8
        if rolloff > 6000:  # Повышенный порог для спектрального спада
            negative_score += 0.8
            
        # Оценка радости (более мягкие критерии)
        if 0.2 < norm_energy < 0.8:
            joy_score += 1.2
        if norm_centroid > 0.3:
            joy_score += 1.2
        if rolloff > 4000:
            joy_score += 1
        if pitch_stability < 0.3:
            joy_score += 1
        if flatness < 0.3:
            joy_score += 1
            
        # Оценка нейтральных эмоций
        if abs(norm_energy) < 0.3:
            neutral_score += 1.2
        if abs(norm_zcr) < 0.3:
            neutral_score += 1.2
        if abs(norm_centroid) < 0.3:
            neutral_score += 1.2
        if pitch_stability < 0.2:
            neutral_score += 1
        if flatness > 0.4:
            neutral_score += 1
        
        # Выбираем эмоцию с наибольшим счетом
        scores = {
            'негативные': negative_score,
            'радость': joy_score,
            'нейтрально': neutral_score
        }
        
        # Добавляем небольшой бонус к предыдущей эмоции для сглаживания
        if i > 0 and emotions[-1] in scores:
            scores[emotions[-1]] += 0.3
            
        # Требуем более высокого порога для негативных эмоций
        if negative_score < 5:  # Увеличенный минимальный порог для негативных эмоций
            scores['негативные'] = 0
            
        emotion = max(scores.items(), key=lambda x: x[1])[0]
        emotions.append(emotion)
    
    return emotions

def smooth_emotions(emotions, window_size=3):
    """
    Сглаживает последовательность эмоций.
    
    Args:
        emotions (list): Список эмоций
        window_size (int): Размер окна сглаживания
        
    Returns:
        list: Сглаженный список эмоций
    """
    if len(emotions) <= window_size:
        return emotions
        
    smoothed = []
    for i in range(len(emotions)):
        start = max(0, i - window_size // 2)
        end = min(len(emotions), i + window_size // 2 + 1)
        window = emotions[start:end]
        
        # Находим наиболее частую эмоцию в окне
        emotion_counts = {}
        for emotion in window:
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            else:
                emotion_counts[emotion] = 1
                
        most_common = max(emotion_counts.items(), key=lambda x: x[1])[0]
        smoothed.append(most_common)
    
    return smoothed
