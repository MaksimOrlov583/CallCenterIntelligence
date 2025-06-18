import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
import pydub
import os

def separate_channels(audio_path):
    """
    Разделение стерео аудио на каналы оператора и клиента.
    Если моно, пытается разделить на основе характеристик голоса.
    
    Args:
        audio_path (str): Путь к аудио файлу
        
    Returns:
        tuple: (аудио_оператора, аудио_клиента, частота_дискретизации)
    """
    try:
        # Загрузка аудио файла
        audio, sample_rate = librosa.load(audio_path, sr=None, mono=False)
        
        # Проверка, является ли аудио стерео (имеет 2 канала)
        if isinstance(audio, np.ndarray) and audio.ndim > 1 and audio.shape[0] == 2:
            # Стерео файл - используем каналы напрямую
            operator_audio = audio[0]
            customer_audio = audio[1]
            return operator_audio, customer_audio, sample_rate
        else:
            # Моно файл - используем технику разделения голосов
            return separate_speakers_from_mono(audio, sample_rate)
            
    except Exception as e:
        # Обработка проблем с форматом
        if "unknown format" in str(e).lower() or "no backend" in str(e).lower():
            # Пробуем использовать pydub для более надёжной обработки форматов
            return process_with_pydub(audio_path)
        else:
            raise Exception(f"Ошибка обработки аудио файла: {str(e)}")

def separate_speakers_from_mono(audio, sample_rate):
    """
    Разделение говорящих из моно записи с использованием временного анализа.
    Использует анализ энергии сигнала для разделения голосов.
    
    Args:
        audio (numpy.ndarray): Моно аудио сигнал
        sample_rate (int): Частота дискретизации аудио
        
    Returns:
        tuple: (аудио_оператора, аудио_клиента, частота_дискретизации)
    """
    try:
        # Убеждаемся, что у нас моно файл
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        # Нормализуем входной сигнал
        audio = normalize_audio(audio)
        
        # Разбиваем сигнал на короткие сегменты
        frame_length = int(sample_rate * 0.025)  # 25 мс
        hop_length = int(sample_rate * 0.010)    # 10 мс
        
        # Вычисляем энергию для каждого сегмента
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energy = np.sum(frames ** 2, axis=0)
        
        # Нормализуем энергию
        energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
        
        # Создаём маски для оператора и клиента
        operator_mask = np.zeros_like(audio)
        customer_mask = np.zeros_like(audio)
        
        # Определяем порог для разделения
        threshold = np.mean(energy) * 1.2
        
        # Создаём маски на основе энергии
        for i in range(len(energy)):
            start = i * hop_length
            end = min(start + frame_length, len(audio))
            
            if energy[i] > threshold:
                # Высокая энергия - вероятно оператор
                operator_mask[start:end] = 0.8
                customer_mask[start:end] = 0.2
            else:
                # Низкая энергия - вероятно клиент
                operator_mask[start:end] = 0.2
                customer_mask[start:end] = 0.8
        
        # Применяем плавные переходы между сегментами
        transition_length = int(sample_rate * 0.005)  # 5 мс
        for i in range(1, len(energy)):
            start = i * hop_length
            end = min(start + transition_length, len(audio))
            
            # Плавный переход для оператора
            operator_mask[start:end] = np.linspace(
                operator_mask[start-1],
                operator_mask[start],
                end - start
            )
            
            # Плавный переход для клиента
            customer_mask[start:end] = np.linspace(
                customer_mask[start-1],
                customer_mask[start],
                end - start
            )
        
        # Применяем маски
        operator_audio = audio * operator_mask
        customer_audio = audio * customer_mask
        
        # Нормализуем результаты
        operator_audio = normalize_audio(operator_audio)
        customer_audio = normalize_audio(customer_audio)
        
        return operator_audio, customer_audio, sample_rate
        
    except Exception as e:
        print(f"Предупреждение: Разделение голосов не удалось, возвращаем исходный сигнал. Ошибка: {str(e)}")
        # Возвращаем исходное аудио для обоих каналов при неудаче разделения
        return audio, audio, sample_rate

def bandpass_filter(data, lowcut, highcut, sample_rate, order=4):
    """
    Применяет полосовой фильтр к аудио сигналу с улучшенными проверками безопасности
    
    Args:
        data (numpy.ndarray): Аудио сигнал
        lowcut (float): Нижняя частота среза
        highcut (float): Верхняя частота среза
        sample_rate (int): Частота дискретизации аудио
        order (int): Порядок фильтра (уменьшен до 4 для стабильности)
        
    Returns:
        numpy.ndarray: Отфильтрованный аудио сигнал
    """
    try:
        nyquist = 0.5 * sample_rate
        
        # Очень консервативные частотные пределы
        max_allowed_freq = nyquist * 0.25  # Максимум 25% от частоты Найквиста
        
        # Обеспечиваем частоты в чрезвычайно безопасных диапазонах
        lowcut = max(20, min(lowcut, max_allowed_freq * 0.8))
        highcut = max(lowcut + 30, min(highcut, max_allowed_freq))
        
        # Преобразуем в нормализованные частоты
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Дополнительные проверки безопасности для нормализованных частот
        low = max(0.001, min(low, 0.249))  # Максимум 25% от частоты Найквиста
        high = max(low + 0.001, min(high, 0.249))
        
        # Обеспечиваем минимальное разделение между низкой и высокой частотой
        if high - low < 0.01:  # Минимум 1% разделения
            mid = (low + high) / 2
            low = mid - 0.005
            high = mid + 0.005
        
        # Применяем фильтр с уменьшенным порядком для стабильности
        b, a = butter(order, [low, high], btype='band')
        filtered = lfilter(b, a, data)
        
        # Нормализуем отфильтрованный сигнал
        if np.max(np.abs(filtered)) > 0:
            filtered = filtered / np.max(np.abs(filtered))
        
        return filtered
        
    except Exception as e:
        print(f"Предупреждение: Фильтрация не удалась, возвращаем исходный сигнал. Ошибка: {str(e)}")
        return data

def normalize_audio(audio):
    """
    Нормализация аудио для максимальной амплитуды 1
    
    Args:
        audio (numpy.ndarray): Аудио сигнал
        
    Returns:
        numpy.ndarray: Нормализованный аудио сигнал
    """
    if np.max(np.abs(audio)) > 0:
        return audio / np.max(np.abs(audio))
    return audio

def process_with_pydub(audio_path):
    """
    Обработка аудио с использованием pydub для большей совместимости форматов
    
    Args:
        audio_path (str): Путь к аудио файлу
        
    Returns:
        tuple: (аудио_оператора, аудио_клиента, частота_дискретизации)
    """
    # Загружаем аудио с помощью pydub
    audio = pydub.AudioSegment.from_file(audio_path)
    
    # Получаем частоту дискретизации
    sample_rate = audio.frame_rate
    
    # Преобразуем в массив numpy
    samples = np.array(audio.get_array_of_samples())
    
    # Проверяем, является ли стерео
    if audio.channels == 2:
        # Изменяем форму для стерео
        samples = samples.reshape((-1, 2))
        operator_audio = samples[:, 0].astype(np.float32) / 32768.0  # Нормализация
        customer_audio = samples[:, 1].astype(np.float32) / 32768.0  # Нормализация
    else:
        # Если моно, разделяем с помощью другой функции
        samples = samples.astype(np.float32) / 32768.0  # Нормализация
        return separate_speakers_from_mono(samples, sample_rate)
    
    return operator_audio, customer_audio, sample_rate

def resample_audio(audio, orig_sr, target_sr):
    """
    Изменяет частоту дискретизации аудио до целевой частоты
    
    Args:
        audio (numpy.ndarray): Аудио сигнал
        orig_sr (int): Исходная частота дискретизации
        target_sr (int): Целевая частота дискретизации
        
    Returns:
        numpy.ndarray: Аудио сигнал с измененной частотой дискретизации
    """
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

def analyze_and_enhance_audio(audio, sample_rate):
    """
    Анализирует и улучшает качество аудио для лучшего распознавания речи.
    
    Args:
        audio (numpy.ndarray): Аудио сигнал
        sample_rate (int): Частота дискретизации
        
    Returns:
        tuple: (улучшенное_аудио, словарь_с_результатами_анализа)
    """
    analysis_results = {
        'has_speech': False,
        'is_too_quiet': False,
        'is_unclear': False,
        'original_rms': 0,
        'enhanced_rms': 0
    }
    
    try:
        # Проверка наличия речи
        # Используем энергию сигнала как индикатор наличия речи
        rms = librosa.feature.rms(y=audio)[0]
        mean_rms = np.mean(rms)
        analysis_results['original_rms'] = mean_rms
        
        # Проверка на тихий звук
        if mean_rms < 0.01:  # Пороговое значение для тихого звука
            analysis_results['is_too_quiet'] = True
            # Усиление тихого звука
            audio = audio * 5.0  # Увеличиваем громкость в 5 раз
        
        # Применяем полосовой фильтр для улучшения разборчивости речи
        # Диапазон частот человеческой речи: 85-255 Гц
        audio = bandpass_filter(audio, 85, 255, sample_rate)
        
        # Нормализация после обработки
        audio = normalize_audio(audio)
        
        # Проверяем результат после обработки
        enhanced_rms = np.mean(librosa.feature.rms(y=audio)[0])
        analysis_results['enhanced_rms'] = enhanced_rms
        
        # Определяем наличие речи после обработки
        if enhanced_rms > 0.02:  # Пороговое значение для определения речи
            analysis_results['has_speech'] = True
        
        # Проверка на неразборчивость
        # Используем спектральную центроиду как индикатор четкости
        # Исправляем параметры для spectral_centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, n_fft=2048, hop_length=512)[0]
        if np.std(spectral_centroid) < 500:  # Низкая вариация может указывать на неразборчивость
            analysis_results['is_unclear'] = True
        
        return audio, analysis_results
        
    except Exception as e:
        print(f"Ошибка при анализе и улучшении аудио: {str(e)}")
        return audio, analysis_results

def get_audio_quality_report(audio_path):
    """
    Создает отчет о качестве аудио файла.
    
    Args:
        audio_path (str): Путь к аудио файлу
        
    Returns:
        dict: Словарь с результатами анализа
    """
    try:
        # Проверка существования файла
        if not os.path.exists(audio_path):
            return {
                'error': f"Файл не найден: {audio_path}",
                'recommendations': ["Убедитесь, что путь к файлу указан правильно и файл существует."]
            }
            
        # Проверка расширения файла
        valid_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
        if not any(audio_path.lower().endswith(ext) for ext in valid_extensions):
            return {
                'error': f"Неподдерживаемый формат файла: {audio_path}",
                'recommendations': [f"Поддерживаемые форматы: {', '.join(valid_extensions)}"]
            }
        
        # Загружаем аудио
        audio, sample_rate = librosa.load(audio_path, sr=None)
        
        # Анализируем и улучшаем аудио
        enhanced_audio, analysis_results = analyze_and_enhance_audio(audio, sample_rate)
        
        # Добавляем рекомендации
        recommendations = []
        
        if analysis_results['is_too_quiet']:
            recommendations.append("Аудио слишком тихое. Рекомендуется увеличить громкость записи.")
        
        if analysis_results['is_unclear']:
            recommendations.append("Речь может быть неразборчивой. Проверьте качество записи и фоновый шум.")
        
        if not analysis_results['has_speech']:
            recommendations.append("В файле не обнаружено речи. Проверьте, что файл содержит запись разговора.")
        
        analysis_results['recommendations'] = recommendations
        
        return analysis_results
        
    except FileNotFoundError:
        return {
            'error': f"Файл не найден: {audio_path}",
            'recommendations': ["Убедитесь, что путь к файлу указан правильно и файл существует."]
        }
    except Exception as e:
        return {
            'error': str(e),
            'recommendations': ["Не удалось проанализировать аудио файл. Проверьте формат и доступность файла."]
        }
