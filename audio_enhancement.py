import numpy as np
from pydub import AudioSegment
import librosa
from scipy import signal
import logging
import tempfile
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioEnhancer:
    def __init__(self):
        """
        Инициализация модуля улучшения качества аудио
        """
        self.sample_rate = 16000  # Стандартная частота дискретизации для распознавания речи
        
    def enhance_audio(self, audio_path: str) -> str:
        """
        Основная функция улучшения качества аудио
        
        Args:
            audio_path (str): Путь к исходному аудио файлу
            
        Returns:
            str: Путь к улучшенному аудио файлу
        """
        try:
            logger.info(f"Начало улучшения аудио: {audio_path}")
            
            # Загружаем аудио
            audio = AudioSegment.from_file(audio_path)
            logger.info(f"Исходная длительность: {len(audio)/1000:.2f} сек")
            
            # Применяем последовательность улучшений
            audio = self._normalize_audio(audio)
            audio = self._remove_background_noise(audio)
            audio = self._enhance_speech(audio)
            audio = self._apply_compression(audio)
            
            # Создаем временный файл для результата
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav.close()
            
            # Экспортируем с оптимальными параметрами
            audio.export(temp_wav.name, format="wav", parameters=[
                "-ac", "1",  # моно
                "-ar", str(self.sample_rate),  # частота дискретизации
                "-acodec", "pcm_s16le",  # кодек
                "-sample_fmt", "s16",  # формат сэмпла
                "-af", "volume=1.5"  # небольшое усиление
            ])
            
            logger.info(f"Аудио успешно улучшено и сохранено в: {temp_wav.name}")
            return temp_wav.name
            
        except Exception as e:
            logger.error(f"Ошибка при улучшении аудио: {str(e)}")
            return audio_path
            
    def _normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        """
        Нормализация громкости и базовое улучшение
        
        Args:
            audio (AudioSegment): Исходное аудио
            
        Returns:
            AudioSegment: Нормализованное аудио
        """
        try:
            logger.info("Нормализация аудио")
            
            # Проверяем исходные параметры
            original_samples = np.array(audio.get_array_of_samples())
            original_avg = np.mean(np.abs(original_samples))
            original_max = np.max(np.abs(original_samples))
            logger.info(f"Исходные параметры - Средняя амплитуда: {original_avg:.2f}, Максимальная: {original_max}")
            
            # Нормализация громкости
            audio = audio.normalize()
            
            # Усиление тихих частей
            audio = audio + 30  # Увеличиваем громкость на 30 дБ
            
            # Применяем компрессор для выравнивания динамического диапазона
            audio = audio.compress_dynamic_range()
            
            # Проверяем результат
            processed_samples = np.array(audio.get_array_of_samples())
            processed_avg = np.mean(np.abs(processed_samples))
            processed_max = np.max(np.abs(processed_samples))
            logger.info(f"После нормализации - Средняя амплитуда: {processed_avg:.2f}, Максимальная: {processed_max}")
            
            return audio
        except Exception as e:
            logger.error(f"Ошибка при нормализации: {str(e)}")
            return audio
            
    def _remove_background_noise(self, audio: AudioSegment) -> AudioSegment:
        """
        Удаление фонового шума
        
        Args:
            audio (AudioSegment): Исходное аудио
            
        Returns:
            AudioSegment: Очищенное аудио
        """
        try:
            logger.info("Удаление фонового шума")
            
            # Конвертируем в numpy массив
            samples = np.array(audio.get_array_of_samples())
            
            # Применяем фильтр высоких частот для удаления низкочастотных шумов
            nyquist = audio.frame_rate / 2
            high_pass = 400  # Увеличиваем частоту среза для низких частот
            b, a = signal.butter(4, high_pass/nyquist, btype='high')
            filtered = signal.filtfilt(b, a, samples)
            
            # Применяем фильтр низких частот для удаления высокочастотных шумов
            low_pass = 3000  # Частота среза для высоких частот
            b, a = signal.butter(4, low_pass/nyquist, btype='low')
            filtered = signal.filtfilt(b, a, filtered)
            
            # Применяем шумоподавление с более агрессивными параметрами
            noise_reduced = librosa.effects.preemphasis(filtered, coef=0.97)
            
            # Конвертируем обратно в AudioSegment
            cleaned_audio = AudioSegment(
                noise_reduced.tobytes(),
                frame_rate=audio.frame_rate,
                sample_width=audio.sample_width,
                channels=audio.channels
            )
            
            return cleaned_audio
        except Exception as e:
            logger.error(f"Ошибка при удалении шума: {str(e)}")
            return audio
            
    def _enhance_speech(self, audio: AudioSegment) -> AudioSegment:
        """
        Улучшение качества речи
        
        Args:
            audio (AudioSegment): Исходное аудио
            
        Returns:
            AudioSegment: Улучшенное аудио
        """
        try:
            logger.info("Улучшение качества речи")
            
            # Конвертируем в numpy массив
            samples = np.array(audio.get_array_of_samples())
            
            # Применяем эквалайзер для усиления частот речи
            nyquist = audio.frame_rate / 2
            
            # Усиливаем частоты речи (300-3400 Гц)
            speech_low = 300 / nyquist
            speech_high = 3400 / nyquist
            b, a = signal.butter(4, [speech_low, speech_high], btype='band')
            enhanced = signal.filtfilt(b, a, samples)
            
            # Применяем предыскажение для улучшения разборчивости
            enhanced = librosa.effects.preemphasis(enhanced, coef=0.97)
            
            # Конвертируем обратно в AudioSegment
            enhanced_audio = AudioSegment(
                enhanced.tobytes(),
                frame_rate=audio.frame_rate,
                sample_width=audio.sample_width,
                channels=audio.channels
            )
            
            return enhanced_audio
        except Exception as e:
            logger.error(f"Ошибка при улучшении речи: {str(e)}")
            return audio
            
    def _apply_compression(self, audio: AudioSegment) -> AudioSegment:
        """
        Применение компрессии для выравнивания громкости
        
        Args:
            audio (AudioSegment): Исходное аудио
            
        Returns:
            AudioSegment: Сжатое аудио
        """
        try:
            logger.info("Применение компрессии")
            
            # Применяем компрессор для выравнивания динамического диапазона
            audio = audio.compress_dynamic_range()
            
            # Дополнительное выравнивание громкости
            audio = audio.normalize()
            
            # Проверяем результат
            processed_samples = np.array(audio.get_array_of_samples())
            processed_avg = np.mean(np.abs(processed_samples))
            processed_max = np.max(np.abs(processed_samples))
            logger.info(f"После компрессии - Средняя амплитуда: {processed_avg:.2f}, Максимальная: {processed_max}")
            
            return audio
        except Exception as e:
            logger.error(f"Ошибка при компрессии: {str(e)}")
            return audio 