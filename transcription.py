import numpy as np
from typing import Tuple, List, Dict
import os
from pydub import AudioSegment
import tempfile
import logging
from scipy import signal
import librosa
import wave
import json
import requests
import time
from audio_enhancement import AudioEnhancer

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Transcriber:
    def __init__(self):
        """
        Инициализация транскрайбера
        """
        logger.info("Инициализация распознавателя речи...")
        self.api_key = os.getenv("YANDEX_API_KEY")
        if not self.api_key:
            logger.warning("YANDEX_API_KEY не найден в переменных окружения")
        self.audio_enhancer = AudioEnhancer()
        logger.info("Распознаватель речи инициализирован")
        
    def _normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        """
        Нормализация аудио
        
        Args:
            audio (AudioSegment): Исходное аудио
            
        Returns:
            AudioSegment: Нормализованное аудио
        """
        try:
            logger.info("Начало нормализации аудио")
            
            # Проверяем исходные параметры
            original_duration = len(audio) / 1000
            original_samples = np.array(audio.get_array_of_samples())
            original_avg = np.mean(np.abs(original_samples))
            original_max = np.max(np.abs(original_samples))
            logger.info(f"Исходные параметры - Длительность: {original_duration:.2f} сек, Средняя амплитуда: {original_avg:.2f}, Максимальная: {original_max}")
            
            # Нормализация громкости
            audio = audio.normalize()
            
            # Усиление тихих частей
            audio = audio + 30  # Увеличиваем громкость на 30 дБ
            
            # Применяем компрессор для выравнивания динамического диапазона
            audio = audio.compress_dynamic_range()
            
            # Проверяем параметры после обработки
            processed_samples = np.array(audio.get_array_of_samples())
            processed_avg = np.mean(np.abs(processed_samples))
            processed_max = np.max(np.abs(processed_samples))
            logger.info(f"После нормализации - Средняя амплитуда: {processed_avg:.2f}, Максимальная: {processed_max}")
            
            logger.info("Аудио успешно нормализовано")
            return audio
        except Exception as e:
            logger.error(f"Ошибка при нормализации аудио: {str(e)}")
            return audio

    def _remove_noise(self, audio: AudioSegment) -> AudioSegment:
        """
        Удаление шумов из аудио
        
        Args:
            audio (AudioSegment): Исходное аудио
            
        Returns:
            AudioSegment: Очищенное аудио
        """
        try:
            logger.info("Начало удаления шумов")
            
            # Конвертируем в numpy массив для обработки
            samples = np.array(audio.get_array_of_samples())
            
            # Применяем фильтр высоких частот для удаления низкочастотных шумов
            nyquist = audio.frame_rate / 2
            high_pass = 400  # Увеличиваем частоту среза для более агрессивного удаления низкочастотных шумов
            b, a = signal.butter(4, high_pass/nyquist, btype='high')
            filtered = signal.filtfilt(b, a, samples)
            
            # Применяем фильтр низких частот для удаления высокочастотных шумов
            low_pass = 3000  # Снижаем частоту среза для более агрессивного удаления высокочастотных шумов
            b, a = signal.butter(4, low_pass/nyquist, btype='low')
            filtered = signal.filtfilt(b, a, filtered)
            
            # Применяем шумоподавление
            noise_reduced = librosa.effects.preemphasis(filtered)
            
            # Проверяем параметры после шумоподавления
            noise_reduced_avg = np.mean(np.abs(noise_reduced))
            noise_reduced_max = np.max(np.abs(noise_reduced))
            logger.info(f"После шумоподавления - Средняя амплитуда: {noise_reduced_avg:.2f}, Максимальная: {noise_reduced_max}")
            
            # Конвертируем обратно в AudioSegment
            cleaned_audio = AudioSegment(
                noise_reduced.tobytes(),
                frame_rate=audio.frame_rate,
                sample_width=audio.sample_width,
                channels=audio.channels
            )
            
            logger.info("Шумы успешно удалены")
            return cleaned_audio
        except Exception as e:
            logger.error(f"Ошибка при удалении шумов: {str(e)}")
            return audio
        
    def _convert_to_wav(self, audio_path: str) -> str:
        """
        Конвертирует аудио файл в формат WAV с предварительной обработкой
        
        Args:
            audio_path (str): Путь к исходному аудио файлу
            
        Returns:
            str: Путь к конвертированному WAV файлу
        """
        try:
            logger.info(f"Начало конвертации файла: {audio_path}")
            # Создаем временный файл
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav.close()
            
            # Загружаем аудио
            audio = AudioSegment.from_file(audio_path)
            logger.info(f"Аудио загружено, длительность: {len(audio)/1000:.2f} сек")
            
            # Предварительная обработка
            audio = self._normalize_audio(audio)
            audio = self._remove_noise(audio)
            
            # Конвертируем в моно и экспортируем в WAV
            audio = audio.set_channels(1)  # Конвертируем в моно
            audio = audio.set_frame_rate(16000)  # Устанавливаем частоту дискретизации
            
            # Проверяем параметры перед экспортом
            final_samples = np.array(audio.get_array_of_samples())
            final_avg = np.mean(np.abs(final_samples))
            final_max = np.max(np.abs(final_samples))
            logger.info(f"Перед экспортом - Средняя амплитуда: {final_avg:.2f}, Максимальная: {final_max}")
            
            # Экспортируем с максимальным качеством
            audio.export(temp_wav.name, format="wav", parameters=[
                "-ac", "1",  # моно
                "-ar", "16000",  # частота дискретизации
                "-acodec", "pcm_s16le",  # кодек
                "-sample_fmt", "s16",  # формат сэмпла
                "-af", "volume=2.0"  # дополнительное усиление
            ])
            
            logger.info(f"Аудио успешно конвертировано в: {temp_wav.name}")
            return temp_wav.name
        except Exception as e:
            logger.error(f"Ошибка при конвертации аудио: {str(e)}")
            return audio_path
        
    def transcribe_audio(self, audio_path: str) -> Dict:
        """
        Транскрибация аудио файла с помощью Yandex SpeechKit
        
        Args:
            audio_path (str): Путь к аудио файлу
            
        Returns:
            Dict: Результаты транскрибации
        """
        try:
            logger.info(f"Начало транскрибации файла: {audio_path}")
            
            # Проверяем существование файла
            if not os.path.exists(audio_path):
                logger.error(f"Файл не найден: {audio_path}")
                return {"segments": [], "error": "Файл не найден"}
            
            # Проверяем размер файла
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                logger.error("Файл пустой")
                return {"segments": [], "error": "Файл пустой"}
            
            # Проверяем формат файла
            if not audio_path.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a')):
                logger.error("Неподдерживаемый формат файла")
                return {"segments": [], "error": "Неподдерживаемый формат файла"}
            
            # Улучшаем качество аудио перед транскрибацией
            enhanced_audio_path = self.audio_enhancer.enhance_audio(audio_path)
            logger.info(f"Аудио улучшено и сохранено в: {enhanced_audio_path}")
            
            # Конвертируем в WAV если нужно
            if not enhanced_audio_path.endswith('.wav'):
                wav_path = self._convert_to_wav(enhanced_audio_path)
                if wav_path != enhanced_audio_path:
                    os.unlink(enhanced_audio_path)
                enhanced_audio_path = wav_path
            
            # Проверяем наличие API ключа
            if not self.api_key:
                logger.error("API ключ не найден")
                return {"segments": [], "error": "API ключ не найден"}
            
            # Проверяем качество конвертированного аудио
            try:
                audio = AudioSegment.from_file(enhanced_audio_path)
                if len(audio) == 0:
                    logger.error("Конвертированный файл пустой")
                    return {"segments": [], "error": "Конвертированный файл пустой"}
                
                # Проверяем уровень сигнала
                samples = np.array(audio.get_array_of_samples())
                signal_level = np.mean(np.abs(samples))
                if signal_level < 100:  # Минимальный порог уровня сигнала
                    logger.error(f"Слишком низкий уровень сигнала: {signal_level}")
                    return {"segments": [], "error": "Слишком низкий уровень сигнала"}
                
            except Exception as e:
                logger.error(f"Ошибка при проверке конвертированного аудио: {str(e)}")
                return {"segments": [], "error": "Ошибка при проверке аудио"}
            
            # Подготавливаем аудио для отправки
            with open(enhanced_audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Формируем запрос к API
            url = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
            headers = {
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "audio/x-wav"
            }
            
            # Отправляем запрос
            try:
                response = requests.post(url, headers=headers, data=audio_data)
                response.raise_for_status()
                
                # Обрабатываем ответ
                result = response.json()
                if "result" not in result:
                    logger.error(f"Неожиданный ответ от API: {result}")
                    return {"segments": [], "error": "Ошибка распознавания речи"}
                
                # Форматируем результат
                transcript = result["result"]
                if not transcript.strip():
                    logger.error("Пустой результат распознавания")
                    return {"segments": [], "error": "Не удалось распознать речь"}
                
                # Разбиваем на сегменты
                segments = [{"text": transcript, "start": 0, "end": len(audio)/1000}]
                
                return {"segments": segments, "error": None}
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Ошибка при отправке запроса к API: {str(e)}")
                return {"segments": [], "error": "Ошибка при отправке запроса к API"}
            
        except Exception as e:
            logger.error(f"Ошибка при транскрибации: {str(e)}")
            return {"segments": [], "error": str(e)}
        finally:
            # Очищаем временные файлы
            try:
                if enhanced_audio_path != audio_path:
                    os.unlink(enhanced_audio_path)
            except:
                pass
    
    def separate_speakers(self, segments: List[Dict], operator_audio: np.ndarray, 
                         customer_audio: np.ndarray, sample_rate: int) -> Tuple[List[Dict], List[Dict]]:
        """
        Разделение транскрипции на оператора и клиента
        
        Args:
            segments (List[Dict]): Сегменты транскрипции
            operator_audio (np.ndarray): Аудио оператора
            customer_audio (np.ndarray): Аудио клиента
            sample_rate (int): Частота дискретизации
            
        Returns:
            Tuple[List[Dict], List[Dict]]: Транскрипции оператора и клиента
        """
        if not segments:
            logger.warning("Нет сегментов для разделения")
            return [], []
            
        operator_segments = []
        customer_segments = []
        
        for segment in segments:
            if not isinstance(segment, dict) or "text" not in segment:
                continue
                
            # Разделяем текст на предложения
            sentences = [s.strip() for s in segment["text"].split(".") if s.strip()]
            logger.info(f"Разделено на {len(sentences)} предложений")
            
            for i, sentence in enumerate(sentences):
                if not sentence:
                    continue
                    
                # Создаем новый сегмент
                new_segment = {
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": sentence + "."
                }
                
                # Чередуем оператора и клиента
                if i % 2 == 0:
                    new_segment["speaker"] = "Оператор"
                    operator_segments.append(new_segment)
                else:
                    new_segment["speaker"] = "Клиент"
                    customer_segments.append(new_segment)
        
        logger.info(f"Разделено на {len(operator_segments)} сегментов оператора и {len(customer_segments)} сегментов клиента")
        return operator_segments, customer_segments
    
    def format_transcript(self, segments: List[Dict]) -> str:
        """
        Форматирование транскрипции в читаемый текст
        
        Args:
            segments (List[Dict]): Сегменты транскрипции
            
        Returns:
            str: Отформатированный текст
        """
        if not segments:
            logger.warning("Нет сегментов для форматирования")
            return "Транскрибация недоступна"
            
        formatted_text = ""
        for segment in segments:
            if not isinstance(segment, dict) or "speaker" not in segment or "text" not in segment:
                continue
            speaker = segment["speaker"]
            text = segment["text"]
            formatted_text += f"{speaker}: {text}\n"
        
        logger.info(f"Отформатирован текст длиной {len(formatted_text)} символов")
        return formatted_text 