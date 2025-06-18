import os
from audio_processing import get_audio_quality_report, analyze_and_enhance_audio
import librosa
import soundfile as sf

def analyze_audio_file(audio_path):
    """
    Анализирует аудио файл и выводит подробный отчет о его качестве.
    
    Args:
        audio_path (str): Путь к аудио файлу
    """
    print(f"\nАнализ файла: {audio_path}")
    print("-" * 50)
    
    # Получаем отчет о качестве
    report = get_audio_quality_report(audio_path)
    
    # Проверяем наличие ошибок
    if 'error' in report:
        print(f"Ошибка: {report['error']}")
        print("\nРекомендации:")
        for rec in report['recommendations']:
            print(f"- {rec}")
        return
    
    # Выводим результаты анализа
    print("\nРезультаты анализа:")
    print(f"- Наличие речи: {'Да' if report['has_speech'] else 'Нет'}")
    print(f"- Уровень громкости: {'Слишком тихий' if report['is_too_quiet'] else 'Нормальный'}")
    print(f"- Четкость речи: {'Неразборчивая' if report['is_unclear'] else 'Разборчивая'}")
    print(f"- Исходный уровень сигнала: {report['original_rms']:.4f}")
    print(f"- Уровень сигнала после обработки: {report['enhanced_rms']:.4f}")
    
    print("\nРекомендации:")
    for rec in report['recommendations']:
        print(f"- {rec}")
    
    # Если есть проблемы, предлагаем улучшить качество
    if report['is_too_quiet'] or report['is_unclear'] or not report['has_speech']:
        print("\nХотите улучшить качество аудио? (y/n)")
        if input().lower() == 'y':
            # Загружаем и улучшаем аудио
            audio, sample_rate = librosa.load(audio_path, sr=None)
            enhanced_audio, _ = analyze_and_enhance_audio(audio, sample_rate)
            
            # Создаем имя для улучшенного файла
            base_name = os.path.splitext(audio_path)[0]
            enhanced_path = f"{base_name}_enhanced.wav"
            
            # Сохраняем улучшенное аудио
            sf.write(enhanced_path, enhanced_audio, sample_rate)
            print(f"\nУлучшенное аудио сохранено в: {enhanced_path}")

if __name__ == "__main__":
    # Запрашиваем путь к аудио файлу
    audio_path = input("Введите путь к аудио файлу: ")
    analyze_audio_file(audio_path) 