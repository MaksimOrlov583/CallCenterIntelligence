import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import streamlit as st
from typing import Tuple, List
import io
import base64
import pandas as pd

# Цветовая схема для эмоций
EMOTION_COLORS = {
    'радость': '#66BB6A',    # Зеленый
    'негативные': '#EF5350',  # Красный
    'нейтрально': '#BDBDBD'   # Серый
}

def create_emotion_timeline(timestamps: List[float], operator_emotions: List[str], 
                           customer_emotions: List[str]) -> str:
    """
    Создает визуализацию временной шкалы эмоций для оператора и клиента.
    
    Аргументы:
        timestamps (List[float]): Список временных точек
        operator_emotions (List[str]): Список эмоций оператора
        customer_emotions (List[str]): Список эмоций клиента
        
    Возвращает:
        str: Base64 строка с изображением графика
    """
    try:
        # Создание DataFrame для построения графика
        df = pd.DataFrame({
            'Время': timestamps,
            'Оператор': operator_emotions,
            'Клиент': customer_emotions
        })
        
        # Создание фигуры
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[1, 1])
        
        # Функция для построения графика эмоций
        def plot_emotions(ax, emotions, title):
            # Создаем цветовую карту для эмоций
            colors = [EMOTION_COLORS[emotion] for emotion in emotions]
            
            # Создаем столбчатую диаграмму
            bars = ax.bar(range(len(emotions)), [1] * len(emotions), color=colors)
            
            # Настройка графика
            ax.set_title(title, fontsize=12, pad=10)
            ax.set_xlabel('Время (секунды)', fontsize=10)
            ax.set_xticks(range(0, len(emotions), max(1, len(emotions)//10)))
            ax.set_xticklabels([f'{t:.1f}' for t in timestamps[::max(1, len(emotions)//10)]])
            ax.set_yticks([])
            ax.grid(True, alpha=0.3)
            
            # Добавляем подписи к столбцам
            for i, emotion in enumerate(emotions):
                if i % max(1, len(emotions)//10) == 0:  # Добавляем подписи только для каждого 10-го столбца
                    ax.text(i, 0.5, emotion, ha='center', va='center', color='black', fontsize=8)
        
        # Построение графиков для оператора и клиента
        plot_emotions(ax1, df['Оператор'], 'Эмоции оператора')
        plot_emotions(ax2, df['Клиент'], 'Эмоции клиента')
        
        # Добавление легенды
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=emotion.capitalize())
                          for emotion, color in EMOTION_COLORS.items()]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Настройка общего вида
        plt.tight_layout()
        
        # Сохраняем график в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Конвертируем в base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        
        return img_str
    except Exception as e:
        st.error(f"Ошибка при создании графика: {str(e)}")
        return ""

def create_emotion_distribution(emotions: List[str]) -> str:
    """
    Создает круговую диаграмму, показывающую распределение эмоций.
    
    Аргументы:
        emotions (List[str]): Список эмоций
        
    Возвращает:
        str: Base64 строка с изображением графика
    """
    try:
        # Подсчет вхождений каждой эмоции
        emotion_counts = {}
        for emotion in emotions:
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
            emotion_counts[emotion] += 1
        
        # Создание списков для круговой диаграммы
        labels = []
        values = []
        colors = []
        
        for emotion in EMOTION_COLORS.keys():
            count = emotion_counts.get(emotion, 0)
            if count > 0:
                labels.append(emotion.capitalize())
                values.append(count)
                colors.append(EMOTION_COLORS[emotion])
        
        # Создание круговой диаграммы
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        
        # Настройка графика
        ax.set_title('Распределение эмоций', fontsize=14, pad=20)
        
        # Сохраняем график в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Конвертируем в base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        
        return img_str
    except Exception as e:
        st.error(f"Ошибка при создании графика: {str(e)}")
        return ""

def plot_waveform(audio_path: str) -> str:
    """
    Создает волновой график аудио файла
    
    Args:
        audio_path (str): Путь к аудио файлу
        
    Returns:
        str: Base64 строка с изображением графика
    """
    try:
        # Загружаем аудио
        y, sr = librosa.load(audio_path, sr=None)
        
        # Создаем фигуру
        plt.figure(figsize=(12, 4))
        
        # Строим волновой график
        librosa.display.waveshow(y, sr=sr, color='#1f77b4')
        
        # Настраиваем внешний вид
        plt.title('Волновая форма аудио', fontsize=14, pad=20)
        plt.xlabel('Время (секунды)', fontsize=12)
        plt.ylabel('Амплитуда', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Сохраняем график в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Конвертируем в base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        
        return img_str
    except Exception as e:
        st.error(f"Ошибка при создании графика: {str(e)}")
        return ""

def plot_spectrogram(audio_path: str) -> str:
    """
    Создает спектрограмму аудио файла
    
    Args:
        audio_path (str): Путь к аудио файлу
        
    Returns:
        str: Base64 строка с изображением графика
    """
    try:
        # Загружаем аудио
        y, sr = librosa.load(audio_path, sr=None)
        
        # Создаем фигуру
        plt.figure(figsize=(12, 4))
        
        # Строим спектрограмму
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        
        # Настраиваем внешний вид
        plt.colorbar(format='%+2.0f dB')
        plt.title('Спектрограмма', fontsize=14, pad=20)
        plt.xlabel('Время (секунды)', fontsize=12)
        plt.ylabel('Частота (Гц)', fontsize=12)
        
        # Сохраняем график в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Конвертируем в base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        
        return img_str
    except Exception as e:
        st.error(f"Ошибка при создании спектрограммы: {str(e)}")
        return ""

def plot_emotions(emotions: List[Tuple[str, float]]) -> str:
    """
    Создает график эмоций
    
    Args:
        emotions (List[Tuple[str, float]]): Список эмоций и их значений
        
    Returns:
        str: Base64 строка с изображением графика
    """
    try:
        # Создаем фигуру
        plt.figure(figsize=(10, 6))
        
        # Подготавливаем данные
        labels = [e[0] for e in emotions]
        values = [e[1] for e in emotions]
        
        # Строим столбчатую диаграмму
        bars = plt.bar(labels, values, color='#1f77b4')
        
        # Добавляем значения над столбцами
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        # Настраиваем внешний вид
        plt.title('Распределение эмоций', fontsize=14, pad=20)
        plt.xlabel('Эмоция', fontsize=12)
        plt.ylabel('Интенсивность', fontsize=12)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Сохраняем график в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Конвертируем в base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        
        return img_str
    except Exception as e:
        st.error(f"Ошибка при создании графика эмоций: {str(e)}")
        return ""

def create_sample_visualization() -> str:
    """
    Создает пример визуализации для демонстрации.
    
    Returns:
        str: Base64 строка с изображением графика
    """
    try:
        # Создаем пример данных
        timestamps = np.linspace(0, 60, 60)  # 60 секунд
        operator_emotions = ['нейтрально'] * 60
        customer_emotions = ['нейтрально'] * 60
        
        # Добавляем некоторые изменения эмоций
        for i in range(10, 20):
            operator_emotions[i] = 'радость'
            customer_emotions[i] = 'радость'
        
        for i in range(30, 40):
            operator_emotions[i] = 'негативные'
            customer_emotions[i] = 'негативные'
        
        # Создаем визуализацию
        return create_emotion_timeline(timestamps, operator_emotions, customer_emotions)
    except Exception as e:
        st.error(f"Ошибка при создании примера визуализации: {str(e)}")
        return ""

def plot_linear_waveform(audio_path: str) -> str:
    """
    Создает линейный график аудио волны
    
    Args:
        audio_path (str): Путь к аудио файлу
        
    Returns:
        str: Base64 строка с изображением графика
    """
    try:
        # Загружаем аудио
        y, sr = librosa.load(audio_path, sr=None)
        
        # Создаем фигуру
        plt.figure(figsize=(12, 4))
        
        # Строим линейный график
        plt.plot(np.linspace(0, len(y)/sr, len(y)), y, color='#1f77b4', linewidth=0.5)
        
        # Настраиваем внешний вид
        plt.title('Линейный график аудио', fontsize=12, pad=20)
        plt.xlabel('Время (секунды)', fontsize=10)
        plt.ylabel('Амплитуда', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Устанавливаем пределы осей
        plt.xlim(0, len(y)/sr)
        plt.ylim(-1, 1)
        
        # Сохраняем в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # Конвертируем в base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        return img_str
        
    except Exception as e:
        st.error(f"Ошибка при создании линейного графика: {str(e)}")
        return None
