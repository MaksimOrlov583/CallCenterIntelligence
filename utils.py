import pandas as pd
from typing import List, Dict, Any
import numpy as np

def count_emotion_sequences(emotions: List[str], target_emotions: List[str], min_length: int = 2) -> int:
    """
    Подсчитывает последовательности целевых эмоций длиной не менее min_length.
    
    Аргументы:
        emotions (List[str]): Список эмоций
        target_emotions (List[str]): Список эмоций для подсчета
        min_length (int): Минимальная длина последовательности
        
    Возвращает:
        int: Количество последовательностей
    """
    count = 0
    current_streak = 0
    
    for emotion in emotions:
        if emotion in target_emotions:
            current_streak += 1
        else:
            if current_streak >= min_length:
                count += 1
            current_streak = 0
    
    # Проверка, закончились ли мы последовательностью
    if current_streak >= min_length:
        count += 1
    
    return count

def get_predominant_emotion(emotions: List[str]) -> str:
    """
    Получает наиболее частую эмоцию в списке.
    
    Аргументы:
        emotions (List[str]): Список эмоций
        
    Возвращает:
        str: Наиболее частая эмоция
    """
    emotion_counts = {}
    for emotion in emotions:
        if emotion not in emotion_counts:
            emotion_counts[emotion] = 0
        emotion_counts[emotion] += 1
    
    return max(emotion_counts.items(), key=lambda x: x[1])[0]

def calculate_call_quality(df: pd.DataFrame) -> float:
    """
    Рассчитывает общую оценку качества разговора (0-10).
    
    Аргументы:
        df (pd.DataFrame): DataFrame с эмоциями
        
    Возвращает:
        float: Оценка качества
    """
    # Инициализация базовой оценки
    score = 5.0
    
    # Штрафы за негативные эмоции
    operator_negative = count_emotion_sequences(df['Эмоция оператора'], ['негативные'])
    customer_negative = count_emotion_sequences(df['Эмоция клиента'], ['негативные'])
    
    # Штраф за чрезмерные негативные эмоции оператора
    score -= min(2.0, operator_negative * 0.3)
    
    # Небольшой штраф за негативные эмоции клиента (вне контроля оператора)
    score -= min(1.0, customer_negative * 0.1)
    
    # Бонусы за позитивные эмоции
    operator_positive = count_emotion_sequences(df['Эмоция оператора'], ['радость'])
    customer_positive = count_emotion_sequences(df['Эмоция клиента'], ['радость'])
    
    # Награда за позитивные эмоции оператора
    score += min(1.0, operator_positive * 0.2)
    
    # Большая награда за позитивные эмоции клиента (показывает хорошее обслуживание)
    score += min(2.0, customer_positive * 0.4)
    
    # Анализ соответствия эмоций
    emotion_match_rate = (df['Эмоция оператора'] == df['Эмоция клиента']).mean()
    
    # Награда за высокое соответствие эмоций с нейтральными или позитивными эмоциями
    positive_match = 0
    for i in range(len(df)):
        if df['Эмоция оператора'][i] == df['Эмоция клиента'][i] and \
           df['Эмоция оператора'][i] in ['нейтрально', 'радость']:
            positive_match += 1
    
    positive_match_rate = positive_match / len(df) if len(df) > 0 else 0
    score += min(2.0, positive_match_rate * 3.0)
    
    # Проверка на отражение оператором негативных эмоций
    negative_mirror = 0
    for i in range(1, len(df)):
        if df['Эмоция клиента'][i-1] == 'негативные' and \
           df['Эмоция оператора'][i] == 'негативные':
            negative_mirror += 1
    
    # Штраф за отражение негативных эмоций
    score -= min(1.0, negative_mirror * 0.3)
    
    # Проверка на восстановление - клиент начинает негативно, заканчивает позитивно
    if len(df) > 10:
        start_segment = df.iloc[:len(df)//3]
        end_segment = df.iloc[-len(df)//3:]
        
        start_customer_negative = sum(1 for e in start_segment['Эмоция клиента'] 
                                     if e == 'негативные') / len(start_segment)
        end_customer_positive = sum(1 for e in end_segment['Эмоция клиента'] 
                                   if e in ['радость', 'нейтрально']) / len(end_segment)
        
        # Награда за превращение негативного разговора в позитивный
        if start_customer_negative > 0.3 and end_customer_positive > 0.7:
            score += 2.0
    
    # Обеспечение оценки в диапазоне 0-10
    score = max(0.0, min(10.0, score))
    
    return score

def extract_key_moments(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Извлекает ключевые моменты из разговора.
    
    Аргументы:
        df (pd.DataFrame): DataFrame с эмоциями
        
    Возвращает:
        Dict[str, Any]: Словарь ключевых моментов
    """
    key_moments = {}
    
    # Определение моментов, когда клиент становится негативным
    negative_points = []
    for i in range(1, len(df)):
        if df['Эмоция клиента'][i] == 'негативные' and df['Эмоция клиента'][i-1] != 'негативные':
            negative_points.append(df['Время'][i])
    
    key_moments['негатив_клиента'] = negative_points
    
    # Определение моментов, когда оператор не адаптируется
    bad_response_points = []
    for i in range(1, len(df)):
        if df['Эмоция клиента'][i-1] == 'негативные' and \
           df['Эмоция оператора'][i] not in ['нейтрально', 'радость']:
            bad_response_points.append(df['Время'][i])
    
    key_moments['плохие_ответы'] = bad_response_points
    
    # Определение моментов, когда оператор успешно успокаивает клиента
    good_responses = []
    for i in range(2, len(df)):
        if df['Эмоция клиента'][i-2] == 'негативные' and \
           df['Эмоция оператора'][i-1] in ['нейтрально', 'радость'] and \
           df['Эмоция клиента'][i] in ['нейтрально', 'радость']:
            good_responses.append(df['Время'][i])
    
    key_moments['хорошие_ответы'] = good_responses
    
    return key_moments
