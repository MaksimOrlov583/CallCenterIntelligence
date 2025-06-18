import streamlit as st
import os
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime

import audio_processing
import emotion_analysis
import visualization
import utils
import transcription

# Настройка конфигурации страницы
st.set_page_config(
    page_title="Система мониторинга качества колл-центра",
    page_icon="📞",
    layout="wide"
)

# Инициализация переменных состояния сессии, если они не существуют
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'transcriber' not in st.session_state:
    st.session_state.transcriber = transcription.Transcriber()

# Главный заголовок и описание
st.title("🎧 Система мониторинга качества колл-центра")
st.write("""
Эта система анализирует взаимодействия оператора и клиента на основе записей колл-центра, визуализируя эмоциональные модели с течением времени для улучшения качества обслуживания.
""")

# Создание макета с боковой панелью
with st.sidebar:
    st.header("📁 Загрузка и управление")
    
    # Загрузчик файлов
    uploaded_file = st.file_uploader(
        "Загрузить запись разговора",
        type=["wav", "mp3"],
        help="Загрузите аудиофайлы записей колл-центра (формат WAV или MP3)"
    )
    
    # Кнопка обработки загруженного файла
    if uploaded_file and st.button("Обработать запись"):
        with st.spinner("Обработка аудиофайла..."):
            # Сохранение загруженного файла во временное место
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
            temp_file.write(uploaded_file.getvalue())
            temp_file.close()
            
            try:
                # Обработка аудиофайла
                operator_audio, customer_audio, sample_rate = audio_processing.separate_channels(temp_file.name)
                
                # Инициализируем анализатор эмоций
                emotion_analyzer = emotion_analysis.EmotionAnalyzer()
                
                # Анализ эмоций
                operator_emotions = emotion_analyzer.analyze_emotions(operator_audio, sample_rate)
                customer_emotions = emotion_analyzer.analyze_emotions(customer_audio, sample_rate)
                
                # Генерация временных меток
                duration = len(operator_audio) / sample_rate
                timestamps = np.linspace(0, duration, len(operator_emotions))
                
                # Транскрибация аудио
                with st.spinner("Транскрибация разговора..."):
                    try:
                        transcription_result = st.session_state.transcriber.transcribe_audio(temp_file.name)
                        if transcription_result and "segments" in transcription_result and transcription_result["segments"]:
                            operator_segments, customer_segments = st.session_state.transcriber.separate_speakers(
                                transcription_result["segments"],
                                operator_audio,
                                customer_audio,
                                sample_rate
                            )
                            transcript = st.session_state.transcriber.format_transcript(
                                sorted(operator_segments + customer_segments, key=lambda x: x.get("start", 0))
                            )
                            if not transcript or transcript == "Транскрибация недоступна":
                                st.warning("Не удалось распознать речь в аудиофайле. Возможные причины:\n"
                                         "- Слишком тихий звук\n"
                                         "- Неразборчивая речь\n"
                                         "- Отсутствие речи в файле")
                                transcript = "Не удалось распознать речь в аудиофайле"
                        else:
                            st.warning("Не удалось распознать речь в аудиофайле. Возможные причины:\n"
                                     "- Слишком тихий звук\n"
                                     "- Неразборчивая речь\n"
                                     "- Отсутствие речи в файле")
                            transcript = "Не удалось распознать речь в аудиофайле"
                    except Exception as e:
                        st.error(f"Ошибка при транскрибации: {str(e)}\n"
                                "Пожалуйста, убедитесь, что:\n"
                                "- Аудиофайл не поврежден\n"
                                "- Формат файла поддерживается (WAV или MP3)\n"
                                "- Есть доступ к интернету для работы сервиса распознавания речи")
                        transcript = "Ошибка при транскрибации"
                
                # Сохранение результатов
                file_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                result = {
                    "id": file_id,
                    "filename": uploaded_file.name,
                    "duration": duration,
                    "operator_emotions": operator_emotions,
                    "customer_emotions": customer_emotions,
                    "timestamps": timestamps,
                    "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "transcript": transcript
                }
                
                st.session_state.analysis_results[file_id] = result
                
                if file_id not in st.session_state.processed_files:
                    st.session_state.processed_files.append(file_id)
                
                st.session_state.current_file = file_id
                st.success(f"Успешно обработано: {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"Ошибка при обработке файла: {str(e)}")
            
            finally:
                # Удаление временного файла
                os.unlink(temp_file.name)
    
    # Отображение списка обработанных файлов
    if st.session_state.processed_files:
        st.subheader("Обработанные записи")
        for file_id in st.session_state.processed_files:
            file_info = st.session_state.analysis_results[file_id]
            if st.button(f"{file_info['filename']} ({file_info['processed_at']})", key=f"btn_{file_id}"):
                st.session_state.current_file = file_id
                st.rerun()

# Основная область контента
if st.session_state.current_file:
    result = st.session_state.analysis_results[st.session_state.current_file]
    
    st.header(f"Анализ: {result['filename']}")
    st.write(f"Длительность: {result['duration']:.2f} секунд")
    
    # Создание вкладок для различных визуализаций
    tab1, tab2, tab3, tab4 = st.tabs(["Временная шкала эмоций", "Общее распределение", "Детальный анализ", "Транскрибация"])
    
    with tab1:
        st.subheader("Временная шкала эмоций")
        emotion_timeline = visualization.create_emotion_timeline(
            result["timestamps"],
            result["operator_emotions"],
            result["customer_emotions"]
        )
        if emotion_timeline:
            st.image(f"data:image/png;base64,{emotion_timeline}", use_container_width=True)
    
    with tab2:
        st.subheader("Общее распределение эмоций")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Эмоции оператора")
            operator_dist = visualization.create_emotion_distribution(result["operator_emotions"])
            if operator_dist:
                st.image(f"data:image/png;base64,{operator_dist}", use_container_width=True)
        
        with col2:
            st.write("Эмоции клиента")
            customer_dist = visualization.create_emotion_distribution(result["customer_emotions"])
            if customer_dist:
                st.image(f"data:image/png;base64,{customer_dist}", use_container_width=True)
    
    with tab3:
        st.subheader("Детальный анализ")
        
        # Создание DataFrame для детального анализа
        df = pd.DataFrame({
            'Время': result["timestamps"],
            'Эмоция оператора': result["operator_emotions"],
            'Эмоция клиента': result["customer_emotions"]
        })
        
        # Показ интерактивной таблицы
        st.dataframe(df)
        
        # Показ статистики совпадения эмоций
        st.write("#### Совпадение эмоций")
        emotion_match_rate = (df['Эмоция оператора'] == df['Эмоция клиента']).mean() * 100
        st.metric("Совпадение эмоций", f"{emotion_match_rate:.1f}%", 
                 delta=None)
        
        # Выявление потенциальных проблем
        st.write("#### Потенциальные проблемы")
        negative_operator = utils.count_emotion_sequences(df['Эмоция оператора'], ['негативные'])
        negative_customer = utils.count_emotion_sequences(df['Эмоция клиента'], ['негативные'])
        
        if negative_operator > 3:
            st.warning(f"Оператор проявлял негативные эмоции {negative_operator} раз во время разговора")
        
        if negative_customer > 3:
            st.warning(f"Клиент проявлял негативные эмоции {negative_customer} раз во время разговора")
        
        # Анализ качества обслуживания
        quality_score = utils.calculate_call_quality(df)
        st.write("#### Оценка качества обслуживания")
        st.metric("Общая оценка", f"{quality_score:.1f}/10")
        
        # Рекомендации по улучшению
        st.write("#### Рекомендации по улучшению")
        if quality_score < 7:
            if negative_operator > 2:
                st.info("Рекомендация: Оператору следует лучше контролировать свои эмоции и сохранять профессиональный тон")
            if negative_customer > 3:
                st.info("Рекомендация: Стоит улучшить навыки деэскалации конфликтных ситуаций")
            if emotion_match_rate < 50:
                st.info("Рекомендация: Оператору следует лучше адаптироваться к эмоциональному состоянию клиента")
        else:
            st.success("Общее качество обслуживания хорошее. Продолжайте поддерживать высокий уровень!")
    
    with tab4:
        st.subheader("Транскрибация разговора")
        if result["transcript"] and result["transcript"] != "Транскрибация недоступна":
            st.text_area("Транскрибация", result["transcript"], height=400)
            st.info("💡 Подсказка: Текст разделен на реплики оператора и клиента")
        else:
            st.warning("Транскрибация недоступна для этого файла")
            st.info("Возможные причины:\n"
                   "- Слишком тихий звук\n"
                   "- Неразборчивая речь\n"
                   "- Отсутствие речи в файле\n"
                   "- Проблемы с доступом к сервису распознавания речи")

# Визуализация аудио
if uploaded_file is not None:
    st.subheader("Визуализация аудио")
    
    # Сохраняем загруженный файл во временное место
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
    temp_file.write(uploaded_file.getvalue())
    temp_file.close()
    
    try:
        # Линейный график
        linear_img = visualization.plot_linear_waveform(temp_file.name)
        if linear_img:
            st.image(f"data:image/png;base64,{linear_img}", use_container_width=True)
        
        # Волновой график и спектрограмма
        col1, col2 = st.columns(2)
        with col1:
            waveform_img = visualization.plot_waveform(temp_file.name)
            if waveform_img:
                st.image(f"data:image/png;base64,{waveform_img}", use_container_width=True)
        
        with col2:
            spectrogram_img = visualization.plot_spectrogram(temp_file.name)
            if spectrogram_img:
                st.image(f"data:image/png;base64,{spectrogram_img}", use_container_width=True)
    finally:
        # Удаляем временный файл
        os.unlink(temp_file.name)

else:
    # Отображение инструкций, когда файл не выбран
    st.info("👈 Пожалуйста, загрузите и обработайте запись разговора для просмотра анализа")
    
    # Пример визуализации
    st.subheader("Пример визуализации")
    sample_viz = visualization.create_sample_visualization()
    if sample_viz:
        st.image(f"data:image/png;base64,{sample_viz}", use_container_width=True)
    
    # Информация о системе
    st.write("""
    ### Как это работает:
    1. **Загрузите** аудиофайлы записей колл-центра
    2. Система **разделит** аудио на каналы оператора и клиента
    3. **Проанализирует эмоции** с использованием библиотеки Anyamore
    4. **Транскрибирует** разговор с помощью Whisper
    5. **Визуализирует** эмоциональные паттерны во времени
    6. Просмотрите результаты для улучшения качества колл-центра
    
    ### Поддерживаемые функции:
    - Обработка аудио форматов WAV и MP3
    - Разделение каналов для различения оператора и клиента
    - Анализ эмоций для определения: радость, грусть, гнев, страх, нейтрально
    - Автоматическая транскрибация с определением говорящих
    - Временная визуализация изменений эмоций
    - Общая оценка качества разговора
    """)
