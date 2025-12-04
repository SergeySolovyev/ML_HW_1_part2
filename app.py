import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from pathlib import Path

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

# --- Minimalistic global styling ---
def apply_minimal_style():
    st.markdown(
        """
        <style>
        /* Фон приложения */
        [data-testid="stAppViewContainer"] {
            background-color: #f3f4f6;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1100px;
            padding-top: 2.5rem;
            padding-bottom: 2.5rem;
        }

        /* Типографика */
        h1, h2, h3 {
            font-family: -apple-system, system-ui, BlinkMacSystemFont, "SF Pro Text", sans-serif;
            letter-spacing: -0.02em;
        }

        h1 {
            font-size: 2.2rem;
            font-weight: 600;
        }

        h2 {
            font-size: 1.4rem;
            font-weight: 500;
        }

        /* Вкладки */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.4rem 0.9rem;
            background-color: #e5e7eb;
            color: #4b5563;
            font-size: 0.9rem;
        }
        .stTabs [aria-selected="true"] {
            background-color: #111827;
            color: #f9fafb;
        }

        /* Метрики как карточки */
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            padding: 0.9rem 1.1rem;
            border-radius: 0.9rem;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.08);
        }

        /* Таблицы */
        .stDataFrame {
            border-radius: 0.75rem;
            overflow: hidden;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_minimal_style()

# Загрузка модели
@st.cache_resource
def load_model():
    """Загружаем модель через pickle"""
    try:
        with open('model.pickle', 'rb') as f:
            model_data = pickle.load(f)
        return model_data['scaler'], model_data['model'], model_data['feature_names']
    except FileNotFoundError:
        st.error("❌ Файл model.pickle не найден! Убедитесь, что модель сохранена.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        st.stop()


def prepare_features(df, feature_names, scaler):
    """Приводим данные к формату обучения модели."""
    df_proc = df.copy()
    
    # Проверяем наличие всех признаков
    missing_cols = set(feature_names) - set(df_proc.columns)
    if missing_cols:
        st.error(f"❌ Отсутствуют признаки: {missing_cols}")
        st.stop()
    
    # Выбираем только нужные признаки в правильном порядке
    X = df_proc[feature_names]
    
    # Стандартизация через scaler
    X_scaled = scaler.transform(X)
    
    return X_scaled


def preprocess_data_for_eda(df):
    """Предобработка данных для EDA - извлечение чисел из строк"""
    df_proc = df.copy()
    
    # Обрабатываем mileage, engine, max_power если они строки
    if 'mileage' in df_proc.columns and df_proc['mileage'].dtype == 'object':
        df_proc['mileage'] = df_proc['mileage'].str.extract(r'(\d+\.?\d*)', expand=False).astype(float)
    
    if 'engine' in df_proc.columns and df_proc['engine'].dtype == 'object':
        df_proc['engine'] = df_proc['engine'].str.extract(r'(\d+\.?\d*)', expand=False).astype(float)
    
    if 'max_power' in df_proc.columns and df_proc['max_power'].dtype == 'object':
        df_proc['max_power'] = df_proc['max_power'].str.extract(r'(\d+\.?\d*)', expand=False).astype(float)
    
    return df_proc


# Загружаем модель
try:
    SCALER, MODEL, FEATURE_NAMES = load_model()
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()


# --- Основной интерфейс ---
# --- Hero section ---
st.markdown(
    """
    <div style="margin-bottom: 1.75rem;">
        <div style="
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: .16em;
            color: #9ca3af;
            margin-bottom: .35rem;
        ">
            ML · regression
        </div>
        <h1 style="margin: 0 0 .5rem 0;">
            Предсказание стоимости автомобиля
        </h1>
        <p style="margin: 0; font-size: 0.95rem; color: #6b7280;">
            Загрузите датасет или введите параметры вручную, а затем изучите модель и её веса.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Создаем вкладки
tab1, tab2, tab3 = st.tabs(["EDA", "Предсказание", "Веса модели"])

# --- Вкладка 1: EDA ---
with tab1:
    st.header("Exploratory Data Analysis")
    
    # Загрузка данных для EDA
    @st.cache_data
    def load_train_data():
        df = pd.read_csv('cars_train.csv')
        # Предобработка для EDA
        df = preprocess_data_for_eda(df)
        return df
    
    try:
        df_train = load_train_data()
        st.success(f"✅ Данные загружены успешно! Размер датасета: {len(df_train)} строк, {len(df_train.columns)} столбцов")
    except FileNotFoundError:
        st.error("❌ Файл cars_train.csv не найден!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Ошибка загрузки данных: {e}")
        st.stop()
    
    # График 1: Распределение целевой переменной
    st.subheader("Распределение цены (selling_price)")
    fig1 = px.histogram(df_train, x='selling_price', nbins=50, 
                        title="Гистограмма распределения цен",
                        labels={'selling_price': 'Цена (₽)', 'count': 'Количество'})
    st.plotly_chart(fig1, use_container_width=True)
    
    # График 2: Распределение числовых признаков
    st.subheader("Распределение числовых признаков")
    numeric_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
    # Фильтруем только те, что есть в данных и числовые
    available_numeric = [col for col in numeric_cols if col in df_train.columns and pd.api.types.is_numeric_dtype(df_train[col])]
    
    if available_numeric:
        selected_feature = st.selectbox("Выберите признак для анализа", available_numeric)
        fig2 = px.histogram(df_train, x=selected_feature, nbins=30,
                           title=f"Распределение признака: {selected_feature}")
        st.plotly_chart(fig2, use_container_width=True)
    
    # График 3: Корреляционная матрица
    st.subheader("Матрица корреляций")
    # Используем только числовые столбцы
    numeric_for_corr = df_train[available_numeric].select_dtypes(include=[np.number])
    if len(numeric_for_corr.columns) > 1:
        corr_matrix = numeric_for_corr.corr()
        fig3 = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                         title="Корреляционная матрица числовых признаков",
                         color_continuous_scale='RdBu')
        st.plotly_chart(fig3, use_container_width=True)
    
    # График 4: Зависимость цены от важных признаков
    st.subheader("Зависимость цены от признаков")
    scatter_features = [col for col in ['year', 'km_driven', 'mileage', 'engine', 'max_power'] 
                       if col in available_numeric]
    if scatter_features:
        feature_x = st.selectbox("Признак X", scatter_features, key="scatter_x")
        fig4 = px.scatter(df_train, x=feature_x, y='selling_price',
                         title=f"Цена vs {feature_x}",
                         labels={'selling_price': 'Цена (₽)'})
        st.plotly_chart(fig4, use_container_width=True)
    
    # График 5: Boxplot по категориальным признакам
    st.subheader("Распределение цены по категориальным признакам")
    categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
    available_cat = [col for col in categorical_cols if col in df_train.columns]
    if available_cat:
        cat_feature = st.selectbox("Категориальный признак", available_cat, key="cat_feature")
        fig5 = px.box(df_train, x=cat_feature, y='selling_price',
                      title=f"Распределение цены по {cat_feature}",
                      labels={'selling_price': 'Цена (₽)'})
        st.plotly_chart(fig5, use_container_width=True)


# --- Вкладка 2: Предсказание ---
with tab2:
    st.header("Предсказание стоимости автомобиля")
    
    # Выбор способа ввода данных
    input_method = st.radio(
        "Выберите способ ввода данных:",
        ["Загрузить CSV файл", "Ручной ввод"],
        horizontal=True
    )
    
    if input_method == "Загрузить CSV файл":
        uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])
        
        if uploaded_file is None:
            st.info("Загрузите CSV файл для начала работы")
        else:
            try:
                df_input = pd.read_csv(uploaded_file)
                st.success(f"✅ Загружено {len(df_input)} строк")
                
                # Проверка наличия нужных признаков
                missing_cols = set(FEATURE_NAMES) - set(df_input.columns)
                if missing_cols:
                    st.error(f"❌ Отсутствуют признаки: {missing_cols}")
                else:
                    try:
                        # Подготовка данных
                        X_scaled = prepare_features(df_input, FEATURE_NAMES, SCALER)
                        
                        # Предсказание
                        predictions = MODEL.predict(X_scaled)
                        
                        # Добавляем предсказания в датафрейм
                        df_input['predicted_price'] = predictions
                        
                        # Отображение результатов
                        st.subheader("Результаты")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Количество автомобилей", len(df_input))
                        with col2:
                            avg_price = df_input['predicted_price'].mean()
                            st.metric("Средняя предсказанная цена", f"{avg_price:,.0f} ₽")
                        with col3:
                            min_price = df_input['predicted_price'].min()
                            max_price = df_input['predicted_price'].max()
                            st.metric("Диапазон цен", f"{min_price:,.0f} - {max_price:,.0f} ₽")
                        
                        # Визуализации
                        st.subheader("Визуализации")
                        
                        fig1 = px.histogram(df_input, x='predicted_price', nbins=30, 
                                           title="Распределение предсказанных цен")
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # Таблица с результатами
                        display_cols = ['predicted_price'] + FEATURE_NAMES
                        st.dataframe(df_input[display_cols].style.format({'predicted_price': '{:,.0f}'}), use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Ошибка при предсказании: {e}")
                        st.exception(e)
                        
            except Exception as e:
                st.error(f"❌ Ошибка при чтении файла: {e}")
                st.exception(e)
    
    else:  # Ручной ввод
        st.subheader("Сделать предсказание для нового автомобиля")
        
        # Загружаем данные для получения уникальных значений (если нужны категориальные)
        @st.cache_data
        def load_sample_data():
            return pd.read_csv('cars_train.csv')
        
        try:
            df_sample = load_sample_data()
        except:
            df_sample = None
        
        with st.form("prediction_form"):
            col_left, col_right = st.columns(2)
            input_data = {}
            
            with col_left:
                st.write("**Основные параметры:**")
                input_data['year'] = st.number_input("Год выпуска", min_value=1900, max_value=2024, value=2015, key="year")
                input_data['km_driven'] = st.number_input("Пробег (км)", min_value=0, value=50000, key="km_driven")
                input_data['mileage'] = st.number_input("Расход топлива (kmpl)", min_value=0.0, value=20.0, step=0.1, key="mileage")
            
            with col_right:
                st.write("**Характеристики двигателя:**")
                input_data['engine'] = st.number_input("Объем двигателя (CC)", min_value=0, value=1200, key="engine")
                input_data['max_power'] = st.number_input("Максимальная мощность (bhp)", min_value=0.0, value=80.0, step=0.1, key="max_power")
                input_data['seats'] = st.number_input("Количество мест", min_value=2, max_value=14, value=5, key="seats")
            
            submitted = st.form_submit_button("Предсказать", use_container_width=True)
        
        if submitted:
            try:
                input_df = pd.DataFrame([input_data])
                prepared_input = prepare_features(input_df, FEATURE_NAMES, SCALER)
                prediction = MODEL.predict(prepared_input)[0]
                
                st.success(f"**Предсказанная стоимость автомобиля: {prediction:,.0f} ₽**")
                st.progress(min(prediction / 2000000, 1.0), text=f"Оценка: {prediction:,.0f} ₽")
            except Exception as e:
                st.error(f"❌ Ошибка при предсказании: {e}")


# --- Вкладка 3: Веса модели ---
with tab3:
    st.header("Веса (коэффициенты) обученной модели")
    
    # Получаем коэффициенты модели
    coefficients = MODEL.coef_
    
    # Создаем DataFrame для удобства
    coef_df = pd.DataFrame({
        'Признак': FEATURE_NAMES,
        'Коэффициент': coefficients
    })
    
    # Сортируем по модулю коэффициента
    coef_df['|Коэффициент|'] = coef_df['Коэффициент'].abs()
    coef_df = coef_df.sort_values('|Коэффициент|', ascending=False)
    
    st.subheader("Результаты")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Всего признаков", len(coef_df))
    with col2:
        churn_rate = (coef_df['Коэффициент'] > 0).sum()
        st.metric("Положительных коэффициентов", churn_rate)
    with col3:
        avg_prob = (coef_df['Коэффициент'] < 0).sum()
        st.metric("Отрицательных коэффициентов", avg_prob)
    
    # Визуализации
    st.subheader("Визуализации")
    
    # График 1: Bar chart коэффициентов
    fig_coef = px.bar(
        coef_df, 
        x='Признак', 
        y='Коэффициент',
        title="Коэффициенты модели ElasticNet",
        color='Коэффициент',
        color_continuous_scale='RdYlGn',
        labels={'Коэффициент': 'Значение коэффициента', 'Признак': 'Признак'}
    )
    fig_coef.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_coef, use_container_width=True)
    
    # График 2: Горизонтальный bar chart (по модулю)
    fig_importance = px.bar(
        coef_df, 
        x='|Коэффициент|', 
        y='Признак',
        orientation='h',
        title="Важность признаков (абсолютные значения коэффициентов)",
        color='|Коэффициент|',
        color_continuous_scale='Blues',
        labels={'|Коэффициент|': 'Абсолютное значение коэффициента', 'Признак': 'Признак'}
    )
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Таблица
    st.subheader("Таблица коэффициентов")
    st.dataframe(coef_df[['Признак', 'Коэффициент']].style.format({'Коэффициент': '{:,.2f}'}), use_container_width=True)
