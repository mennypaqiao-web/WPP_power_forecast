import io
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
import altair as alt
import os

# Импорт функций из power_forecast.py (предполагаем, что они в том же файле)
# Для простоты скопируем функции сюда или импортируем

st.set_page_config(page_title="Прогноз мощности ВЭС", page_icon="⚡")
st.title("⚡ Прогноз мощности ветряной электростанции")
st.write(
    """
    Это приложение позволяет прогнозировать мощность ветряной электростанции на основе данных о погоде от OpenWeatherMap.
    Введите параметры, загрузите таблицу мощности и нажмите "Запустить прогноз".
    """
)

# Функции из power_forecast.py
def get_weather(api_key, lat, lon):
    url = (
        f"https://api.openweathermap.org/data/2.5/forecast"
        f"?lat={lat}&lon={lon}&units=metric&appid={api_key}"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    if data.get("cod") == "401":
        raise ValueError(f"Ошибка API: {data.get('message')}")
    if "list" not in data:
        raise ValueError(f"Некорректный ответ API: {data}")

    return data

def load_power_table_from_df(df_raw):
    header_row_index = -1
    for i, row in df_raw.iterrows():
        if any(
            pd.notna(cell) and str(cell).strip() == "Скорость ветра (м/с)"
            for cell in row
        ):
            header_row_index = i
            break

    if header_row_index == -1:
        raise ValueError(
            "Не удалось найти строку с заголовками 'Скорость ветра (м/с)' в файле."
        )

    header_row = df_raw.iloc[header_row_index]

    power_curve_dfs = []
    for col_idx in range(len(header_row)):
        if (
            pd.notna(header_row.iloc[col_idx])
            and str(header_row.iloc[col_idx]).strip() == "Скорость ветра (м/с)"
        ):
            if (
                col_idx + 1 < len(df_raw.columns)
                and pd.notna(header_row.iloc[col_idx + 1])
                and str(header_row.iloc[col_idx + 1]).strip() == "Мощность (кВт)"
            ):
                temp_df = df_raw.iloc[header_row_index + 1 :, [col_idx, col_idx + 1]].copy()
                temp_df.columns = ["Скорость ветра (м/с)", "Мощность (кВт)"]
                power_curve_dfs.append(temp_df)

    if not power_curve_dfs:
        raise ValueError(
            "Не удалось найти ни одной пары столбцов 'Скорость ветра (м/с)' и 'Мощность (кВт)'."
        )

    power_table_df = pd.concat(power_curve_dfs, ignore_index=True)
    power_table_df["Скорость ветра (м/с)"] = pd.to_numeric(
        power_table_df["Скорость ветра (м/с)"], errors="coerce"
    )
    power_table_df["Мощность (кВт)"] = pd.to_numeric(
        power_table_df["Мощность (кВт)"], errors="coerce"
    )

    power_table_df = power_table_df.dropna(
        subset=["Скорость ветра (м/с)", "Мощность (кВт)"]
    )
    power_table_df = power_table_df.sort_values(
        by="Скорость ветра (м/с)"
    ).reset_index(drop=True)

    return power_table_df

def prepare_hourly_wind(data):
    all_times = []
    all_wind_gusts = []

    for forecast_item in data["list"]:
        dt_object = datetime.fromtimestamp(forecast_item["dt"], timezone.utc)
        all_times.append(dt_object)
        all_wind_gusts.append(
            forecast_item["wind"].get("gust", forecast_item["wind"]["speed"])
        )

    df_3hourly = pd.DataFrame({
        "Время_UTC": all_times,
        "Порывы_ветра_м_с": all_wind_gusts,
    })

    df_3hourly = df_3hourly.set_index("Время_UTC")
    df_hourly = df_3hourly.resample("h").mean().interpolate(method="linear").reset_index()
    df_hourly["Время_UTC+1"] = df_hourly["Время_UTC"] + timedelta(hours=1)

    return df_hourly

def calculate_power(wind_speed, power_df):
    power_df_sorted = power_df.sort_values(by="Скорость ветра (м/с)").reset_index(drop=True)

    wind_speeds_table = power_df_sorted["Скорость ветра (м/с)"].values
    power_values_table = power_df_sorted["Мощность (кВт)"].values

    if wind_speed < wind_speeds_table.min() or wind_speed > wind_speeds_table.max():
        return 0
    else:
        return np.interp(wind_speed, wind_speeds_table, power_values_table)

def predict_power(df_hourly, power_table_df, num_generators):
    df_hourly["Мощность_кВт"] = df_hourly["Порывы_ветра_м_с"].apply(
        lambda x: calculate_power(x, power_table_df)
    )
    df_hourly["Общая_мощность_кВт"] = df_hourly["Мощность_кВт"] * num_generators
    return df_hourly

# Интерфейс
# Sidebar для настроек
st.sidebar.header("Настройки")
api_key = st.sidebar.text_input("API-ключ OpenWeatherMap", type="password")
lat = st.sidebar.number_input("Широта (LAT)", value=50.334336, format="%.6f")
lon = st.sidebar.number_input("Долгота (LON)", value=58.612371, format="%.6f")
num_generators = st.sidebar.number_input("Количество генераторов", value=24, min_value=1)

uploaded_file = st.sidebar.file_uploader("Загрузите Excel-файл с таблицей мощности", type=["xlsx"])

# Кэширование API
@st.cache_data
def cached_get_weather(api_key, lat, lon):
    return get_weather(api_key, lat, lon)

if st.sidebar.button("Запустить прогноз"):
    if not api_key:
        st.error("Введите API-ключ!")
    elif not uploaded_file:
        st.error("Загрузите файл с таблицей мощности!")
    else:
        try:
            # Загрузка таблицы мощности
            df_raw = pd.read_excel(uploaded_file, header=None)
            power_table_df = load_power_table_from_df(df_raw)
            
            # Показать таблицу мощности
            st.subheader("Таблица мощности генератора")
            st.dataframe(power_table_df)
            
            # Получение погоды
            data = cached_get_weather(api_key, lat, lon)
            
            # Подготовка данных
            df_hourly = prepare_hourly_wind(data)
            df_hourly = predict_power(df_hourly, power_table_df, num_generators)
            
            # Отображение результатов
            st.success("Прогноз готов!")
            
            # Таблица
            st.subheader("Результаты прогноза")
            st.dataframe(df_hourly[["Время_UTC+1", "Порывы_ветра_м_с", "Мощность_кВт", "Общая_мощность_кВт"]])

            def to_excel_bytes(df: pd.DataFrame) -> bytes:
                df_export = df.copy()
                df_export["Время_UTC"] = df_export["Время_UTC"].dt.tz_localize(None)
                df_export["Время_UTC+1"] = df_export["Время_UTC+1"].dt.tz_localize(None)

                daily_summary = (
                    df_export
                    .groupby(df_export["Время_UTC+1"].dt.date)
                    .agg(
                        Средняя_скорость_ветра_м_с=("Порывы_ветра_м_с", "mean"),
                        Средняя_мощность_кВт=("Мощность_кВт", "mean"),
                        Суммарная_мощность_кВт=("Общая_мощность_кВт", "sum"),
                    )
                    .reset_index()
                )
                daily_summary["Дата"] = daily_summary["Время_UTC+1"].astype(str)
                daily_summary = daily_summary[["Дата", "Средняя_скорость_ветра_м_с", "Средняя_мощность_кВт", "Суммарная_мощность_кВт"]]

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df_export.to_excel(writer, index=False, sheet_name="5-дневный прогноз")
                    daily_summary.to_excel(writer, index=False, sheet_name="Итоги по дням")

                return output.getvalue()

            excel_bytes = to_excel_bytes(df_hourly)
            st.download_button(
                label="Скачать прогноз за 5 дней в Excel",
                data=excel_bytes,
                file_name="forecast_5day.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # Графики
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("График скорости ветра")
                wind_chart = alt.Chart(df_hourly).mark_line(color="blue").encode(
                    x=alt.X(
                        "Время_UTC+1:T",
                        title="Дата",
                        axis=alt.Axis(format="%Y-%m-%d %H:%M", labelAngle=-45, labelFlush=True),
                    ),
                    y=alt.Y("Порывы_ветра_м_с:Q", title="Скорость ветра (м/с)"),
                ).properties(height=300)
                st.altair_chart(wind_chart, use_container_width=True)
            
            with col2:
                st.subheader("График мощности")
                power_chart = alt.Chart(df_hourly).mark_line(color="green").encode(
                    x=alt.X(
                        "Время_UTC+1:T",
                        title="Дата",
                        axis=alt.Axis(format="%Y-%m-%d %H:%M", labelAngle=-45, labelFlush=True),
                    ),
                    y=alt.Y("Общая_мощность_кВт:Q", title="Общая мощность (кВт)"),
                ).properties(height=300)
                st.altair_chart(power_chart, use_container_width=True)
            
            # Дополнительный график: кривая мощности
            st.subheader("Кривая мощности генератора")
            curve_chart = alt.Chart(power_table_df).mark_line(point=True).encode(
                x=alt.X("Скорость ветра (м/с):Q", title="Скорость ветра (м/с)"),
                y=alt.Y("Мощность (кВт):Q", title="Мощность (кВт)"),
            ).properties(height=300)
            st.altair_chart(curve_chart, use_container_width=True)
            
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")