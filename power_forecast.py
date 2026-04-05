import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import os

# =====================
# НАСТРОЙКИ
# =====================
API_KEY = "36a44e780e97e0cdf7d9eec08749abd0"
LAT = 50.334336
LON = 58.612371
NUM_GENERATORS = 24
POWER_FILE = "Таблица мощности для прогноза.xlsx"
OUTPUT_FILE = "hourly_power_forecast.xlsx"

# =====================
# ЗАГРУЗКА ПОГОДЫ
# =====================
def get_weather():
    url = (
        f"https://api.openweathermap.org/data/2.5/forecast"
        f"?lat={LAT}&lon={LON}&units=metric&appid={API_KEY}"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    if data.get("cod") == "401":
        raise ValueError(f"Ошибка API: {data.get('message')}")
    if "list" not in data:
        raise ValueError(f"Некорректный ответ API: {data}")

    return data

# =====================
# ЗАГРУЗКА ТАБЛИЦЫ МОЩНОСТИ
# =====================
def load_power_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Файл '{path}' не найден. Положи его в ту же папку, где лежит скрипт."
        )

    df_raw = pd.read_excel(path, header=None)

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
            "Не удалось найти строку с заголовками 'Скорость ветра (м/с)' в Excel-файле."
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

# =====================
# ПОДГОТОВКА ДАННЫХ О ВЕТРЕ
# =====================
def prepare_hourly_wind(data: dict) -> pd.DataFrame:
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

# =====================
# ТВОЙ РАСЧЁТ МОЩНОСТИ — БЕЗ ИЗМЕНЕНИЙ
# =====================
def calculate_power(wind_speed, power_df):
    power_df_sorted = power_df.sort_values(by="Скорость ветра (м/с)").reset_index(drop=True)

    wind_speeds_table = power_df_sorted["Скорость ветра (м/с)"].values
    power_values_table = power_df_sorted["Мощность (кВт)"].values

    if wind_speed < wind_speeds_table.min() or wind_speed > wind_speeds_table.max():
        return 0
    else:
        return np.interp(wind_speed, wind_speeds_table, power_values_table)

# =====================
# ПРОГНОЗ МОЩНОСТИ
# =====================
def predict_power(df_hourly: pd.DataFrame, power_table_df: pd.DataFrame, num_generators: int = None) -> pd.DataFrame:
    if num_generators is None:
        num_generators = NUM_GENERATORS
    df_hourly["Мощность_кВт"] = df_hourly["Порывы_ветра_м_с"].apply(
        lambda x: calculate_power(x, power_table_df)
    )
    df_hourly["Общая_мощность_кВт"] = df_hourly["Мощность_кВт"] * num_generators
    return df_hourly

# =====================
# СОХРАНЕНИЕ
# =====================
def save_result(df_hourly: pd.DataFrame, output_path: str) -> None:
    df_export = df_hourly.copy()

    # Excel не любит timezone-aware datetime
    df_export["Время_UTC"] = df_export["Время_UTC"].dt.tz_localize(None)
    df_export["Время_UTC+1"] = df_export["Время_UTC+1"].dt.tz_localize(None)

    df_export.to_excel(output_path, index=False)

# =====================
# MAIN
# =====================
def main():
    print("Запуск прогноза ВЭС...")

    data = get_weather()
    power_table_df = load_power_table(POWER_FILE)
    df_hourly = prepare_hourly_wind(data)
    df_hourly = predict_power(df_hourly, power_table_df)
    save_result(df_hourly, OUTPUT_FILE)

    print(f"Готово. Файл сохранён: {OUTPUT_FILE}")
    print(df_hourly[["Время_UTC+1", "Порывы_ветра_м_с", "Мощность_кВт", "Общая_мощность_кВт"]].head(10))

if __name__ == "__main__":
    main()