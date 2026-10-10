"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    # Шаг 1. Загружаем таблицу из CSV-файла
    df = pd.read_csv(data.csv_path)

    # Количество строк (пассажиров) в таблице
    row_count = len(df)

    # Шаг 2. Пропуски в каждом столбце
    missing_by_column = {}
    for column in df.columns:
        missing_count = df[column].isna().sum()        # сколько пустых ячеек
        missing_by_column[column] = int(missing_count)

    # Шаг 3. Сколько пассажиров старше 30 лет
    # (там, где возраст неизвестен, сравнение даёт False - такие не считаются)
    is_over_30 = df["Age"] > 30
    adults_over_30_count = int(is_over_30.sum())

    # Шаг 4. Средний возраст по классам
    # groupby собирает пассажиров по Pclass, mean() пропуски пропускает сам
    mean_age = df.groupby("Pclass")["Age"].mean()
    mean_age_by_pclass = {}
    for pclass, value in mean_age.items():
        mean_age_by_pclass[int(pclass)] = float(value)

    # Шаг 5. Доля выживших по классам
    # Survived это 0 или 1, поэтому среднее и есть доля выживших
    survival = df.groupby("Pclass")["Survived"].mean()
    survival_rate_by_pclass = {}
    for pclass, value in survival.items():
        survival_rate_by_pclass[int(pclass)] = float(value)

    # Шаг 6. Пять наибольших тарифов по убыванию
    sorted_fares = df["Fare"].sort_values(ascending=False)   # от большего к меньшему
    highest_fares = sorted_fares.head(5).tolist()            # первые пять

    return TitanicSummary(
        row_count=row_count,
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares,
    )