"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

# Признаки, которые сравниваем с MRI_Count
FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]


def correlations_with_mri(group: pd.DataFrame) -> dict[str, float]:
    """Вспомогательная функция: корреляции Пирсона признаков с MRI_Count."""
    result = {}
    for feature in FEATURES:
        # corr() по умолчанию считает коэффициент Пирсона
        # и сам пропускает строки, где есть пропуск (NaN)
        value = group[feature].corr(group["MRI_Count"])
        result[feature] = float(value)
    return result


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    # Шаг 1. Загружаем файл: разделитель табуляция, "NA" значит "нет данных"
    df = pd.read_csv(data.csv_path, sep="\t", na_values="NA")

    # Шаг 2. Делим на мужчин и женщин
    men = df[df["Gender"] == "Male"]
    women = df[df["Gender"] == "Female"]

    # Шаг 3. Корреляции в каждой группе
    men_correlation = correlations_with_mri(men)
    women_correlation = correlations_with_mri(women)

    # Шаг 4. Ищем признак с самым большим модулем корреляции среди обеих групп
    strongest_feature = ""
    strongest_value = -1.0
    for correlations in (men_correlation, women_correlation):
        for feature, value in correlations.items():
            if abs(value) > strongest_value:      # abs() это модуль числа
                strongest_value = abs(value)
                strongest_feature = feature

    return BrainCorrelationSummary(
        men_count=len(men),
        women_count=len(women),
        women_mri_correlation=women_correlation,
        men_mri_correlation=men_correlation,
        strongest_mri_feature=strongest_feature,
    )