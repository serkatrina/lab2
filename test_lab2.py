from pathlib import Path

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput
from grader_contracts.correlation_tasks import BrainDataInput
from pandas_tasks import analyze_titanic
from correlation_tasks import analyze_brain_correlations

# Файлы с данными лежат рядом с этим файлом
FOLDER = Path(__file__).parent
TITANIC = str(FOLDER / "titanic.csv")
BRAIN = str(FOLDER / "brainsize.txt")


# ---------- Часть 1. Titanic ----------
def test_titanic_row_count():
    result = analyze_titanic(TitanicInput(TITANIC))
    assert result.row_count == 891


def test_titanic_missing_values():
    result = analyze_titanic(TitanicInput(TITANIC))
    assert result.missing_by_column["Age"] == 177
    assert result.missing_by_column["Cabin"] == 687
    assert result.missing_by_column["Embarked"] == 2
    assert result.missing_by_column["Survived"] == 0


def test_titanic_over_30():
    result = analyze_titanic(TitanicInput(TITANIC))
    # сверяем с независимым расчётом
    df = pd.read_csv(TITANIC)
    assert result.adults_over_30_count == len(df[df["Age"] > 30])


def test_titanic_mean_age_by_class():
    result = analyze_titanic(TitanicInput(TITANIC))
    assert set(result.mean_age_by_pclass.keys()) == {1, 2, 3}      # ключи это int
    assert abs(result.mean_age_by_pclass[1] - 38.23) < 0.01
    # возраст 0 вместо пропусков испортил бы среднее, поэтому оно должно быть больше 20
    assert result.mean_age_by_pclass[3] > 20


def test_titanic_survival_rate():
    result = analyze_titanic(TitanicInput(TITANIC))
    assert abs(result.survival_rate_by_pclass[1] - 0.6296) < 0.001
    assert abs(result.survival_rate_by_pclass[3] - 0.2424) < 0.001
    for rate in result.survival_rate_by_pclass.values():
        assert 0 <= rate <= 1


def test_titanic_highest_fares():
    result = analyze_titanic(TitanicInput(TITANIC))
    assert len(result.highest_fares) == 5
    assert result.highest_fares == sorted(result.highest_fares, reverse=True)
    assert result.highest_fares[0] == 512.3292


# ---------- Часть 2. Корреляции ----------
def test_brain_counts():
    result = analyze_brain_correlations(BrainDataInput(BRAIN))
    assert result.men_count == 20
    assert result.women_count == 20


def test_brain_correlation_keys_and_range():
    result = analyze_brain_correlations(BrainDataInput(BRAIN))
    expected_keys = {"FSIQ", "VIQ", "PIQ", "Weight", "Height"}
    assert set(result.men_mri_correlation.keys()) == expected_keys
    assert set(result.women_mri_correlation.keys()) == expected_keys
    for value in list(result.men_mri_correlation.values()) + list(result.women_mri_correlation.values()):
        assert -1 <= value <= 1          # корреляция всегда между -1 и 1


def test_brain_correlation_known_value():
    result = analyze_brain_correlations(BrainDataInput(BRAIN))
    # пропуски NA не должны ломать расчёт (в Weight у мужчин есть NA)
    assert abs(result.men_mri_correlation["PIQ"] - 0.568) < 0.001
    assert result.men_mri_correlation["Weight"] == result.men_mri_correlation["Weight"]  # не NaN


def test_brain_strongest_feature():
    result = analyze_brain_correlations(BrainDataInput(BRAIN))
    all_values = {}
    for name, value in result.men_mri_correlation.items():
        all_values["men " + name] = abs(value)
    for name, value in result.women_mri_correlation.items():
        all_values["women " + name] = abs(value)
    best = max(all_values, key=all_values.get)       # ключ с наибольшим значением
    assert result.strongest_mri_feature == best.split(" ")[1]