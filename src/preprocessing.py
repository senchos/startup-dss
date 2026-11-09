"""Підготовка даних: перетворює сирі записи про стартапи (CSV або форму застосунку) на
таблицю ознак для ML-моделі та цільову змінну. Однакова обробка використовується і під час
навчання, і під час аналізу окремої компанії (SPEC.md §2.1)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder

# Рік, відносно якого обчислюється вік компанії
REFERENCE_YEAR = 2026

# Порядок стадій фінансування. У датасеті startups_dataset.csv є Seed < Series A < B < C < D+;
# Pre-seed та IPO залишено в порядку, бо їх знають модуль угоди й інтерфейс, хоча у
# навчальних даних їх немає (порядковий код екстраполюється на крок нижче Seed / вище Series D+).
FUNDING_STAGE_ORDER = ["Pre-seed", "Seed", "Series A", "Series B", "Series C", "Series D+", "IPO"]

# Відповідність «країна → регіон» (SPEC.md §2.1.1). Поточний датасет уже містить колонку
# Region (North America, Europe, Asia, South America, Oceania, Middle East & Africa), і вона
# використовується як є. Ця таблиця потрібна лише як запасний варіант для викликів, що знають
# тільки країну (демонстраційний скрипт, застарілий CSV): усе, чого немає в таблиці,
# наприклад "Ukraine", отримує значення "Other", яке one-hot-кодувальник трактує як невідому
# категорію (усі нулі, без регіонального ефекту).
REGION_MAP = {
    # North America
    "USA": "North America", "United States": "North America", "Canada": "North America",
    "Mexico": "North America",
    # Europe
    "UK": "Europe", "United Kingdom": "Europe", "Germany": "Europe", "France": "Europe",
    "Netherlands": "Europe", "Sweden": "Europe", "Spain": "Europe", "Italy": "Europe",
    # Asia
    "China": "Asia", "India": "Asia", "Singapore": "Asia", "Japan": "Asia", "South Korea": "Asia",
    # South America
    "Brazil": "South America", "Argentina": "South America",
    # Middle East & Africa
    "Israel": "Middle East & Africa", "UAE": "Middle East & Africa", "South Africa": "Middle East & Africa",
    # Oceania
    "Australia": "Oceania", "New Zealand": "Oceania",
}

# Межі шкали Success Score (SPEC.md §2.1.2): у startups_dataset.csv це неперервна оцінка
# від 1 до 10 (фактично 1.1–10.0), тому номінальні межі відображають її на шкалу [0, 100].
SCORE_MIN, SCORE_MAX = 1, 10

# Категоріальні ознаки для one-hot-кодування
ONEHOT_FEATURES = ["Region", "Industry", "Business Model"]
# Порядкова ознака: стадія фінансування
ORDINAL_FEATURES = ["Funding Stage"]
# Грошові суми та лічильники: log1p, потім заповнення пропусків медіаною з прапорцем пропуску
LOG_FEATURES = [
    "Total Funding ($M)",
    "Number of Employees",
    "Annual Revenue ($M)",
    "Valuation ($M)",
    "Customer Base (Thousands)",
    "Social Media Followers",
]
# Відношення, що обчислюються в add_derived_features (перші три вимагає SPEC.md §2.1;
# valuation_per_revenue — мультиплікатор виручки, доданий 07.10.2026 за результатами крос-валідації)
RATIO_FEATURES = [
    "funding_per_employee",
    "revenue_per_funding",
    "customers_per_employee",
    "valuation_per_revenue",
]
# Ознаки, що вже мають придатну шкалу: логарифм зростання (може бути від'ємним, тому log1p
# до нього не застосовується повторно) та вік компанії
LINEAR_FEATURES = ["growth_log", "age"]
# Повний перелік ознак моделі у фіксованому порядку
FEATURE_COLUMNS = ONEHOT_FEATURES + ORDINAL_FEATURES + LOG_FEATURES + RATIO_FEATURES + LINEAR_FEATURES

# Назва колонки з вихідною оцінкою успіху в датасеті
TARGET_RAW_COL = "Success Score"


def add_derived_features(df):
    """Додає похідні ознаки: вік компанії (age), чотири відношення (funding_per_employee,
    revenue_per_funding, customers_per_employee, valuation_per_revenue) та логарифм
    зростання виручки (growth_log)."""
    df = df.copy()
    # Вік компанії та відношення між грошовими показниками
    df["age"] = REFERENCE_YEAR - df["Founded Year"]
    revenue = df["Annual Revenue ($M)"]
    df["funding_per_employee"] = df["Total Funding ($M)"] / df["Number of Employees"]
    df["revenue_per_funding"] = revenue / df["Total Funding ($M)"]
    df["customers_per_employee"] = df["Customer Base (Thousands)"] / df["Number of Employees"]
    # Мультиплікатор виручки визначений лише за додатної виручки
    df["valuation_per_revenue"] = df["Valuation ($M)"] / revenue.where(revenue > 0)
    # Ділення на нуль дає нескінченність, яку замінюємо на пропуск
    df[RATIO_FEATURES] = df[RATIO_FEATURES].replace([np.inf, -np.inf], np.nan)

    # У даних відсутнє значення Revenue Growth не випадкове: у кожної компанії з нульовою
    # виручкою зростання не вказано. Цей пропуск структурний («немає виручки» = немає
    # зростання), тому його заповнено нулем; лише решту пропусків (виручка невідома або
    # зростання відсутнє попри наявну виручку) залишено імп'ютеру як справжні пропуски.
    # Логарифм log1p згладжує значення зростання аж до 684 %.
    growth = df["Revenue Growth (%)"].where(revenue != 0, 0.0)
    df["growth_log"] = np.log1p(growth.clip(lower=-99) / 100)
    return df


def add_region(df):
    """Визначає Region за Country через REGION_MAP, якщо колонки Region ще немає (датасет
    і форма застосунку передають регіон напряму). Невідомі країни отримують значення "Other"."""
    df = df.copy()
    if "Region" in df.columns:
        df["Region"] = df["Region"].fillna("Other")
        return df
    df["Region"] = df["Country"].map(REGION_MAP).fillna("Other")
    return df


def add_target(df):
    """Будує цільову змінну target ∈ [0, 100] зі Success Score (1–10) за формулою
    (score − 1) / 9 · 100. Ця колонка існує лише в пам'яті, у CSV її немає."""
    df = df.copy()
    df["target"] = (df[TARGET_RAW_COL] - SCORE_MIN) / (SCORE_MAX - SCORE_MIN) * 100
    return df


def target_to_raw_score(scaled):
    """Зворотне до add_target перетворення (SPEC.md §2.1.2): значення з [0, 100] назад у
    шкалу Success Score 1–10 (використовується в «Деталях розрахунку»)."""
    return scaled / 100 * (SCORE_MAX - SCORE_MIN) + SCORE_MIN


def build_feature_frame(raw_df):
    """Перетворює сирі рядки компаній на рівно ті 16 колонок FEATURE_COLUMNS, яких очікує
    ML-конвеєр. Використовується і під час навчання, і під час кожного аналізу. Пропуски
    залишаються NaN: їх заповнює конвеєр, позначаючи прапорцем, що значення було відсутнім."""
    df = raw_df.copy()
    for col in LOG_FEATURES + ["Revenue Growth (%)", "Founded Year"]:
        if col in df.columns:  # None або "" з форми стають NaN, а не колонкою типу object
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = add_derived_features(df)
    df = add_region(df)
    return df[FEATURE_COLUMNS]


def _log_then_impute():
    """Логарифм log1p, потім заповнення пропусків медіаною з прапорцем пропуску."""
    # Спочатку логарифм, щоб прапорці пропусків лишилися 0/1, а не були прологарифмовані
    return make_pipeline(
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
        SimpleImputer(strategy="median", add_indicator=True),
    )


def build_preprocessor():
    """Будує ColumnTransformer, що задає обробку кожної групи колонок:
    категорії (Region, Industry, Business Model) — one-hot, невідома категорія дає нулі;
    стадія (Funding Stage) — число за порядком Pre-seed < Seed < … < Series D+ < IPO;
    суми, лічильники та відношення — log1p, заповнення медіаною та прапорець пропуску;
    growth_log і age — заповнення медіаною та прапорець пропуску."""
    return ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), ONEHOT_FEATURES),
            (
                "ordinal",
                make_pipeline(
                    OrdinalEncoder(
                        categories=[FUNDING_STAGE_ORDER],
                        handle_unknown="use_encoded_value",
                        unknown_value=np.nan,
                    ),
                    SimpleImputer(strategy="median"),
                ),
                ORDINAL_FEATURES,
            ),
            ("log", _log_then_impute(), LOG_FEATURES + RATIO_FEATURES),
            ("linear", SimpleImputer(strategy="median", add_indicator=True), LINEAR_FEATURES),
        ],
    )


def prepare_dataset(path):
    """Читає CSV і повертає пару (X, y): таблицю ознак та цільову змінну. Викликається
    під час навчання моделі."""
    df = pd.read_csv(path)
    df = add_target(df)
    X = build_feature_frame(df)
    y = df["target"]
    return X, y
