"""ML-модуль: навчання, порівняння та збереження регресійних моделей, що прогнозують
Success Score стартапу за кількісними показниками (SPEC.md §3)."""
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor, prepare_dataset

# Фіксоване зерно випадковості для відтворюваності результатів
RANDOM_STATE = 42
# Кількість блоків крос-валідації
CV_FOLDS = 5


def make_models():
    """Повертає нові ненавчені екземпляри трьох моделей: лінійної регресії, Random Forest
    та Gradient Boosting. Нові об'єкти створюються при кожному виклику, бо спільний словник
    екземплярів перенавчався б при кожному запуску й непомітно змінював би конвеєри
    попередніх запусків. Гіперпараметри лісу та бустингу підібрано 5-кратною
    крос-валідацією на навчальній вибірці (results_log.md, «Етап 2»: ліс із неглибокими
    листками, 500 повільних неглибоких дерев бустингу)."""
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=10, max_features=0.5, random_state=RANDOM_STATE
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.7,
            min_samples_leaf=10, random_state=RANDOM_STATE,
        ),
    }


def build_pipeline(model):
    """Об'єднує препроцесор і модель в один Pipeline. Саме такий об'єкт зберігається й потім
    викликається: на вході таблиця ознак, на виході число 0–100."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def evaluate(pipeline, X, y):
    """Метрики якості прогнозу: RMSE, MAE та R²."""
    y_pred = pipeline.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
    }


def cross_validate_pipelines(pipelines, X, y):
    """5-кратна крос-валідація кожного конвеєра на (X, y): середнє та стандартне відхилення
    RMSE і R²."""
    cv = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, pipeline in pipelines.items():
        scores = cross_validate(
            pipeline, X, y, cv=cv, n_jobs=-1,
            scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
        )
        results[name] = {
            "rmse": float(-scores["test_rmse"].mean()),
            "rmse_std": float(scores["test_rmse"].std()),
            "r2": float(scores["test_r2"].mean()),
            "r2_std": float(scores["test_r2"].std()),
        }
    return results


def train_and_compare(csv_path, test_size=0.2):
    """Головна функція навчання. Ділить дані на навчальну та тестову частини (80/20),
    обирає найкращу модель за RMSE крос-валідації лише на навчальній частині, потім
    навчає всі моделі на всій навчальній частині та один раз оцінює на тестовій. Вибір
    за тестовим RMSE (як було раніше) призводив б до витоку тестових даних у вибір.
    Повертає метрики (CV, тест, навчання) і навчені конвеєри."""
    # Підготовка даних і розподіл на навчальну та тестову частини
    X, y = prepare_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    # Крос-валідація всіх моделей і вибір найкращої за CV RMSE
    pipelines = {name: build_pipeline(model) for name, model in make_models().items()}
    cv_results = cross_validate_pipelines(pipelines, X_train, y_train)
    best_name = min(cv_results, key=lambda n: cv_results[n]["rmse"])

    # Фінальне навчання та єдина оцінка на тестовій вибірці
    results = {}
    train_results = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        results[name] = evaluate(pipeline, X_test, y_test)
        train_results[name] = evaluate(pipeline, X_train, y_train)

    return {
        "results": results,
        "train_results": train_results,
        "cv_results": cv_results,
        "pipelines": pipelines,
        "best_name": best_name,
        "best_pipeline": pipelines[best_name],
        "X_train": X_train,
        "y_train": y_train,
    }


def feature_diagnostics(train_out, top_n=15):
    """Діагностика нелінійного сигналу (SPEC.md §2.1.2, §3): взаємна інформація кожної
    перетвореної ознаки з ціллю та важливість ознак (feature_importances_) для Random Forest
    і Gradient Boosting. Модель із R² близько нуля все одно може «знаходити» структуру, на якій
    перенавчається; взаємна інформація та важливості — додаткова нелінійна перевірка того,
    чи є ця структура справжнім сигналом, чи шумом."""
    preprocessor = train_out["pipelines"]["random_forest"].named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    # Перетворена навчальна вибірка (розріджену матрицю робимо щільною)
    X_train_t = preprocessor.transform(train_out["X_train"])
    if hasattr(X_train_t, "toarray"):
        X_train_t = X_train_t.toarray()

    # Топ-N ознак за взаємною інформацією, потім за важливістю в деревних моделях
    mi = mutual_info_regression(X_train_t, train_out["y_train"], random_state=RANDOM_STATE)
    report = {
        "mutual_info": dict(
            sorted(zip(feature_names, mi), key=lambda kv: -kv[1])[:top_n]
        )
    }
    for name in ("random_forest", "gradient_boosting"):
        model = train_out["pipelines"][name].named_steps["model"]
        report[f"{name}_importances"] = dict(
            sorted(zip(feature_names, model.feature_importances_), key=lambda kv: -kv[1])[:top_n]
        )
    return report


def save_model(pipeline, path):
    """Зберігає навчений конвеєр у файл (joblib)."""
    joblib.dump(pipeline, path)


def load_model(path):
    """Завантажує збережений конвеєр із файлу."""
    return joblib.load(path)
