"""Агрегатор: об'єднує оцінки ML-, нечіткого та угодного модулів у підсумковий Score
з вагами AHP і визначає категорію інвестиційної привабливості (SPEC.md §1).
Модуль не залежить ні від інтерфейсу, ні від бази даних."""
import numpy as np
import pandas as pd

from src import ahp_module, deal_module, fuzzy_module
from src.preprocessing import SCORE_MAX, SCORE_MIN, build_feature_frame, target_to_raw_score

# Текстові рекомендації для кожної категорії привабливості
RECOMMENDATIONS = {
    "низька": "Низька інвестиційна привабливість за сукупністю показників. Рекомендується поглиблений due diligence або відмова від угоди на поточних умовах.",
    "помірна": "Помірна інвестиційна привабливість. Проєкт має окремі сильні сторони, але потребує додаткової перевірки ключових факторів перед прийняттям рішення.",
    "висока": "Висока інвестиційна привабливість. Компанія демонструє сильні показники за більшістю компонентів оцінки — рекомендується розглянути угоду.",
    "дуже висока": "Дуже висока інвестиційна привабливість. Виняткові показники за всіма компонентами оцінки — пріоритетний кандидат для інвестування.",
}
# Позначка категорії для відображення
CATEGORY_ICON = {"низька": "🔴", "помірна": "🟡", "висока": "🟢", "дуже висока": "🟢"}


def categorize(score):
    """Відображає підсумковий Score з діапазону [0, 100] на категорію (SPEC.md §1).
    Пороги: 40, 70 та 90."""
    if score < 40:
        return "низька"
    if score < 70:
        return "помірна"
    if score < 90:
        return "висока"
    return "дуже висока"


def effective_weights(weights, has_deal):
    """Ваги, з якими фактично обчислюється Score. Якщо оцінка угоди G(D) відсутня
    (або її вага дорівнює 0), ваги ML та Fuzzy перенормовуються так, щоб їх сума дорівнювала 1
    (SPEC.md §1)."""
    w_ml, w_fuzzy, w_deal = weights.get("ml", 0), weights.get("fuzzy", 0), weights.get("deal", 0)
    if not has_deal or w_deal == 0:
        total = w_ml + w_fuzzy
        return {"ml": w_ml / total, "fuzzy": w_fuzzy / total, "deal": 0.0}
    return {"ml": w_ml, "fuzzy": w_fuzzy, "deal": w_deal}


def aggregate_scores(ml_score, fuzzy_score, deal_score, weights):
    """Підсумковий Score = w1·M(X) + w2·F(Y) + w3·G(D) (SPEC.md, формули 2.8–2.10).
    Якщо G(D) відсутня, ваги перенормовуються лише між ML та Fuzzy.
    Повертає пару (Score, категорія)."""
    # Ефективні ваги з урахуванням наявності умов угоди
    w = effective_weights(weights, deal_score is not None)
    score = w["ml"] * ml_score + w["fuzzy"] * fuzzy_score
    if deal_score is not None:
        score += w["deal"] * deal_score
    # Обмеження результату межами шкали [0, 100]
    score = float(np.clip(score, 0, 100))
    return score, categorize(score)


def format_formula(details):
    """Формує рядок формули з підставленими числами, наприклад:
    'Score = 0.25×56.7 + 0.60×79.1 + 0.15×96.7 = 76.1'."""
    f = details["formula"]
    terms = " + ".join(f"{t['weight']:.2f}×{t['value']:.1f}" for t in f["terms"])
    return f"Score = {terms} = {f['score']:.1f}"


def _round(value, digits=4):
    """Округлення для запису в JSON; None залишається None."""
    return None if value is None else round(float(value), digits)


def build_details(ml_pred, ml_score, fuzzy_result, inputs, deal_result, weight_preset,
                  base_weights, weights, ahp_meta, score, category, top_rules=5):
    """Збирає всі проміжні значення одного аналізу (SPEC.md §4.4): саме їх показують
    блок «Деталі розрахунку» та PDF-звіт. Результат серіалізується в JSON і зберігається
    разом з аналізом."""
    # Складові підсумкової формули: назва, вага та значення кожного компонента
    terms = [
        {"name": "ml", "label": "M(X)", "weight": weights["ml"], "value": ml_score},
        {"name": "fuzzy", "label": "F(Y)", "weight": weights["fuzzy"], "value": fuzzy_result["F"]},
    ]
    if deal_result is not None:
        terms.append({"name": "deal", "label": "G(D)", "weight": weights["deal"], "value": deal_result["G"]})

    # Сили спрацювання всіх 27 правил нечіткого модуля
    firings = fuzzy_result["rules"]
    return {
        # Блок ML: прогноз у шкалі 0–100, ознака обрізання та вихідна оцінка за шкалою 1–10
        "ml": {
            "predicted_scaled": _round(ml_pred),
            "score": _round(ml_score),
            "clipped": bool(ml_pred != ml_score),
            "raw_score": _round(target_to_raw_score(ml_pred)),
            "score_min": SCORE_MIN,
            "score_max": SCORE_MAX,
        },
        # Блок Fuzzy: експертні оцінки, проміжні змінні каскаду, нормалізація та найсильніші правила
        "fuzzy": {
            "inputs": {k: _round(v, 2) for k, v in inputs.items()},
            "execution_capability": _round(fuzzy_result["execution_capability"]),
            "market_attractiveness": _round(fuzzy_result["market_attractiveness"]),
            "favorability": _round(fuzzy_result["favorability"]),
            "F_raw": _round(fuzzy_result["F_raw"]),
            "F": _round(fuzzy_result["F"]),
            "F_RAW_MIN": fuzzy_module.F_RAW_MIN,
            "F_RAW_MAX": fuzzy_module.F_RAW_MAX,
            "rules_total": len(firings),
            "rules_fired": sum(1 for f in firings if f["strength"] > 0),
            "top_rules": [
                {**r, "strength": _round(r["strength"])}
                for r in fuzzy_module.strongest_rules(firings, top_rules)
            ],
        },
        # Блок AHP: профіль ваг, базові та ефективні ваги, показники узгодженості CI/CR
        "ahp": {
            "preset": weight_preset,
            "base_weights": {k: _round(v) for k, v in base_weights.items()},
            "weights": {k: _round(v) for k, v in weights.items()},
            "renormalized": any(abs(weights[k] - base_weights.get(k, 0)) > 1e-9 for k in weights),
            "ci": _round(ahp_meta["ci"]) if ahp_meta else None,
            "cr": _round(ahp_meta["cr"]) if ahp_meta else None,
            "consistent": ahp_meta["consistent"] if ahp_meta else None,
        },
        # Блок угоди: порожній, якщо умови угоди не вказано
        "deal": None
        if deal_result is None
        else {k: (_round(v) if isinstance(v, float) else v) for k, v in deal_result.items()},
        # Підсумкова формула та категорія
        "formula": {
            "terms": [{**t, "weight": _round(t["weight"]), "value": _round(t["value"])} for t in terms],
            "score": _round(score),
            "category": category,
        },
    }


def analyze_startup(
    company_raw,
    qualitative,
    ml_pipeline,
    deal_terms=None,
    weight_preset="balanced",
    custom_matrix=None,
    custom_labels=None,
):
    """Повний цикл аналізу стартапу (SPEC.md §1): M(X), F(Y) та за наявності G(D),
    далі ваги AHP і підсумкові Score та категорія.

    company_raw — кількісні дані компанії; qualitative — чотири експертні оцінки (0–10);
    ml_pipeline — навчена модель; deal_terms — умови угоди (необов'язково);
    weight_preset, custom_matrix, custom_labels — профіль ваг інвестора."""
    # 1. ML-компонент: підготовка ознак і прогноз моделі, обмежений шкалою [0, 100]
    company_df = pd.DataFrame([company_raw])
    X = build_feature_frame(company_df)
    ml_pred = float(ml_pipeline.predict(X)[0])
    ml_score = float(np.clip(ml_pred, 0, 100))

    # 2. Нечіткий компонент: каскад Мамдані за чотирма експертними оцінками
    fuzzy_inputs = {
        "team_quality": qualitative["team_quality"],
        "traction": qualitative["traction"],
        "market_size": qualitative["market_size"],
        "competition": qualitative["competition"],
    }
    fuzzy_result = fuzzy_module.evaluate(**fuzzy_inputs, explain=True)
    fuzzy_score = fuzzy_result["F"]

    # 3. Компонент угоди: враховується лише за наявності умов угоди
    deal_result = None
    deal_score = None
    if deal_terms is not None:
        deal_result = deal_module.evaluate_deal(
            company_raw["Funding Stage"],
            deal_terms["funding_ask"],
            deal_terms["pre_money_valuation"],
        )
        deal_score = deal_result["G"]

    # 4. Ваги профілю інвестора (пресет або матриця попарних порівнянь AHP)
    base_weights, ahp_meta = ahp_module.resolve_weights(
        weight_preset,
        has_deal=deal_score is not None,
        custom_matrix=custom_matrix,
        custom_labels=custom_labels,
    )

    # resolve_weights уже перенормовує пресети за відсутності угоди; повний пресет
    # зберігається як «ваги профілю», щоб у деталях було видно вибір інвестора.
    if weight_preset in ahp_module.PRESETS:
        base_weights = dict(ahp_module.PRESETS[weight_preset])
    # 5. Ефективні ваги, підсумковий Score і категорія
    weights = effective_weights(base_weights, deal_score is not None)
    score, category = aggregate_scores(ml_score, fuzzy_score, deal_score, base_weights)
    # 6. Проміжні значення для екрана й PDF-звіту
    details = build_details(
        ml_pred, ml_score, fuzzy_result, fuzzy_inputs, deal_result, weight_preset,
        base_weights, weights, ahp_meta, score, category,
    )

    return {
        "ml_score": ml_score,
        "fuzzy": fuzzy_result,
        "deal": deal_result,
        "weights": weights,
        "weights_base": base_weights,
        "ahp_meta": ahp_meta,
        "score": score,
        "category": category,
        "details": details,
    }
