"""Перетворює словник `details` одного аналізу (aggregator.build_details) на секції,
готові до відображення. Блок «Деталі розрахунку» в Streamlit і PDF-звіт будують вміст з
тих самих секцій, тому покроковий розрахунок на екрані та в друкованому звіті
(SPEC.md §4.4) не може розійтися. У тексті уникається символ '$', щоб Streamlit
не сприйняв його як початок формули LaTeX."""
from src import fuzzy_module
from src.aggregator import format_formula

# Українські назви профілів ваг
PRESET_LABELS = {
    "conservative": "Консервативний",
    "growth": "Зростання",
    "balanced": "Збалансований",
    "custom": "Власний (попарне порівняння)",
}
# Скорочені назви компонентів для рядка ваг
COMPONENT_LABELS = {"ml": "ML", "fuzzy": "Fuzzy", "deal": "Deal"}


def _weights_line(weights):
    """Рядок виду «ML 0.25 / Fuzzy 0.60 / Deal 0.15»; нульова вага угоди не виводиться."""
    return " / ".join(f"{COMPONENT_LABELS[k]} {v:.2f}" for k, v in weights.items() if k == "ml" or k == "fuzzy" or v)


def _ml_section(ml):
    """Секція 1: перемасштабування прогнозу ML зі шкали Success Score на шкалу 0–100."""
    lo, hi = ml["score_min"], ml["score_max"]
    lines = [
        f"Сира прогнозна оцінка Success Score (шкала {lo}–{hi}): {ml['raw_score']:.2f}",
        f"Перемасштабування: M(X) = (оцінка − {lo}) / ({hi} − {lo}) × 100 = {ml['predicted_scaled']:.2f}",
    ]
    if ml["clipped"]:
        lines.append(f"Значення поза [0, 100] обрізано до {ml['score']:.2f}")
    else:
        lines.append(f"M(X) = {ml['score']:.2f}")
    return {"title": "1. ML-модуль — M(X)", "lines": lines, "table": None}


def _fuzzy_section(fz):
    """Секція 2: кроки каскаду Мамдані, нормалізація F(Y) та таблиця найсильніших правил."""
    # Підписи вхідних змінних беруться з файлу правил
    variables = fuzzy_module.get_rules()["variables"]
    inputs = ", ".join(f"{variables[k]['label']} {v:g}" for k, v in fz["inputs"].items())
    lines = [
        f"Експертні оцінки (0–10): {inputs}",
        f"Рівень 1: Виконавча спроможність = {fz['execution_capability']:.2f} / 10; "
        f"Ринкова привабливість = {fz['market_attractiveness']:.2f} / 10",
        f"Рівень 2: Сприятливість (centroid) = {fz['favorability']:.2f} / 10, "
        f"F_raw = {fz['favorability']:.2f} × 10 = {fz['F_raw']:.2f}",
        f"Нормалізація: F(Y) = (F_raw − {fz['F_RAW_MIN']}) / ({fz['F_RAW_MAX']} − {fz['F_RAW_MIN']}) × 100 = {fz['F']:.2f}",
        f"Спрацювало правил: {fz['rules_fired']} з {fz['rules_total']}. Найсильніші:",
    ]
    # Рядки таблиці: номер, рівень, умова, висновок і сила спрацювання правила
    rows = [
        [r["id"], r["level"], r["if"], r["then"], f"{r['strength']:.3f}"]
        for r in fz["top_rules"]
    ]
    return {
        "title": "2. Fuzzy-модуль — F(Y)",
        "lines": lines,
        "table": {"headers": ["№", "Рівень", "ЯКЩО", "ТО", "Сила"], "rows": rows},
    }


def _ahp_section(ahp):
    """Секція 3: профіль ваг, базові та перенормовані ваги, узгодженість матриці (CI, CR)."""
    lines = [
        f"Профіль ваг: {PRESET_LABELS.get(ahp['preset'], ahp['preset'])}",
        f"Ваги профілю: {_weights_line(ahp['base_weights'])}",
    ]
    if ahp["renormalized"]:
        lines.append(f"Умов угоди немає — ваги перенормовано: {_weights_line(ahp['weights'])}")
    if ahp["cr"] is not None:
        verdict = "судження узгоджені" if ahp["consistent"] else "CR > 0.10 — судження неузгоджені"
        lines.append(f"CI = {ahp['ci']:.4f}, CR = {ahp['cr']:.4f} ({verdict})")
    else:
        lines.append("CI / CR не застосовуються: пресет задає ваги напряму, без матриці порівнянь")
    return {"title": "3. AHP-модуль — ваги", "lines": lines, "table": None}


def _deal_section(deal):
    """Секція 4: часткові оцінки умов угоди та їх порівняння з діапазонами стадії."""
    if deal is None:
        return {
            "title": "4. Умови угоди — G(D)",
            "lines": ["Умови угоди не вказано, компонент G(D) не враховується."],
            "table": None,
        }
    r = deal["ranges"]

    def rng(values, fmt):
        # Діапазон стадії у вигляді «мінімум / медіана / максимум»
        return " / ".join(fmt(v) for v in values)

    # Форматування грошових сум та відсотків
    money = lambda v: f"${v:g}M"
    pct = lambda v: f"{v * 100:g}%"
    lines = [
        f"Стадія: {deal['stage']}. Post-money = {deal['pre_money_valuation']:g} + {deal['funding_ask']:g} = "
        f"{deal['post_money']:g} (млн); частка інвестора = {deal['dilution_pct'] * 100:.1f}%",
        f"G(D) = середнє трьох часткових оцінок = {deal['G']:.2f}",
    ]
    rows = [
        ["Funding ask", money(deal["funding_ask"]), rng(r["funding_ask"], money), f"{deal['funding_score']:.1f}"],
        ["Pre-money valuation", money(deal["pre_money_valuation"]),
         rng(r["pre_money_valuation"], money), f"{deal['valuation_score']:.1f}"],
        ["Dilution", pct(deal["dilution_pct"]), rng(r["dilution"], pct), f"{deal['dilution_score']:.1f}"],
    ]
    return {
        "title": "4. Умови угоди — G(D)",
        "lines": lines,
        "table": {
            "headers": ["Показник", "Значення", "Типовий діапазон стадії (low / median / high)", "Часткова оцінка"],
            "rows": rows,
        },
    }


def _formula_section(details):
    """Секція 5: підсумкова формула з підставленими числами та категорія."""
    return {
        "title": "5. Підсумкова формула",
        "lines": [
            format_formula(details),
            f"Категорія: {details['formula']['category']} (пороги 40 / 70 / 90)",
        ],
        "table": None,
    }


def build_sections(details):
    """П'ять секцій розрахунку (ML, Fuzzy, AHP, угода, підсумкова формула). Кожна секція —
    словник {'title': str, 'lines': [str], 'table': {'headers': [...], 'rows': [[...]]} | None}."""
    return [
        _ml_section(details["ml"]),
        _fuzzy_section(details["fuzzy"]),
        _ahp_section(details["ahp"]),
        _deal_section(details["deal"]),
        _formula_section(details),
    ]
