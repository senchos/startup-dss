"""Модуль оцінки умов угоди G(D): порівнює суму раунду, оцінку компанії до інвестиції та
частку інвестора з типовими діапазонами для стадії фінансування (SPEC.md §5)."""
import numpy as np

# Типові діапазони (мінімум, медіана, максимум) у млн $. Значення funding_ask для
# Pre-seed/Seed/Series A взято з дипломної роботи (SPEC.md §1.1); решта — явні ринкові
# припущення на основі звітів Carta та PitchBook-NVCA (SPEC.md §5.1).
STAGE_RANGES = {
    "Pre-seed": {"funding_ask": (0.1, 0.5, 1), "pre_money_valuation": (2, 3, 5)},
    "Seed": {"funding_ask": (1, 3, 5), "pre_money_valuation": (5, 10, 20)},
    "Series A": {"funding_ask": (10, 20, 30), "pre_money_valuation": (40, 80, 120)},
    "Series B": {"funding_ask": (20, 35, 50), "pre_money_valuation": (100, 180, 280)},
    "Series C": {"funding_ask": (40, 60, 90), "pre_money_valuation": (250, 400, 600)},
    # Series D+ (назва стадії з startups_dataset.csv): припущення між Series C та IPO, SPEC.md §5.1
    "Series D+": {"funding_ask": (50, 90, 150), "pre_money_valuation": (500, 900, 1500)},
    "IPO": {"funding_ask": (75, 150, 300), "pre_money_valuation": (800, 1500, 3000)},
}

# Припущення (SPEC.md §5.1): однаковий типовий діапазон частки інвестора за раунд для всіх стадій.
DILUTION_RANGE = (0.10, 0.20, 0.30)


def compute_deal_metrics(funding_ask, pre_money_valuation):
    """Оцінка компанії після інвестиції (post-money) та частка інвестора (dilution)."""
    post_money = pre_money_valuation + funding_ask
    dilution_pct = funding_ask / post_money
    return post_money, dilution_pct


def _deviation_score(value, low, median, high):
    """Часткова оцінка відхилення від медіани: 100 у медіані, 50 на межі діапазону та 0 на
    подвоєній відстані від медіани."""
    # Півширина діапазону береться з того боку від медіани, де лежить значення
    half_range = (median - low) if value <= median else (high - median)
    if half_range <= 0:
        return 100.0 if value == median else 0.0
    score = 100.0 * (1 - abs(value - median) / (2 * half_range))
    return float(np.clip(score, 0, 100))


def evaluate_deal(funding_stage, funding_ask, pre_money_valuation):
    """Оцінка угоди G(D) ∈ [0, 100] як середнє трьох часткових оцінок (SPEC.md §5)."""
    if funding_stage not in STAGE_RANGES:
        raise ValueError(f"unknown funding_stage: {funding_stage}")

    # Діапазони стадії та похідні показники угоди
    ranges = STAGE_RANGES[funding_stage]
    post_money, dilution_pct = compute_deal_metrics(funding_ask, pre_money_valuation)

    # Часткові оцінки: сума раунду, оцінка до інвестиції та частка інвестора
    funding_score = _deviation_score(funding_ask, *ranges["funding_ask"])
    valuation_score = _deviation_score(pre_money_valuation, *ranges["pre_money_valuation"])
    dilution_score = _deviation_score(dilution_pct, *DILUTION_RANGE)

    # Підсумок — середнє арифметичне часткових оцінок
    g = (funding_score + valuation_score + dilution_score) / 3

    return {
        "stage": funding_stage,
        "funding_ask": funding_ask,
        "pre_money_valuation": pre_money_valuation,
        "ranges": {
            "funding_ask": list(ranges["funding_ask"]),
            "pre_money_valuation": list(ranges["pre_money_valuation"]),
            "dilution": list(DILUTION_RANGE),
        },
        "post_money": post_money,
        "dilution_pct": dilution_pct,
        "funding_score": funding_score,
        "valuation_score": valuation_score,
        "dilution_score": dilution_score,
        "G": g,
    }
