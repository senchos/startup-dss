"""Модуль AHP (метод аналізу ієрархій): обчислює ваги компонентів Score за готовим
пресетом або за матрицею попарних порівнянь і перевіряє її узгодженість (SPEC.md §6)."""
import numpy as np

# Випадковий індекс узгодженості Сааті для матриць розміру n
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58}
# Допустиме значення коефіцієнта узгодженості CR
CR_THRESHOLD = 0.10

# Готові профілі ваг (SPEC.md §6). Історія ваги ML: 0.35/0.45/0.55 → 0.10/0.12/0.15
# (01.10.2026, ML не мав сигналу на тодішньому датасеті) → 0.20/0.25/0.30 (05.10.2026:
# після нормалізації F(Y) на повний діапазон [0, 100], §4.3, Fuzzy не потребує такої
# домінантної частки).
PRESETS = {
    "conservative": {"ml": 0.20, "fuzzy": 0.65, "deal": 0.15},
    "growth": {"ml": 0.30, "fuzzy": 0.55, "deal": 0.15},
    "balanced": {"ml": 0.25, "fuzzy": 0.60, "deal": 0.15},
}


def geometric_mean_weights(matrix):
    """Вектор ваг AHP методом середнього геометричного (SPEC.md, формула 2.1)."""
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    # Добуток елементів кожного рядка, корінь n-го степеня та нормування на суму
    row_products = np.prod(matrix, axis=1)
    geometric_means = row_products ** (1.0 / n)
    return geometric_means / geometric_means.sum()


def consistency_ratio(matrix, weights):
    """Індекс узгодженості CI та коефіцієнт узгодженості CR (SPEC.md, формули 2.2–2.3)."""
    matrix = np.asarray(matrix, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = matrix.shape[0]
    # Оцінка головного власного числа матриці
    weighted_sum = matrix @ weights
    lambda_max = np.mean(weighted_sum / weights)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = RANDOM_INDEX.get(n, 0.0)
    # Для матриць розміру 1–2 випадковий індекс дорівнює нулю, тому CR вважається нульовим
    cr = ci / ri if ri > 0 else 0.0
    return ci, cr


def build_matrix_from_judgments(n, judgments):
    """Будує обернено-симетричну матрицю попарних порівнянь (шкала Сааті 1–9) за
    судженнями верхнього трикутника.

    `judgments` відображає пару (i, j), де i < j, у значення a_ij за шкалою Сааті
    (критерій i проти критерію j); елемент a_ji заповнюється як обернена величина.
    """
    # Діагональ дорівнює одиниці; решта елементів задається судженнями
    matrix = np.ones((n, n))
    for (i, j), value in judgments.items():
        matrix[i, j] = value
        matrix[j, i] = 1.0 / value
    return matrix


def evaluate_pairwise(matrix, labels):
    """Повна оцінка за методом AHP: ваги середнім геометричним, CI, CR та прапорець
    узгодженості consistent (SPEC.md §6)."""
    weights = geometric_mean_weights(matrix)
    ci, cr = consistency_ratio(matrix, weights)
    return {
        "weights": dict(zip(labels, weights)),
        "ci": ci,
        "cr": cr,
        "consistent": cr <= CR_THRESHOLD,
    }


def get_preset_weights(preset_name, has_deal):
    """Ваги пресету (SPEC.md §6). За відсутності умов угоди вони перенормовуються лише
    між ML та Fuzzy."""
    preset = PRESETS[preset_name]
    if has_deal:
        return dict(preset)
    total = preset["ml"] + preset["fuzzy"]
    return {"ml": preset["ml"] / total, "fuzzy": preset["fuzzy"] / total, "deal": 0.0}


def resolve_weights(weight_preset, has_deal, custom_matrix=None, custom_labels=None):
    """Точка входу відповідно до профілю інвестора (SPEC.md §2.4). Для пресету повертає
    його ваги (без угоди — перенормовані на дві компоненти), для "custom" обчислює ваги
    з матриці порівнянь. Повертає пару (ваги, результат AHP або None для пресету)."""
    # Власний профіль: ваги обчислюються з матриці попарних порівнянь
    if weight_preset == "custom":
        if custom_matrix is None or custom_labels is None:
            raise ValueError("custom preset requires custom_matrix and custom_labels")
        result = evaluate_pairwise(custom_matrix, custom_labels)
        return result["weights"], result
    # Готовий пресет: CI та CR не застосовуються
    if weight_preset not in PRESETS:
        raise ValueError(f"unknown weight_preset: {weight_preset}")
    return get_preset_weights(weight_preset, has_deal), None
