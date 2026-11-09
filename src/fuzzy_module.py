"""Нечіткий модуль (SPEC.md §4): каскад Мамдані з трьох систем, 27 правил якого зберігаються
у файлі data/fuzzy_rules.json, щоб базу правил можна було переглядати й редагувати
(екран «Адміністратор»), а не зашивати в код."""
import copy
import datetime
import itertools
import json
from pathlib import Path

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Універсум усіх лінгвістичних змінних: шкала 0–10 з кроком 0.1
UNIVERSE = np.arange(0, 10.01, 0.1)
# Шлях до файлу з правилами та межами нормалізації
RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "fuzzy_rules.json"

# Терми лінгвістичних змінних: низький, середній, високий
TERMS = ("low", "medium", "high")
# Функції належності термів: трапеція, трикутник, трапеція
TERM_MF = {
    "low": (fuzz.trapmf, [0, 0, 3, 5]),
    "medium": (fuzz.trimf, [3, 5, 7]),
    "high": (fuzz.trapmf, [5, 7, 10, 10]),
}
# Рівні каскаду: два проміжні та підсумковий
LEVEL_NAMES = ["execution_capability", "market_attractiveness", "favorability"]
# Сітка значень для пошуку меж нормалізації: 5⁴ = 625 комбінацій
BOUNDS_GRID = [0, 2.5, 5, 7.5, 10]

# Межі нормалізації перезавантажуються з файлу правил функцією reload(); їх слід читати як
# атрибути модуля (fuzzy_module.F_RAW_MIN), а не імпортувати через `from ... import`.
F_RAW_MIN = 0.0
F_RAW_MAX = 100.0
# Поточна конфігурація правил і побудовані системи керування
_STATE = {}

# Кеш масивів функцій належності, щоб не обчислювати їх повторно
_MF_CACHE = {}


def _term_mf(term):
    """Значення функції належності терма на універсумі (з кешем)."""
    if term not in _MF_CACHE:
        function, params = TERM_MF[term]
        _MF_CACHE[term] = function(UNIVERSE, params)
    return _MF_CACHE[term]


def validate_rules(cfg):
    """Перевіряє повноту й узгодженість бази правил; за порушення викликає ValueError."""
    # Рівні мають бути саме в порядку каскаду
    variables = cfg.get("variables", {})
    levels = cfg.get("levels", [])
    if [level.get("name") for level in levels] != LEVEL_NAMES:
        raise ValueError(f"levels must be exactly {LEVEL_NAMES} in this order")

    # Кожен рівень: відомі змінні, 2 входи та рівно 9 правил (по одному на комбінацію термів)
    seen_ids = set()
    for level in levels:
        for name in [*level["inputs"], level["output"]]:
            if name not in variables:
                raise ValueError(f"variable '{name}' has no entry in 'variables'")
        if len(level["inputs"]) != 2 or level["output"] != level["name"]:
            raise ValueError(f"level '{level['name']}' must have 2 inputs and output named after itself")

        combos = set()
        for rule in level["rules"]:
            if rule["id"] in seen_ids:
                raise ValueError(f"duplicate rule id {rule['id']}")
            seen_ids.add(rule["id"])
            if len(rule["if"]) != 2 or any(t not in TERMS for t in rule["if"]) or rule["then"] not in TERMS:
                raise ValueError(f"rule {rule['id']}: terms must be one of {TERMS}")
            combos.add(tuple(rule["if"]))
        expected = set(itertools.product(TERMS, repeat=2))
        if len(level["rules"]) != 9 or combos != expected:
            raise ValueError(f"level '{level['name']}' must have exactly one rule per term combination (9)")

    # Межі нормалізації мають бути впорядковані
    norm = cfg.get("normalization", {})
    if not norm.get("F_RAW_MIN", 0) < norm.get("F_RAW_MAX", 0):
        raise ValueError("normalization requires F_RAW_MIN < F_RAW_MAX")
    return cfg


def load_rules(path=None):
    """Читає файл правил і повертає перевірену конфігурацію."""
    with open(path or RULES_PATH, encoding="utf-8") as f:
        return validate_rules(json.load(f))


def _build_system(level):
    """Будує систему керування skfuzzy для одного рівня каскаду (змінні, терми, правила)."""
    # Вхідні змінні та вихідна змінна з дефазифікацією за центроїдом
    inputs = [ctrl.Antecedent(UNIVERSE, name) for name in level["inputs"]]
    output = ctrl.Consequent(UNIVERSE, level["output"], defuzzify_method="centroid")
    for variable in [*inputs, output]:
        for term in TERMS:
            variable[term] = _term_mf(term)

    # Правило «ЯКЩО перший вхід І другий вхід, ТО вихід»; кон'юнкція — мінімум
    rules = [
        ctrl.Rule(inputs[0][rule["if"][0]] & inputs[1][rule["if"][1]], output[rule["then"]])
        for rule in level["rules"]
    ]
    return ctrl.ControlSystem(rules)


def reload(path=None):
    """Читає JSON, перевіряє його та (пере)будує три системи skfuzzy і межі нормалізації.
    Викликається під час імпорту модуля та після збереження правил."""
    global F_RAW_MIN, F_RAW_MAX
    cfg = load_rules(path)
    _STATE["cfg"] = cfg
    _STATE["systems"] = [_build_system(level) for level in cfg["levels"]]
    F_RAW_MIN = cfg["normalization"]["F_RAW_MIN"]
    F_RAW_MAX = cfg["normalization"]["F_RAW_MAX"]
    return cfg


def get_rules():
    """Копія поточної конфігурації правил (змінювати оригінал не можна)."""
    return copy.deepcopy(_STATE["cfg"])


def _simulate(system, inputs, output_name):
    """Обчислює чітке значення виходу однієї системи для заданих входів."""
    sim = ctrl.ControlSystemSimulation(system)
    for name, value in inputs.items():
        sim.input[name] = value
    sim.compute()
    return sim.output[output_name]


def evaluate_raw(team_quality, traction, market_size, competition):
    """Каскад без підсумкової нормалізації (SPEC.md §4)."""
    exec_system, market_system, fav_system = _STATE["systems"]
    # Рівень 1: виконавча спроможність (команда, traction)
    execution_capability = _simulate(
        exec_system, {"team_quality": team_quality, "traction": traction}, "execution_capability"
    )
    # Рівень 1: ринкова привабливість (розмір ринку, конкуренція)
    market_attractiveness = _simulate(
        market_system, {"market_size": market_size, "competition": competition}, "market_attractiveness"
    )
    # Рівень 2: сприятливість за двома проміжними змінними
    favorability = _simulate(
        fav_system,
        {"execution_capability": execution_capability, "market_attractiveness": market_attractiveness},
        "favorability",
    )
    return {
        "execution_capability": execution_capability,
        "market_attractiveness": market_attractiveness,
        "favorability": favorability,
        "F_raw": favorability * 10,
    }


def normalize_favorability(f_raw):
    """Мін-макс нормалізація підсумкового виходу на [0, 100] з обрізанням (SPEC.md §4.3):
    розтягує реальний діапазон сирого результату (близько 20–80) на всю шкалу."""
    scaled = (f_raw - F_RAW_MIN) / (F_RAW_MAX - F_RAW_MIN) * 100
    return float(min(100.0, max(0.0, scaled)))


def rule_text(level, rule, variables=None):
    """Текст правила українською: пара (умова, висновок)."""
    variables = variables or _STATE["cfg"]["variables"]
    parts = [
        f"{variables[name]['label']} = {variables[name]['terms'][term]}"
        for name, term in zip(level["inputs"], rule["if"])
    ]
    out = variables[level["output"]]
    return " І ".join(parts), f"{out['label']} = {out['terms'][rule['then']]}"


def rule_firings(values):
    """Сила спрацювання кожного правила для чітких значень усіх шести змінних каскаду.
    Кон'юнкція Мамдані — мінімум, що відповідає оператору `&` у skfuzzy. Повертає словник
    для кожного правила (використовується в «Деталях розрахунку»)."""
    cfg = _STATE["cfg"]
    firings = []
    for level in cfg["levels"]:
        for rule in level["rules"]:
            # Сила правила — мінімум значень належності обох умов
            strength = min(
                float(fuzz.interp_membership(UNIVERSE, _term_mf(term), values[name]))
                for name, term in zip(level["inputs"], rule["if"])
            )
            condition, conclusion = rule_text(level, rule)
            firings.append(
                {
                    "id": rule["id"],
                    "level": cfg["variables"][level["name"]]["label"],
                    "if": condition,
                    "then": conclusion,
                    "strength": strength,
                }
            )
    return firings


def strongest_rules(firings, n=5):
    """Не більше n найсильніших правил, що спрацювали (сила > 0)."""
    fired = [f for f in firings if f["strength"] > 0]
    return sorted(fired, key=lambda f: (-f["strength"], f["id"]))[:n]


def evaluate(team_quality, traction, market_size, competition, explain=False):
    """Головний виклик модуля. Повертає execution_capability, market_attractiveness,
    favorability, F_raw та нормалізовану оцінку F(Y) ∈ [0, 100] (SPEC.md §4, §4.3);
    проміжні змінні лишаються сирими. З explain=True додатково повертає під ключем 'rules'
    силу спрацювання всіх 27 правил (для «Деталей розрахунку», SPEC.md §4.4)."""
    result = evaluate_raw(team_quality, traction, market_size, competition)
    result["F"] = normalize_favorability(result["F_raw"])
    if explain:
        result["rules"] = rule_firings(
            {
                "team_quality": team_quality,
                "traction": traction,
                "market_size": market_size,
                "competition": competition,
                "execution_capability": result["execution_capability"],
                "market_attractiveness": result["market_attractiveness"],
            }
        )
    return result


def compute_bounds(grid=BOUNDS_GRID):
    """Знаходить межі F_RAW_MIN та F_RAW_MAX перебором сітки grid⁴ (SPEC.md §4.3): для
    сітки за замовчуванням {0, 2.5, 5, 7.5, 10} це 625 комбінацій входів."""
    lo = hi = None
    for combo in itertools.product(grid, repeat=4):
        raw = evaluate_raw(*combo)["favorability"] * 10
        if lo is None or raw < lo[0]:
            lo = (raw, combo)
        if hi is None or raw > hi[0]:
            hi = (raw, combo)
    return {
        "F_RAW_MIN": round(lo[0], 4),
        "F_RAW_MAX": round(hi[0], 4),
        "argmin": list(lo[1]),
        "argmax": list(hi[1]),
        "grid": list(grid),
        "combinations": len(grid) ** 4,
    }


def rule_rows(cfg=None):
    """Плоска таблиця всіх правил, готова до показу (екран «Адміністратор»)."""
    cfg = cfg or _STATE["cfg"]
    rows = []
    for level in cfg["levels"]:
        for rule in level["rules"]:
            condition, conclusion = rule_text(level, rule, cfg["variables"])
            rows.append(
                {
                    "id": rule["id"],
                    "level": cfg["variables"][level["name"]]["label"],
                    "if": condition,
                    "then": conclusion,
                    "then_term": rule["then"],
                }
            )
    return rows


def with_consequents(cfg, consequents):
    """Копія конфігурації із заміненими висновками правил; consequents відображає номер правила в терм."""
    new_cfg = copy.deepcopy(cfg)
    for level in new_cfg["levels"]:
        for rule in level["rules"]:
            if rule["id"] in consequents:
                rule["then"] = consequents[rule["id"]]
    return validate_rules(new_cfg)


def save_rules(cfg, path=None):
    """Перевіряє конфігурацію, зберігає попередній файл як <ім'я>.prev.json, записує нову,
    перебудовує системи та перераховує межі нормалізації (вони залежать від правил, §4.3).
    Повертає нові межі. Якщо щось не вдалося, зокрема коли вихід бази правил ніколи не
    змінюється (немає придатного діапазону F), попередній файл відновлюється, а помилка
    передається далі."""
    path = Path(path or RULES_PATH)
    validate_rules(cfg)
    previous = path.read_text(encoding="utf-8") if path.exists() else None
    if previous is not None:
        path.with_suffix(".prev.json").write_text(previous, encoding="utf-8")

    def write(config):
        # Запис конфігурації у файл правил
        path.write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")

    try:
        # Спочатку запис правил і перебудова систем, потім розрахунок меж за новими правилами
        write(cfg)
        reload(path)
        bounds = compute_bounds()
        if not bounds["F_RAW_MIN"] < bounds["F_RAW_MAX"]:
            raise ValueError(
                "Ці правила дають сталий вихід каскаду (F не залежить від входів): нормалізація неможлива."
            )
        # Остаточний файл містить нові межі нормалізації з датою розрахунку
        final = copy.deepcopy(cfg)
        final["normalization"] = {
            "F_RAW_MIN": bounds["F_RAW_MIN"],
            "F_RAW_MAX": bounds["F_RAW_MAX"],
            "grid": bounds["grid"],
            "combinations": bounds["combinations"],
            "computed_on": datetime.date.today().isoformat(),
        }
        write(final)
        reload(path)
        return bounds
    except Exception:
        # Відкат до попередньої версії файлу
        if previous is not None:
            path.write_text(previous, encoding="utf-8")
            reload(path)
        raise


# Початкове завантаження правил під час імпорту модуля
reload()
