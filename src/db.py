"""Шар даних на SQLite: схема з семи таблиць (SPEC.md §7), міграції та функції
збереження, видалення й читання історії аналізів. Модуль не залежить від інших
модулів проєкту."""
import datetime
import json
import sqlite3
from pathlib import Path

# Схема бази даних; усі таблиці створюються лише за відсутності (IF NOT EXISTS)
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY, name TEXT, email TEXT
);

CREATE TABLE IF NOT EXISTS investor_profiles (
  id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id),
  weight_preset TEXT, custom_weights TEXT, -- власні ваги у форматі JSON
  search_industries TEXT, search_stages TEXT,
  ticket_size_min REAL, ticket_size_max REAL
);

CREATE TABLE IF NOT EXISTS startups (
  id INTEGER PRIMARY KEY, name TEXT, founded_year INTEGER, country TEXT,
  industry TEXT, funding_stage TEXT, total_funding REAL, employees INTEGER,
  revenue REAL, valuation REAL, customer_base REAL, social_media_followers INTEGER,
  business_model TEXT, revenue_growth REAL
  -- 06.10.2026 (новий датасет): valuation у млн $, customer_base у тисячах;
  -- у полі `country` зберігається регіон; business_model та revenue_growth — нові поля
);

CREATE TABLE IF NOT EXISTS qualitative_assessments (
  id INTEGER PRIMARY KEY, analysis_id INTEGER REFERENCES analyses(id),
  team_quality REAL, traction REAL, market_size REAL, competition REAL
);

CREATE TABLE IF NOT EXISTS deal_terms (
  id INTEGER PRIMARY KEY, analysis_id INTEGER REFERENCES analyses(id),
  funding_ask REAL, pre_money_valuation REAL
);

CREATE TABLE IF NOT EXISTS analyses (
  id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id),
  startup_id INTEGER REFERENCES startups(id),
  date TEXT, ml_score REAL, fuzzy_score REAL, deal_score REAL,
  final_score REAL, category TEXT, state TEXT,
  details_json TEXT -- деталі розрахунку (SPEC.md §4.4), додано 05.10.2026
);

CREATE TABLE IF NOT EXISTS reports (
  id INTEGER PRIMARY KEY, analysis_id INTEGER REFERENCES analyses(id),
  generated_date TEXT, file_path TEXT
);
"""


def get_connection(db_path):
    """Відкриває з'єднання з базою: з увімкненими зовнішніми ключами та доступом до полів за іменем."""
    # check_same_thread=False: одне довготривале з'єднання (кешований ресурс Streamlit)
    # використовується з різних потоків перезапуску сценарію; виклики не повинні
    # виконуватися на ньому одночасно.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def _add_missing_columns(conn, table, columns):
    """Додає до таблиці стовпці зі словника {ім'я: тип}, яких у ній ще немає."""
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, sql_type in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")


def _migrate(conn):
    """Додає стовпці, що з'явилися вже після створення бази: analyses.details_json
    (05.10.2026) та startups.business_model, revenue_growth (06.10.2026). Зауваження:
    одиниці startups.valuation і customer_base змінилися 06.10.2026 (млрд $ → млн $,
    мільйони → тисячі); старі рядки не перераховуються, демонстраційну базу слід
    створити заново."""
    _add_missing_columns(conn, "analyses", {"details_json": "TEXT"})
    _add_missing_columns(conn, "startups", {"business_model": "TEXT", "revenue_growth": "REAL"})


def init_db(db_path):
    """Створює (за потреби) базу зі схемою, виконує міграції та повертає з'єднання."""
    # Каталог бази може бути відсутнім (git не зберігає порожні каталоги)
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def ensure_default_user(conn, user_id=1):
    """Гарантує наявність користувача за замовчуванням (автентифікації в системі немає)."""
    conn.execute(
        "INSERT OR IGNORE INTO users (id, name, email) VALUES (?, ?, ?)",
        (user_id, "Investor", ""),
    )
    conn.commit()


def _none_if_nan(value):
    """Замінює NaN на None, щоб у базі зберігався NULL (NaN != NaN)."""
    return None if value is None or value != value else value


def save_analysis(conn, company_raw, qualitative, deal_terms, result, user_id=1, date=None):
    """Зберігає один завершений аналіз: компанію, сам аналіз, експертні оцінки та умови
    угоди. Повертає ідентифікатор нового аналізу."""
    # Компанія; у поле country записується регіон, який бачила модель
    cur = conn.execute(
        """INSERT INTO startups
           (name, founded_year, country, industry, funding_stage, total_funding, employees,
            revenue, valuation, customer_base, social_media_followers,
            business_model, revenue_growth)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            company_raw["Startup Name"], company_raw["Founded Year"], company_raw.get("Region") or company_raw["Country"],
            company_raw["Industry"], company_raw["Funding Stage"], company_raw["Total Funding ($M)"],
            company_raw["Number of Employees"], company_raw["Annual Revenue ($M)"],
            company_raw["Valuation ($M)"], company_raw["Customer Base (Thousands)"],
            company_raw["Social Media Followers"],
            company_raw["Business Model"], _none_if_nan(company_raw.get("Revenue Growth (%)")),
        ),
    )
    startup_id = cur.lastrowid

    # Аналіз: оцінки компонентів, підсумковий Score, категорія та деталі розрахунку у JSON
    cur = conn.execute(
        """INSERT INTO analyses
           (user_id, startup_id, date, ml_score, fuzzy_score, deal_score, final_score,
            category, state, details_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id, startup_id, date or datetime.date.today().isoformat(),
            result["ml_score"], result["fuzzy"]["F"],
            result["deal"]["G"] if result["deal"] else None,
            result["score"], result["category"], "completed",
            json.dumps(result.get("details"), ensure_ascii=False) if result.get("details") else None,
        ),
    )
    analysis_id = cur.lastrowid

    # Чотири експертні оцінки
    conn.execute(
        """INSERT INTO qualitative_assessments
           (analysis_id, team_quality, traction, market_size, competition)
           VALUES (?, ?, ?, ?, ?)""",
        (analysis_id, qualitative["team_quality"], qualitative["traction"],
         qualitative["market_size"], qualitative["competition"]),
    )

    # Умови угоди записуються лише якщо вони були вказані
    if deal_terms is not None:
        conn.execute(
            "INSERT INTO deal_terms (analysis_id, funding_ask, pre_money_valuation) VALUES (?, ?, ?)",
            (analysis_id, deal_terms["funding_ask"], deal_terms["pre_money_valuation"]),
        )

    conn.commit()
    return analysis_id


def delete_analysis(conn, analysis_id):
    """Видаляє один аналіз разом з пов'язаними записами (експертні оцінки, умови угоди,
    відомості про звіти) та рядком компанії, якщо на неї більше не посилається жоден
    аналіз. Згенеровані PDF-файли на диску не видаляються.
    Повертає True, якщо такий аналіз існував."""
    row = conn.execute("SELECT startup_id FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    if row is None:
        return False
    # Спочатку залежні таблиці, потім аналіз, щоб не порушити зовнішні ключі
    for table in ("qualitative_assessments", "deal_terms", "reports"):
        conn.execute(f"DELETE FROM {table} WHERE analysis_id = ?", (analysis_id,))
    conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    startup_id = row["startup_id"]
    # Компанію видаляємо лише якщо в неї не лишилося інших аналізів
    if not conn.execute("SELECT 1 FROM analyses WHERE startup_id = ?", (startup_id,)).fetchone():
        conn.execute("DELETE FROM startups WHERE id = ?", (startup_id,))
    conn.commit()
    return True


def fetch_history(conn):
    """Усі аналізи від найновішого до найстаршого разом з даними компанії, експертними
    оцінками та умовами угоди. Поле 'details' — розібраний словник деталей розрахунку
    або None для записів, збережених до появи цього поля."""
    rows = conn.execute(
        """SELECT a.id, s.name, s.founded_year, s.country, s.industry, s.funding_stage,
                  s.total_funding, s.employees, s.revenue, s.valuation,
                  s.customer_base, s.social_media_followers, s.business_model, s.revenue_growth,
                  a.date, a.ml_score, a.fuzzy_score, a.deal_score, a.final_score, a.category,
                  a.details_json,
                  d.funding_ask, d.pre_money_valuation,
                  q.team_quality, q.traction, q.market_size, q.competition
           FROM analyses a
           JOIN startups s ON s.id = a.startup_id
           LEFT JOIN deal_terms d ON d.analysis_id = a.id
           LEFT JOIN qualitative_assessments q ON q.analysis_id = a.id
           ORDER BY a.id DESC"""
    ).fetchall()
    # JSON з деталями розбирається у словник
    history = []
    for row in rows:
        item = dict(row)
        item["details"] = json.loads(item.pop("details_json")) if item["details_json"] else None
        history.append(item)
    return history
