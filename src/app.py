"""Інтерфейс Streamlit: сторінки нового аналізу, результату, історії, порівняння, профілю
інвестора, PDF-звіту та режиму адміністратора. Містить лише зв'язування введення з
логікою й відображення результатів; розрахунки виконують інші модулі."""
import datetime
import json
import shutil
import sys
from pathlib import Path

# Додаємо корінь проєкту до шляху імпорту, щоб працював запуск `streamlit run src/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src import ahp_module, db as db_module, details as details_module, fuzzy_module, ml_module
from src import report as report_module, theme
from src.aggregator import RECOMMENDATIONS, analyze_startup
from src.db import fetch_history, save_analysis
from src.ml_module import load_model
from src.preprocessing import FUNDING_STAGE_ORDER, REGION_MAP

# Шляхи до бази даних і файлу навчальної моделі; користувач за замовчуванням
DB_PATH = "data/processed/startup_dss.db"
MODEL_PATH = "models/model.pkl"
DEFAULT_USER_ID = 1

# ML-конвеєр бачить лише Region, а не Country (SPEC.md §2.1.1), тому форма запитує регіон
# напряму. Спочатку йдуть шість регіонів із startups_dataset.csv, потім інші значення з
# REGION_MAP, наприкінці "Other" (невідоме навченому кодувальнику значення: нейтральний,
# нульовий регіональний внесок).
_REGION_ORDER = ["North America", "Europe", "Asia", "South America", "Oceania", "Middle East & Africa"]
REGIONS = _REGION_ORDER + sorted(set(REGION_MAP.values()) - set(_REGION_ORDER)) + ["Other"]
# Допустимі значення полів форми
INDUSTRIES = ["SaaS", "AI", "FinTech", "HealthTech", "E-commerce", "EdTech", "Gaming", "CleanTech", "Logistics", "Cybersecurity"]
BUSINESS_MODELS = ["B2B", "B2C"]
FUNDING_STAGES = FUNDING_STAGE_ORDER
# Назви профілів ваг для вибору в інтерфейсі
WEIGHT_PRESET_LABELS = {
    "conservative": "Консервативний",
    "growth": "Зростання",
    "balanced": "Збалансований",
    "custom": "Власний (попарне порівняння)",
}
# Шкала Сааті для попарних порівнянь (включно з оберненими значеннями)
SAATY_SCALE = [1 / 9, 1 / 7, 1 / 5, 1 / 3, 1, 3, 5, 7, 9]
# Каталог для згенерованих PDF-звітів
REPORTS_DIR = Path("data/processed/reports")


@st.cache_resource
def get_db_connection():
    """З'єднання з базою даних; кешується між перезапусками сценарію Streamlit."""
    conn = db_module.init_db(DB_PATH)
    db_module.ensure_default_user(conn, DEFAULT_USER_ID)
    return conn


@st.cache_resource
def get_ml_pipeline():
    """Завантажує навчений конвеєр (кешується). Повертає None, якщо models/model.pkl ще немає
    (свіжий клон репозиторію): застосунок усе одно запускається, а модель можна навчити на
    екрані «Адміністратор»."""
    return load_model(MODEL_PATH) if Path(MODEL_PATH).exists() else None


def get_investor_profile(conn):
    """Профіль інвестора з бази; за його відсутності — профіль за замовчуванням («Збалансований»)."""
    row = conn.execute(
        "SELECT * FROM investor_profiles WHERE user_id = ?", (DEFAULT_USER_ID,)
    ).fetchone()
    if row is None:
        return {
            "weight_preset": "balanced",
            "custom_weights": None,
            "search_industries": [],
            "search_stages": [],
            "ticket_size_min": None,
            "ticket_size_max": None,
        }
    return {
        "weight_preset": row["weight_preset"],
        "custom_weights": json.loads(row["custom_weights"]) if row["custom_weights"] else None,
        "search_industries": json.loads(row["search_industries"]) if row["search_industries"] else [],
        "search_stages": json.loads(row["search_stages"]) if row["search_stages"] else [],
        "ticket_size_min": row["ticket_size_min"],
        "ticket_size_max": row["ticket_size_max"],
    }


def save_investor_profile(conn, profile):
    """Зберігає профіль інвестора: оновлює наявний запис або створює новий."""
    existing = conn.execute(
        "SELECT id FROM investor_profiles WHERE user_id = ?", (DEFAULT_USER_ID,)
    ).fetchone()
    payload = (
        profile["weight_preset"],
        json.dumps(profile["custom_weights"]) if profile["custom_weights"] else None,
        json.dumps(profile["search_industries"]),
        json.dumps(profile["search_stages"]),
        profile["ticket_size_min"],
        profile["ticket_size_max"],
    )
    if existing:
        conn.execute(
            """UPDATE investor_profiles SET weight_preset=?, custom_weights=?,
               search_industries=?, search_stages=?, ticket_size_min=?, ticket_size_max=?
               WHERE user_id=?""",
            payload + (DEFAULT_USER_ID,),
        )
    else:
        conn.execute(
            """INSERT INTO investor_profiles
               (user_id, weight_preset, custom_weights, search_industries, search_stages,
                ticket_size_min, ticket_size_max)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (DEFAULT_USER_ID,) + payload,
        )
    conn.commit()


def _company_raw_from_row(row):
    """Відновлює словник даних компанії з рядка історії (для відображення картки)."""
    return {
        "Startup Name": row["name"], "Founded Year": row["founded_year"], "Region": row["country"],
        "Industry": row["industry"], "Funding Stage": row["funding_stage"],
        "Total Funding ($M)": row["total_funding"], "Number of Employees": row["employees"],
        "Annual Revenue ($M)": row["revenue"], "Valuation ($M)": row["valuation"],
        "Business Model": row["business_model"], "Revenue Growth (%)": row["revenue_growth"],
    }


def _qualitative_from_row(row):
    """Експертні оцінки з рядка історії."""
    return {
        "team_quality": row["team_quality"], "traction": row["traction"],
        "market_size": row["market_size"], "competition": row["competition"],
    }


def _deal_terms_from_row(row):
    """Умови угоди з рядка історії; None, якщо їх не було."""
    if row["funding_ask"] is None:
        return None
    return {"funding_ask": row["funding_ask"], "pre_money_valuation": row["pre_money_valuation"]}


def matches_search_profile(row, profile):
    """Чи відповідає аналіз пошуковому профілю інвестора: індустрії, стадії та розмір чека."""
    if profile["search_industries"] and row["industry"] not in profile["search_industries"]:
        return False
    if profile["search_stages"] and row["funding_stage"] not in profile["search_stages"]:
        return False
    ticket = row["funding_ask"]
    if profile["ticket_size_min"] is not None and ticket is not None and ticket < profile["ticket_size_min"]:
        return False
    if profile["ticket_size_max"] is not None and ticket is not None and ticket > profile["ticket_size_max"]:
        return False
    return True


def _growth_text(value):
    """Текст темпу зростання зі знаком; «—» для невідомого значення."""
    return "—" if value is None or value != value else f"{value:+.0f}%"


def render_snapshot_card(company_raw, qualitative, deal_terms=None):
    """Картка компанії: теги, ключові показники, умови угоди та радар експертних оцінок."""
    region = company_raw.get("Region") or company_raw.get("Country", "—")
    tags_html = "".join(
        theme.tag(t)
        for t in [
            company_raw["Industry"], company_raw["Funding Stage"], company_raw.get("Business Model") or "—",
            f"Регіон: {region}", f"Засн. {company_raw['Founded Year']}",
        ]
    )
    stats_html = "".join(
        theme.stat(label, value)
        for label, value in [
            ("Funding", f"${company_raw['Total Funding ($M)']:.1f}M"),
            ("Employees", f"{company_raw['Number of Employees']}"),
            ("Revenue", f"${company_raw['Annual Revenue ($M)']:.1f}M"),
            ("Growth", _growth_text(company_raw.get("Revenue Growth (%)"))),
            ("Valuation", f"${company_raw['Valuation ($M)']:.0f}M"),
        ]
    )
    deal_html = ""
    if deal_terms:
        post_money = deal_terms["pre_money_valuation"] + deal_terms["funding_ask"]
        dilution = deal_terms["funding_ask"] / post_money * 100 if post_money else 0.0
        deal_html = '<div class="deal-strip">{}{}{}</div>'.format(
            theme.stat("Funding ask", f"${deal_terms['funding_ask']:.1f}M"),
            theme.stat("Pre-money", f"${deal_terms['pre_money_valuation']:.1f}M"),
            theme.stat("Dilution", f"{dilution:.1f}%"),
        )
    radar_html = theme.radar_svg(
        qualitative["team_quality"], qualitative["traction"],
        qualitative["market_size"], qualitative["competition"],
    )

    # HTML складається конкатенацією, а не багаторядковим f-рядком: підстановка на окремому
    # рядку може дати порожній рядок (наприклад, deal_html == "" без умов угоди), а парсер
    # markdown у Streamlit на першому порожньому рядку перестає вважати блок HTML і виводить
    # решту як звичайний текст.
    html = (
        '<div class="card">'
        '<p class="eyebrow">Компанія</p>'
        f'<div class="company-name">{company_raw["Startup Name"]}</div>'
        f'<div class="tags">{tags_html}</div>'
        f'<div class="stat-grid">{stats_html}</div>'
        f"{deal_html}"
        '<p class="eyebrow">Експертні оцінки, 0–10</p>'
        f"{radar_html}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_score_card(result, startup_name=""):
    """Картка результату: Score, категорія, рекомендація, внески компонентів і формула."""
    weights = result.get("weights") or {}
    bars_html = theme.component_bar("ML-прогноз M(X)", "--ml", result["ml_score"], weights.get("ml"))
    bars_html += theme.component_bar("Fuzzy-оцінка F(Y)", "--fuzzy", result["fuzzy"]["F"], weights.get("fuzzy"))

    deal_score = result["deal"]["G"] if result["deal"] else None
    if deal_score is not None:
        bars_html += theme.component_bar("Умови угоди G(D)", "--deal", deal_score, weights.get("deal"))

    # Формула з підставленими числами показується, лише якщо відомі всі потрібні ваги
    formula_html = ""
    required = ["ml", "fuzzy"] + (["deal"] if deal_score is not None else [])
    if all(weights.get(k) is not None for k in required):
        terms = [f"<b>{weights['ml']:.2f}</b>×{result['ml_score']:.1f}", f"<b>{weights['fuzzy']:.2f}</b>×{result['fuzzy']['F']:.1f}"]
        if deal_score is not None:
            terms.append(f"<b>{weights['deal']:.2f}</b>×{deal_score:.1f}")
        formula_html = f'<div class="formula mono tab">Score = {" + ".join(terms)} = <b>{result["score"]:.1f}</b></div>'

    html = (
        '<div class="card">'
        '<p class="eyebrow">Результат аналізу</p>'
        '<div class="score-row">'
        f'<div><span class="score-value mono tab">{result["score"]:.1f}</span><span class="score-max mono"> / 100</span></div>'
        f"{theme.status_pill(result['category'])}"
        "</div>"
        f'<p class="recommendation">{RECOMMENDATIONS[result["category"]]}</p>'
        '<p class="eyebrow">Розбивка за компонентами</p>'
        f"{bars_html}"
        f"{formula_html}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_details(details):
    """Покроковий розрахунок одного аналізу (SPEC.md §4.4): ML, Fuzzy, AHP, угода, формула."""
    if not details:
        st.info("Деталі розрахунку недоступні: цей аналіз збережено до їх появи.")
        return
    for section in details_module.build_sections(details):
        st.markdown(f"**{section['title']}**")
        for line in section["lines"]:
            st.markdown(f"- {line}")
        table = section["table"]
        if table:
            st.dataframe(
                [dict(zip(table["headers"], row)) for row in table["rows"]],
                hide_index=True, width="stretch",
            )


def render_result_card(result, startup_name="", company_raw=None, qualitative=None, deal_terms=None):
    """Результат аналізу: картка компанії (якщо відомі її дані), картка Score та деталі розрахунку."""
    if company_raw and qualitative:
        col1, col2 = st.columns([1, 1.3])
        with col1:
            render_snapshot_card(company_raw, qualitative, deal_terms)
        with col2:
            render_score_card(result, startup_name)
    else:
        render_score_card(result, startup_name)

    with st.expander("Деталі розрахунку"):
        render_details(result.get("details"))


def page_new_analysis(conn, pipeline):
    """Сторінка «Новий аналіз»: форма введення, запуск розрахунку та збереження результату."""
    st.header("Новий аналіз")
    # Без навченої моделі аналіз неможливий
    if pipeline is None:
        st.error(
            f"ML-модель `{MODEL_PATH}` ще не створена. Увімкніть «Режим адміністратора» в бічній панелі "
            "й натисніть «Перенавчити» (потрібен датасет у `data/raw/`, див. README)."
        )
        return
    profile = get_investor_profile(conn)

    # Прапорець поза формою, щоб він одразу розблоковував поля угоди
    include_deal = st.checkbox("Додати умови угоди (funding ask, pre-money valuation)")

    with st.form("new_analysis_form"):
        st.subheader("Кількісні дані компанії")
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Назва стартапу", value="Startup_X")
            founded_year = st.number_input("Рік заснування", min_value=1990, max_value=2026, value=2020)
            region = st.selectbox("Регіон", REGIONS)
            industry = st.selectbox("Індустрія", INDUSTRIES)
            business_model = st.selectbox("Бізнес-модель", BUSINESS_MODELS)
        with c2:
            funding_stage = st.selectbox("Funding Stage", FUNDING_STAGES, index=1)
            total_funding = st.number_input("Total Funding ($M)", min_value=0.0, value=5.0)
            employees = st.number_input("Кількість співробітників", min_value=1, value=50)
            revenue = st.number_input("Annual Revenue ($M)", min_value=0.0, value=1.0)
            growth = st.number_input("Revenue Growth (%)", value=100.0, step=10.0)
            growth_unknown = st.checkbox("Темп зростання невідомий (немає виручки)")
        with c3:
            valuation = st.number_input("Valuation ($M)", min_value=0.0, value=20.0)
            customer_base = st.number_input("Customer Base (тис.)", min_value=0.0, value=5.0)
            social_media = st.number_input("Social Media Followers", min_value=0, value=10000)

        st.subheader("Експертні оцінки (0–10)")
        c4, c5, c6, c7 = st.columns(4)
        with c4:
            team_quality = st.slider("Команда", 0.0, 10.0, 5.0)
        with c5:
            traction = st.slider("Traction", 0.0, 10.0, 5.0)
        with c6:
            market_size = st.slider("Розмір ринку", 0.0, 10.0, 5.0)
        with c7:
            competition = st.slider("Конкуренція", 0.0, 10.0, 5.0)

        st.subheader("Умови угоди (опційно)")
        c8, c9 = st.columns(2)
        with c8:
            funding_ask = st.number_input("Funding Ask ($M)", min_value=0.0, value=3.0, disabled=not include_deal)
        with c9:
            pre_money = st.number_input("Pre-money Valuation ($M)", min_value=0.0, value=10.0, disabled=not include_deal)

        submitted = st.form_submit_button("Провести аналіз")

    if submitted:
        # Збираємо введені дані у структури, яких очікує analyze_startup
        company_raw = {
            "Startup Name": name, "Founded Year": int(founded_year),
            "Country": region, "Region": region,
            "Industry": industry, "Funding Stage": funding_stage,
            "Total Funding ($M)": total_funding, "Number of Employees": int(employees),
            "Annual Revenue ($M)": revenue, "Valuation ($M)": valuation,
            "Customer Base (Thousands)": customer_base, "Social Media Followers": int(social_media),
            "Business Model": business_model,
            "Revenue Growth (%)": None if growth_unknown else growth,
        }
        qualitative = {
            "team_quality": team_quality, "traction": traction,
            "market_size": market_size, "competition": competition,
        }
        deal_terms = {"funding_ask": funding_ask, "pre_money_valuation": pre_money} if include_deal else None

        # Профіль ваг береться з профілю інвестора
        weight_preset = profile["weight_preset"]
        custom_matrix, custom_labels = None, None
        if weight_preset == "custom":
            if profile["custom_weights"]:
                custom_matrix = profile["custom_weights"]["matrix"]
                custom_labels = profile["custom_weights"]["labels"]
            else:
                st.warning("Профіль ваг «Власний» ще не налаштовано — використано «Збалансований».")
                weight_preset = "balanced"

        # Розрахунок, збереження в базу й запам'ятовування останнього аналізу в сесії
        result = analyze_startup(
            company_raw, qualitative, pipeline, deal_terms=deal_terms,
            weight_preset=weight_preset, custom_matrix=custom_matrix, custom_labels=custom_labels,
        )

        analysis_id = save_analysis(conn, company_raw, qualitative, deal_terms, result)
        st.session_state["last_result"] = result
        st.session_state["last_startup_name"] = name
        st.session_state["last_analysis_id"] = analysis_id

        st.success(f"Аналіз збережено (ID {analysis_id}). Результат — на вкладці «Результат аналізу».")
        render_result_card(result, name, company_raw=company_raw, qualitative=qualitative, deal_terms=deal_terms)


def page_result(conn, pipeline):
    """Сторінка «Результат аналізу»: вибір збереженого аналізу та його картка з деталями."""
    st.header("Результат аналізу")

    history = fetch_history(conn)
    if not history:
        st.info("Ще немає жодного аналізу. Почніть із вкладки «Новий аналіз».")
        return

    # Типово показується останній виконаний аналіз
    options = {f"#{r['id']} — {r['name']} ({r['date']})": r["id"] for r in history}
    default_label = None
    if "last_analysis_id" in st.session_state:
        for label, aid in options.items():
            if aid == st.session_state["last_analysis_id"]:
                default_label = label
    chosen_label = st.selectbox(
        "Аналіз", list(options.keys()),
        index=list(options.keys()).index(default_label) if default_label else 0,
    )
    row = next(r for r in history if r["id"] == options[chosen_label])

    # Результат відновлюється із збережених значень, без повторного розрахунку
    details = row["details"]
    result = {
        "score": row["final_score"], "category": row["category"], "ml_score": row["ml_score"],
        "fuzzy": {"F": row["fuzzy_score"]},
        "deal": {"G": row["deal_score"]} if row["deal_score"] is not None else None,
        "weights": details["ahp"]["weights"] if details else None,
        "details": details,
    }

    render_result_card(
        result, row["name"],
        company_raw=_company_raw_from_row(row),
        qualitative=_qualitative_from_row(row),
        deal_terms=_deal_terms_from_row(row),
    )


def page_history(conn, pipeline):
    """Сторінка «Історія аналізів»: таблиця збережених аналізів та видалення вибраних."""
    st.header("Історія аналізів")
    profile = get_investor_profile(conn)
    history = fetch_history(conn)
    if not history:
        st.info("Історія порожня.")
        return

    # Рядки таблиці; позначка показує відповідність пошуковому профілю
    rows_html = ""
    for row in history:
        match = "✅" if matches_search_profile(row, profile) else "—"
        rows_html += f"""<tr>
          <td>{row['id']}</td>
          <td class="name">{row['name']}</td>
          <td>{row['industry']}</td>
          <td>{row['funding_stage']}</td>
          <td>{row['date']}</td>
          <td class="score mono tab">{row['final_score']:.1f}</td>
          <td>{theme.chip(row['category'])}</td>
          <td>{match}</td>
        </tr>"""

    st.markdown(
        f"""<div class="table-wrap">
          <table>
            <thead><tr>
              <th>ID</th><th>Стартап</th><th>Індустрія</th><th>Стадія</th>
              <th>Дата</th><th>Score</th><th>Категорія</th><th>Профіль</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>""",
        unsafe_allow_html=True,
    )

    # Видалення вимагає підтвердження, бо воно незворотне
    with st.expander("Видалити аналізи з історії"):
        options = {f"#{r['id']} — {r['name']} ({r['date']})": r["id"] for r in history}
        to_delete = st.multiselect("Аналізи для видалення", list(options.keys()))
        confirmed = st.checkbox("Підтверджую: видалення незворотне", key="confirm_delete")
        if st.button("Видалити вибране", disabled=not (to_delete and confirmed)):
            ids = [options[label] for label in to_delete]
            for analysis_id in ids:
                db_module.delete_analysis(conn, analysis_id)
            if st.session_state.get("last_analysis_id") in ids:
                st.session_state.pop("last_analysis_id")
            st.session_state.pop("confirm_delete", None)
            st.success(f"Видалено аналізів: {len(ids)}.")
            st.rerun()


def page_comparison(conn, pipeline):
    """Сторінка «Порівняння»: до трьох збережених аналізів поруч."""
    st.header("Порівняння")
    history = fetch_history(conn)
    if len(history) < 2:
        st.info("Для порівняння потрібно щонайменше 2 збережені аналізи.")
        return

    options = {f"#{r['id']} — {r['name']}": r for r in history}
    chosen = st.multiselect("Оберіть 2–3 аналізи", list(options.keys()), max_selections=3)

    if len(chosen) < 2:
        st.info("Оберіть щонайменше 2 аналізи.")
        return

    # Одна колонка-картка на кожен вибраний аналіз
    cols = st.columns(len(chosen))
    for col, label in zip(cols, chosen):
        row = options[label]
        deal_line = f'<div class="stat-value" style="margin-top:4px">Deal {row["deal_score"]:.1f}</div>' if row["deal_score"] is not None else ""
        with col:
            ml_html = theme.stat("ML", f"{row['ml_score']:.1f}")
            fuzzy_html = theme.stat("Fuzzy", f"{row['fuzzy_score']:.1f}")
            html = (
                '<div class="card">'
                f'<p class="eyebrow">{row["industry"]} · {row["funding_stage"]}</p>'
                f'<div class="company-name" style="font-size:17px">{row["name"]}</div>'
                f'<div class="score-value mono tab" style="font-size:34px; margin:10px 0 6px">{row["final_score"]:.1f}</div>'
                f"{theme.status_pill(row['category'])}"
                '<div class="stat-grid" style="margin-top:16px">'
                f"{ml_html}{fuzzy_html}"
                "</div>"
                f"{deal_line}"
                "</div>"
            )
            st.markdown(html, unsafe_allow_html=True)


def page_investor_profile(conn, pipeline):
    """Сторінка «Профіль інвестора»: профіль ваг (пресет або попарні порівняння AHP) та пошуковий профіль."""
    st.header("Профіль інвестора")
    profile = get_investor_profile(conn)

    preset = st.selectbox(
        "Пресет ваг", list(WEIGHT_PRESET_LABELS.keys()),
        format_func=lambda k: WEIGHT_PRESET_LABELS[k],
        index=list(WEIGHT_PRESET_LABELS.keys()).index(profile["weight_preset"]),
    )

    # Власні ваги: судження користувача перетворюються на матрицю й перевіряються на узгодженість
    custom_weights_payload = profile["custom_weights"]
    if preset == "custom":
        st.subheader("Ручне попарне порівняння (шкала Сааті)")
        include_deal = st.checkbox("Включити «Умови угоди» як третій критерій", value=True)
        labels = ["ml", "fuzzy", "deal"] if include_deal else ["ml", "fuzzy"]
        display_labels = {"ml": "ML", "fuzzy": "Fuzzy", "deal": "Deal"}
        n = len(labels)

        judgments = {}
        if n == 2:
            judgments[(0, 1)] = st.select_slider("ML проти Fuzzy", SAATY_SCALE, value=1)
        else:
            judgments[(0, 1)] = st.select_slider("ML проти Fuzzy", SAATY_SCALE, value=1)
            judgments[(0, 2)] = st.select_slider("ML проти Deal", SAATY_SCALE, value=1)
            judgments[(1, 2)] = st.select_slider("Fuzzy проти Deal", SAATY_SCALE, value=1)

        matrix = ahp_module.build_matrix_from_judgments(n, judgments)
        eval_result = ahp_module.evaluate_pairwise(matrix, labels)

        st.write("Ваги:", {display_labels[k]: round(v, 3) for k, v in eval_result["weights"].items()})
        st.write(f"CI = {eval_result['ci']:.4f}, CR = {eval_result['cr']:.4f}")
        if not eval_result["consistent"]:
            st.warning("CR > 0.10 — судження неузгоджені. Рекомендується переглянути попарні порівняння.")

        weights_full = {"ml": eval_result["weights"].get("ml", 0), "fuzzy": eval_result["weights"].get("fuzzy", 0), "deal": eval_result["weights"].get("deal", 0)}
        custom_weights_payload = {"weights": weights_full, "cr": eval_result["cr"], "matrix": matrix.tolist(), "labels": labels}

    # Пошуковий профіль впливає лише на позначку в історії
    st.subheader("Профіль пошуку (лише для позначки в історії)")
    search_industries = st.multiselect("Індустрії", INDUSTRIES, default=profile["search_industries"])
    search_stages = st.multiselect("Стадії фінансування", FUNDING_STAGES, default=profile["search_stages"])
    c1, c2 = st.columns(2)
    with c1:
        ticket_min = st.number_input("Ticket size, мін ($M)", min_value=0.0, value=float(profile["ticket_size_min"] or 0.0))
    with c2:
        ticket_max = st.number_input("Ticket size, макс ($M)", min_value=0.0, value=float(profile["ticket_size_max"] or 50.0))

    if st.button("Зберегти профіль"):
        save_investor_profile(conn, {
            "weight_preset": preset,
            "custom_weights": custom_weights_payload,
            "search_industries": search_industries,
            "search_stages": search_stages,
            "ticket_size_min": ticket_min,
            "ticket_size_max": ticket_max,
        })
        st.success("Профіль збережено.")


def save_report(conn, analysis_id, file_path):
    """Записує в базу відомості про згенерований PDF-звіт."""
    conn.execute(
        "INSERT INTO reports (analysis_id, generated_date, file_path) VALUES (?, ?, ?)",
        (analysis_id, datetime.date.today().isoformat(), str(file_path)),
    )
    conn.commit()


def page_export_pdf(conn, pipeline):
    """Сторінка «Звіт»: генерація PDF-звіту за вибраним аналізом."""
    st.header("Звіт")

    history = fetch_history(conn)
    if not history:
        st.info("Ще немає жодного аналізу. Почніть із вкладки «Новий аналіз».")
        return

    options = {f"#{r['id']} — {r['name']} ({r['date']})": r for r in history}
    chosen_label = st.selectbox("Аналіз для звіту", list(options.keys()))
    row = options[chosen_label]

    # Дані для звіту беруться зі збереженого аналізу
    details = row["details"]
    weights = details["ahp"]["weights"] if details else None

    context = {
        "name": row["name"], "industry": row["industry"], "funding_stage": row["funding_stage"],
        "date": row["date"], "score": row["final_score"], "category": row["category"],
        "ml_score": row["ml_score"], "fuzzy_score": row["fuzzy_score"], "deal_score": row["deal_score"],
        "weights": weights, "team_quality": row["team_quality"], "traction": row["traction"],
        "market_size": row["market_size"], "competition": row["competition"],
        "funding_ask": row["funding_ask"], "pre_money_valuation": row["pre_money_valuation"],
        "details": details,
    }

    if st.button("Згенерувати PDF"):
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = REPORTS_DIR / f"analysis_{row['id']}_{timestamp}.pdf"
        report_module.generate_pdf_report(context, file_path)
        save_report(conn, row["id"], file_path)

        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        st.success(f"Звіт збережено: {file_path}")
        st.download_button(
            "Завантажити PDF", data=pdf_bytes,
            file_name=file_path.name, mime="application/pdf",
        )


# Шляхи для перенавчання моделі: файл моделі, підсумок останнього навчання та датасет
MODEL_PATH_OBJ = Path(MODEL_PATH)
LAST_TRAINING_PATH = Path("models/last_training.json")
DATASET_PATH = Path("data/raw/startups_dataset.csv")


def retrain_model():
    """UC14: перенавчає модель на сирому датасеті, зберігає попередню як models/model_prev.pkl,
    записує найкращу як models/model.pkl і скидає кеш конвеєра. Повертає підсумок навчання."""
    out = ml_module.train_and_compare(str(DATASET_PATH))
    if MODEL_PATH_OBJ.exists():
        shutil.copy(MODEL_PATH_OBJ, MODEL_PATH_OBJ.with_name("model_prev.pkl"))
    ml_module.save_model(out["best_pipeline"], MODEL_PATH)
    summary = {
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "best_model": out["best_name"],
        "n_train": len(out["X_train"]),
        "cv": out["cv_results"],
        "test": out["results"],
        "train": out["train_results"],
    }
    LAST_TRAINING_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    get_ml_pipeline.clear()
    return summary


def _training_table(summary):
    """Таблиця метрик моделей за підсумком навчання; найкраща модель позначена зірочкою."""
    return [
        {
            "Модель": ("★ " if name == summary["best_model"] else "") + name,
            "CV R²": round(summary["cv"][name]["r2"], 4) if "cv" in summary else None,
            "Test RMSE": round(m["rmse"], 3), "Test MAE": round(m["mae"], 3),
            "Test R²": round(m["r2"], 4), "Train R²": round(summary["train"][name]["r2"], 4),
        }
        for name, m in summary["test"].items()
    ]


def page_admin(conn, pipeline):
    """Сторінка «Адміністратор»: перенавчання ML-моделі та редагування бази правил Fuzzy."""
    st.header("Адміністратор")
    st.caption(
        "Режим без авторизації (UC14 — перенавчання моделі, UC15 — база правил). "
        "Зміни діють на нові аналізи; збережена історія не перераховується."
    )

    # Блок ML-моделі: стан файлу, кнопка перенавчання та метрики останнього навчання
    st.subheader("ML-модель")
    if MODEL_PATH_OBJ.exists():
        modified = datetime.datetime.fromtimestamp(MODEL_PATH_OBJ.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        st.write(f"Файл `{MODEL_PATH}`, оновлено {modified}.")
    else:
        st.warning(f"Файл моделі `{MODEL_PATH}` не знайдено.")
    if "training_summary" not in st.session_state and LAST_TRAINING_PATH.exists():
        st.session_state["training_summary"] = json.loads(LAST_TRAINING_PATH.read_text(encoding="utf-8"))

    if st.button("Перенавчити"):
        if not DATASET_PATH.exists():
            st.error(f"Немає датасету `{DATASET_PATH}`. Покладіть CSV у `data/raw/` (див. README).")
        else:
            with st.spinner("Крос-валідація й навчання трьох моделей на 80% даних (≈ 1 хв)…"):
                st.session_state["training_summary"] = retrain_model()
            st.success("Модель перенавчено й збережено; нові аналізи використовуватимуть її.")

    summary = st.session_state.get("training_summary")
    if summary:
        st.caption(
            f"Останнє навчання: {summary['trained_at']}, навчальна вибірка {summary['n_train']} рядків, "
            f"обрано за 5-fold CV RMSE на навчальній частині: {summary['best_model']}."
        )
        st.dataframe(_training_table(summary), hide_index=True, width="stretch")

    # Блок бази правил Fuzzy: перегляд, зміна висновків і збереження з перерахунком меж
    st.subheader("База правил Fuzzy")
    cfg = fuzzy_module.get_rules()
    labels = cfg["variables"]["favorability"]["terms"]
    to_term = {label: term for term, label in labels.items()}
    norm = cfg["normalization"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Правил", sum(len(level["rules"]) for level in cfg["levels"]))
    c2.metric("F_RAW_MIN", norm["F_RAW_MIN"])
    c3.metric("F_RAW_MAX", norm["F_RAW_MAX"])

    # Таблиця правил; редагується лише стовпець висновку
    original = [
        {"№": r["id"], "Рівень": r["level"], "ЯКЩО": r["if"], "ТО (висновок)": labels[r["then_term"]]}
        for r in fuzzy_module.rule_rows(cfg)
    ]
    edited = st.data_editor(
        original, hide_index=True, width="stretch", height=38 + 35 * len(original),
        disabled=["№", "Рівень", "ЯКЩО"],
        column_config={
            "ТО (висновок)": st.column_config.SelectboxColumn(options=list(labels.values()), required=True)
        },
        key=f"rules_editor_{st.session_state.get('rules_version', 0)}",
    )
    edited_rows = edited.to_dict("records") if hasattr(edited, "to_dict") else list(edited)
    # Змінені висновки: номер правила → новий терм
    changes = {
        row["№"]: to_term[row["ТО (висновок)"]]
        for row, before in zip(edited_rows, original)
        if row["ТО (висновок)"] != before["ТО (висновок)"]
    }

    st.caption(
        "Редагується лише висновок правила; кожна комбінація умов має рівно одне правило. "
        "Після збереження межі нормалізації F(Y) перераховуються по сітці 5⁴ = 625 комбінацій (≈ 5 с); "
        "попередня версія файлу лишається як `data/fuzzy_rules.prev.json`."
    )
    if st.button(f"Зберегти правила ({len(changes)} змін) і перерахувати межі", disabled=not changes):
        try:
            with st.spinner("Перерахунок меж нормалізації…"):
                bounds = fuzzy_module.save_rules(fuzzy_module.with_consequents(cfg, changes))
        except ValueError as error:
            st.error(f"Правила не збережено, попередню версію відновлено. {error}")
        else:
            st.session_state["rules_version"] = st.session_state.get("rules_version", 0) + 1
            st.session_state["admin_message"] = (
                f"Збережено {len(changes)} змін. Нові межі: F_RAW_MIN = {bounds['F_RAW_MIN']}, "
                f"F_RAW_MAX = {bounds['F_RAW_MAX']}."
            )
            st.rerun()
    if "admin_message" in st.session_state:
        st.success(st.session_state.pop("admin_message"))


# Сторінки застосунку в порядку відображення в меню
PAGES = {
    "Новий аналіз": page_new_analysis,
    "Результат аналізу": page_result,
    "Історія аналізів": page_history,
    "Порівняння": page_comparison,
    "Профіль інвестора": page_investor_profile,
    "Звіт": page_export_pdf,
}


def main():
    """Точка входу: налаштування сторінки, ресурси, бічне меню та виклик вибраної сторінки."""
    st.set_page_config(page_title="СППР оцінки стартапів", layout="wide")
    theme.inject(st)
    conn = get_db_connection()
    pipeline = get_ml_pipeline()

    # Сторінка «Адміністратор» з'являється лише в режимі адміністратора
    pages = dict(PAGES)
    if st.session_state.get("admin_mode"):
        pages["Адміністратор"] = page_admin

    st.sidebar.title("СППР стартапів")
    page = st.sidebar.radio("Навігація", list(pages.keys()))
    st.sidebar.divider()
    st.sidebar.toggle("Режим адміністратора", key="admin_mode")
    pages[page](conn, pipeline)


if __name__ == "__main__":
    main()
