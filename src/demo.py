"""Helpers shared by the seed script, the benchmark and the tests for demo companies
(plain dicts in the seed script's / fixture file's field names)."""
import json
from pathlib import Path

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "section_4_2_companies.json"


def to_inputs(c, country="Ukraine"):
    """Demo-company dict -> (company_raw, qualitative, deal_terms) as analyze_startup expects.
    A company with its own "region" is analysed in that region; the others use `country`
    (Ukraine is not in REGION_MAP, so they fall into "Other")."""
    company_raw = {
        "Startup Name": c["name"], "Founded Year": c["founded_year"], "Country": country,
        "Industry": c["industry"], "Funding Stage": c["funding_stage"],
        "Total Funding ($M)": c["total_funding"], "Number of Employees": c["employees"],
        "Annual Revenue ($M)": c["revenue"], "Valuation ($M)": c["valuation"],
        "Customer Base (Thousands)": c["customer_base"], "Social Media Followers": c["followers"],
        "Business Model": c["business_model"],
        "Revenue Growth (%)": c.get("revenue_growth"),  # None -> missing, as in the dataset
    }
    if c.get("region"):
        company_raw["Region"] = c["region"]
    qualitative = {
        "team_quality": c["team_quality"], "traction": c["traction"],
        "market_size": c["market_size"], "competition": c["competition"],
    }
    return company_raw, qualitative, c.get("deal")


def load_section_4_2(path=FIXTURE_PATH):
    """The two fixture companies for thesis section 4.2 (with their expected values)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["companies"]
