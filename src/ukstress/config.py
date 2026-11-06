"""Project paths, source series and portfolio definitions."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BOE_DIR = RAW_DIR / "boe"
ONS_DIR = RAW_DIR / "ons"
HMLR_DIR = RAW_DIR / "hmlr"
STRESS_DIR = RAW_DIR / "boe_stress"
BCST_PATHS_FILE = STRESS_DIR / "bcst2025_variable_paths.xlsx"
UKHPI_FILE = HMLR_DIR / "ukhpi_united_kingdom.csv"
PROCESSED_DIR = DATA_DIR / "processed"
PANEL_PATH = PROCESSED_DIR / "quarterly_panel.parquet"
MACRO_PATH = PROCESSED_DIR / "macro_history.parquet"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

# Portfolios modelled. Write-offs (Bankstats C2.1, sterling, NSA, £m in the quarter) are divided
# by the amount outstanding at the END OF THE PREVIOUS QUARTER, as the Bank of England advises.
PORTFOLIOS: dict[str, dict[str, str]] = {
    "credit_card": {"write_off": "RPQTFHE", "balance": "RPQB74D", "label": "Credit cards"},
    "other_consumer": {
        "write_off": "RPQTFHF",
        "balance": "RPQB75D",
        "label": "Other consumer credit",
    },
    "mortgage": {"write_off": "RPQTFHD", "balance": "RPQB73D", "label": "Mortgages"},
    "corporate": {"write_off": "RPQTFHB", "balance": "RPQB69D", "label": "Corporate (PNFC)"},
}

# Macro series: code -> (file, column name in the panel)
ONS_SERIES = {
    "MGSX": ("MGSX_unemployment.csv", "unemployment"),
    "ABMI": ("ABMI_real_gdp.csv", "real_gdp"),
    "D7G7": ("D7G7_cpi_inflation.csv", "cpi_inflation"),
}
BOE_RATES = {"IUQABEDR": "bank_rate", "IUMBV34": "mortgage_rate_2y_fixed"}

FIRST_QUARTER = "1993Q1"  # write-off data start
