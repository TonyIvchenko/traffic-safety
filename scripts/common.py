from __future__ import annotations

from datetime import datetime
from pathlib import Path
import math


SERVICE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = SERVICE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "fars"
TRAFFIC_RAW_DIR = DATA_DIR / "raw" / "traffic"
US_ACCIDENTS_RAW_DIR = TRAFFIC_RAW_DIR / "us_accidents"
TIGER_RAW_DIR = TRAFFIC_RAW_DIR / "tiger"
TIGER_PRISEC_DIR = TIGER_RAW_DIR / "prisecroads"
CENSUS_RAW_DIR = DATA_DIR / "raw" / "census"
NOAA_RAW_DIR = DATA_DIR / "raw" / "noaa"
NOAA_ISD_LITE_DIR = NOAA_RAW_DIR / "isd-lite"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_WEATHER_DIR = PROCESSED_DIR / "weather"
WEATHER_HOURLY_DIR = PROCESSED_WEATHER_DIR / "hourly"
PROCESSED_SEGMENTS_DIR = PROCESSED_DIR / "segments"
PROCESSED_GEO_DIR = PROCESSED_DIR / "geo"
PROCESSED_SAFETY_DIR = PROCESSED_DIR / "safety"
MODELS_DIR = SERVICE_DIR / "models"
TILES_DIR = SERVICE_DIR / "tiles"

ACCIDENTS_CLEAN_PATH = PROCESSED_DIR / "accidents_clean.csv.gz"
CANDIDATE_CELLS_PATH = PROCESSED_DIR / "candidate_cells.csv.gz"
WEEKLY_COUNTS_PATH = PROCESSED_DIR / "weekly_counts.csv.gz"
VRU_ACCIDENTS_CLEAN_PATH = PROCESSED_DIR / "vru_accidents_clean.csv.gz"
VRU_CANDIDATE_CELLS_PATH = PROCESSED_DIR / "vru_candidate_cells.csv.gz"
VRU_WEEKLY_COUNTS_PATH = PROCESSED_DIR / "vru_weekly_counts.csv.gz"
VRU_MODEL_BUNDLE_PATH = MODELS_DIR / "traffic_safety_vru.joblib"
SEGMENT_GEOID_PATH = PROCESSED_GEO_DIR / "segment_geoid.csv.gz"
CELL_GEOID_PATH = PROCESSED_GEO_DIR / "cell_geoid.csv.gz"
HIGH_INJURY_NETWORK_PATH = PROCESSED_SAFETY_DIR / "high_injury_network.parquet"
STATION_HISTORY_PATH = NOAA_RAW_DIR / "isd-history.csv"
REPRESENTATIVE_STATIONS_PATH = PROCESSED_WEATHER_DIR / "representative_stations.csv.gz"
CELL_WEATHER_STATIONS_PATH = PROCESSED_WEATHER_DIR / "cell_weather_stations.csv.gz"
WEATHER_CLIMATOLOGY_PATH = PROCESSED_WEATHER_DIR / "weather_climatology.csv.gz"
MODEL_BUNDLE_PATH = MODELS_DIR / "traffic_safety.joblib"
US_ACCIDENTS_PATH = US_ACCIDENTS_RAW_DIR / "US_Accidents.csv"
ROAD_SEGMENTS_PATH = PROCESSED_SEGMENTS_DIR / "road_segments.parquet"
ACTIVE_ROAD_SEGMENTS_PATH = PROCESSED_SEGMENTS_DIR / "active_road_segments.parquet"
SEGMENT_EVENTS_PATH = PROCESSED_SEGMENTS_DIR / "segment_events.parquet"
SEGMENT_HOURLY_COUNTS_PATH = PROCESSED_SEGMENTS_DIR / "segment_hourly_counts.parquet"
SEGMENT_MODEL_BUNDLE_PATH = MODELS_DIR / "traffic_safety_segments.joblib"
SEGMENT_RISK_SNAPSHOT_PATH = TILES_DIR / "segment_risk_snapshot.parquet"
OVERLAY_NPZ_PATH = TILES_DIR / "overlay.npz"
OVERLAY_JSON_PATH = TILES_DIR / "overlay.json"

DEFAULT_YEARS = list(range(2016, 2025))
DEFAULT_TRAIN_YEARS = list(range(2018, 2024))
DEFAULT_EVAL_YEAR = 2024
H3_RESOLUTION = 5
NEGATIVE_RATIO = 4
RANDOM_SEED = 42
WEATHER_REPRESENTATION_H3_RES = 2

LAT_MIN = 18.0
LAT_MAX = 72.0
LON_MIN = -179.0
LON_MAX = -66.0
OVERLAY_HEIGHT = 360
OVERLAY_WIDTH = 760
CELL_PAINT_RADIUS = 2
SEGMENT_MAX_LENGTH_KM = 0.5
SEGMENT_MATCH_CANDIDATES = 8
SEGMENT_SERVE_LIMIT = 3000

WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
STATE_FIPS = [
    "01",
    "02",
    "04",
    "05",
    "06",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "40",
    "41",
    "42",
    "44",
    "45",
    "46",
    "47",
    "48",
    "49",
    "50",
    "51",
    "53",
    "54",
    "55",
    "56",
]


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TRAFFIC_RAW_DIR.mkdir(parents=True, exist_ok=True)
    US_ACCIDENTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    TIGER_RAW_DIR.mkdir(parents=True, exist_ok=True)
    TIGER_PRISEC_DIR.mkdir(parents=True, exist_ok=True)
    CENSUS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    NOAA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    NOAA_ISD_LITE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    WEATHER_HOURLY_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_GEO_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_SAFETY_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TILES_DIR.mkdir(parents=True, exist_ok=True)


def fars_zip_url(year: int) -> str:
    return (
        "https://static.nhtsa.gov/nhtsa/downloads/FARS/"
        f"{year}/National/FARS{year}NationalCSV.zip"
    )


def fars_zip_path(year: int) -> Path:
    return RAW_DIR / f"FARS{year}NationalCSV.zip"


def tiger_prisecroads_url(state_fips: str, year: int = 2024) -> str:
    return (
        f"https://www2.census.gov/geo/tiger/TIGER{year}/PRISECROADS/"
        f"tl_{year}_{state_fips}_prisecroads.zip"
    )


def tiger_prisecroads_path(state_fips: str, year: int = 2024) -> Path:
    return TIGER_PRISEC_DIR / f"tl_{year}_{state_fips}_prisecroads.zip"


def tiger_county_url(year: int = 2024) -> str:
    return f"https://www2.census.gov/geo/tiger/TIGER{year}/COUNTY/tl_{year}_us_county.zip"


def tiger_county_path(year: int = 2024) -> Path:
    return CENSUS_RAW_DIR / f"tl_{year}_us_county.zip"


def tiger_tract_url(state_fips: str, year: int = 2024) -> str:
    return (
        f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/"
        f"tl_{year}_{state_fips}_tract.zip"
    )


def tiger_tract_path(state_fips: str, year: int = 2024) -> Path:
    return CENSUS_RAW_DIR / f"tl_{year}_{state_fips}_tract.zip"


def noaa_isd_lite_url(station_id: str, year: int) -> str:
    return f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{station_id}-{year}.gz"


def noaa_isd_lite_path(station_id: str, year: int) -> Path:
    return NOAA_ISD_LITE_DIR / str(year) / f"{station_id}-{year}.gz"


def month_sin_cos(month: int) -> tuple[float, float]:
    angle = 2.0 * math.pi * float(month) / 12.0
    return math.sin(angle), math.cos(angle)


def hour_sin_cos(hour: int) -> tuple[float, float]:
    angle = 2.0 * math.pi * float(hour) / 24.0
    return math.sin(angle), math.cos(angle)


def dow_sin_cos(day_of_week: int) -> tuple[float, float]:
    angle = 2.0 * math.pi * float(day_of_week - 1) / 7.0
    return math.sin(angle), math.cos(angle)


def local_hour_of_week_label(frame_idx: int) -> str:
    day_idx = frame_idx // 24
    hour = frame_idx % 24
    return f"{WEEKDAY_LABELS[day_idx]} {hour:02d}:00"


def weekly_frame_labels() -> list[str]:
    return [local_hour_of_week_label(idx) for idx in range(24 * 7)]


def weekly_ticks() -> list[dict[str, int | str]]:
    return [
        {"label": weekday, "frame_idx": idx * 24}
        for idx, weekday in enumerate(WEEKDAY_LABELS)
    ]


def current_month() -> int:
    return datetime.now().month
