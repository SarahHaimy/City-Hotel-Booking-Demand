import numpy as np
import pandas as pd

# ---- Leakage registry (document each in the report's Leakage Audit) ----
OUTCOME_COLS      = ["IsCanceled", "ReservationStatus", "ReservationStatusDate"]
OUTCOME_PROXY     = ["DepositType"]
POST_BOOKING_COLS = ["AssignedRoomType", "BookingChanges", "DaysInWaitingList"]
FORBIDDEN = set(OUTCOME_COLS + OUTCOME_PROXY + POST_BOOKING_COLS)

# ---- Candidate feature sets ----
FEATURE_SET = {
    "num": ["LeadTime", "TotalNights", "TotalGuests", "ADR", "TotalOfSpecialRequests"],
    "bin": ["HasChildren", "HasParking", "IsRepeatedGuest"],
    "cat": ["MarketSegment", "CustomerType", "Meal"],
}

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic row-wise features: nothing is learned from the data, so no leakage."""
    out = df.copy()
    # Children is stored as a string in the CSV, so coerce it before doing arithmetic
    out["Children"] = pd.to_numeric(out["Children"], errors="coerce").fillna(0)
    out["TotalNights"] = out["StaysInWeekendNights"] + out["StaysInWeekNights"]
    out["TotalGuests"] = out["Adults"] + out["Children"] + out["Babies"]
    out["HasChildren"] = ((out["Children"] + out["Babies"]) > 0).astype(int)
    out["HasParking"]  = (out["RequiredCarParkingSpaces"] > 0).astype(int)
    out["IsDomestic"]  = (out["Country"] == "PRT").astype(int)   # optional variant
    return out

def get_feature_frame(df: pd.DataFrame):
    fs = FEATURE_SET
    cols = fs["num"] + fs["bin"] + fs["cat"]

    # Hard guard: fail loudly if someone adds a forbidden column
    leaked = FORBIDDEN & set(cols)
    assert not leaked, f"Leakage risk, forbidden columns in feature set: {leaked}"

    X = add_engineered_features(df)[cols]
    assert X.isna().sum().sum() == 0, "Unexpected NaNs in clustering features"
    return X, fs