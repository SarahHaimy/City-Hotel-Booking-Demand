"""Schema, source variable definitions, and leakage rules for City Hotel (H2).

All column names follow the exact verified source labels in Section 4 of the
CSE382 project specification. Documented code-level renames (e.g. Babies -> HasBabies)
are tracked explicitly in DOCUMENTED_RENAMES.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ColumnKind = Literal["numeric_int", "numeric_float", "binary", "categorical", "id", "date"]


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: ColumnKind
    meaning: str
    default: Any = None
    min_val: float | None = None
    max_val: float | None = None
    categories: tuple[str, ...] = field(default_factory=tuple)
    nullable: bool = False


MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

TOP_COUNTRIES = (
    "PRT", "FRA", "DEU", "GBR", "ESP", "ITA", "BEL", "BRA", "USA", "NLD",
    "CHE", "IRL", "AUT", "CHN", "SWE", "POL", "RUS", "NOR", "ISR", "FIN", "Other",
)

TOP_AGENTS = (
    "9", "7", "1", "14", "28", "6", "8", "19", "27", "83", "85", "138", "229", "null",
)

# All 31 verified H2 source variables from Section 4 of the project specification.
H2_COLUMNS: dict[str, ColumnSpec] = {
    "IsCanceled": ColumnSpec(
        "IsCanceled", "binary",
        "1 if the booking was canceled; 0 otherwise.",
        default=0, min_val=0, max_val=1,
    ),
    "LeadTime": ColumnSpec(
        "LeadTime", "numeric_int",
        "Days between the booking entering the PMS and the arrival date.",
        default=65, min_val=0, max_val=730,
    ),
    "ArrivalDateYear": ColumnSpec(
        "ArrivalDateYear", "numeric_int",
        "Year of arrival (2015–2017 in H2).",
        default=2016, min_val=2015, max_val=2030,
    ),
    "ArrivalDateMonth": ColumnSpec(
        "ArrivalDateMonth", "categorical",
        "Month of arrival.",
        default="July", categories=MONTHS,
    ),
    "ArrivalDateWeekNumber": ColumnSpec(
        "ArrivalDateWeekNumber", "numeric_int",
        "Week number of the arrival date (1–53).",
        default=28, min_val=1, max_val=53,
    ),
    "ArrivalDateDayOfMonth": ColumnSpec(
        "ArrivalDateDayOfMonth", "numeric_int",
        "Day of month of the arrival date (1–31).",
        default=15, min_val=1, max_val=31,
    ),
    "StaysInWeekendNights": ColumnSpec(
        "StaysInWeekendNights", "numeric_int",
        "Number of weekend nights (Sat/Sun) booked.",
        default=1, min_val=0, max_val=30,
    ),
    "StaysInWeekNights": ColumnSpec(
        "StaysInWeekNights", "numeric_int",
        "Number of weekday nights (Mon–Fri) booked.",
        default=2, min_val=0, max_val=60,
    ),
    "Adults": ColumnSpec(
        "Adults", "numeric_int",
        "Number of adults.",
        default=2, min_val=0, max_val=20,
    ),
    "Children": ColumnSpec(
        "Children", "numeric_int",
        "Number of children.",
        default=0, min_val=0, max_val=10, nullable=True,
    ),
    "Babies": ColumnSpec(
        "Babies", "numeric_int",
        "Number of babies (mapped to binary HasBabies during cleaning).",
        default=0, min_val=0, max_val=10,
    ),
    "Meal": ColumnSpec(
        "Meal", "categorical",
        "Meal package booked (BB=Bed & Breakfast, HB=Half Board, FB=Full Board, SC=Self-Catering).",
        default="BB", categories=("BB", "HB", "SC", "FB"),
    ),
    "Country": ColumnSpec(
        "Country", "categorical",
        "ISO country of origin code.",
        default="PRT", categories=TOP_COUNTRIES, nullable=True,
    ),
    "MarketSegment": ColumnSpec(
        "MarketSegment", "categorical",
        "Market segment designation (e.g. Online TA, Offline TA/TO, Direct, Groups, Corporate).",
        default="Online TA",
        categories=("Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate", "Complementary", "Aviation"),
    ),
    "DistributionChannel": ColumnSpec(
        "DistributionChannel", "categorical",
        "Booking distribution channel (TA/TO, Direct, Corporate, GDS).",
        default="TA/TO", categories=("TA/TO", "Direct", "Corporate", "GDS"),
    ),
    "IsRepeatedGuest": ColumnSpec(
        "IsRepeatedGuest", "binary",
        "1 for a repeated guest; 0 otherwise.",
        default=0, min_val=0, max_val=1,
    ),
    "PreviousCancellations": ColumnSpec(
        "PreviousCancellations", "numeric_int",
        "Number of previous bookings canceled by the customer prior to current booking.",
        default=0, min_val=0, max_val=50,
    ),
    "PreviousBookingsNotCanceled": ColumnSpec(
        "PreviousBookingsNotCanceled", "numeric_int",
        "Number of previous bookings not canceled by the customer prior to current booking.",
        default=0, min_val=0, max_val=100,
    ),
    "ReservedRoomType": ColumnSpec(
        "ReservedRoomType", "categorical",
        "Code of the room type originally reserved by the guest.",
        default="A", categories=("A", "B", "C", "D", "E", "F", "G"),
    ),
    "AssignedRoomType": ColumnSpec(
        "AssignedRoomType", "categorical",
        "Code of the room type ultimately assigned by operations (post-booking).",
        default="A", categories=("A", "B", "C", "D", "E", "F", "G", "K"),
    ),
    "BookingChanges": ColumnSpec(
        "BookingChanges", "numeric_int",
        "Cumulative amendments made between booking entry and check-in/cancellation.",
        default=0, min_val=0, max_val=30,
    ),
    "DepositType": ColumnSpec(
        "DepositType", "categorical",
        "Deposit arrangement (No Deposit, Non Refund, Refundable).",
        default="No Deposit", categories=("No Deposit", "Non Refund", "Refundable"),
    ),
    "Agent": ColumnSpec(
        "Agent", "id",
        "Travel-agency identifier (categorical string ID; may be NULL).",
        default="9", categories=TOP_AGENTS, nullable=True,
    ),
    "Company": ColumnSpec(
        "Company", "id",
        "Corporate entity identifier (~95% missing in H2; dropped in cleaning).",
        default="null", nullable=True,
    ),
    "DaysInWaitingList": ColumnSpec(
        "DaysInWaitingList", "numeric_int",
        "Days the booking remained on the waiting list before confirmation.",
        default=0, min_val=0, max_val=400,
    ),
    "CustomerType": ColumnSpec(
        "CustomerType", "categorical",
        "Type of booking/customer (Transient, Transient-Party, Contract, Group).",
        default="Transient", categories=("Transient", "Transient-Party", "Contract", "Group"),
    ),
    "ADR": ColumnSpec(
        "ADR", "numeric_float",
        "Average Daily Rate (EUR per night).",
        default=105.0, min_val=0.0, max_val=1000.0,
    ),
    "RequiredCarParkingSpaces": ColumnSpec(
        "RequiredCarParkingSpaces", "numeric_int",
        "Number of car-parking spaces requested by the customer.",
        default=0, min_val=0, max_val=5,
    ),
    "TotalOfSpecialRequests": ColumnSpec(
        "TotalOfSpecialRequests", "numeric_int",
        "Number of special requests made by the customer (e.g. high floor, twin bed).",
        default=1, min_val=0, max_val=10,
    ),
    "ReservationStatus": ColumnSpec(
        "ReservationStatus", "categorical",
        "Final reservation status (Check-Out, Canceled, No-Show) — direct outcome!",
        default="Check-Out", categories=("Check-Out", "Canceled", "No-Show"),
    ),
    "ReservationStatusDate": ColumnSpec(
        "ReservationStatusDate", "date",
        "Date on which the final reservation status was set — post-outcome timestamp!",
        default="2016-07-17",
    ),
}

# Documented 1-to-1 rename in src/data/data_cleaning.py (Spec Section 4).
DOCUMENTED_RENAMES: dict[str, str] = {"Babies": "HasBabies"}


def _to_snake(name: str) -> str:
    out = []
    for idx, ch in enumerate(name):
        if ch.isupper() and idx > 0 and (not name[idx - 1].isupper()):
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


# Automatic normalization map: lowercase / snake_case -> canonical H2 label.
COLUMN_ALIASES: dict[str, str] = {}
for _col in H2_COLUMNS:
    COLUMN_ALIASES[_col.lower()] = _col
    COLUMN_ALIASES[_to_snake(_col)] = _col
    COLUMN_ALIASES[_col.lower().replace("_", "")] = _col
COLUMN_ALIASES["adr"] = "ADR"
COLUMN_ALIASES["is_cancelled"] = "IsCanceled"
COLUMN_ALIASES["iscancelled"] = "IsCanceled"
COLUMN_ALIASES["previous_cancellations"] = "PreviousCancellations"
COLUMN_ALIASES["has_babies"] = "HasBabies"
COLUMN_ALIASES["hasbabies"] = "HasBabies"


# Per-task leakage & feature-usage audit matrix (PDF Section 16).
# Assumed prediction point across supervised tasks: Moment of initial reservation entry in the PMS.
LEAKAGE_AUDIT_MATRIX: list[dict[str, str]] = [
    {
        "column": "IsCanceled",
        "regression": "excluded_outcome",
        "classification": "target",
        "clustering": "excluded_outcome",
        "rationale": "Final cancellation outcome. Target for Task B; direct future outcome leakage if supplied to ADR regression or unsupervised clustering.",
    },
    {
        "column": "ADR",
        "regression": "target",
        "classification": "excluded_design",
        "clustering": "used",
        "rationale": "Average Daily Rate. Target for Task A; used in customer segmentation to distinguish budget vs premium stays; omitted in baseline Bayes cancellation model.",
    },
    {
        "column": "ReservationStatus",
        "regression": "excluded_outcome",
        "classification": "excluded_outcome",
        "clustering": "excluded_outcome",
        "rationale": "Direct 1-to-1 encoding of final outcome (Canceled / No-Show -> IsCanceled=1; Check-Out -> IsCanceled=0). Severe direct target leakage.",
    },
    {
        "column": "ReservationStatusDate",
        "regression": "excluded_outcome",
        "classification": "excluded_outcome",
        "clustering": "excluded_outcome",
        "rationale": "Timestamp of check-out or cancellation event. Comparing against ArrivalDate reveals whether a guest checked out or canceled early.",
    },
    {
        "column": "AssignedRoomType",
        "regression": "excluded_post_booking",
        "classification": "excluded_post_booking",
        "clustering": "excluded_post_booking",
        "rationale": "Assigned by hotel front-desk/operations closer to check-in (upgrades/overbooking), not fixed at initial booking entry.",
    },
    {
        "column": "BookingChanges",
        "regression": "excluded_post_booking",
        "classification": "excluded_post_booking",
        "clustering": "excluded_post_booking",
        "rationale": "Accumulates amendments over the entire booking lifecycle up to arrival or cancellation; unavailable at initial reservation time.",
    },
    {
        "column": "DaysInWaitingList",
        "regression": "excluded_post_booking",
        "classification": "excluded_post_booking",
        "clustering": "excluded_post_booking",
        "rationale": "Elapsed waiting-list duration is only finalized after confirmation or cancellation, not at initial booking submission.",
    },
    {
        "column": "DepositType",
        "regression": "used",
        "classification": "excluded_proxy",
        "clustering": "excluded_proxy",
        "rationale": "Rate-plan attribute known at booking for ADR pricing, but in PMS exports 'Non Refund' exhibits >99% cancellation rate due to post-hoc fee/group coding — excluded as indirect outcome proxy in classification and clustering.",
    },
    {
        "column": "Company",
        "regression": "excluded_sparse",
        "classification": "excluded_sparse",
        "clustering": "excluded_sparse",
        "rationale": "~95% NULL in City Hotel (H2); dropped deterministically during structural cleaning.",
    },
    {
        "column": "LeadTime",
        "regression": "used",
        "classification": "used",
        "clustering": "used",
        "rationale": "Exact days between PMS entry and scheduled arrival; known immediately when the reservation is placed.",
    },
    {
        "column": "ArrivalDateYear",
        "regression": "used",
        "classification": "excluded_design",
        "clustering": "excluded_design",
        "rationale": "Captures multi-year price inflation/trend in ADR regression; omitted from cancellation/clustering to generalize across calendar years.",
    },
    {
        "column": "ArrivalDateMonth",
        "regression": "used",
        "classification": "used",
        "clustering": "excluded_design",
        "rationale": "Known at reservation time; strong seasonal driver of room pricing and holiday cancellation behavior.",
    },
    {
        "column": "ArrivalDateWeekNumber",
        "regression": "excluded_design",
        "classification": "excluded_design",
        "clustering": "excluded_design",
        "rationale": "Collinear with ArrivalDateMonth + ArrivalDateDayOfMonth.",
    },
    {
        "column": "ArrivalDateDayOfMonth",
        "regression": "used",
        "classification": "excluded_design",
        "clustering": "excluded_design",
        "rationale": "Used with Month & Year in regression to derive exact day-of-week and day-of-year harmonic seasonality.",
    },
    {
        "column": "StaysInWeekendNights",
        "regression": "used",
        "classification": "used",
        "clustering": "engineered",
        "rationale": "Known at reservation; combined into TotalNights / WeekendShare for regression and clustering.",
    },
    {
        "column": "StaysInWeekNights",
        "regression": "used",
        "classification": "used",
        "clustering": "engineered",
        "rationale": "Known at reservation; combined with weekend nights into TotalNights.",
    },
    {
        "column": "Adults",
        "regression": "used",
        "classification": "used",
        "clustering": "engineered",
        "rationale": "Party size booked at reservation; drives room occupancy pricing and TotalGuests.",
    },
    {
        "column": "Children",
        "regression": "used",
        "classification": "used",
        "clustering": "engineered",
        "rationale": "Known at reservation (4 missing values imputed to 0); feeds TotalGuests and HasChildren.",
    },
    {
        "column": "Babies",
        "regression": "excluded_design",
        "classification": "excluded_design",
        "clustering": "engineered",
        "rationale": "Renamed 1-to-1 to binary HasBabies in clean_data(); contributes to TotalGuests / HasChildren in clustering.",
    },
    {
        "column": "Meal",
        "regression": "used",
        "classification": "used",
        "clustering": "used",
        "rationale": "Selected meal plan (BB/HB/FB/SC) directly shifts ADR and characterizes traveler type.",
    },
    {
        "column": "Country",
        "regression": "excluded_design",
        "classification": "excluded_design",
        "clustering": "excluded_design",
        "rationale": "Guest nationality in PMS is often populated at check-in from ID documents for non-canceled stays, creating subtle missingness/accuracy asymmetry vs canceled bookings.",
    },
    {
        "column": "MarketSegment",
        "regression": "used",
        "classification": "used",
        "clustering": "used",
        "rationale": "Contracted booking channel/segment recorded at reservation time.",
    },
    {
        "column": "DistributionChannel",
        "regression": "used",
        "classification": "excluded_design",
        "clustering": "excluded_design",
        "rationale": "Used in ADR pricing; dropped in classification/clustering due to high overlap with MarketSegment.",
    },
    {
        "column": "ReservedRoomType",
        "regression": "used",
        "classification": "used",
        "clustering": "excluded_design",
        "rationale": "Room category requested by guest at booking time — primary legitimate driver of ADR (unlike AssignedRoomType).",
    },
    {
        "column": "IsRepeatedGuest",
        "regression": "used",
        "classification": "used",
        "clustering": "used",
        "rationale": "Historical PMS profile flag known prior to the current booking.",
    },
    {
        "column": "PreviousCancellations",
        "regression": "used",
        "classification": "used",
        "clustering": "excluded_design",
        "rationale": "Prior customer history strictly predating the current reservation.",
    },
    {
        "column": "PreviousBookingsNotCanceled",
        "regression": "excluded_design",
        "classification": "used",
        "clustering": "excluded_design",
        "rationale": "Prior completed stays predating current booking; strong loyalty signal for cancellation risk.",
    },
    {
        "column": "Agent",
        "regression": "used",
        "classification": "excluded_design",
        "clustering": "excluded_design",
        "rationale": "Booking agency ID recorded at creation; grouped with min_frequency thresholding in regression.",
    },
    {
        "column": "CustomerType",
        "regression": "used",
        "classification": "used",
        "clustering": "used",
        "rationale": "Reservation classification (Transient, Transient-Party, Contract, Group) known at creation.",
    },
    {
        "column": "RequiredCarParkingSpaces",
        "regression": "excluded_design",
        "classification": "excluded_design",
        "clustering": "engineered",
        "rationale": "Converted to binary HasParking for guest segmentation; omitted in regression/classification.",
    },
    {
        "column": "TotalOfSpecialRequests",
        "regression": "excluded_design",
        "classification": "excluded_design",
        "clustering": "used",
        "rationale": "Special requests can be added after initial booking up to check-in, so excluded from strict booking-entry supervised models; included in behavioral clustering.",
    },
]


TASK_FORBIDDEN_COLUMNS: dict[str, list[str]] = {
    "regression": [
        "ADR", "IsCanceled", "ReservationStatus", "ReservationStatusDate",
        "AssignedRoomType", "BookingChanges", "DaysInWaitingList",
    ],
    "classification": [
        "IsCanceled", "ReservationStatus", "ReservationStatusDate",
        "AssignedRoomType", "BookingChanges", "DaysInWaitingList", "DepositType",
    ],
    "clustering": [
        "IsCanceled", "ReservationStatus", "ReservationStatusDate",
        "AssignedRoomType", "BookingChanges", "DaysInWaitingList", "DepositType",
    ],
}


# Preset realistic bookings for the single-booking Predict form.
BOOKING_PRESETS: dict[str, dict[str, Any]] = {
    "Summer Couple Leisure (Online TA)": {
        "LeadTime": 85,
        "ArrivalDateYear": 2016,
        "ArrivalDateMonth": "August",
        "ArrivalDateWeekNumber": 33,
        "ArrivalDateDayOfMonth": 14,
        "StaysInWeekendNights": 2,
        "StaysInWeekNights": 3,
        "Adults": 2,
        "Children": 0,
        "Babies": 0,
        "Meal": "BB",
        "Country": "FRA",
        "MarketSegment": "Online TA",
        "DistributionChannel": "TA/TO",
        "IsRepeatedGuest": 0,
        "PreviousCancellations": 0,
        "PreviousBookingsNotCanceled": 0,
        "ReservedRoomType": "D",
        "DepositType": "No Deposit",
        "Agent": "9",
        "CustomerType": "Transient",
        "ADR": 145.0,
        "RequiredCarParkingSpaces": 0,
        "TotalOfSpecialRequests": 2,
    },
    "Last-Minute Business Solo (Corporate)": {
        "LeadTime": 4,
        "ArrivalDateYear": 2017,
        "ArrivalDateMonth": "March",
        "ArrivalDateWeekNumber": 11,
        "ArrivalDateDayOfMonth": 15,
        "StaysInWeekendNights": 0,
        "StaysInWeekNights": 2,
        "Adults": 1,
        "Children": 0,
        "Babies": 0,
        "Meal": "BB",
        "Country": "PRT",
        "MarketSegment": "Corporate",
        "DistributionChannel": "Corporate",
        "IsRepeatedGuest": 1,
        "PreviousCancellations": 0,
        "PreviousBookingsNotCanceled": 5,
        "ReservedRoomType": "A",
        "DepositType": "No Deposit",
        "Agent": "null",
        "CustomerType": "Transient",
        "ADR": 95.0,
        "RequiredCarParkingSpaces": 0,
        "TotalOfSpecialRequests": 1,
    },
    "Family Holiday Suite (Direct)": {
        "LeadTime": 120,
        "ArrivalDateYear": 2017,
        "ArrivalDateMonth": "July",
        "ArrivalDateWeekNumber": 29,
        "ArrivalDateDayOfMonth": 20,
        "StaysInWeekendNights": 2,
        "StaysInWeekNights": 5,
        "Adults": 2,
        "Children": 2,
        "Babies": 0,
        "Meal": "HB",
        "Country": "GBR",
        "MarketSegment": "Direct",
        "DistributionChannel": "Direct",
        "IsRepeatedGuest": 0,
        "PreviousCancellations": 0,
        "PreviousBookingsNotCanceled": 0,
        "ReservedRoomType": "F",
        "DepositType": "No Deposit",
        "Agent": "14",
        "CustomerType": "Transient",
        "ADR": 210.0,
        "RequiredCarParkingSpaces": 1,
        "TotalOfSpecialRequests": 3,
    },
    "High-Risk Long-Lead Group Booking": {
        "LeadTime": 290,
        "ArrivalDateYear": 2016,
        "ArrivalDateMonth": "October",
        "ArrivalDateWeekNumber": 42,
        "ArrivalDateDayOfMonth": 12,
        "StaysInWeekendNights": 1,
        "StaysInWeekNights": 2,
        "Adults": 2,
        "Children": 0,
        "Babies": 0,
        "Meal": "BB",
        "Country": "PRT",
        "MarketSegment": "Groups",
        "DistributionChannel": "TA/TO",
        "IsRepeatedGuest": 0,
        "PreviousCancellations": 2,
        "PreviousBookingsNotCanceled": 0,
        "ReservedRoomType": "A",
        "DepositType": "No Deposit",
        "Agent": "1",
        "CustomerType": "Transient-Party",
        "ADR": 80.0,
        "RequiredCarParkingSpaces": 0,
        "TotalOfSpecialRequests": 0,
    },
}
