"""Schema-driven reservation input form with presets, calendar date sync, and JSON/CSV paste (emoji-free)."""
from __future__ import annotations

import datetime as dt
import io
import json
from typing import Any

import pandas as pd
import streamlit as st

from app.components.style import glass_container
from src.data.schema import BOOKING_PRESETS, H2_COLUMNS, MONTHS
from src.data.validation import ensure_task_splits, normalize_columns


def _apply_booking_dict(data: dict[str, Any]) -> None:
    cur = dict(st.session_state.get("predict_booking", {}))
    cur.update(data)
    st.session_state["predict_booking"] = cur


def render_booking_form(key_prefix: str = "bf") -> dict[str, Any]:
    if "predict_booking" not in st.session_state:
        st.session_state["predict_booking"] = dict(next(iter(BOOKING_PRESETS.values())))

    # Baseline-aligned preset & quick-fill toolbar (zero spacer hacks)
    p1, p2, p3, p4 = st.columns([2.2, 0.95, 1.25, 1.15], vertical_alignment="bottom")
    with p1:
        preset_choice = st.selectbox(
            "Load Booking Archetype Preset",
            options=["(Keep current values)", *BOOKING_PRESETS.keys()],
            key=f"{key_prefix}_preset_sel",
        )
    with p2:
        apply_disabled = preset_choice == "(Keep current values)"
        if st.button("Apply Preset", key=f"{key_prefix}_apply_preset", icon=":material/check:", disabled=apply_disabled, width="stretch"):
            _apply_booking_dict(BOOKING_PRESETS[preset_choice])
            st.rerun()
    with p3:
        if st.button("Sample Validation Row", key=f"{key_prefix}_rand_val", icon=":material/shuffle:", width="stretch"):
            _, val_df, _ = ensure_task_splits("regression")
            sample_row = val_df.sample(n=1).iloc[0].to_dict()
            _apply_booking_dict(sample_row)
            st.rerun()
    with p4:
        with st.popover("Paste JSON / CSV", icon=":material/content_paste:", width="stretch"):
            raw_txt = st.text_area(
                "Paste a JSON object or 2-line CSV (header + values)",
                height=120,
                key=f"{key_prefix}_paste_txt",
            )
            if st.button("Parse & Fill Form", key=f"{key_prefix}_parse_btn", type="primary", width="stretch") and raw_txt.strip():
                try:
                    txt = raw_txt.strip()
                    if txt.startswith("{"):
                        parsed = json.loads(txt)
                        df_one = pd.DataFrame([parsed])
                    else:
                        df_one = pd.read_csv(io.StringIO(txt))
                    norm, _, _ = normalize_columns(df_one)
                    _apply_booking_dict(norm.iloc[0].to_dict())
                    st.success("Loaded booking fields.", icon=":material/check_circle:")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not parse input: {exc}", icon=":material/error:")

    cur = dict(st.session_state["predict_booking"])

    m_name = str(cur.get("ArrivalDateMonth", "July"))
    m_num = (MONTHS.index(m_name) + 1) if m_name in MONTHS else 7
    yr = int(cur.get("ArrivalDateYear", 2016))
    dy = min(int(cur.get("ArrivalDateDayOfMonth", 15)), 28)
    try:
        init_date = dt.date(yr, m_num, dy)
    except ValueError:
        init_date = dt.date(2016, 7, 15)

    c_stay, c_chan, c_hist = st.columns(3, gap="medium")

    with c_stay:
        with glass_container("solid"):
            st.markdown('<div class="section-kicker">Stay & Guest Party</div>', unsafe_allow_html=True)
            arr_date = st.date_input(
                "Scheduled Arrival Date",
                value=init_date,
                min_value=dt.date(2015, 1, 1),
                max_value=dt.date(2030, 12, 31),
                key=f"{key_prefix}_arr_date",
            )
            lead_time = st.number_input(
                "Booking Lead Time (days)",
                min_value=0, max_value=730, value=int(cur.get("LeadTime", 65)), step=1,
                help=H2_COLUMNS["LeadTime"].meaning,
            )
            n1, n2 = st.columns(2)
            with n1:
                week_nights = st.number_input(
                    "Weeknight Stays",
                    min_value=0, max_value=60, value=int(cur.get("StaysInWeekNights", 2)), step=1,
                    help="Monday through Friday nights (`StaysInWeekNights`).",
                )
            with n2:
                wknd_nights = st.number_input(
                    "Weekend Stays",
                    min_value=0, max_value=30, value=int(cur.get("StaysInWeekendNights", 1)), step=1,
                    help="Saturday and Sunday nights (`StaysInWeekendNights`).",
                )
            g1, g2, g3 = st.columns(3)
            with g1:
                adults = st.number_input("Adults", min_value=1, max_value=12, value=max(1, int(cur.get("Adults", 2))))
            with g2:
                children = st.number_input("Children", min_value=0, max_value=10, value=int(cur.get("Children", 0) or 0))
            with g3:
                babies = st.number_input("Infants", min_value=0, max_value=6, value=int(cur.get("Babies", cur.get("HasBabies", 0)) or 0))

    with c_chan:
        with glass_container("solid"):
            st.markdown('<div class="section-kicker">Room & Commercial Channel</div>', unsafe_allow_html=True)
            room_cats = list(H2_COLUMNS["ReservedRoomType"].categories)
            cur_room = str(cur.get("ReservedRoomType", "A"))
            room_type = st.selectbox(
                "Reserved Room Category",
                options=room_cats,
                index=room_cats.index(cur_room) if cur_room in room_cats else 0,
                help=H2_COLUMNS["ReservedRoomType"].meaning,
            )
            meal_cats = list(H2_COLUMNS["Meal"].categories)
            cur_meal = str(cur.get("Meal", "BB"))
            meal = st.selectbox(
                "Meal Plan",
                options=meal_cats,
                index=meal_cats.index(cur_meal) if cur_meal in meal_cats else 0,
                help=H2_COLUMNS["Meal"].meaning,
            )
            seg_cats = list(H2_COLUMNS["MarketSegment"].categories)
            cur_seg = str(cur.get("MarketSegment", "Online TA"))
            market_seg = st.selectbox(
                "Market Segment",
                options=seg_cats,
                index=seg_cats.index(cur_seg) if cur_seg in seg_cats else 0,
            )
            c_dc, c_ct = st.columns(2)
            with c_dc:
                dist_cats = list(H2_COLUMNS["DistributionChannel"].categories)
                cur_dist = str(cur.get("DistributionChannel", "TA/TO"))
                dist_chan = st.selectbox(
                    "Distribution Channel",
                    options=dist_cats,
                    index=dist_cats.index(cur_dist) if cur_dist in dist_cats else 0,
                )
            with c_ct:
                cust_cats = list(H2_COLUMNS["CustomerType"].categories)
                cur_cust = str(cur.get("CustomerType", "Transient"))
                cust_type = st.selectbox(
                    "Customer Profile",
                    options=cust_cats,
                    index=cust_cats.index(cur_cust) if cur_cust in cust_cats else 0,
                )

    with c_hist:
        with glass_container("solid"):
            st.markdown('<div class="section-kicker">Deposit, Agency & History</div>', unsafe_allow_html=True)
            d1, d2 = st.columns([1.25, 1.0])
            with d1:
                dep_cats = list(H2_COLUMNS["DepositType"].categories)
                cur_dep = str(cur.get("DepositType", "No Deposit"))
                dep_type = st.selectbox(
                    "Deposit Policy (ADR only)",
                    options=dep_cats,
                    index=dep_cats.index(cur_dep) if cur_dep in dep_cats else 0,
                    help="Used by ADR pricing models; excluded from cancellation/clustering to prevent PMS outcome-proxy leakage.",
                )
            with d2:
                agent_val = st.text_input(
                    "Booking Agent ID",
                    value=str(cur.get("Agent", "9")) if pd.notna(cur.get("Agent")) else "null",
                    help="Travel agency identifier or 'null' for direct reservations.",
                )
            is_rep = st.selectbox(
                "Guest Loyalty Status",
                options=[0, 1],
                index=1 if int(cur.get("IsRepeatedGuest", 0) or 0) == 1 else 0,
                format_func=lambda v: "Returning Guest (1)" if v == 1 else "First-Time Guest (0)",
            )
            h1, h2 = st.columns(2)
            with h1:
                p_canc = st.number_input(
                    "Prior Cancellations",
                    min_value=0, max_value=50, value=int(cur.get("PreviousCancellations", 0) or 0),
                )
            with h2:
                p_ok = st.number_input(
                    "Prior Completed Stays",
                    min_value=0, max_value=100, value=int(cur.get("PreviousBookingsNotCanceled", 0) or 0),
                )
            r1, r2 = st.columns(2)
            with r1:
                parking = st.number_input(
                    "Parking Spaces",
                    min_value=0, max_value=5, value=int(cur.get("RequiredCarParkingSpaces", 0) or 0),
                )
            with r2:
                special_req = st.number_input(
                    "Special Requests",
                    min_value=0, max_value=10, value=int(cur.get("TotalOfSpecialRequests", 1) or 0),
                )

    booking = {
        "LeadTime": int(lead_time),
        "ArrivalDateYear": int(arr_date.year),
        "ArrivalDateMonth": MONTHS[arr_date.month - 1],
        "ArrivalDateWeekNumber": int(arr_date.isocalendar().week),
        "ArrivalDateDayOfMonth": int(arr_date.day),
        "StaysInWeekendNights": int(wknd_nights),
        "StaysInWeekNights": int(week_nights),
        "Adults": int(adults),
        "Children": int(children),
        "Babies": int(babies),
        "HasBabies": int(1 if babies > 0 else 0),
        "Meal": meal,
        "Country": str(cur.get("Country", "PRT")),
        "MarketSegment": market_seg,
        "DistributionChannel": dist_chan,
        "IsRepeatedGuest": int(is_rep),
        "PreviousCancellations": int(p_canc),
        "PreviousBookingsNotCanceled": int(p_ok),
        "ReservedRoomType": room_type,
        "DepositType": dep_type,
        "Agent": agent_val.strip() or "null",
        "CustomerType": cust_type,
        "RequiredCarParkingSpaces": int(parking),
        "TotalOfSpecialRequests": int(special_req),
    }
    st.session_state["predict_booking"] = booking

    st.caption(
        ":material/lock: **Leakage Protection Active:** Post-booking and outcome variables "
        "(`ReservationStatus`, `ReservationStatusDate`, `AssignedRoomType`, `BookingChanges`, `DaysInWaitingList`) "
        "are excluded because they are unknown when a reservation is initially placed."
    )
    return booking
