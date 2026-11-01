"""Reakcje rynku wokół zdarzeń (format długi) oraz panele czasowe (dzienny, handlowy, godzinowy).

Zasady (bez informacji z przyszłości, bez analizy treści):
- P(t) = cena OTWARCIA ostatniej świecy 5-min rozpoczętej <= t; gdy starsza niż 2 kroki (noc, weekend) -> NaN,
- horyzonty < 1d: koniec okna musi mieć świeżą cenę, inaczej NaN (żadnych "zwrotów weekendowych" pod etykietą t+5m),
- horyzonty >= 1d: gdy koniec wypada w przerwie, bierzemy pierwszą cenę po nim (do 3 dni); kontrola ma tę samą strukturę,
- kontrola = losowe momenty z tego samego (dzień tygodnia, godzina NY), poza oknem +-exclude od zdarzeń,
- zdarzenia o precyzji dziennej mają horyzonty < 1d wyzerowane (NaN); ich P(t) to ostatnia cena PRZED dniem
  zdarzenia (do 4 dni wstecz), więc ogłoszenie z soboty mierzy ruch od piątku do poniedziałku, a nie daje NaN.

Panel dzienny używa JEDNEGO klucza dnia dla wszystkich źródeł: eff_day (data NY, od 16:00 NY następny dzień).
Zwrot dnia d to zmiana ceny od ostatniej świecy dnia poprzedniego do ostatniej świecy dnia d (w tym samym kluczu),
więc wiadomość opublikowana po 16:00 NY trafia do tego samego dnia, w którym rynek mógł na nią zareagować.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import DAY, eff_day
from cleaning_data.horizons import HorizonTools
from .loaders import tag_of

LONG_COLS = ["event_id", "instrument", "horizon", "ret", "abs_ret", "pre_ret", "ctl_ret", "ctl_abs",
             "excess_ret", "excess_abs", "open_at_t", "ctl_n", "ctl_unique_share"]


# --------------------------------------------------------------------------- rynek wokół zdarzeń
def _fwd(times, price: pd.Series, horizons: dict) -> pd.DataFrame:
    short = {l: h for l, h in horizons.items() if pd.Timedelta(h) < DAY}
    longh = {l: h for l, h in horizons.items() if pd.Timedelta(h) >= DAY}
    cols = {}
    for hz, fill in ((short, "last"), (longh, "next")):
        if hz:
            r = HorizonTools.returns_at(times, price, hz, end_fill=fill).reset_index(drop=True)
            cols.update({l: r[l].to_numpy() for l in hz})
    return pd.DataFrame({l: cols[l] for l in horizons})


def _fwd_day(times, price: pd.Series, horizons: dict, anchor_stale: str = "4D", max_wait: str = "3D") -> pd.DataFrame:
    """Zdarzenia z dokładnością dnia (t = północ UTC dnia zdarzenia): P(t) = ostatnia cena <= t do `anchor_stale` wstecz
    (cena sprzed dnia zdarzenia, także gdy dzień wypada w weekend); koniec horyzontu jak w _fwd dla h >= 1d."""
    pi, px, step = HorizonTools._prep_price(price)
    t = pd.DatetimeIndex(pd.to_datetime(pd.Series(times).reset_index(drop=True), utc=True))
    out = pd.DataFrame(np.nan, index=range(len(t)), columns=list(horizons))
    if len(pi) == 0 or len(t) == 0:
        return out
    e = HorizonTools._ns(t)
    p0 = HorizonTools._last(e, pi, px, int(pd.Timedelta(anchor_stale).value))
    tol, wait = 2 * step, int(pd.Timedelta(max_wait).value)
    for l, h in horizons.items():
        hn = int(pd.Timedelta(h).value)
        if hn < DAY.value:
            continue
        p1 = HorizonTools._last(e + hn, pi, px, tol)
        miss = np.isnan(p1)
        if miss.any():
            p1[miss] = HorizonTools._next(e[miss] + hn, pi, px, wait)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[l] = np.log(p1 / p0)
    return out


def _pre(times, price: pd.Series, horizons: dict, day_precision: bool = False) -> pd.DataFrame:
    """log(P(t)/P(t-h)): ruch tuż PRZED zdarzeniem (kontrola wycieku informacji i trendu poprzedzającego).
    Obie ceny muszą być świeże; dla zdarzeń dziennych P(t) jest ceną sprzed dnia zdarzenia (jak w _fwd_day)."""
    t = pd.DatetimeIndex(pd.to_datetime(pd.Series(times).reset_index(drop=True), utc=True))
    if not day_precision:
        cols = {}
        for l, h in horizons.items():
            hd = pd.Timedelta(h)
            r = HorizonTools.returns_at(t - hd, price, {l: hd}, end_fill="last")
            cols[l] = r[l].to_numpy()
        return pd.DataFrame(cols, index=range(len(t)))
    pi, px, step = HorizonTools._prep_price(price)
    out = pd.DataFrame(np.nan, index=range(len(t)), columns=list(horizons))
    if len(pi) == 0 or len(t) == 0:
        return out
    e = HorizonTools._ns(t)
    p1 = HorizonTools._last(e, pi, px, int(pd.Timedelta("4D").value))
    for l, h in horizons.items():
        hn = int(pd.Timedelta(h).value)
        if hn < DAY.value:
            continue
        p0 = HorizonTools._last(e - hn, pi, px, int(pd.Timedelta("4D").value))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[l] = np.log(p1 / p0)
    return out


def _one_instrument(ev: pd.DataFrame, inst: str, price: pd.Series, horizons: dict, bounds, k: int, seed: int, exclude: str,
                    day_precision: bool) -> pd.DataFrame:
    n = len(ev)
    t = pd.DatetimeIndex(ev["t"])
    fwd = _fwd_day if day_precision else _fwd
    R, P = fwd(t, price, horizons), _pre(t, price, horizons, day_precision=day_precision)
    pl = HorizonTools.matched_placebo(t, price, bounds, k=k, seed=seed, exclude=exclude)
    if len(pl):
        rc = fwd(pl["time"], price, horizons)
        pos = pl["event_pos"].to_numpy()
        c_sgn = rc.groupby(pos).mean().reindex(range(n))
        c_abs = rc.abs().groupby(pos).mean().reindex(range(n))
        c_n = np.bincount(pos, minlength=n)[:n]
        uniq = pl["time"].nunique() / len(pl)
    else:
        c_sgn = c_abs = pd.DataFrame(np.nan, index=range(n), columns=list(horizons))
        c_n, uniq = np.zeros(n, int), np.nan
    pi, px, step = HorizonTools._prep_price(price)
    fresh = ~np.isnan(HorizonTools._last(HorizonTools._ns(t), pi, px, 2 * step))
    out = []
    for lab, h in horizons.items():
        blind = day_precision and pd.Timedelta(h) < DAY
        r = np.full(n, np.nan) if blind else R[lab].to_numpy()
        p = np.full(n, np.nan) if blind else P[lab].to_numpy()
        cs = np.full(n, np.nan) if blind else c_sgn[lab].to_numpy()
        ca = np.full(n, np.nan) if blind else c_abs[lab].to_numpy()
        out.append(pd.DataFrame({"event_id": ev["event_id"].to_numpy(), "instrument": inst, "horizon": lab, "ret": r,
                                 "abs_ret": np.abs(r), "pre_ret": p, "ctl_ret": cs, "ctl_abs": ca,
                                 "excess_ret": r - cs, "excess_abs": np.abs(r) - ca,
                                 "open_at_t": False if day_precision else fresh, "ctl_n": c_n,
                                 "ctl_unique_share": uniq}))
    return pd.concat(out, ignore_index=True)


def event_market_long(events: pd.DataFrame, prices: dict, horizons: dict, bounds, k: int = 20, seed: int = 0,
                      exclude_exact: str = "2h", exclude_day: str = "1D") -> pd.DataFrame:
    """Format długi: wiersz = zdarzenie x instrument x horyzont. `prices`: etykieta -> Series cen otwarcia (indeks UTC).
    Kolumna ctl_n = ile momentów kontrolnych faktycznie wylosowano dla zdarzenia (0 => brak kontroli, nadwyżka NaN)."""
    parts = []
    # Placebo osobno dla KAŻDEGO rodzaju zdarzeń: wykluczamy tylko momenty blisko zdarzeń tego samego rodzaju.
    # Wspólne wykluczanie (np. +-1 dzień wokół wszystkich dat GTA i FR naraz) zostawia ~10% świec z kilku tygodni,
    # kontrole się powtarzają i obciążają nadwyżkę (sprawdzone na danych syntetycznych bez efektu).
    for kind in sorted(events["kind"].unique()):
        ev = events[events["kind"] == kind].sort_values(["t", "event_id"]).reset_index(drop=True)
        prec = ev["precision"].iat[0]
        excl = exclude_day if prec == "day" else exclude_exact
        for inst, price in prices.items():
            parts.append(_one_instrument(ev, inst, price.dropna(), horizons, bounds, k, seed, excl, prec == "day"))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=LONG_COLS)
    return out


# --------------------------------------------------------------------------- agregaty rynkowe
def _bar_day(idx) -> pd.Series:
    """Klucz dnia świecy = eff_day jej początku (świece od 16:00 NY należą do następnego dnia)."""
    return pd.Series(eff_day(pd.Series(idx)), index=idx)


def _intra(c: pd.Series) -> pd.Series:
    """Zwrot log między kolejnymi świecami; NaN po przerwie (noc, weekend), żeby luki nie udawały zwrotu wewnątrz sesji."""
    lr = np.log(c).diff()
    dt = c.index.to_series().diff()
    return lr.where(dt <= 2 * dt.median())


def market_daily(close: pd.Series, days: pd.DatetimeIndex) -> pd.DataFrame:
    """ret = log(ostatnia cena dnia / ostatnia cena poprzedniego dnia w kluczu eff_day), z przerwą weekendową;
    rv = suma kwadratów zwrotów 5-min wewnątrz sesji. Dane powinny sięgać kilka dni przed pierwszym dniem panelu."""
    c = close.dropna().sort_index()
    key = _bar_day(c.index)
    last = c.groupby(key).last()
    ret = np.log(last).diff()
    rv = (_intra(c) ** 2).groupby(key).sum(min_count=1)
    return pd.DataFrame({"ret": ret, "absret": ret.abs(), "rv": rv}).reindex(days)


def market_hourly(close: pd.Series, hours: pd.DatetimeIndex) -> pd.DataFrame:
    c = close.dropna().sort_index()
    hr = c.index.floor("h")
    lr = _intra(c)
    return pd.DataFrame({"ret": lr.groupby(hr).sum(min_count=1), "rv": (lr ** 2).groupby(hr).sum(min_count=1)}).reindex(hours)


# --------------------------------------------------------------------------- panele
def _span(bounds):
    lo = min(a for a, _ in bounds).tz_localize(None)
    hi = max(b for _, b in bounds).tz_localize(None)
    return lo, hi


def finalize(P: pd.DataFrame) -> pd.DataFrame:
    """Stawki i udziały liczone z sum (poprawne także po agregacji do dni handlowych)."""
    P = P.copy()
    z = lambda s: s.replace(0, np.nan)
    if {"tw_topic_n", "tw_n"} <= set(P):
        P["tw_share"] = P["tw_topic_n"] / z(P["tw_n"])
    if "fn_n" in P and "fn_all_n" in P:
        P["fn_sector_share"] = P["fn_sector_n"] / z(P["fn_all_n"]) if "fn_sector_n" in P else np.nan
    if "gkg_files_n" in P:
        P["gkg_rate"] = P["gkg_topic_n"] / z(P["gkg_files_n"])
    return P


def build_daily_panel(bounds, tr_all: pd.DataFrame, posts: pd.DataFrame, fn_daily, gkg, gkg_files, events: pd.DataFrame,
                      closes: dict) -> pd.DataFrame:
    """Panel dzienny w kluczu eff_day. tr_all = wszystkie oryginalne posty (mianownik udziału), posts = posty o cłach.
    Dni bez źródła to 0 w licznikach, ale flagi pokrycia (gkg_files_n, fn_all_n) pozwalają odróżnić 'brak newsów' od 'brak danych'."""
    lo, hi = _span(bounds)
    days = pd.date_range(lo, hi - pd.Timedelta(days=1), freq="D", name="day")
    cnt = lambda s: pd.Series(s).value_counts().reindex(days, fill_value=0).astype(float)
    P = pd.DataFrame(index=days)
    P["tw_n"] = cnt(pd.Series(eff_day(tr_all["time_utc"])))
    P["tw_topic_n"] = cnt(posts["eff_day"])
    if fn_daily is not None and len(fn_daily):
        P = P.join(fn_daily.reindex(days).fillna(0.0))
    for kind, pre in (("gta_announced", "gta_ann"), ("gta_implemented", "gta_impl"), ("fr", "fr"), ("fr_effective", "fr_eff")):
        e = events[events["kind"] == kind]
        P[f"{pre}_n"] = cnt(e["eff_day"])
        if kind.startswith("gta"):
            P[f"{pre}_products_n"] = e.groupby("eff_day")["weight"].sum().reindex(days, fill_value=0).astype(float)
    if gkg is not None and gkg_files is not None and len(gkg_files):
        P["gkg_files_n"] = cnt(pd.Series(eff_day(pd.Series(gkg_files))))
        P["gkg_topic_n"] = cnt(pd.Series(eff_day(gkg["time_utc"])))
    for label, close in closes.items():
        m = market_daily(close, days)
        for c in m:
            P[f"{tag_of(label)}_{c}"] = m[c]
    return finalize(P)


def build_hourly_panel(bounds, posts: pd.DataFrame, gkg, gkg_files, closes: dict, tr_all: pd.DataFrame | None = None) -> pd.DataFrame:
    """Panel godzinowy w UTC. Godziny bez paczki GKG mają gkg_files_n = 0 (GKG to próbka), a bez notowań NaN w rynku."""
    lo, hi = _span(bounds)
    idx = pd.date_range(lo.tz_localize("UTC"), (hi - pd.Timedelta(hours=1)).tz_localize("UTC"), freq="h", name="hour_utc")
    cnt = lambda s: pd.Series(s).value_counts().reindex(idx, fill_value=0).astype(float)
    P = pd.DataFrame(index=idx)
    if tr_all is not None:
        P["tw_n"] = cnt(tr_all["time_utc"].dt.floor("h"))
    P["tw_topic_n"] = cnt(posts["time_utc"].dt.floor("h"))
    if gkg is not None and gkg_files is not None and len(gkg_files):
        P["gkg_files_n"] = cnt(pd.Series(gkg_files.floor("h")))
        P["gkg_topic_n"] = cnt(gkg["time_utc"].dt.floor("h"))
        P["gkg_rate"] = P["gkg_topic_n"] / P["gkg_files_n"].replace(0, np.nan)
    for label, close in closes.items():
        m = market_hourly(close, idx)
        P[f"{tag_of(label)}_ret"], P[f"{tag_of(label)}_rv"] = m["ret"], m["rv"]
    ny = idx.tz_convert("America/New_York")
    mins = ny.hour * 60 + ny.minute
    P["us_session"] = ((ny.dayofweek < 5) & (mins >= 570) & (mins < 960)).astype(int)
    P["weekend"] = (ny.dayofweek >= 5).astype(int)
    return P


def trading_pos(days, td) -> np.ndarray:
    """Pozycja pierwszego dnia handlowego >= dzień (weekend/święto -> następny dzień handlowy)."""
    return np.searchsorted(pd.DatetimeIndex(td).values, pd.DatetimeIndex(days).values, side="left")


def to_trading_days(P: pd.DataFrame) -> pd.DataFrame:
    """Dni nie-handlowe (sobota, niedziela, święta bez notowań) wpadają do następnego dnia handlowego.
    Liczniki (_n), zwroty logarytmiczne (_ret) i zmienność zrealizowaną (_rv) SUMUJEMY (zwroty log są addytywne);
    |zwrot| liczymy od nowa z sumy; udziały i stawki liczymy od nowa z sum.
    Dzień handlowy = dzień Pon-Pt z notowaniem któregokolwiek instrumentu. Dni po ostatnim dniu handlowym wypadają."""
    mk = [c for c in P if c.endswith(("_ret", "_rv"))]
    if not mk:
        raise ValueError("Brak kolumn rynkowych (_ret/_rv): nie da się wyznaczyć dni handlowych.")
    has = P[mk].notna().any(axis=1) & (P.index.dayofweek < 5)
    td = pd.DatetimeIndex(P.index[has])
    j = trading_pos(P.index, td)
    keep = j < len(td)
    grp = pd.Series(td[j[keep]], index=P.index[keep])
    A = P[keep]
    cols = [c for c in A if c.endswith(("_n", "_ret", "_rv"))]
    out = A[cols].groupby(grp).sum(min_count=1).reindex(td)
    for c in [c for c in out if c.endswith("_ret")]:
        out[c.replace("_ret", "_absret")] = out[c].abs()
    out.index.name = "day"
    return finalize(out)