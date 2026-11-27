"""Hipoteza wyjściowa: rynek reaguje na OFICJALNE działania celne (ogłoszenie / wejście w życie), a Trump tylko je nagłaśnia.
Hipoteza przeformułowana po wynikach: rynek reaguje na PIERWSZĄ WIARYGODNĄ INFORMACJĘ o cłach, niezależnie od kanału
(rząd albo tweet Trumpa); potwierdzenie i wejście w życie są już wycenione. Wnioski: analysis/out/SKROT_HIPOTEZA.md.

Uruchom z katalogu projektu:  python analysis/hypothesis_official_vs_trump.py
Wyniki: analysis/out/hip_*.csv, analysis/out/fig_hip_*.png, analysis/out/hip_summary.txt

Rynki: S&P 500 (SPY, zwrot dzienny) i chińskie ETF-y (koszyk 5 ETF, zwrot nadzwyczajny vs SPY, z china_etfs.py).
Wszystko dziennie, w kalendarzu NYSE (ten sam dla obu rynków).

Dwie analizy:
  A) SYSTEMATYCZNA, wszystkie sesje 2018-2019. Typ dnia: oficjalne ogłoszenie (GTA + Federal Register + kronika),
     wejście w życie, intensywny dzień tweetów Trumpa o cłach. Miara: wielkość ruchu |r| (kierunek oficjalnych
     zdarzeń z GTA/FR nie jest znany). Regresja z błędami Neweya-Westa.
  B) KRONIKA kluczowych epizodów USA-Chiny (ręcznie, ok. 20). Dla każdego epizodu: dzień sygnału Trumpa, dzień
     oficjalnego ogłoszenia, dzień wejścia w życie. Miara: zwrot w oknie 2 sesji [d0, d0+1] (odporny na niepewną
     godzinę ogłoszenia), ze znakiem względem kierunku epizodu, oraz |zwrot|.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import annotation_market_study as ams  # noqa: E402
from cleaning_data.base import Config  # noqa: E402
from cleaning_data.china_etfs import ChinaETFs  # noqa: E402
from cleaning_data.trump_truth import TrumpExplorer  # noqa: E402

OUT = ams.OUT
LOG = ams.LOG
log, section = ams.log, ams.section
NY = "America/New_York"
RNG = np.random.default_rng(7)

MARKETS = {"SPY_ret": "S&P 500 (SPY)", "CHN_ALL_ar": "Chińskie ETF-y (AR vs SPY)"}

# ---------------------------------------------------------------------------------------------------------------
# KRONIKA. Czas w strefie Nowego Jorku. kier: +1 eskalacja, -1 deeskalacja.
# trump  = pierwszy publiczny sygnał Trumpa (tweet lub jego oświadczenie), zweryfikowany z data/trump_tweets.csv
# oficj  = oficjalne ogłoszenie rządu (USTR / proklamacja / Biały Dom / MOFCOM / Federal Register)
# wejscie = wejście ceł w życie
# "~" w uwagach = godzina przybliżona; dlatego główną miarą jest okno 2 sesji.
EPISODES = [
    ("E01", "Stal i aluminium (sekcja 232)", +1, "2018-03-01 07:12", "2018-03-08 15:30", "2018-03-23 00:01", "tweet 1.03; proklamacja 8.03"),
    ("E02", "Memorandum 301 wobec Chin", +1, None, "2018-03-22 12:30", None, "brak wcześniejszego tweeta"),
    ("E03", "Lista 301: 50 mld USD (projekt)", +1, "2018-04-04 07:22", "2018-04-03 17:00", "2018-07-06 00:01", "USTR po sesji 3.04; tweet rano 4.04"),
    ("E04", "Groźba ceł na kolejne 100 mld USD", +1, "2018-04-05 19:00", None, None, "oświadczenie Trumpa, bez dokumentu"),
    ("E05", "Lista 34 mld USD: decyzja ostateczna", +1, None, "2018-06-15 08:00", "2018-07-06 00:01", "~godzina"),
    ("E06", "Lista 200 mld USD (projekt)", +1, None, "2018-07-10 18:30", None, "USTR wieczorem"),
    ("E07", "Lista 16 mld USD: decyzja", +1, None, "2018-08-07 17:00", "2018-08-23 00:01", "~godzina"),
    ("E08", "Lista 200 mld USD: decyzja (10%)", +1, "2018-09-17 06:11", "2018-09-17 18:30", "2018-09-24 00:01", "tweet rano, ogłoszenie po sesji"),
    ("E09", "Rozejm G20 w Buenos Aires", -1, "2018-12-02 23:00", "2018-12-01 20:00", None, "sobota; komunikat Białego Domu"),
    ("E10", "Odroczenie podwyżki z 1 marca", -1, "2019-02-24 17:39", "2019-03-05 08:45", None, "oficjalnie: Federal Register"),
    ("E11", "Podwyżka 10% -> 25% na 200 mld USD", +1, "2019-05-05 12:08", "2019-05-09 08:45", "2019-05-10 00:01", "oficjalnie: Federal Register"),
    ("E12", "Lista 4 (300 mld USD): projekt", +1, None, "2019-05-13 18:00", None, "~godzina"),
    ("E13", "10% na 300 mld USD od 1 września", +1, "2019-08-01 13:26", "2019-08-20 08:45", "2019-09-01 00:01", "oficjalnie: Federal Register"),
    ("E14", "USTR odracza część listy do 15 grudnia", -1, None, "2019-08-13 10:00", None, "komunikat USTR"),
    ("E15", "Podwyżki 25->30% i 10->15%", +1, "2019-08-23 17:00", "2019-08-30 08:45", None, "oficjalnie: Federal Register"),
    ("E16", "Odroczenie podwyżki z 1 października", -1, "2019-09-11 19:17", None, None, "tylko tweet"),
    ("E17", "Faza 1: porozumienie wstępne", -1, "2019-10-11 10:15", None, None, "tylko Trump"),
    ("E18", "Faza 1: umowa, odwołanie ceł z 15 grudnia", -1, "2019-12-12 09:35", "2019-12-13 10:30", None, "~godzina komunikatu USTR"),
    ("C1", "Chiny: odwet na 50 mld USD (zapowiedź)", +1, None, "2018-04-04 03:00", "2018-07-06 00:01", "MOFCOM, noc czasu NY"),
    ("C2", "Chiny: odwet na 60 mld USD", +1, None, "2018-09-18 07:00", "2018-09-24 00:01", "~godzina"),
    ("C3", "Chiny: podwyżka ceł odwetowych", +1, None, "2019-05-13 08:00", "2019-06-01 00:01", "~godzina"),
    ("C4", "Chiny: odwet na 75 mld USD", +1, None, "2019-08-23 09:00", "2019-09-01 00:01", "~godzina"),
]
STAGES = {"trump": "Sygnał Trumpa", "oficj": "Oficjalne ogłoszenie", "wejscie": "Wejście w życie"}
COL = {"trump": "#eb6834", "oficj": "#2a78d6", "wejscie": "#1baf7a"}     # paleta zweryfikowana validate_palette.js
MARK = {"trump": "o", "oficj": "s", "wejscie": "^"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

FR_SUBST = re.compile(r"Notice of Action|Modification of Section 301 Action|Implementing Modification|Proposed (Determination|Modification)|"
                      r"Notice of Determination|Actions by the United States", re.I)


# ============================================================================ dane
def trading_day(days: pd.DatetimeIndex, ny_times: pd.Series) -> pd.Series:
    """Czas NY -> sesja NYSE, której zwrot obejmuje informację (po 16:00 lub w weekend -> następna sesja)."""
    t = pd.to_datetime(ny_times)
    d = t.dt.normalize() + pd.to_timedelta((t.dt.hour >= 16).astype(int), unit="D")
    i = days.searchsorted(d.to_numpy())
    return pd.Series(np.where(i < len(days), days[np.clip(i, 0, len(days) - 1)], pd.NaT), index=ny_times.index)


def load_daily(cfg: Config) -> pd.DataFrame:
    R = ChinaETFs(cfg).returns()
    return R.loc["2018-01-01":"2019-12-31", ["SPY_ret", "CHN_ALL_ar", "CHN_TECH_ar", "KWEB_ret"]].dropna()


def trump_counts(days: pd.DatetimeIndex) -> pd.Series:
    tw = pd.read_csv(ROOT / "data" / "trump_tweets.csv", dtype={"id": str})
    tw = tw[(tw["isRetweet"] == "f") & ~tw["text"].str.startswith("RT @")]
    tw["ny"] = pd.to_datetime(tw["date"], utc=True).dt.tz_convert(NY).dt.tz_localize(None)
    tw = tw[(tw["ny"] >= "2017-12-29") & (tw["ny"] < "2020-01-01")]
    t = tw["text"].fillna("")
    rel = t.str.contains(TrumpExplorer.TOPIC_RE) | (t.str.contains(TrumpExplorer.CHINA_RE) & t.str.contains(TrumpExplorer.TRADE_CTX_RE))
    tw = tw[rel]
    td = trading_day(days, tw["ny"])
    return td.value_counts().reindex(days, fill_value=0).rename("trump_n")


def official_days(days: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Dni oficjalnych ogłoszeń i wejść w życie: GTA (duże albo USA-Chiny) + Federal Register (merytoryczne) + kronika."""
    E = pd.read_csv(ROOT / "clean_out" / "merged" / "events.csv")
    E["eff_day"] = pd.to_datetime(E["eff_day"])
    big = (E["weight"].fillna(0) >= 100) | (E["thread"] == "USA-Chiny")
    ann = E[((E["kind"] == "gta_announced") & big) | ((E["kind"] == "fr") & E["title"].fillna("").str.contains(FR_SUBST))]
    imp = E[((E["kind"] == "gta_implemented") & big) | (E["kind"] == "fr_effective")]
    rows = []
    for k, d in (("ann", ann), ("imp", imp)):
        i = days.searchsorted(d["eff_day"].to_numpy())
        for ii, r in zip(i, d.itertuples()):
            if ii < len(days):
                rows.append({"typ": k, "zrodlo": r.kind, "sesja": days[ii], "opis": str(r.title)[:90]})
    K = chronicle(days)
    for r in K.itertuples():
        if pd.notna(r.oficj_day):
            rows.append({"typ": "ann", "zrodlo": "kronika", "sesja": r.oficj_day, "opis": r.opis})
        if pd.notna(r.wejscie_day):
            rows.append({"typ": "imp", "zrodlo": "kronika", "sesja": r.wejscie_day, "opis": r.opis})
    O = pd.DataFrame(rows)
    a = pd.Series(days.isin(O.loc[O["typ"] == "ann", "sesja"]), index=days).astype(int)
    m = pd.Series(days.isin(O.loc[O["typ"] == "imp", "sesja"]), index=days).astype(int)
    return a, m, O


def chronicle(days: pd.DatetimeIndex) -> pd.DataFrame:
    K = pd.DataFrame(EPISODES, columns=["id", "opis", "kier", "trump", "oficj", "wejscie", "uwagi"])
    for s in STAGES:
        K[f"{s}_day"] = trading_day(days, pd.to_datetime(K[s]))
    return K


# ============================================================================ statystyka
def newey_west(y, X, lags=5):
    X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    u = X * e[:, None]
    S = u.T @ u
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    se = np.sqrt(np.diag(V))
    return b, se, 2 * stats.t.sf(np.abs(b / se), n - k)


def boot_mean_ci(x, B=5000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    m = RNG.choice(x, (B, len(x))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def perm_vs_base(x, base, B=10000):
    """p: czy średnia z n dni zdarzeń różni się od średniej z n losowych dni bazowych."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    base = np.asarray(base, float); base = base[np.isfinite(base)]
    if len(x) == 0:
        return np.nan
    sims = RNG.choice(base, (B, len(x))).mean(1)
    return (np.sum(np.abs(sims - base.mean()) >= abs(x.mean() - base.mean())) + 1) / (B + 1)


def window(D: pd.DataFrame, col: str, day, n=2) -> float:
    if pd.isna(day):
        return np.nan
    i = D.index.get_indexer([day])[0]
    if i < 0 or i + n > len(D):
        return np.nan
    return D[col].iloc[i:i + n].sum()


# ============================================================================ main
def main():
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 70)
    LOG.clear()
    cfg = Config()
    D = load_daily(cfg)
    days = D.index
    D["trump_n"] = trump_counts(days)
    D["ann"], D["imp"], O = official_days(days)
    q = D["trump_n"].quantile(0.90)
    D["trump_heavy"] = (D["trump_n"] >= max(q, 3)).astype(int)
    D["ann_lag"] = D["ann"].shift(1).fillna(0)   # ogłoszenie z dnia poprzedniego (niepewna godzina)
    D["imp_lag"] = D["imp"].shift(1).fillna(0)
    near_off = (D["ann"] + D["ann_lag"] + D["ann"].shift(-1).fillna(0) + D["imp"] + D["imp_lag"]) > 0

    section("A. SYSTEMATYCZNIE: wszystkie sesje 2018-2019 (n=%d)" % len(D))
    log(f"Dni z >=1 tweetem Trumpa o cłach: {(D['trump_n'] > 0).sum()} ({100 * (D['trump_n'] > 0).mean():.0f}% sesji) - za często, "
        f"by traktować każdy jako zdarzenie.\nDzień intensywny Trumpa = >= {max(q, 3):.0f} tweetów o cłach (górne 10%): {D['trump_heavy'].sum()} sesji.")
    log(f"Dni oficjalnych ogłoszeń: {D['ann'].sum()}, wejść w życie: {D['imp'].sum()} "
        f"(źródła: {O.groupby(['typ', 'zrodlo']).size().to_dict()})")
    O.sort_values("sesja").to_csv(OUT / "hip_00_dni_oficjalne.csv", sep=";", index=False, encoding="utf-8-sig")

    D["typ_dnia"] = np.select(
        [(D["ann"] + D["ann_lag"] > 0) & (D["trump_heavy"] == 1), D["ann"] + D["ann_lag"] > 0, D["imp"] + D["imp_lag"] > 0,
         (D["trump_heavy"] == 1) & ~near_off, ~near_off & (D["trump_n"] == 0)],
        ["Ogłoszenie + Trump", "Tylko ogłoszenie", "Wejście w życie", "Tylko Trump (intensywnie)", "Spokojny dzień"], "inne")
    order = ["Spokojny dzień", "Tylko Trump (intensywnie)", "Tylko ogłoszenie", "Ogłoszenie + Trump", "Wejście w życie"]
    rows = []
    for mk, lab in MARKETS.items():
        base = D.loc[D["typ_dnia"] == "Spokojny dzień", mk].abs() * 1e4
        for t in order:
            x = D.loc[D["typ_dnia"] == t, mk].abs() * 1e4
            lo, hi = boot_mean_ci(x)
            rows.append({"rynek": lab, "typ_dnia": t, "n": len(x), "śr_|r|_pb": x.mean(), "CI_lo": lo, "CI_hi": hi,
                         "mediana_pb": x.median(), "p_vs_spokojny": perm_vs_base(x, base) if t != "Spokojny dzień" else np.nan})
    TD = pd.DataFrame(rows)
    log("\nŚrednia wielkość ruchu |r| wg typu dnia (pb):\n" + TD.round(2).to_string(index=False))
    TD.to_csv(OUT / "hip_01_typy_dni.csv", sep=";", index=False, encoding="utf-8-sig")

    log("\nRegresja |r_t| (pb) = a + b*zdarzenia + c*|r_t-1| ; błędy Neweya-Westa (5 lagów):")
    regs = []
    xs = ["trump_heavy", "ann", "ann_lag", "imp", "imp_lag"]
    names = {"trump_heavy": "Intensywny dzień Trumpa", "ann": "Ogłoszenie (dzień t)", "ann_lag": "Ogłoszenie (dzień t+1)",
             "imp": "Wejście w życie (t)", "imp_lag": "Wejście w życie (t+1)"}
    for mk, lab in MARKETS.items():
        y = D[mk].abs() * 1e4
        X = np.column_stack([D[c] for c in xs] + [y.shift(1).fillna(y.mean())])
        b, se, p = newey_west(y.to_numpy(), X)
        for i, c in enumerate(xs, 1):
            regs.append({"rynek": lab, "zmienna": names[c], "beta_pb": b[i], "SE": se[i], "CI_lo": b[i] - 1.96 * se[i],
                         "CI_hi": b[i] + 1.96 * se[i], "p": p[i]})
        # wariant: liczba tweetów (log) zamiast progu
        X2 = np.column_stack([np.log1p(D["trump_n"])] + [D[c] for c in xs[1:]] + [y.shift(1).fillna(y.mean())])
        b2, se2, p2 = newey_west(y.to_numpy(), X2)
        regs.append({"rynek": lab, "zmienna": "log(1+liczba tweetów) [wariant]", "beta_pb": b2[1], "SE": se2[1],
                     "CI_lo": b2[1] - 1.96 * se2[1], "CI_hi": b2[1] + 1.96 * se2[1], "p": p2[1]})
    RG = pd.DataFrame(regs)
    log(RG.round(3).to_string(index=False))
    RG.to_csv(OUT / "hip_02_regresja.csv", sep=";", index=False, encoding="utf-8-sig")

    section("B. KRONIKA KLUCZOWYCH EPIZODÓW: sygnał Trumpa vs ogłoszenie vs wejście w życie (okno 2 sesji)")
    K = chronicle(days)
    long = []
    for r in K.itertuples():
        dts = {s: getattr(r, f"{s}_day") for s in STAGES}
        known = {s: d for s, d in dts.items() if s != "wejscie" and pd.notna(d)}
        first = min(known.values()) if known else pd.NaT
        for s, d in dts.items():
            if pd.isna(d):
                continue
            rec = {"id": r.id, "opis": r.opis, "kier": r.kier, "etap": s, "sesja": d, "uwagi": r.uwagi,
                   "pierwsza_info": bool(s != "wejscie" and d == first),
                   "ta_sama_sesja_co_inny_etap": any(dd == d for ss, dd in dts.items() if ss != s and pd.notna(dd))}
            for mk in MARKETS:
                w = window(D, mk, d, 2)
                rec[f"{mk}_2d"] = 1e4 * w
                rec[f"{mk}_2d_oczek"] = -r.kier * 1e4 * w      # >0 = ruch w oczekiwanym kierunku (eskalacja -> spadek)
                rec[f"{mk}_d0"] = 1e4 * window(D, mk, d, 1)
            long.append(rec)
    L = pd.DataFrame(long)
    L.to_csv(OUT / "hip_03_epizody.csv", sep=";", index=False, encoding="utf-8-sig")
    K.to_csv(OUT / "hip_03_kronika.csv", sep=";", index=False, encoding="utf-8-sig")
    show = ["id", "opis", "etap", "sesja", "pierwsza_info"] + [f"{m}_2d" for m in MARKETS]
    log(L[show].round(0).to_string(index=False))

    # baza: losowe okna 2-sesyjne
    base2 = {mk: (D[mk].rolling(2).sum().shift(-1) * 1e4).dropna() for mk in MARKETS}
    rows = []
    for mk, lab in MARKETS.items():
        for grp, m in [("Sygnał Trumpa", L["etap"] == "trump"), ("Oficjalne ogłoszenie", L["etap"] == "oficj"),
                       ("Wejście w życie", L["etap"] == "wejscie"),
                       ("Pierwsza informacja (dowolne źródło)", L["pierwsza_info"]),
                       ("Potwierdzenie (ogłoszenie po sygnale)", (L["etap"] == "oficj") & ~L["pierwsza_info"]),
                       ("Ogłoszenie jako pierwsze", (L["etap"] == "oficj") & L["pierwsza_info"]),
                       ("Trump jako pierwszy", (L["etap"] == "trump") & L["pierwsza_info"])]:
            S = L[m].drop_duplicates(["sesja"])   # ta sama sesja = jedna obserwacja
            x, a = S[f"{mk}_2d_oczek"], S[f"{mk}_2d"].abs()
            lo, hi = boot_mean_ci(x); alo, ahi = boot_mean_ci(a)
            rows.append({"rynek": lab, "grupa": grp, "n": len(S), "ruch_w_oczek_kier_pb": x.mean(), "CI_lo": lo, "CI_hi": hi,
                         "p_kier(perm vs losowe dni)": perm_vs_base(x, base2[mk] - base2[mk].mean()),
                         "trafny_kierunek_%": 100 * (x > 0).mean(), "śr_|ruch|_pb": a.mean(), "|ruch|_CI_lo": alo, "|ruch|_CI_hi": ahi,
                         "p_|ruch|(perm vs losowe dni)": perm_vs_base(a, base2[mk].abs()), "baza_|ruch|_pb": base2[mk].abs().mean()})
    ST = pd.DataFrame(rows)
    log("\nPodsumowanie etapów (okno 2 sesji; 'ruch w oczekiwanym kierunku' > 0 = eskalacja->spadek / deeskalacja->wzrost):\n"
        + ST.round(2).to_string(index=False))
    ST.to_csv(OUT / "hip_04_etapy.csv", sep=";", index=False, encoding="utf-8-sig")
    log("\nUwaga: w epizodach, gdzie Trump był pierwszy, oficjalne ogłoszenie przychodzi po 4-19 dniach i jest już znane rynkowi.")
    order_tab = L[L["etap"] != "wejscie"].pivot_table(index="id", columns="etap", values="sesja", aggfunc="first")
    order_tab["kto_pierwszy"] = np.where(order_tab["trump"].isna(), "tylko oficjalnie",
                                np.where(order_tab["oficj"].isna(), "tylko Trump",
                                np.where(order_tab["trump"] < order_tab["oficj"], "Trump",
                                np.where(order_tab["trump"] > order_tab["oficj"], "oficjalnie", "ta sama sesja"))))
    log("Kto pierwszy podał informację:\n" + order_tab["kto_pierwszy"].value_counts().to_string())

    plots(D, TD, RG, L, ST, K, order)
    (OUT / "hip_summary.txt").write_text("\n".join(LOG), encoding="utf-8")
    log(f"\nZapisano wyniki w {OUT} (hip_*)")


# ============================================================================ wykresy
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 11,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 12.5, "axes.titlecolor": INK,
        "legend.frameon": False, "font.family": "DejaVu Sans"})
    return plt


def _title(fig, title, sub):
    fig.text(0.01, 0.985, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.935, sub, fontsize=10.5, color=INK2, va="top")


def plots(D, TD, RG, L, ST, K, order):
    plt = _style()
    from matplotlib.lines import Line2D

    # 1. typy dni ----------------------------------------------------------------------------------------------
    cols = {"Spokojny dzień": "#b9b8b2", "Tylko Trump (intensywnie)": COL["trump"], "Tylko ogłoszenie": COL["oficj"],
            "Ogłoszenie + Trump": "#6250d6", "Wejście w życie": COL["wejscie"]}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    for ax, lab in zip(axes, MARKETS.values()):
        T = TD[TD["rynek"] == lab].set_index("typ_dnia").loc[order]
        y = np.arange(len(T))[::-1]
        ax.barh(y, T["śr_|r|_pb"], color=[cols[t] for t in T.index], height=0.6, edgecolor=SURF, linewidth=2)
        ax.errorbar(T["śr_|r|_pb"], y, xerr=[T["śr_|r|_pb"] - T["CI_lo"], T["CI_hi"] - T["śr_|r|_pb"]], fmt="none",
                    ecolor=INK2, elinewidth=1.2, capsize=3)
        base = T.loc["Spokojny dzień", "śr_|r|_pb"]
        ax.axvline(base, color=INK2, ls="--", lw=1)
        for yi, (t, r) in zip(y, T.iterrows()):
            pv = "" if np.isnan(r["p_vs_spokojny"]) else f"   p={r['p_vs_spokojny']:.2f}"
            ax.text(r["CI_hi"] + 2, yi, f"{r['śr_|r|_pb']:.0f} pb  (n={int(r['n'])}){pv}", va="center", fontsize=9.5, color=INK)
        ax.set_yticks(y, T.index)
        ax.set_xlim(0, T["CI_hi"].max() * 1.55)
        ax.set_title(lab, loc="left")
        ax.set_xlabel("średnia wielkość ruchu w dniu sesji |r| (pb; 100 pb = 1%)")
        ax.grid(axis="y", visible=False)
    _title(fig, "S&P 500 rusza się mocniej w dni oficjalnych ogłoszeń; intensywne tweety Trumpa nie zwiększają ruchu",
           "Wszystkie sesje 2018–2019. Słupek = średnia |zwrot|, wąsy = 95% przedział ufności (bootstrap), linia przerywana = spokojny dzień. "
           "p = test permutacyjny vs spokojne dni.")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(OUT / "fig_hip_1_typy_dni.png", dpi=150); plt.close(fig)

    # 2. regresja ----------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    vars_ = ["Intensywny dzień Trumpa", "log(1+liczba tweetów) [wariant]", "Ogłoszenie (dzień t)", "Ogłoszenie (dzień t+1)",
             "Wejście w życie (t)", "Wejście w życie (t+1)"]
    vc = [COL["trump"], COL["trump"], COL["oficj"], COL["oficj"], COL["wejscie"], COL["wejscie"]]
    for ax, lab in zip(axes, MARKETS.values()):
        R = RG[RG["rynek"] == lab].set_index("zmienna").loc[vars_]
        y = np.arange(len(R))[::-1]
        ax.axvline(0, color=INK2, lw=1)
        for yi, (v, r), c, mk in zip(y, R.iterrows(), vc, ["o", "D", "s", "s", "^", "^"]):
            ax.plot([r["CI_lo"], r["CI_hi"]], [yi, yi], color=c, lw=2.2, solid_capstyle="round")
            ax.plot(r["beta_pb"], yi, marker=mk, ms=9, color=c, mec=SURF, mew=2)
            ax.text(r["CI_hi"] + 2, yi, f"{r['beta_pb']:+.0f} pb  p={r['p']:.2f}", va="center", fontsize=9.5, color=INK)
        ax.set_yticks(y, vars_)
        ax.set_title(lab, loc="left")
        ax.set_xlabel("dodatkowy ruch |r| względem zwykłego dnia (pb)")
        ax.grid(axis="y", visible=False)
        lo, hi = R["CI_lo"].min(), R["CI_hi"].max()
        ax.set_xlim(lo - 10, hi + 45)
    _title(fig, "Ile dodatkowego ruchu daje każdy typ zdarzenia (regresja, wszystkie sesje 2018–2019)",
           "|r_t| = a + zdarzenia + |r_t−1|; kropka = efekt, kreska = 95% CI (błędy Neweya-Westa). "
           "Ogłoszenie t+1: gdy ogłoszono po sesji, reakcja przychodzi następnego dnia.")
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(OUT / "fig_hip_2_regresja.png", dpi=150); plt.close(fig)

    # 3. epizody: dot plot ------------------------------------------------------------------------------------
    ids = [e[0] for e in EPISODES]
    lab_ep = {e[0]: f"{e[1]}  {'▲' if e[2] > 0 else '▼'}" for e in EPISODES}
    fig, axes = plt.subplots(1, 2, figsize=(16, 10.5), sharey=True)
    for ax, (mk, lab) in zip(axes, MARKETS.items()):
        y = {i: len(ids) - k for k, i in enumerate(ids)}
        ax.axvline(0, color=INK2, lw=1)
        for i in ids:
            pts = L[L["id"] == i]
            if len(pts) > 1:
                ax.plot([pts[f"{mk}_2d"].min(), pts[f"{mk}_2d"].max()], [y[i]] * 2, color=GRID, lw=3, zorder=1)
        for s in STAGES:
            P = L[L["etap"] == s]
            ax.scatter(P[f"{mk}_2d"], P["id"].map(y), s=95, marker=MARK[s], color=COL[s], edgecolor=SURF, linewidth=2,
                       zorder=3, label=STAGES[s])
            F = P[P["pierwsza_info"]]
            ax.scatter(F[f"{mk}_2d"], F["id"].map(y), s=260, marker="o", facecolor="none", edgecolor=INK, linewidth=1.2, zorder=4)
        ax.set_yticks(list(y.values()), [lab_ep[i] for i in y])
        ax.set_title(lab, loc="left")
        ax.set_xlabel("zwrot w oknie 2 sesji od zdarzenia (pb)")
        ax.grid(axis="y", color=GRID, lw=0.5)
    h = [Line2D([], [], marker=MARK[s], color=COL[s], ls="", ms=10, label=STAGES[s]) for s in STAGES] + \
        [Line2D([], [], marker="o", mfc="none", mec=INK, ls="", ms=14, label="pierwsza publiczna informacja")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.01, 0.915), ncol=4, fontsize=10.5)
    _title(fig, "Kluczowe epizody wojny celnej: kiedy rynek naprawdę zareagował?",
           "▲ eskalacja (oczekiwany spadek), ▼ deeskalacja (oczekiwany wzrost). Każdy punkt = zwrot w 2 sesjach od danego etapu. "
           "Obwódka = etap, który jako pierwszy podał informację.")
    fig.tight_layout(rect=(0, 0, 1, 0.885)); fig.savefig(OUT / "fig_hip_3_epizody.png", dpi=150); plt.close(fig)

    # 4. podsumowanie etapów ---------------------------------------------------------------------------------
    groups = ["Sygnał Trumpa", "Oficjalne ogłoszenie", "Wejście w życie", "Pierwsza informacja (dowolne źródło)",
              "Potwierdzenie (ogłoszenie po sygnale)"]
    gcol = [COL["trump"], COL["oficj"], COL["wejscie"], INK, "#9ac0ec"]
    glab = ["Sygnał\nTrumpa", "Oficjalne\nogłoszenie", "Wejście\nw życie", "Pierwsza\ninformacja\n(dowolne źródło)",
            "Ogłoszenie po\nwcześniejszym\nsygnale Trumpa"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, lab in zip(axes, MARKETS.values()):
        S = ST[ST["rynek"] == lab].set_index("grupa").loc[groups]
        x = np.arange(len(S))
        ax.bar(x, S["ruch_w_oczek_kier_pb"], color=gcol, width=0.62, edgecolor=SURF, linewidth=2)
        ax.errorbar(x, S["ruch_w_oczek_kier_pb"], yerr=[S["ruch_w_oczek_kier_pb"] - S["CI_lo"], S["CI_hi"] - S["ruch_w_oczek_kier_pb"]],
                    fmt="none", ecolor=INK2, elinewidth=1.2, capsize=4)
        ax.axhline(0, color=INK2, lw=1)
        for xi, (g, r) in zip(x, S.iterrows()):
            top = max(r["CI_hi"], 0)
            ax.text(xi, top + 8, f"{r['ruch_w_oczek_kier_pb']:+.0f} pb\nn={int(r['n'])}, trafnie {r['trafny_kierunek_%']:.0f}%",
                    ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(x, glab, fontsize=9.5)
        ax.set_title(lab, loc="left")
        ax.set_ylabel("ruch w oczekiwanym kierunku, 2 sesje (pb)")
        ax.set_ylim(min(S["CI_lo"].min(), 0) - 30, S["CI_hi"].max() + 90)
        ax.grid(axis="x", visible=False)
    _title(fig, "Ruch w oczekiwanym kierunku: wyraźny tylko dla chińskich ETF-ów po sygnale Trumpa",
           "Kronika 22 epizodów USA–Chiny 2018–2019, zwrot w 2 sesjach. Wartość > 0: eskalacja → spadek, deeskalacja → wzrost.\n"
           "Wąsy = 95% CI (bootstrap); 'trafnie' = % epizodów z ruchem w oczekiwanym kierunku.")
    fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig(OUT / "fig_hip_4_etapy.png", dpi=150); plt.close(fig)

    # 5. oś czasu --------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(16, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.1]})
    ax = axes[0]
    spy = np.exp(D["SPY_ret"].cumsum()); kweb = np.exp(D["KWEB_ret"].cumsum())
    ax.plot(D.index, spy, color="#2c2c2a", lw=2, label="S&P 500 (SPY)")
    ax.plot(D.index, kweb, color="#6250d6", lw=2, label="Chiński internet (KWEB)")
    ax.text(D.index[-1], spy.iloc[-1], "  SPY", va="center", color="#2c2c2a", fontsize=10, fontweight="bold")
    ax.text(D.index[-1], kweb.iloc[-1], "  KWEB", va="center", color="#6250d6", fontsize=10, fontweight="bold")
    ax.set_ylabel("wartość (2018-01 = 1)")
    for r in K.itertuples():
        for s, yy in (("trump", 1.37), ("oficj", 1.32), ("wejscie", 1.27)):
            d = getattr(r, f"{s}_day")
            if pd.notna(d):
                ax.axvline(d, color=COL[s], lw=0.7, alpha=0.35, zorder=0)
                ax.scatter([d], [yy], marker=MARK[s], s=55, color=COL[s], edgecolor=SURF, linewidth=1.5, zorder=3)
    for s, yy in (("trump", 1.37), ("oficj", 1.32), ("wejscie", 1.27)):
        ax.text(D.index[0] - pd.Timedelta(days=4), yy, STAGES[s], ha="right", va="center", fontsize=9.5, color=INK2)
    ax.set_ylim(0.6, 1.41)
    ax.legend(loc="lower left", fontsize=10)
    ax2 = axes[1]
    ax2.bar(D.index, D["trump_n"], color=COL["trump"], width=1.2)
    ax2.set_ylabel("tweety Trumpa\no cłach / sesję")
    _title(fig, "Oś czasu wojny celnej 2018–2019: sygnały Trumpa, oficjalne ogłoszenia i wejście ceł w życie",
           "Górny panel: skumulowany zwrot SPY i KWEB; znaczniki = etapy epizodów z kroniki. Dolny panel: liczba tweetów Trumpa o cłach.")
    fig.tight_layout(rect=(0.06, 0, 1, 0.92)); fig.savefig(OUT / "fig_hip_5_os_czasu.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
