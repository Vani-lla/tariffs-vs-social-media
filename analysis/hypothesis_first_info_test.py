"""Test wprost hipotezy: rynek reaguje na PIERWSZĄ WIARYGODNĄ INFORMACJĘ o cłach, niezależnie od kanału.

Uruchom z katalogu projektu:  python analysis/hypothesis_first_info_test.py
(korzysta z danych ETF z cleaning_data/china_etfs.py i kroniki z hypothesis_official_vs_trump.py)
Wyniki: analysis/out/test_*.csv, analysis/out/fig_hip_6_*.png, fig_hip_7_*.png, analysis/out/test_summary.txt

Układ 2 x 2: NOWOŚĆ informacji x KANAŁ. Klasy ustalone z góry, przed patrzeniem na zwroty:
  Nowa  / Trump      pierwsza publiczna informacja o decyzji z kroniki, podana tweetem/oświadczeniem Trumpa
  Nowa  / oficjalnie pierwsza publiczna informacja z kroniki, podana przez rząd (USTR, proklamacja, MOFCOM, FR)
  Stara / oficjalnie oficjalne ogłoszenie decyzji, którą Trump podał wcześniej (potwierdzenie)
  Stara / Trump      tweety Trumpa o cłach z próby gold (etykieta ESKALACJA/DEESKALACJA), które NIE przypadają na
                     dzień żadnego etapu kroniki (±1 sesja): komentarz, groźba, optymizm, bez nowej decyzji
Dodatkowo (tylko wielkość ruchu, bez kierunku): oficjalne dokumenty administracyjne Federal Register
(wykluczenia produktów, procedury, przedłużenia), czyli kanał oficjalny bez nowej decyzji.

Przewidywania hipotezy: efekt NOWOŚCI > 0; efekt KANAŁU ≈ 0; interakcja ≈ 0.
Miara: zwrot w 2 sesjach ze znakiem względem kierunku (eskalacja -> spadek = dodatni), oraz |zwrot|.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import annotation_market_study as ams  # noqa: E402
import hypothesis_official_vs_trump as hip  # noqa: E402
from cleaning_data.base import Config  # noqa: E402

OUT = ams.OUT
LOG = ams.LOG
log, section = ams.log, ams.section
MARKETS = hip.MARKETS
FR_ADMIN = re.compile(r"Product Exclusion|Procedures|Extension|Submissions of Exclusion|Exclusions Portal|Petition", re.I)
CLASSES = ["Nowa / Trump", "Nowa / oficjalnie", "Stara / Trump", "Stara / oficjalnie"]


def build_units(D: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    days = D.index
    K = hip.chronicle(days)
    rows = []
    stage_days = set()
    for r in K.itertuples():
        dts = {s: getattr(r, f"{s}_day") for s in hip.STAGES}
        stage_days |= {d for d in dts.values() if pd.notna(d)}
        known = {s: d for s, d in dts.items() if s != "wejscie" and pd.notna(d)}
        if not known:
            continue
        first = min(known.values())
        for s, d in known.items():
            kanal = "Trump" if s == "trump" else "oficjalnie"
            if d == first:
                if s == "oficj" and dts.get("trump") == d:
                    continue                       # ta sama sesja: liczymy raz, jako Trump (kolejność w sesji nieznana)
                nowosc = "Nowa"
            elif s == "oficj":
                nowosc = "Stara"
            else:
                continue                           # Trump po ogłoszeniu oficjalnym: w kronice nie występuje
            rows.append({"klasa": f"{nowosc} / {kanal}", "nowa": int(nowosc == "Nowa"), "trump": int(kanal == "Trump"),
                         "sesja": d, "kier": r.kier, "zrodlo": f"kronika {r.id}", "opis": r.opis})
    # tweety z próby gold bez nowej decyzji
    near = set()
    for d in stage_days:
        i = days.get_indexer([d])[0]
        near |= set(days[max(i - 1, 0):i + 2])
    g = gold[gold["etykieta4"].isin(["ESKALACJA", "DEESKALACJA"])].copy()
    g["ny"] = pd.to_datetime(g["time_utc"], utc=True).dt.tz_convert(hip.NY).dt.tz_localize(None)
    g["sesja"] = hip.trading_day(days, g["ny"])
    g = g[g["sesja"].notna() & ~g["sesja"].isin(near)]
    for r in g.itertuples():
        rows.append({"klasa": "Stara / Trump", "nowa": 0, "trump": 1, "sesja": r.sesja,
                     "kier": 1 if r.etykieta4 == "ESKALACJA" else -1, "zrodlo": f"gold {r.item_id}", "opis": str(r.text)[:90]})
    U = pd.DataFrame(rows)
    U["sesja"] = pd.to_datetime(U["sesja"])
    # jedna obserwacja na (klasa, sesja, kierunek); sesje z przeciwnymi kierunkami w tej samej klasie odpadają
    U = U.drop_duplicates(["klasa", "sesja", "kier"])
    U = U[~U.duplicated(["klasa", "sesja"], keep=False)]
    for mk in MARKETS:
        w2 = U["sesja"].map(lambda d: hip.window(D, mk, d, 2)) * 1e4
        w1 = U["sesja"].map(lambda d: hip.window(D, mk, d, 1)) * 1e4
        U[f"{mk}_oczek"] = -U["kier"] * w2
        U[f"{mk}_abs"] = w2.abs()
        U[f"{mk}_oczek_d0"] = -U["kier"] * w1
    return U.sort_values("sesja")


def fr_admin_units(D: pd.DataFrame) -> pd.DataFrame:
    E = pd.read_csv(ROOT / "clean_out" / "merged" / "events.csv")
    E = E[(E["kind"] == "fr") & E["title"].fillna("").str.contains(FR_ADMIN)].copy()
    i = D.index.searchsorted(pd.to_datetime(E["eff_day"]).to_numpy())
    E = E[i < len(D)]
    E["sesja"] = D.index[i[i < len(D)]]
    E = E.drop_duplicates("sesja")
    for mk in MARKETS:
        E[f"{mk}_abs"] = E["sesja"].map(lambda d: abs(hip.window(D, mk, d, 2)) * 1e4)
    return E[["sesja", "title"] + [f"{mk}_abs" for mk in MARKETS]]


def contrast(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    lo, hi = ams.boot_diff_ci(x, y)
    return {"n_A": len(x), "n_B": len(y), "śr_A": x.mean(), "śr_B": y.mean(), "różnica": x.mean() - y.mean(),
            "CI_lo": lo, "CI_hi": hi, "p_perm": ams.perm_p(x, y, 10000)}


def main():
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 80)
    LOG.clear()
    cfg = Config()
    D = hip.load_daily(cfg)
    L, _ = ams.load_labels()
    _, _, W = ams.agreement(L)
    GOLD = ams.build_gold(L, W)

    section("T1. JEDNOSTKI ANALIZY (klasy ustalone przed patrzeniem na zwroty)")
    U = build_units(D, GOLD["MG_pierwszy"])
    log(U.groupby("klasa").size().reindex(CLASSES).to_string())
    U.to_csv(OUT / "test_00_jednostki.csv", sep=";", index=False, encoding="utf-8-sig")
    A = fr_admin_units(D)
    log(f"Dokumenty administracyjne FR (kanał oficjalny bez nowej decyzji, tylko |ruch|): {len(A)} sesji")

    section("T2. ŚREDNIE W KLASACH (2 sesje; 'oczek' > 0 = ruch w oczekiwanym kierunku)")
    rows = []
    for mk, lab in MARKETS.items():
        for k in CLASSES:
            s = U[U["klasa"] == k]
            for miara, col in (("kierunek", f"{mk}_oczek"), ("|ruch|", f"{mk}_abs"), ("kierunek, 1 sesja", f"{mk}_oczek_d0")):
                x = s[col].dropna()
                lo, hi = hip.boot_mean_ci(x)
                rows.append({"rynek": lab, "klasa": k, "miara": miara, "n": len(x), "średnia_pb": x.mean(), "CI_lo": lo, "CI_hi": hi,
                             "trafnie_%": 100 * (x > 0).mean() if miara.startswith("kier") else np.nan})
        x = A[f"{mk}_abs"].dropna(); lo, hi = hip.boot_mean_ci(x)
        rows.append({"rynek": lab, "klasa": "Stara / oficjalnie (FR administracyjne)", "miara": "|ruch|", "n": len(x),
                     "średnia_pb": x.mean(), "CI_lo": lo, "CI_hi": hi})
        base = (D[mk].rolling(2).sum().shift(-1) * 1e4).abs().dropna()
        rows.append({"rynek": lab, "klasa": "Losowe 2 sesje (baza)", "miara": "|ruch|", "n": len(base), "średnia_pb": base.mean()})
    M = pd.DataFrame(rows)
    log(M.round(1).to_string(index=False))
    M.to_csv(OUT / "test_01_klasy.csv", sep=";", index=False, encoding="utf-8-sig")

    section("T3. KONTRASTY (różnica średnich, pb; CI z bootstrapu; p permutacyjne)")
    cons = [("NOWOŚĆ w kanale Trumpa", "Nowa / Trump", "Stara / Trump"),
            ("NOWOŚĆ w kanale oficjalnym", "Nowa / oficjalnie", "Stara / oficjalnie"),
            ("KANAŁ przy nowej informacji (Trump - oficjalnie)", "Nowa / Trump", "Nowa / oficjalnie"),
            ("KANAŁ przy starej informacji (Trump - oficjalnie)", "Stara / Trump", "Stara / oficjalnie")]
    rows = []
    for mk, lab in MARKETS.items():
        for miara in ("oczek", "abs"):
            for name, a, b in cons:
                r = contrast(U.loc[U["klasa"] == a, f"{mk}_{miara}"], U.loc[U["klasa"] == b, f"{mk}_{miara}"])
                rows.append({"rynek": lab, "miara": "kierunek" if miara == "oczek" else "|ruch|", "kontrast": name, **r})
            # nowość ogółem
            r = contrast(U.loc[U["nowa"] == 1, f"{mk}_{miara}"], U.loc[U["nowa"] == 0, f"{mk}_{miara}"])
            rows.append({"rynek": lab, "miara": "kierunek" if miara == "oczek" else "|ruch|", "kontrast": "NOWOŚĆ ogółem (nowa - stara)", **r})
    C = pd.DataFrame(rows)
    C["q_FDR"] = ams.bh(C["p_perm"])
    log(C.round(3).to_string(index=False))
    C.to_csv(OUT / "test_02_kontrasty.csv", sep=";", index=False, encoding="utf-8-sig")

    section("T4. REGRESJA 2x2: y = a + b1*NOWA + b2*TRUMP + b3*NOWA*TRUMP (błędy HC3)")
    regs = []
    for mk, lab in MARKETS.items():
        for miara in ("oczek", "abs", "oczek_d0"):
            d = U[["nowa", "trump", f"{mk}_{miara}"]].dropna()
            X = np.column_stack([d["nowa"], d["trump"], d["nowa"] * d["trump"]])
            b, se, p, r2 = ams.ols(d[f"{mk}_{miara}"].to_numpy(), X)
            for i, nm in enumerate(["stała (stara / oficjalnie)", "NOWOŚĆ", "KANAŁ Trump", "NOWOŚĆ x Trump"]):
                regs.append({"rynek": lab, "y": {"oczek": "kierunek, 2 sesje", "abs": "|ruch|, 2 sesje", "oczek_d0": "kierunek, 1 sesja"}[miara],
                             "zmienna": nm, "beta_pb": b[i], "SE": se[i], "CI_lo": b[i] - 1.96 * se[i], "CI_hi": b[i] + 1.96 * se[i],
                             "p": p[i], "n": len(d)})
    RG = pd.DataFrame(regs)
    log(RG.round(3).to_string(index=False))
    RG.to_csv(OUT / "test_03_regresja.csv", sep=";", index=False, encoding="utf-8-sig")

    section("T5. ODPORNOŚĆ: etykiety 'zgodne' (MG i PK zgodni) zamiast MG_pierwszy")
    U2 = build_units(D, GOLD["zgodne"])
    for mk, lab in MARKETS.items():
        for name, a, b in cons[:1] + [("NOWOŚĆ ogółem", None, None)]:
            if a is None:
                x, y = U2.loc[U2["nowa"] == 1, f"{mk}_oczek"], U2.loc[U2["nowa"] == 0, f"{mk}_oczek"]
            else:
                x, y = U2.loc[U2["klasa"] == a, f"{mk}_oczek"], U2.loc[U2["klasa"] == b, f"{mk}_oczek"]
            r = contrast(x, y)
            log(f"{lab:28s} {name:28s} n={r['n_A']}/{r['n_B']}  różnica={r['różnica']:+.1f} pb  p={r['p_perm']:.3f}")

    plots(M, RG, U, A)
    (OUT / "test_summary.txt").write_text("\n".join(LOG), encoding="utf-8")
    log(f"\nZapisano wyniki w {OUT} (test_*)")


def plots(M, RG, U, A):
    plt = hip._style()
    from matplotlib.lines import Line2D
    COLC = {"Trump": hip.COL["trump"], "oficjalnie": hip.COL["oficj"]}

    # 6. macierz 2x2: kierunek -------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    pos = {"Nowa / Trump": 0, "Nowa / oficjalnie": 1, "Stara / Trump": 2.6, "Stara / oficjalnie": 3.6}
    for ax, lab in zip(axes, MARKETS.values()):
        S = M[(M["rynek"] == lab) & (M["miara"] == "kierunek")].set_index("klasa")
        mk = [k for k, v in MARKETS.items() if v == lab][0]
        for k, x in pos.items():
            r = S.loc[k]
            kan = k.split(" / ")[1]
            nowa = k.startswith("Nowa")
            ax.bar(x, r["średnia_pb"], width=0.82, color=COLC[kan], alpha=1 if nowa else 0.45, edgecolor=hip.SURF, linewidth=2)
            ax.errorbar(x, r["średnia_pb"], yerr=[[r["średnia_pb"] - r["CI_lo"]], [r["CI_hi"] - r["średnia_pb"]]],
                        fmt="none", ecolor=hip.INK2, elinewidth=1.2, capsize=4)
            pts = U.loc[U["klasa"] == k, f"{mk}_oczek"].dropna()
            ax.scatter(x + hip.RNG.uniform(-0.25, 0.25, len(pts)), pts, s=14, color=hip.INK, alpha=0.35, zorder=3, linewidths=0)
            ax.text(x, max(r["CI_hi"], 0) + 12, f"{r['średnia_pb']:+.0f} pb\nn={int(r['n'])}, trafnie {r['trafnie_%']:.0f}%",
                    ha="center", va="bottom", fontsize=9.5, color=hip.INK, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", fc=hip.SURF, ec="none", alpha=0.85))
        ax.axhline(0, color=hip.INK2, lw=1)
        ax.set_xticks(list(pos.values()), ["Trump", "Oficjalnie", "Trump", "Oficjalnie"])
        ax.text(0.5, -0.14, "NOWA informacja\n(pierwsza zapowiedź decyzji)", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=10.5, fontweight="bold", color=hip.INK)
        ax.text(3.1, -0.14, "STARA / brak nowej decyzji\n(potwierdzenie, komentarz)", transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=10.5, fontweight="bold", color=hip.INK2)
        ax.set_title(lab, loc="left")
        ax.set_ylabel("ruch w oczekiwanym kierunku, 2 sesje (pb)")
        lo = np.nanmin(U[f"{mk}_oczek"]); hi = np.nanmax(U[f"{mk}_oczek"])
        ax.set_ylim(lo - 25, hi + 25)
        ax.grid(axis="x", visible=False)
    h = [Line2D([], [], marker="s", ls="", ms=12, color=COLC["Trump"], label="kanał: Trump"),
         Line2D([], [], marker="s", ls="", ms=12, color=COLC["oficjalnie"], label="kanał: oficjalny"),
         Line2D([], [], marker="o", ls="", ms=5, color=hip.INK, alpha=0.4, label="pojedyncza sesja")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.01, 0.885), ncol=3, fontsize=10.5)
    hip._title(fig, "Test wprost: nowa informacja przesuwa rynek bardziej niż stara, ale przy tej próbie różnice są nieistotne",
               "Słupek = średni ruch w oczekiwanym kierunku (eskalacja → spadek, deeskalacja → wzrost), wąsy = 95% CI, kropki = sesje. "
               "Pełny kolor = nowa informacja, blady = stara.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.84)); fig.savefig(OUT / "fig_hip_6_test_2x2.png", dpi=150); plt.close(fig)

    # 7. regresja 2x2 ---------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
    vars_ = ["NOWOŚĆ", "KANAŁ Trump", "NOWOŚĆ x Trump"]
    vcol = [hip.INK, hip.COL["trump"], "#6250d6"]
    ys = ["kierunek, 2 sesje", "kierunek, 1 sesja", "|ruch|, 2 sesje"]
    off = {"kierunek, 2 sesje": 0.22, "kierunek, 1 sesja": 0, "|ruch|, 2 sesje": -0.22}
    mks = {"kierunek, 2 sesje": "o", "kierunek, 1 sesja": "D", "|ruch|, 2 sesje": "s"}
    for ax, lab in zip(axes, MARKETS.values()):
        ax.axvline(0, color=hip.INK2, lw=1)
        for j, v in enumerate(vars_):
            yb = len(vars_) - 1 - j
            for yv in ys:
                r = RG[(RG["rynek"] == lab) & (RG["zmienna"] == v) & (RG["y"] == yv)].iloc[0]
                yy = yb + off[yv]
                ax.plot([r["CI_lo"], r["CI_hi"]], [yy, yy], color=vcol[j], lw=2, alpha=0.9 if yv != "|ruch|, 2 sesje" else 0.45)
                ax.plot(r["beta_pb"], yy, marker=mks[yv], ms=8, color=vcol[j], mec=hip.SURF, mew=1.5)
                ax.text(r["CI_hi"] + 6, yy, f"{r['beta_pb']:+.0f}  p={r['p']:.2f}", va="center", fontsize=8.5, color=hip.INK)
        ax.set_yticks(range(len(vars_))[::-1], vars_)
        ax.set_title(lab, loc="left")
        ax.set_xlabel("efekt (pb)")
        ax.grid(axis="y", visible=False)
        R = RG[RG["rynek"] == lab]
        ax.set_xlim(R["CI_lo"].min() - 20, R["CI_hi"].max() + 110)
    h = [Line2D([], [], marker=mks[y], ls="", color=hip.INK2, ms=8, label=y) for y in ys]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.01, 0.87), ncol=3, fontsize=10)
    hip._title(fig, "Regresja 2×2: efekt nowości, efekt kanału i ich interakcja",
               "y = a + b1·NOWA + b2·TRUMP + b3·NOWA·TRUMP; kreska = 95% CI (HC3). Hipoteza przewiduje: b1 > 0, b2 ≈ 0, b3 ≈ 0.")
    fig.tight_layout(rect=(0, 0, 1, 0.82)); fig.savefig(OUT / "fig_hip_7_regresja_2x2.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
