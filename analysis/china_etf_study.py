"""Tweety Trumpa o cłach vs chińskie ETF-y notowane w USA (KWEB, CQQQ, FXI, MCHI, ASHR), 2018-2019.

Uruchom z katalogu projektu:  python analysis/china_etf_study.py
Wyniki: analysis/out/china_*.csv, analysis/out/fig_china_*.png, analysis/out/china_summary.txt

Miara główna: zwrot nadzwyczajny AR = r - beta*r_SPY, beta z modelu rynkowego oszacowanego na 2017 (bez alfy). Odfiltrowuje to
ogólny ruch rynku USA, więc AR mierzy reakcję specyficzną dla Chin. ETF-y handlują w godzinach NYSE, czyli w tych
samych godzinach, w których Trump tweetował (inaczej niż Hang Seng). Etykiety, testy i klasyfikator są
reużyte z annotation_market_study.py, żeby wyniki były porównywalne z analizą S&P 500.
"""
from __future__ import annotations

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

OUT = ams.OUT
LOG = ams.LOG          # ams.log() dopisuje tutaj
log, section, perm_p, bh = ams.log, ams.section, ams.perm_p, ams.bh

SERIES = ["CHN_ALL_ar", "CHN_TECH_ar", "KWEB_ar", "CQQQ_ar", "FXI_ar", "MCHI_ar", "ASHR_ar"]
MAIN = ["CHN_ALL_ar", "CHN_TECH_ar"]


def post_features(posts: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    """Dla każdego posta: zwroty skumulowane w sesjach NYSE względem dnia efektywnego (post po 16:00 NY -> następna sesja)."""
    days = R.index
    di = days.searchsorted(ams.eff_day(pd.to_datetime(posts["time_utc"], utc=True)).to_numpy())
    out = pd.DataFrame(index=posts.index)
    out["eff_trading_day"] = days[np.clip(di, 0, len(days) - 1)]
    for s in SERIES + ["KWEB_ret", "FXI_ret", "SPY_ret"]:
        arr = R[s].to_numpy()

        def csum(o, n):
            res = []
            for i in di:
                a, b = i + o, i + o + n
                res.append(np.nan if a < 0 or b > len(arr) or np.isnan(arr[a:b]).any() else arr[a:b].sum())
            return np.array(res)

        out[f"{s}_d0"] = csum(0, 1)
        out[f"{s}_d1"] = csum(1, 1)
        out[f"{s}_d0_2"] = csum(0, 3)
        out[f"{s}_d0_4"] = csum(0, 5)
        out[f"{s}_pre_d5"] = csum(-5, 5)
        out[f"{s}_d0_abs"] = np.abs(out[f"{s}_d0"])
    return out


def main():
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
    LOG.clear()
    cfg = Config()
    E = ChinaETFs(cfg)

    section("C1. DANE ETF (yfinance, ceny skorygowane) I MODEL RYNKOWY")
    long = E.download()
    rep = E.check(long)
    log(rep.to_string())
    R = E.returns(long)
    log("\nModel rynkowy r_ETF = alfa + beta * r_SPY (AR = r - beta*r_SPY, alfy nie odejmujemy), okno estymacji " + " - ".join(cfg.china_estimation) + ":")
    log(E.params.round(3).to_string(index=False))
    rep.to_csv(OUT / "china_00_pokrycie.csv", sep=";", encoding="utf-8-sig")
    E.params.to_csv(OUT / "china_00_model_rynkowy.csv", sep=";", index=False, encoding="utf-8-sig")
    P = R.loc["2018-01-01":"2019-12-31"]
    log("\nZwroty 2018-2019 (pb/dzień): " + ", ".join(
        f"{c.replace('_ret', '')}: śr={1e4 * P[c].mean():.1f}, sd={1e4 * P[c].std():.0f}"
        for c in [f"{t}_ret" for t in cfg.china_etfs] + ["SPY_ret"]))
    log("Korelacja AR między ETF-ami (2018-2019):\n" + P[[f"{t}_ar" for t in cfg.china_etfs]].corr().round(2).to_string())

    section("C2. KONTROLA SENSU: AR w znanych dniach eskalacji/deeskalacji (pb)")
    ke = pd.DataFrame([{"zdarzenie": k, "dzień": d} for k, d in cfg.key_events.items()])
    ke["sesja"] = R.index[R.index.searchsorted(pd.to_datetime(ke["dzień"]))]
    for c in ("SPY_ret", "KWEB_ret", "CHN_TECH_ar", "CHN_ALL_ar"):
        ke[c] = (1e4 * R[c].reindex(ke["sesja"])).round(0).to_numpy()
    for extra, d in (("2019-05-06 pon. po tweecie 10->25%", "2019-05-06"), ("2019-08-02 pt. po tweecie 10% na 300 mld", "2019-08-02")):
        row = {"zdarzenie": extra, "dzień": d, "sesja": pd.Timestamp(d)}
        row.update({c: round(1e4 * R.loc[d, c]) for c in ("SPY_ret", "KWEB_ret", "CHN_TECH_ar", "CHN_ALL_ar")})
        ke = pd.concat([ke, pd.DataFrame([row])], ignore_index=True)
    log(ke.sort_values("sesja").to_string(index=False))
    ke.to_csv(OUT / "china_01_kontrola_zdarzen.csv", sep=";", index=False, encoding="utf-8-sig")

    section("C3. ETYKIETY (gold z adnotacji MG + PK, te same co w analizie S&P)")
    L, _ = ams.load_labels()
    _, _, W = ams.agreement(L)
    GOLD = ams.build_gold(L, W)
    ALL = {k: g.reset_index(drop=True).join(post_features(g.reset_index(drop=True), R)) for k, g in GOLD.items()}
    for k, g in ALL.items():
        log(f"{k:12s} n={len(g)}  " + str(g["etykieta4"].value_counts(dropna=False).to_dict()))
    ALL["MG_pierwszy"].drop(columns=["text"]).to_csv(OUT / "china_02_posty_z_etf.csv", sep=";", index=False, encoding="utf-8-sig")

    section("C4. TESTY: ESKALACJA vs DEESKALACJA (i inne kontrasty) dla zwrotów nadzwyczajnych chińskich ETF-ów")
    ret_out = [f"{s}_{h}" for s in SERIES for h in ("d0", "d1", "d0_2", "d0_4")]
    vol_out = [f"{s}_d0_abs" for s in MAIN]
    pre_out = [f"{s}_pre_d5" for s in MAIN]
    raw_out = [f"{s}_{h}" for s in ("KWEB_ret", "FXI_ret") for h in ("d0", "d0_4")]
    Rt = pd.concat([ams.compare(G, ret_out + vol_out + pre_out + raw_out, k) for k, G in ALL.items()], ignore_index=True)
    Rt.to_csv(OUT / "china_03_testy_kontrastow.csv", sep=";", index=False, encoding="utf-8-sig")
    show = ["wariant", "miara", "kontrast", "n_A", "n_B", "śr_A_pb", "śr_B_pb", "różnica_pb", "CI95_pb", "d_Cohena",
            "p_permutacyjny", "q_FDR(perm)"]
    main_out = [f"{s}_{h}" for s in MAIN for h in ("d0", "d1", "d0_2", "d0_4")]
    log("Główny test - ESKALACJA vs DEESKALACJA, AR koszyków (pb):\n" +
        Rt[(Rt["kontrast"] == "ESKALACJA vs DEESKALACJA") & Rt["miara"].isin(main_out)][show].round(3).to_string(index=False))
    log("\nESKALACJA vs posty nie o cłach (tło):\n" +
        Rt[(Rt["kontrast"] == "ESKALACJA vs nie_cla") & Rt["miara"].isin(main_out)][show].round(3).to_string(index=False))
    log("\nPojedyncze ETF-y, ESK vs DEESK, dzień 0 i 5 sesji:\n" +
        Rt[(Rt["kontrast"] == "ESKALACJA vs DEESKALACJA") & Rt["miara"].str.contains("_ar_d0$|_ar_d0_4$") &
           ~Rt["miara"].str.startswith("CHN") & (Rt["wariant"] == "MG_pierwszy")][show].round(3).to_string(index=False))
    log("\nSpearman: wynik kierunku (+1 ESK, 0 NEUTR, -1 DEESK) vs AR (cla=tak):\n" +
        Rt[Rt["kontrast"].str.startswith("Spearman") & Rt["miara"].isin(main_out)]
        [["wariant", "miara", "n_A", "rho", "p_permutacyjny", "q_FDR(perm)"]].round(3).to_string(index=False))
    log("\nZmienność |AR dnia 0| - cła vs nie cła:\n" +
        Rt[Rt["miara"].isin(vol_out) & Rt["kontrast"].isin(["cla=tak vs cla=nie", "ESKALACJA vs nie_cla"])][show].round(3).to_string(index=False))
    log("\nOdwrotna przyczynowość - AR w 5 sesjach PRZED postem:\n" +
        Rt[Rt["miara"].isin(pre_out) & (Rt["kontrast"] == "ESKALACJA vs DEESKALACJA")][show].round(3).to_string(index=False))
    log("\nZwroty surowe (bez odjęcia SPY), ESK vs DEESK:\n" +
        Rt[Rt["miara"].isin(raw_out) & (Rt["kontrast"] == "ESKALACJA vs DEESKALACJA")][show].round(3).to_string(index=False))
    sig = Rt[Rt["q_FDR(perm)"] < 0.10]
    log(f"\nWszystkich testów: {len(Rt)}. Nominalnie p_perm<0.05: {(Rt['p_permutacyjny'] < 0.05).sum()} "
        f"(oczekiwane fałszywe przy braku efektu: ~{0.05 * len(Rt):.0f}). Po FDR q<0.10: {len(sig)}")
    if len(sig):
        log(sig[show].round(3).to_string(index=False))

    section("C5. ODPORNOŚĆ: jeden dzień sesji = jedna obserwacja; posty o Chinach")
    for k in ("MG_pierwszy", "zgodne"):
        G = ALL[k]
        U = G[G["etykieta4"].isin(["ESKALACJA", "DEESKALACJA"])].drop_duplicates(["eff_trading_day", "etykieta4"])
        U = U[~U["eff_trading_day"].duplicated(keep=False)]
        for oc in ("CHN_ALL_ar_d0", "CHN_ALL_ar_d0_4", "CHN_TECH_ar_d0", "CHN_TECH_ar_d0_4"):
            x = U.loc[U["etykieta4"] == "ESKALACJA", oc].dropna(); y = U.loc[U["etykieta4"] == "DEESKALACJA", oc].dropna()
            log(f"[{k:11s}] {oc:17s} dni ESK={len(x):3d} śr={1e4 * x.mean():7.1f} pb | dni DEESK={len(y):3d} "
                f"śr={1e4 * y.mean():7.1f} pb | różnica={1e4 * (x.mean() - y.mean()):7.1f} pb, p_perm={perm_p(x, y, 5000):.3f}")
    G = ALL["MG_pierwszy"]
    t = G[G["cla"] == "tak"].copy()
    t["chiny"] = t["obszar"].str.contains("Chin", case=False, na=False)
    rows = []
    for oc in ("CHN_ALL_ar_d0", "CHN_ALL_ar_d0_4", "CHN_TECH_ar_d0", "CHN_TECH_ar_d0_4"):
        for sub, m in (("Chiny", t["chiny"]), ("nie-Chiny", ~t["chiny"])):
            s = t[m]
            x = s.loc[s["kierunek"] == "ESKALACJA", oc].dropna(); y = s.loc[s["kierunek"] == "DEESKALACJA", oc].dropna()
            if len(x) >= 3 and len(y) >= 3:
                rows.append({"miara": oc, "obszar": sub, "n_ESK": len(x), "n_DEESK": len(y), "ESK_pb": 1e4 * x.mean(),
                             "DEESK_pb": 1e4 * y.mean(), "p_perm": perm_p(x, y, 3000)})
    OB = pd.DataFrame(rows)
    log("\nObszar Chiny vs reszta (cla=tak):\n" + (OB.round(3).to_string(index=False) if len(OB) else "za mało obserwacji"))
    OB.to_csv(OUT / "china_04_obszar.csv", sep=";", index=False, encoding="utf-8-sig")
    log("\nPosty DEESKALACJA (MG_pierwszy) - AR koszyka:")
    for r in G[G["etykieta4"] == "DEESKALACJA"].sort_values("time_utc").itertuples():
        log(f"   {str(r.time_utc)[:16]}  sesja {str(r.eff_trading_day)[:10]}  CHN_ALL d0={1e4 * r.CHN_ALL_ar_d0:7.1f} pb  "
            f"5 sesji={1e4 * r.CHN_ALL_ar_d0_4:7.1f} pb  {r.text[:80]}")

    section("C6. MOC (minimalny wykrywalny efekt, alfa=0.05, moc=0.8)")
    n1 = (G["etykieta4"] == "ESKALACJA").sum(); n2 = (G["etykieta4"] == "DEESKALACJA").sum()
    for oc in ("CHN_ALL_ar_d0", "CHN_TECH_ar_d0", "CHN_ALL_ar_d0_4"):
        sd = G[oc].std()
        log(f"{oc:17s} sd={1e4 * sd:6.1f} pb | ESK(n={n1}) vs DEESK(n={n2}): MDE={1e4 * ams.mde(sd, n1, n2):6.1f} pb")

    section("C7. CAŁY KORPUS: dzienny indeks z klasyfikatora vs AR chińskich ETF-ów")
    CI = ams.corpus_index(GOLD["MG_pierwszy"])
    D = CI["daily"][["e_net", "e_esk", "e_deesk", "tw_topic_n", "n_all"]].join(R[MAIN + ["KWEB_ar", "FXI_ar", "SPY_ret"]], how="inner")
    for s in MAIN:
        D[f"{s}_abs"] = D[s].abs()
    log(f"Dni sesji: {len(D)}")
    LL = pd.concat([ams.lead_lag(D, x, y) for x in ("e_net", "e_esk", "tw_topic_n")
                    for y in ("CHN_ALL_ar", "CHN_TECH_ar", "CHN_ALL_ar_abs")], ignore_index=True)
    LL["q_FDR"] = bh(LL["p_pearson"])
    LL.to_csv(OUT / "china_05_lead_lag.csv", sep=";", index=False, encoding="utf-8-sig")
    log("corr(x_t, y_{t+k}); k>0 = tweety PRZED rynkiem, k<0 = rynek PRZED tweetami. e_net vs CHN_ALL_ar:")
    log(LL[(LL["x"] == "e_net") & (LL["y"] == "CHN_ALL_ar")].round(3).to_string(index=False))
    log("\nNajsilniejsze zależności (|r|):")
    log(LL.reindex(LL["pearson_r"].abs().sort_values(ascending=False).index).head(10).round(3).to_string(index=False))
    GR = pd.DataFrame([ams.granger(D, a, b, p) for p in (1, 2, 5) for a, b in
                       (("e_net", "CHN_ALL_ar"), ("CHN_ALL_ar", "e_net"), ("e_esk", "CHN_ALL_ar_abs"),
                        ("CHN_ALL_ar", "e_esk"), ("tw_topic_n", "CHN_ALL_ar"), ("CHN_ALL_ar", "tw_topic_n"))])
    GR.to_csv(OUT / "china_06_granger.csv", sep=";", index=False, encoding="utf-8-sig")
    log("\nPrzyczynowość Grangera (F-test):\n" + GR.round(4).to_string(index=False))

    plots(ALL["MG_pierwszy"], R, D, cfg)
    (OUT / "china_summary.txt").write_text("\n".join(LOG), encoding="utf-8")
    log(f"\nZapisano wyniki w {OUT} (china_*)")


def plots(G, R, D, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["ESKALACJA", "NEUTRALNE", "DEESKALACJA", "nie_cla"]
    col = {"ESKALACJA": "#c0392b", "NEUTRALNE": "#7f8c8d", "DEESKALACJA": "#27ae60", "nie_cla": "#2c3e50"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (oc, t) in zip(axes, (("CHN_ALL_ar_d0", "Koszyk 5 ETF: AR dnia sesji"),
                                   ("CHN_TECH_ar_d0", "KWEB+CQQQ: AR dnia sesji"),
                                   ("CHN_ALL_ar_d0_4", "Koszyk 5 ETF: CAR 5 sesji"))):
        data = [1e4 * G.loc[G["etykieta4"] == o, oc].dropna() for o in order]
        ax.boxplot(data, showfliers=False, widths=0.5)
        for i, (o, d) in enumerate(zip(order, data), 1):
            ax.scatter(np.full(len(d), i) + ams.RNG.uniform(-0.15, 0.15, len(d)), d, s=10, alpha=0.6, color=col[o])
            ax.scatter(i, d.mean(), marker="D", color="black", zorder=5)
        ax.set_xticks(range(1, 5), [f"{o}\n(n={len(d)})" for o, d in zip(order, data)], fontsize=8)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(t); ax.set_ylabel("pb")
    fig.suptitle("Zwrot nadzwyczajny chińskich ETF-ów (vs SPY) po poście wg etykiety (gold MG_pierwszy; romb = średnia)")
    fig.tight_layout(); fig.savefig(OUT / "fig_china_zwroty_wg_etykiety.png", dpi=130); plt.close(fig)

    P = R.loc["2018-01-01":"2019-12-31"]
    fig, ax = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    ax[0].plot(P.index, np.exp(P["SPY_ret"].fillna(0).cumsum()), color="#2c3e50", label="SPY")
    ax[0].plot(P.index, np.exp(P["KWEB_ret"].fillna(0).cumsum()), color="#c0392b", label="KWEB")
    ax[0].plot(P.index, np.exp(P["CHN_ALL_ar"].fillna(0).cumsum()), color="#8e44ad", label="koszyk 5 ETF: skumul. AR")
    for d in cfg.key_events.values():
        ax[0].axvline(pd.Timestamp(d), color="grey", lw=0.6, ls=":")
    ax[0].legend(); ax[0].set_ylabel("indeks (2018-01=1)")
    ax[1].bar(D.index, D["e_esk"], color="#c0392b", width=1, label="E[eskalacja]")
    ax[1].bar(D.index, -D["e_deesk"], color="#27ae60", width=1, label="-E[deeskalacja]")
    ax[1].legend(); ax[1].set_ylabel("dzienny indeks z klasyfikatora")
    fig.suptitle("Chińskie ETF-y vs dzienny indeks eskalacji/deeskalacji (linie kropkowane = kluczowe zdarzenia)")
    fig.tight_layout(); fig.savefig(OUT / "fig_china_indeks.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
