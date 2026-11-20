"""Analiza adnotacji MG + PK: zgodność (kappa), gold standard, połączenie z rynkiem, testy, indeks dzienny.

Uruchom z katalogu projektu:  python analysis/annotation_market_study.py
Wyniki: analysis/out/*.csv, analysis/out/*.png, analysis/out/summary.txt

Uwaga: opisowe/eksploracyjne. Mała próba (~70 postów o cłach), wiele testów -> korekta FDR (Benjamini-Hochberg).
"""
from __future__ import annotations

import itertools
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "annotations done"
MERGED = ROOT / "clean_out" / "merged"
OUT = ROOT / "analysis" / "out"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(2026)
NY = "America/New_York"
LOG: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def section(t):
    log("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


# ============================================================================ 1. wczytanie i połączenie
def read_sheet(path: Path, who: str) -> pd.DataFrame:
    d = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    d = d.rename(columns={d.columns[6]: "obszar"})
    d["oznaczajacy"] = who
    d["plik"] = path.name
    for c in ("cla", "kierunek", "obszar", "uwagi"):
        d[c] = d[c].str.strip()
    d["cla"] = d["cla"].str.lower()
    d["kierunek"] = d["kierunek"].str.upper()
    return d[["item_id", "etap", "oznaczajacy", "plik", "cla", "kierunek", "obszar", "uwagi"]]


def load_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for p in sorted(ANN.glob("[12]_*_*.csv")):
        m = re.match(r"([12])_(pilotaz|glowna)_([A-Z]+)\.csv$", p.name)
        if m:
            frames.append(read_sheet(p, m.group(3)))
    L = pd.concat(frames, ignore_index=True)
    probs = []
    bad = (L["cla"] == "nie") & (L["kierunek"] != "")
    for r in L[bad].itertuples():
        probs.append((r.plik, r.item_id, "kierunek przy cla=nie -> zignorowano", r.kierunek))
    L.loc[L["cla"] == "nie", "kierunek"] = "n/d"
    L.loc[L["cla"] == "nie", "obszar"] = ""
    miss = (L["cla"] == "tak") & (L["kierunek"] == "")
    for r in L[miss].itertuples():
        probs.append((r.plik, r.item_id, "brak kierunku przy cla=tak -> brak (wykluczone z analiz kierunku)", ""))
    L.loc[miss, "kierunek"] = ""
    # etykieta 4-klasowa: nie / ESKALACJA / DEESKALACJA / NEUTRALNE
    L["etykieta4"] = np.where(L["cla"] == "nie", "nie_cla", L["kierunek"].replace("", np.nan))
    P = pd.read_csv(ROOT / "annotation_files" / "klucz_probki.csv", sep=";", dtype={"post_id": str})
    P["time_utc"] = pd.to_datetime(P["time_utc"], utc=True, format="ISO8601")
    P["hit"] = P["hit"].astype(str).str.lower().isin(["true", "1"])
    L = L.merge(P[["item_id", "post_id", "time_utc", "stratum", "hit", "weight", "oznaczajacy", "text"]]
                .rename(columns={"oznaczajacy": "przydzial"}), on="item_id", how="left", validate="m:1")
    # kontrola: czy osoba była przydzielona do posta
    wrong = L[~L.apply(lambda r: r["oznaczajacy"] in str(r["przydzial"]).split("|"), axis=1)]
    for r in wrong.itertuples():
        probs.append((r.plik, r.item_id, "post nieprzydzielony tej osobie", r.przydzial))
    return L, pd.DataFrame(probs, columns=["plik", "item_id", "problem", "wartosc"])


# ============================================================================ 2. zgodność
def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a) | set(b))
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return np.nan if pe >= 1 else (po - pe) / (1 - pe)


def gwet_ac1(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a) | set(b))
    q = len(cats)
    if q < 2:
        return 1.0
    po = np.mean(a == b)
    pi = np.array([(np.mean(a == c) + np.mean(b == c)) / 2 for c in cats])
    pe = np.sum(pi * (1 - pi)) / (q - 1)
    return (po - pe) / (1 - pe)


def kripp_alpha_nominal(a, b) -> float:
    vals = np.concatenate([a, b])
    cats = sorted(set(vals))
    ix = {c: i for i, c in enumerate(cats)}
    O = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        O[ix[x], ix[y]] += 1
        O[ix[y], ix[x]] += 1
    n_c, n = O.sum(1), O.sum()
    Do = (n - np.trace(O)) / n
    De = (n * n - (n_c ** 2).sum()) / (n * (n - 1))
    return 1.0 if De == 0 else 1 - Do / De


def boot_ci(a, b, f, B=4000):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    v = []
    for _ in range(B):
        i = RNG.integers(0, n, n)
        x = f(a[i], b[i])
        if np.isfinite(x):
            v.append(x)
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)


def landis(k):
    if not np.isfinite(k):
        return "brak"
    return ("brak/ślad" if k < 0.2 else "słaba" if k < 0.4 else "umiarkowana" if k < 0.6
            else "znaczna" if k < 0.8 else "prawie pełna")


def agreement(L: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    W = L[L["oznaczajacy"].isin(["MG", "PK"])].pivot_table(
        index=["item_id", "etap"], columns="oznaczajacy", values=["cla", "kierunek", "etykieta4", "obszar"],
        aggfunc="first").dropna(subset=[("cla", "MG"), ("cla", "PK")])
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    W = W.reset_index()
    W["etap"] = W["etap"].replace({"nakladka": "glowna_nakladka"})
    rows = []
    specs = [
        ("cla (tak/nie)", "cla", None),
        ("kierunek | obaj cla=tak", "kierunek", "both_tak"),
        ("etykieta 4-klasowa (nie/ESK/DEESK/NEUTR)", "etykieta4", None),
        ("ESKALACJA vs reszta (binarnie, wszystkie posty)", "esk_bin", None),
        ("obszar | obaj cla=tak", "obszar", "both_tak"),
    ]
    W["esk_bin_MG"] = (W["etykieta4_MG"] == "ESKALACJA").map({True: "ESK", False: "inne"})
    W["esk_bin_PK"] = (W["etykieta4_PK"] == "ESKALACJA").map({True: "ESK", False: "inne"})
    W["obszar_MG"] = W["obszar_MG"].fillna("").replace("", "brak")
    W["obszar_PK"] = W["obszar_PK"].fillna("").replace("", "brak")
    for stage_name, mask in (("pilotaż", W["etap"] == "pilotaz"), ("główna", W["etap"] != "pilotaz"),
                             ("RAZEM", pd.Series(True, index=W.index))):
        for name, f, cond in specs:
            m = mask.copy()
            if cond == "both_tak":
                m &= (W["cla_MG"] == "tak") & (W["cla_PK"] == "tak")
            a, b = W.loc[m, f + "_MG"].fillna(""), W.loc[m, f + "_PK"].fillna("")
            ok = (a != "") & (b != "")
            a, b = a[ok].to_numpy(), b[ok].to_numpy()
            if len(a) < 3:
                continue
            k = cohen_kappa(a, b)
            lo, hi = boot_ci(a, b, cohen_kappa)
            rows.append({"zbiór": stage_name, "etykieta": name, "n": len(a), "zgodność_%": round(100 * np.mean(a == b), 1),
                         "kappa_Cohena": round(k, 3), "kappa_95CI": f"[{lo:.2f}; {hi:.2f}]",
                         "alfa_Krippendorffa": round(kripp_alpha_nominal(a, b), 3), "AC1_Gweta": round(gwet_ac1(a, b), 3),
                         "PABAK": round(2 * np.mean(a == b) - 1, 3) if len(set(a) | set(b)) <= 2 else np.nan,
                         "interpretacja (Landis-Koch)": landis(k)})
    A = pd.DataFrame(rows)
    C4 = pd.crosstab(W["etykieta4_MG"].fillna("brak"), W["etykieta4_PK"].fillna("brak"), rownames=["MG"], colnames=["PK"])
    dis = W[(W["etykieta4_MG"].fillna("") != W["etykieta4_PK"].fillna(""))]
    return A, C4, W


# ============================================================================ 3. gold standard (3 warianty)
def build_gold(L: pd.DataFrame, W: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Jeden wiersz na post. Warianty:
    MG_pierwszy: etykieta MG, a gdy MG nie oznaczał -> PK;  PK_pierwszy: odwrotnie;
    zgodne: posty pojedyncze + posty, gdzie MG i PK się zgadzają (spory odrzucone)."""
    base = L.drop_duplicates("item_id")[["item_id", "post_id", "time_utc", "stratum", "hit", "weight", "text"]]
    out = {}
    for name, order in (("MG_pierwszy", ["MG", "PK"]), ("PK_pierwszy", ["PK", "MG"])):
        rk = {w: i for i, w in enumerate(order)}
        g = L.assign(_r=L["oznaczajacy"].map(rk)).sort_values("_r").drop_duplicates("item_id")
        out[name] = base.merge(g[["item_id", "cla", "kierunek", "etykieta4", "obszar"]], on="item_id")
    n_ann = L.groupby("item_id")["oznaczajacy"].nunique()
    disagree = set(W.loc[W["etykieta4_MG"].fillna("") != W["etykieta4_PK"].fillna(""), "item_id"])
    g = out["MG_pierwszy"]
    out["zgodne"] = g[~g["item_id"].isin(disagree)].copy()
    for k in out:
        out[k]["n_oznaczajacych"] = out[k]["item_id"].map(n_ann)
    return out


# ============================================================================ 4. reakcja rynku dla każdego posta
INSTR = {"S_P_500": "S&P 500", "Nasdaq": "Nasdaq", "Hang_Seng": "Hang Seng", "EURUSD": "EURUSD", "TLT": "TLT"}


def eff_day(t: pd.Series) -> pd.Series:
    ny = pd.DatetimeIndex(pd.to_datetime(t, utc=True)).tz_convert(NY)
    d = ny.normalize().tz_localize(None) + pd.to_timedelta((ny.hour >= 16).astype(int), unit="D")
    return pd.Series(d, index=t.index)


def market_features(posts: pd.DataFrame) -> pd.DataFrame:
    H = pd.read_csv(MERGED / "hourly.csv", parse_dates=["hour_utc"]).set_index("hour_utc").sort_index()
    T = pd.read_csv(MERGED / "trading.csv", parse_dates=["day"]).set_index("day").sort_index()
    days = T.index
    out = pd.DataFrame(index=posts.index)
    t = pd.to_datetime(posts["time_utc"], utc=True)
    h0 = t.dt.floor("h")
    for code in INSTR:
        hr = H[f"{code}_ret"]
        cum = hr.fillna(0).cumsum()
        has = hr.notna().astype(int).cumsum()
        pos = cum.index.searchsorted(h0)  # indeks godziny zawierającej post

        def window(start_off, length):
            s = np.clip(pos + start_off, 0, len(cum) - 1)
            e = np.clip(pos + start_off + length, 0, len(cum) - 1)
            r = cum.to_numpy()[e - 1] - np.where(s > 0, cum.to_numpy()[np.maximum(s - 1, 0)], 0)
            n = has.to_numpy()[e - 1] - np.where(s > 0, has.to_numpy()[np.maximum(s - 1, 0)], 0)
            return np.where(n > 0, r, np.nan)

        # post: od następnej pełnej godziny (bez częściowej godziny, w której był post)
        for lab, L_ in (("1h", 1), ("3h", 3), ("6h", 6), ("24h", 24)):
            out[f"{code}_post_{lab}"] = window(1, L_)
            out[f"{code}_pre_{lab}"] = window(-L_, L_)
        # dzienne: dzień efektywny (pierwsza sesja >= eff_day)
        dr = T[f"{code}_ret"]
        di = days.searchsorted(eff_day(t).to_numpy())
        arr = dr.to_numpy()

        def dsum(o, n):
            r = []
            for i in di:
                a, b = i + o, i + o + n
                if a < 0 or b > len(arr):
                    r.append(np.nan); continue
                seg = arr[a:b]
                r.append(np.nansum(seg) if np.isfinite(seg).any() else np.nan)
            return np.array(r)

        out[f"{code}_d0"] = dsum(0, 1)
        out[f"{code}_d1"] = dsum(1, 1)
        out[f"{code}_d0_2"] = dsum(0, 3)
        out[f"{code}_d0_4"] = dsum(0, 5)
        out[f"{code}_pre_d5"] = dsum(-5, 5)
        # standaryzacja: zwrot dnia 0 / zmienność 60 sesji przed
        vol = dr.rolling(60, min_periods=20).std().shift(1).to_numpy()
        v = np.array([vol[i] if 0 <= i < len(vol) else np.nan for i in di])
        out[f"{code}_d0_z"] = out[f"{code}_d0"] / v
        out[f"{code}_d0_abs_z"] = out[f"{code}_d0"].abs() / v
        out[f"{code}_post_24h_abs"] = np.abs(out[f"{code}_post_24h"])
    out["eff_trading_day"] = days[np.clip(days.searchsorted(eff_day(t).to_numpy()), 0, len(days) - 1)]
    # nadwyżki z market_long (tylko posty słownikowe, z kontrolą: excess = zwrot - zwrot w oknach kontrolnych)
    M = pd.read_csv(MERGED / "market_long.csv")
    M = M[M["event_id"].str.startswith("tw_")]
    for ins in ("S&P 500", "Hang Seng", "EURUSD"):
        X = M[M["instrument"] == ins].pivot_table(index="event_id", columns="horizon", values=["excess_ret", "excess_abs"])
        X.columns = [f"ML_{ins.replace(' ', '').replace('&', '')}_{a}_{h}" for a, h in X.columns]
        ev = "tw_" + posts["post_id"].astype(str)
        out = out.join(X.reindex(ev.to_numpy()).set_index(posts.index))
    return out


# ============================================================================ 5. testy
def perm_p(x, y, B=10000):
    x, y = np.asarray(x), np.asarray(y)
    obs = x.mean() - y.mean()
    z = np.concatenate([x, y])
    n = len(x)
    cnt = 0
    for _ in range(B):
        RNG.shuffle(z)
        cnt += abs(z[:n].mean() - z[n:].mean()) >= abs(obs) - 1e-15
    return (cnt + 1) / (B + 1)


def boot_diff_ci(x, y, B=4000):
    x, y = np.asarray(x), np.asarray(y)
    v = [RNG.choice(x, len(x)).mean() - RNG.choice(y, len(y)).mean() for _ in range(B)]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return q
    o = np.argsort(pv)
    r = pv[o] * n / np.arange(1, n + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    qq = np.empty(n)
    qq[o] = np.minimum(r, 1)
    q[ok] = qq
    return q


def compare(G: pd.DataFrame, outcomes: list[str], variant: str) -> pd.DataFrame:
    """Kontrasty grup etykiet dla każdej miary rynku."""
    lab = G["etykieta4"]
    groups = {
        "ESKALACJA vs DEESKALACJA": (lab == "ESKALACJA", lab == "DEESKALACJA"),
        "ESKALACJA vs nie_cla": (lab == "ESKALACJA", lab == "nie_cla"),
        "DEESKALACJA vs nie_cla": (lab == "DEESKALACJA", lab == "nie_cla"),
        "cla=tak vs cla=nie": (G["cla"] == "tak", G["cla"] == "nie"),
        "ESKALACJA vs (DEESK+NEUTR)": (lab == "ESKALACJA", lab.isin(["DEESKALACJA", "NEUTRALNE"])),
    }
    rows = []
    for oc in outcomes:
        for gname, (ma, mb) in groups.items():
            x, y = G.loc[ma, oc].dropna().to_numpy(), G.loc[mb, oc].dropna().to_numpy()
            if len(x) < 4 or len(y) < 4:
                continue
            t = stats.ttest_ind(x, y, equal_var=False)
            u = stats.mannwhitneyu(x, y, alternative="two-sided")
            lo, hi = boot_diff_ci(x, y)
            sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
            sc = 1 if "_z" in oc else 1e4  # miary _z są w odchyleniach standardowych, nie w pb
            rows.append({"wariant": variant, "miara": oc, "kontrast": gname, "n_A": len(x), "n_B": len(y),
                         "śr_A_pb": sc * x.mean(), "śr_B_pb": sc * y.mean(), "różnica_pb": sc * (x.mean() - y.mean()),
                         "CI95_pb": f"[{1e4 * lo:.1f}; {1e4 * hi:.1f}]" if "_z" not in oc else f"[{lo:.2f}; {hi:.2f}] (z)",
                         "d_Cohena": (x.mean() - y.mean()) / sd if sd > 0 else np.nan,
                         "p_Welch": t.pvalue, "p_MannWhitney": u.pvalue, "p_permutacyjny": perm_p(x, y, 3000)})
        # korelacja: wynik kierunku (-1/0/+1) wśród cla=tak
        m = G["etykieta4"].isin(["ESKALACJA", "DEESKALACJA", "NEUTRALNE"]) & G[oc].notna()
        if m.sum() >= 8:
            s = G.loc[m, "etykieta4"].map({"ESKALACJA": 1, "NEUTRALNE": 0, "DEESKALACJA": -1})
            r, p = stats.spearmanr(s, G.loc[m, oc])
            rows.append({"wariant": variant, "miara": oc, "kontrast": "Spearman: wynik kierunku (+1 ESK, 0, -1 DEESK)",
                         "n_A": int(m.sum()), "n_B": np.nan, "rho": r, "p_Welch": np.nan, "p_MannWhitney": np.nan,
                         "p_permutacyjny": p})
    R = pd.DataFrame(rows)
    R["p_min"] = R[["p_Welch", "p_MannWhitney", "p_permutacyjny"]].min(axis=1)
    R["q_FDR(perm)"] = bh(R["p_permutacyjny"])
    return R


def ols(y, X, clusters=None):
    """OLS z błędami HC3 albo (gdy podano clusters) klastrowanymi CR1 (np. po dniu sesji)."""
    X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    if clusters is None:
        h = np.einsum("ij,jk,ik->i", X, XtXi, X)
        S = (X.T * (e / (1 - h)) ** 2) @ X
        dof = n - k
    else:
        cl = pd.factorize(np.asarray(clusters))[0]
        G = cl.max() + 1
        S = np.zeros((k, k))
        for g in range(G):
            u = X[cl == g].T @ e[cl == g]
            S += np.outer(u, u)
        S *= G / (G - 1) * (n - 1) / (n - k)
        dof = G - 1
    V = XtXi @ S @ XtXi
    se = np.sqrt(np.diag(V))
    p = 2 * stats.t.sf(np.abs(b / se), dof)
    return b, se, p, 1 - (e @ e) / (((y - y.mean()) ** 2).sum())


def regressions(G: pd.DataFrame, F: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Zwrot S&P = a + b1*ESK + b2*DEESK + b3*NEUTR + b4*zwrot_przed + e  (baza: posty nie o cłach), SE klastrowane po dniu sesji."""
    rows = []
    for y_col, pre_col in (("S_P_500_post_1h", "S_P_500_pre_1h"), ("S_P_500_post_3h", "S_P_500_pre_3h"),
                           ("S_P_500_post_24h", "S_P_500_pre_24h"), ("S_P_500_d0", "S_P_500_pre_d5"),
                           ("S_P_500_d0_2", "S_P_500_pre_d5"), ("S_P_500_d0_z", "S_P_500_pre_d5")):
        d = G[["etykieta4", "eff_trading_day"]].join(F[[y_col, pre_col]]).dropna()
        X = np.column_stack([(d["etykieta4"] == v).astype(float) for v in ("ESKALACJA", "DEESKALACJA", "NEUTRALNE")]
                            + [d[pre_col].to_numpy()])
        b, se, p, r2 = ols(d[y_col].to_numpy(), X, clusters=d["eff_trading_day"].to_numpy())
        sc = 1 if y_col.endswith("_z") else 1e4
        for nm, i in (("ESKALACJA", 1), ("DEESKALACJA", 2), ("NEUTRALNE", 3), ("zwrot_przed", 4)):
            rows.append({"wariant": variant, "y": y_col, "zmienna": nm, "beta": b[i] * (sc if nm != "zwrot_przed" else 1),
                         "jednostka": "z" if sc == 1 else ("pb" if nm != "zwrot_przed" else "współcz."),
                         "SE_klastr": se[i] * (sc if nm != "zwrot_przed" else 1), "p": p[i], "R2": r2, "n": len(d)})
    return pd.DataFrame(rows)


# ============================================================================ 6. indeks dzienny na całym korpusie
def corpus_index(gold: pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline

    def mk():
        return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=True),
                             LogisticRegression(C=4, class_weight="balanced", max_iter=3000))

    res = {}
    g = gold.dropna(subset=["etykieta4"])
    g = g[g["etykieta4"] != ""]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pred_cla = cross_val_predict(mk(), g["text"], g["cla"], cv=cv)
    res["cla_report"] = classification_report(g["cla"], pred_cla, digits=3, zero_division=0)
    res["cla_f1"] = f1_score(g["cla"], pred_cla, pos_label="tak")
    gt = g[g["cla"] == "tak"]
    gt = gt[gt["kierunek"].isin(["ESKALACJA", "DEESKALACJA", "NEUTRALNE"])]
    cv3 = StratifiedKFold(min(5, gt["kierunek"].value_counts().min()), shuffle=True, random_state=0)
    pred_k = cross_val_predict(mk(), gt["text"], gt["kierunek"], cv=cv3)
    res["kier_report"] = classification_report(gt["kierunek"], pred_k, digits=3, zero_division=0)
    res["kier_f1_macro"] = f1_score(gt["kierunek"], pred_k, average="macro")
    res["kier_majority_acc"] = gt["kierunek"].value_counts(normalize=True).max()

    # predykcja na korpusie 2018-2019 (oryginalne posty)
    tw = pd.read_csv(ROOT / "data" / "trump_tweets.csv", dtype={"id": str})
    tw["time_utc"] = pd.to_datetime(tw["date"], utc=True)
    tw = tw[(tw["time_utc"] >= "2018-01-01") & (tw["time_utc"] < "2020-01-01") & (tw["isRetweet"] == "f")]
    tw = tw[~tw["text"].str.startswith("RT @") & (tw["text"].str.strip().str.len() >= 15)].copy()
    m1 = mk().fit(g["text"], g["cla"])
    m2 = mk().fit(gt["text"], gt["kierunek"])
    tw["p_cla"] = m1.predict_proba(tw["text"])[:, list(m1.classes_).index("tak")]
    tw["cla_hat"] = tw["p_cla"] >= 0.5
    pk = m2.predict_proba(tw["text"])
    for i, c in enumerate(m2.classes_):
        tw[f"p_{c}"] = pk[:, i] * tw["p_cla"]
    tw["kier_hat"] = np.where(tw["cla_hat"], m2.classes_[pk.argmax(1)], "nie_cla")
    tw["eff_day"] = eff_day(tw["time_utc"])
    T = pd.read_csv(MERGED / "trading.csv", parse_dates=["day"]).set_index("day").sort_index()
    di = T.index.searchsorted(tw["eff_day"].to_numpy())
    tw["trading_day"] = T.index[np.clip(di, 0, len(T) - 1)]
    agg = tw.groupby("trading_day").agg(
        n_all=("id", "size"), n_cla=("cla_hat", "sum"),
        n_esk=("kier_hat", lambda s: (s == "ESKALACJA").sum()), n_deesk=("kier_hat", lambda s: (s == "DEESKALACJA").sum()),
        e_esk=("p_ESKALACJA", "sum"), e_deesk=("p_DEESKALACJA", "sum"))
    D = T[["S_P_500_ret", "Nasdaq_ret", "Hang_Seng_ret", "EURUSD_ret", "TLT_ret", "S_P_500_absret", "tw_topic_n"]].join(agg).fillna(
        {"n_all": 0, "n_cla": 0, "n_esk": 0, "n_deesk": 0, "e_esk": 0, "e_deesk": 0})
    D["net_esk"] = D["n_esk"] - D["n_deesk"]
    D["e_net"] = D["e_esk"] - D["e_deesk"]
    res["daily"] = D
    res["tweets"] = tw
    return res


def lead_lag(D: pd.DataFrame, x: str, y: str, lags=range(-5, 6)) -> pd.DataFrame:
    """corr(x_t, y_{t+k}). k>0: tweety wyprzedzają rynek; k<0: rynek wyprzedza tweety."""
    rows = []
    for k in lags:
        a = D[x]
        b = D[y].shift(-k)
        m = a.notna() & b.notna()
        r, p = stats.pearsonr(a[m], b[m])
        rs, ps = stats.spearmanr(a[m], b[m])
        rows.append({"x": x, "y": y, "lag_k": k, "n": int(m.sum()), "pearson_r": r, "p_pearson": p, "spearman_rho": rs, "p_spearman": ps})
    return pd.DataFrame(rows)


def granger(D: pd.DataFrame, cause: str, effect: str, p: int) -> dict:
    d = D[[cause, effect]].dropna().copy()
    for i in range(1, p + 1):
        d[f"y{i}"] = d[effect].shift(i)
        d[f"x{i}"] = d[cause].shift(i)
    d = d.dropna()
    y = d[effect].to_numpy()
    Xr = np.column_stack([np.ones(len(d))] + [d[f"y{i}"] for i in range(1, p + 1)])
    Xu = np.column_stack([Xr] + [d[f"x{i}"] for i in range(1, p + 1)])
    rss = lambda X: ((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2).sum()
    rr, ru = rss(Xr), rss(Xu)
    df2 = len(y) - Xu.shape[1]
    F = ((rr - ru) / p) / (ru / df2)
    return {"przyczyna": cause, "skutek": effect, "lagi": p, "F": F, "p": stats.f.sf(F, p, df2), "n": len(y)}


# ============================================================================ 7. moc
def mde(sd, n1, n2, alpha=0.05, power=0.8):
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z * sd * np.sqrt(1 / n1 + 1 / n2)


# ============================================================================ main
def main():
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
    section("1. DANE I KONTROLA JAKOŚCI")
    L, probs = load_labels()
    log("Wiersze adnotacji:", len(L), "| unikalne posty:", L["item_id"].nunique())
    log(L.groupby(["plik"]).agg(n=("item_id", "size"), cla_tak=("cla", lambda s: (s == "tak").sum()),
                                ESK=("kierunek", lambda s: (s == "ESKALACJA").sum()),
                                DEESK=("kierunek", lambda s: (s == "DEESKALACJA").sum()),
                                NEUTR=("kierunek", lambda s: (s == "NEUTRALNE").sum())).to_string())
    log("\nProblemy w arkuszach:\n" + (probs.to_string(index=False) if len(probs) else "brak"))
    probs.to_csv(OUT / "01_problemy_arkuszy.csv", sep=";", index=False, encoding="utf-8-sig")

    section("2. ZGODNOŚĆ MG vs PK (kappa Cohena, alfa Krippendorffa, AC1 Gweta; CI z bootstrapu)")
    A, C4, W = agreement(L)
    log(A.to_string(index=False))
    log("\nMacierz pomyłek (etykieta 4-klasowa, pilotaż + nakładka):\n" + C4.to_string())
    A.to_csv(OUT / "02_kappa.csv", sep=";", index=False, encoding="utf-8-sig")
    C4.to_csv(OUT / "02_macierz_pomylek.csv", sep=";", encoding="utf-8-sig")
    dis = W[W["etykieta4_MG"].fillna("") != W["etykieta4_PK"].fillna("")].merge(
        L.drop_duplicates("item_id")[["item_id", "text"]], on="item_id")
    uw = L.pivot_table(index="item_id", columns="oznaczajacy", values="uwagi", aggfunc="first")
    dis = dis.join(uw.add_prefix("uwagi_"), on="item_id")
    dis[["item_id", "etap", "etykieta4_MG", "etykieta4_PK", "obszar_MG", "obszar_PK", "uwagi_MG", "uwagi_PK", "text"]].to_csv(
        OUT / "02_spory_do_rozstrzygniecia.csv", sep=";", index=False, encoding="utf-8-sig")
    log(f"\nSporne posty: {len(dis)} z {len(W)} -> analysis/out/02_spory_do_rozstrzygniecia.csv")
    # systematyczne różnice (test McNemara-Bowkera dla marginesów ESK)
    both = W[(W["cla_MG"] == "tak") & (W["cla_PK"] == "tak")]
    b01 = ((both["kierunek_MG"] == "ESKALACJA") & (both["kierunek_PK"] != "ESKALACJA")).sum()
    b10 = ((both["kierunek_MG"] != "ESKALACJA") & (both["kierunek_PK"] == "ESKALACJA")).sum()
    log(f"Asymetria ESKALACJA (obaj cla=tak): MG=ESK a PK inne: {b01}, PK=ESK a MG inne: {b10}, "
        f"McNemar exact p={stats.binomtest(int(b01), int(b01 + b10)).pvalue if b01 + b10 else np.nan:.4f}")

    section("3. GOLD STANDARD (3 warianty: MG_pierwszy, PK_pierwszy, zgodne)")
    GOLD = build_gold(L, W)
    for k, g in GOLD.items():
        log(f"{k:12s} n={len(g)}  " + str(g["etykieta4"].value_counts(dropna=False).to_dict()))
        g.drop(columns=["text"]).to_csv(OUT / f"03_gold_{k}.csv", sep=";", index=False, encoding="utf-8-sig")
    # ważony udział (wagi = 1/p wylosowania) -> szacunek dla całej populacji postów 2018-2019
    g = GOLD["MG_pierwszy"]
    for v in ("tak",):
        w = g["weight"]
        log(f"Szacowany odsetek postów o cłach w populacji (ważony, MG_pierwszy): "
            f"{100 * (w * (g['cla'] == v)).sum() / w.sum():.1f}%  (nieważony: {100 * (g['cla'] == v).mean():.1f}%)")
    hit_tab = pd.crosstab(g["hit"], g["cla"], margins=True)
    log("Słownik (hit) vs etykieta cla:\n" + hit_tab.to_string())

    section("4. REAKCJA RYNKU (wszystkie posty z próby; godzinowe i dzienne)")
    feats = market_features(GOLD["MG_pierwszy"].reset_index(drop=True))
    log("Pokrycie miar (niepuste):\n" + feats.notna().sum().filter(like="S_P_500").to_string())
    ALL = {k: g.reset_index(drop=True).join(market_features(g.reset_index(drop=True))) for k, g in GOLD.items()}
    ALL["MG_pierwszy"].drop(columns=["text"]).to_csv(OUT / "04_posty_z_rynkiem_MG_pierwszy.csv", sep=";", index=False, encoding="utf-8-sig")

    section("5. TESTY: czy kierunek ceł wiąże się ze zwrotem S&P 500 (i innych rynków)?")
    sp_out = ["S_P_500_post_1h", "S_P_500_post_3h", "S_P_500_post_6h", "S_P_500_post_24h", "S_P_500_d0", "S_P_500_d1",
              "S_P_500_d0_2", "S_P_500_d0_4", "S_P_500_d0_z"]
    vol_out = ["S_P_500_post_24h_abs", "S_P_500_d0_abs_z"]
    pre_out = ["S_P_500_pre_1h", "S_P_500_pre_3h", "S_P_500_pre_24h", "S_P_500_pre_d5"]
    other = [f"{c}_{h}" for c in ("Nasdaq", "Hang_Seng", "EURUSD", "TLT") for h in ("post_3h", "post_24h", "d0")]
    ml = [c for c in ALL["MG_pierwszy"].columns if c.startswith("ML_SP500_excess_ret")]
    R = pd.concat([compare(G, sp_out + vol_out + pre_out + other + ml, k) for k, G in ALL.items()], ignore_index=True)
    R.to_csv(OUT / "05_testy_kontrastow.csv", sep=";", index=False, encoding="utf-8-sig")
    show = ["wariant", "miara", "kontrast", "n_A", "n_B", "śr_A_pb", "śr_B_pb", "różnica_pb", "CI95_pb", "d_Cohena",
            "p_permutacyjny", "q_FDR(perm)"]
    main_c = R[(R["kontrast"] == "ESKALACJA vs DEESKALACJA") & R["miara"].isin(sp_out)]
    log("\nGłówny test - ESKALACJA vs DEESKALACJA, S&P 500 (pb = punkty bazowe):\n" + main_c[show].round(3).to_string(index=False))
    log("\nSpearman: wynik kierunku vs zwrot S&P (cla=tak):\n" + R[R["kontrast"].str.startswith("Spearman") & R["miara"].isin(sp_out)]
        [["wariant", "miara", "n_A", "rho", "p_permutacyjny", "q_FDR(perm)"]].round(3).to_string(index=False))
    log("\nZmienność (|zwrot|) - cła vs nie cła:\n" + R[R["miara"].isin(vol_out) & R["kontrast"].isin(["cla=tak vs cla=nie", "ESKALACJA vs nie_cla"])][show].round(3).to_string(index=False))
    log("\nOdwrotna przyczynowość - zwrot PRZED postem:\n" + R[R["miara"].isin(pre_out) & (R["kontrast"] == "ESKALACJA vs DEESKALACJA")][show].round(3).to_string(index=False))
    log("\nInne rynki (ESK vs DEESK):\n" + R[R["miara"].isin(other) & (R["kontrast"] == "ESKALACJA vs DEESKALACJA")][show].round(3).to_string(index=False))
    if ml:
        log("\nNadwyżki z market_long (tylko posty słownikowe; zwrot minus okna kontrolne):\n" +
            R[R["miara"].isin(ml) & (R["kontrast"] == "ESKALACJA vs DEESKALACJA")][show].round(3).to_string(index=False))
    sig = R[R["q_FDR(perm)"] < 0.10]
    log(f"\nWszystkich testów: {len(R)}. Nominalnie p_perm<0.05: {(R['p_permutacyjny'] < 0.05).sum()} "
        f"(oczekiwane fałszywe przy braku efektu: ~{0.05 * len(R):.0f}). Po FDR q<0.10: {len(sig)}")
    if len(sig):
        log(sig[show].round(3).to_string(index=False))

    log("\nRegresje OLS (SE klastrowane po dniu sesji), baza = posty nie o cłach, kontrola zwrotu przed postem:")
    RG = pd.concat([regressions(G, G, k) for k, G in ALL.items()], ignore_index=True)
    RG.to_csv(OUT / "05_regresje.csv", sep=";", index=False, encoding="utf-8-sig")
    log(RG[RG["zmienna"] != "zwrot_przed"].round(4).to_string(index=False))


    section("5a. ODPORNOŚĆ: jeden dzień sesji = jedna obserwacja (posty z tego samego dnia dzielą zwrot)")
    for k in ("MG_pierwszy", "zgodne"):
        G = ALL[k]
        log(f"[{k}] postów o cłach: {(G['cla'] == 'tak').sum()}, unikalnych dni sesji z postem o cłach: "
            f"{G.loc[G['cla'] == 'tak', 'eff_trading_day'].nunique()}")
        U = G[G["etykieta4"].isin(["ESKALACJA", "DEESKALACJA"])].drop_duplicates(["eff_trading_day", "etykieta4"])
        both_days = U["eff_trading_day"].duplicated(keep=False)
        U = U[~both_days]  # dni z ESK i DEESK jednocześnie wyrzucone (niejednoznaczne)
        for oc in ("S_P_500_d0", "S_P_500_d0_2", "S_P_500_d0_4", "S_P_500_post_24h"):
            x = U.loc[U["etykieta4"] == "ESKALACJA", oc].dropna(); y = U.loc[U["etykieta4"] == "DEESKALACJA", oc].dropna()
            log(f"   {oc:18s} dni ESK={len(x):3d} śr={1e4 * x.mean():7.1f} pb | dni DEESK={len(y):3d} śr={1e4 * y.mean():7.1f} pb"
                f" | różnica={1e4 * (x.mean() - y.mean()):7.1f} pb, p_perm={perm_p(x, y, 5000):.3f}, "
                f"p_MW={stats.mannwhitneyu(x, y).pvalue:.3f}")
    G = ALL["MG_pierwszy"]
    log("\nPosty DEESKALACJA (MG_pierwszy) - które dni napędzają wynik:")
    dd = G[G["etykieta4"] == "DEESKALACJA"].sort_values("time_utc")
    for r in dd.itertuples():
        log(f"   {str(r.time_utc)[:16]}  sesja {str(r.eff_trading_day)[:10]}  d0={1e4 * r.S_P_500_d0:7.1f} pb  "
            f"24h={1e4 * r.S_P_500_post_24h if pd.notna(r.S_P_500_post_24h) else float('nan'):7.1f} pb  {r.text[:90]}")
    # bez dwóch najbardziej skrajnych dni
    for oc in ("S_P_500_d0", "S_P_500_post_24h"):
        x = G.loc[G["etykieta4"] == "ESKALACJA", oc].dropna(); y = G.loc[G["etykieta4"] == "DEESKALACJA", oc].dropna()
        y2 = y.drop(y.abs().nlargest(2).index); x2 = x.drop(x.abs().nlargest(2).index)
        log(f"Bez 2 najbardziej skrajnych obserwacji w każdej grupie, {oc}: ESK={1e4 * x2.mean():.1f} pb, "
            f"DEESK={1e4 * y2.mean():.1f} pb, p_perm={perm_p(x2, y2, 5000):.3f}")

    section("5b. OBSZAR: Chiny vs reszta (cla=tak)")
    G = ALL["MG_pierwszy"]
    t = G[G["cla"] == "tak"].copy()
    t["chiny"] = t["obszar"].str.contains("Chin", case=False, na=False)
    log(pd.crosstab(t["obszar"].replace("", "brak"), t["kierunek"].replace("", "brak_kierunku")).to_string())
    rows = []
    for oc in ["S_P_500_d0", "S_P_500_post_24h", "Hang_Seng_d0", "Hang_Seng_post_24h"]:
        for sub, m in (("Chiny", t["chiny"]), ("nie-Chiny", ~t["chiny"])):
            s = t[m]
            x = s.loc[s["kierunek"] == "ESKALACJA", oc].dropna(); y = s.loc[s["kierunek"] == "DEESKALACJA", oc].dropna()
            if len(x) >= 3 and len(y) >= 3:
                rows.append({"miara": oc, "obszar": sub, "n_ESK": len(x), "n_DEESK": len(y), "ESK_pb": 1e4 * x.mean(),
                             "DEESK_pb": 1e4 * y.mean(), "p_perm": perm_p(x, y, 3000)})
    OB = pd.DataFrame(rows)
    log(OB.round(3).to_string(index=False) if len(OB) else "za mało obserwacji")
    OB.to_csv(OUT / "05_obszar_chiny.csv", sep=";", index=False, encoding="utf-8-sig")

    section("6. MOC STATYSTYCZNA (minimalny wykrywalny efekt, alfa=0.05, moc=0.8)")
    for oc, lab in (("S_P_500_post_1h", "1h"), ("S_P_500_post_24h", "24h"), ("S_P_500_d0", "dzień 0")):
        sd = G[oc].std()
        n1 = (G["etykieta4"] == "ESKALACJA").sum(); n2 = (G["etykieta4"] == "DEESKALACJA").sum(); n0 = (G["cla"] == "nie").sum()
        log(f"{lab:8s} sd={1e4 * sd:6.1f} pb | ESK(n={n1}) vs DEESK(n={n2}): MDE={1e4 * mde(sd, n1, n2):6.1f} pb "
            f"| ESK vs nie_cla(n={n0}): MDE={1e4 * mde(sd, n1, n0):6.1f} pb")
        need = int(np.ceil(2 * ((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * sd / 0.0025) ** 2)) if lab != "1h" else \
            int(np.ceil(2 * ((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * sd / 0.0005) ** 2))
        log(f"         n na grupę potrzebne do wykrycia {'5' if lab == '1h' else '25'} pb: {need}")

    section("7. INDEKS DZIENNY NA CAŁYM KORPUSIE (klasyfikator TF-IDF+LR uczony na gold 'MG_pierwszy')")
    CI = corpus_index(GOLD["MG_pierwszy"])
    log("Walidacja krzyżowa 5-fold, cla:\n" + CI["cla_report"])
    log("Walidacja krzyżowa, kierunek (wśród cla=tak):\n" + CI["kier_report"])
    log(f"macro-F1 kierunku = {CI['kier_f1_macro']:.3f} (trafność klasy dominującej = {CI['kier_majority_acc']:.3f})")
    D = CI["daily"]
    D.to_csv(OUT / "07_indeks_dzienny.csv", sep=";", encoding="utf-8-sig")
    log(f"Posty w korpusie 2018-2019: {len(CI['tweets'])}, przewidziane cla=tak: {int(CI['tweets']['cla_hat'].sum())}, "
        f"ESK: {(CI['tweets']['kier_hat'] == 'ESKALACJA').sum()}, DEESK: {(CI['tweets']['kier_hat'] == 'DEESKALACJA').sum()}")
    LL = pd.concat([lead_lag(D, x, y) for x in ("e_net", "e_esk", "tw_topic_n") for y in ("S_P_500_ret", "S_P_500_absret", "Hang_Seng_ret")],
                   ignore_index=True)
    LL["q_FDR"] = bh(LL["p_pearson"])
    LL.to_csv(OUT / "07_lead_lag.csv", sep=";", index=False, encoding="utf-8-sig")
    log("Korelacje opóźnione corr(x_t, y_{t+k}); k>0 = tweety PRZED rynkiem, k<0 = rynek PRZED tweetami:")
    log(LL[(LL["x"] == "e_net") & (LL["y"] == "S_P_500_ret")].round(3).to_string(index=False))
    log("\nNajsilniejsze zależności (|r|) we wszystkich kombinacjach:")
    log(LL.reindex(LL["pearson_r"].abs().sort_values(ascending=False).index).head(10).round(3).to_string(index=False))
    GR = pd.DataFrame([granger(D, a, b, p) for p in (1, 2, 5) for a, b in
                       (("e_net", "S_P_500_ret"), ("S_P_500_ret", "e_net"), ("e_esk", "S_P_500_absret"),
                        ("S_P_500_ret", "e_esk"), ("tw_topic_n", "S_P_500_ret"), ("S_P_500_ret", "tw_topic_n"))])
    GR.to_csv(OUT / "07_granger.csv", sep=";", index=False, encoding="utf-8-sig")
    log("\nPrzyczynowość Grangera (F-test):\n" + GR.round(4).to_string(index=False))
    # tygodniowo / miesięcznie: trend eskalacji vs poziom S&P
    Wk = D.resample("W").agg({"e_net": "sum", "e_esk": "sum", "S_P_500_ret": "sum"})
    r, p = stats.spearmanr(Wk["e_net"], Wk["S_P_500_ret"])
    r2, p2 = stats.spearmanr(Wk["e_net"].iloc[:-1].to_numpy(), Wk["S_P_500_ret"].iloc[1:].to_numpy())
    log(f"\nTygodniowo: corr(netto eskalacja_tydz, zwrot S&P ten sam tydzień) rho={r:.3f} p={p:.4f}; "
        f"następny tydzień rho={r2:.3f} p={p2:.4f}  (n={len(Wk)})")
    Mo = D.resample("ME").agg({"e_net": "sum", "e_esk": "sum", "S_P_500_ret": "sum"})
    r3, p3 = stats.spearmanr(Mo["e_esk"], Mo["S_P_500_ret"])
    log(f"Miesięcznie: corr(eskalacja, zwrot S&P) rho={r3:.3f} p={p3:.4f} (n={len(Mo)})")

    plots(ALL["MG_pierwszy"], D, A)
    (OUT / "summary.txt").write_text("\n".join(LOG), encoding="utf-8")
    log(f"\nZapisano wyniki w {OUT}")


def plots(G, D, A):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["ESKALACJA", "NEUTRALNE", "DEESKALACJA", "nie_cla"]
    col = {"ESKALACJA": "#c0392b", "NEUTRALNE": "#7f8c8d", "DEESKALACJA": "#27ae60", "nie_cla": "#2c3e50"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (oc, t) in zip(axes, (("S_P_500_post_1h", "S&P 500: 1h po poście"), ("S_P_500_post_24h", "S&P 500: 24h po poście"),
                                   ("S_P_500_d0", "S&P 500: zwrot dnia sesji"))):
        data = [1e4 * G.loc[G["etykieta4"] == o, oc].dropna() for o in order]
        ax.boxplot(data, showfliers=False, widths=0.5)
        for i, (o, d) in enumerate(zip(order, data), 1):
            ax.scatter(np.full(len(d), i) + RNG.uniform(-0.15, 0.15, len(d)), d, s=10, alpha=0.6, color=col[o])
            ax.scatter(i, d.mean(), marker="D", color="black", zorder=5)
        ax.set_xticks(range(1, 5), [f"{o}\n(n={len(d)})" for o, d in zip(order, data)], fontsize=8)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(t); ax.set_ylabel("pb")
    fig.suptitle("Zwrot S&P 500 po poście wg etykiety (gold MG_pierwszy; romb = średnia)")
    fig.tight_layout(); fig.savefig(OUT / "fig_zwroty_wg_etykiety.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    px = (1 + D["S_P_500_ret"].fillna(0)).cumprod()
    ax[0].plot(px.index, px, color="#2c3e50"); ax[0].set_ylabel("S&P 500 (indeks, 2018-01=1)")
    ax[1].bar(D.index, D["e_esk"], color="#c0392b", width=1, label="E[eskalacja]")
    ax[1].bar(D.index, -D["e_deesk"], color="#27ae60", width=1, label="-E[deeskalacja]")
    ax[1].legend(); ax[1].set_ylabel("dzienny indeks z klasyfikatora")
    fig.suptitle("S&P 500 vs dzienny indeks eskalacji/deeskalacji (cały korpus, klasyfikator)")
    fig.tight_layout(); fig.savefig(OUT / "fig_indeks_vs_sp500.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
