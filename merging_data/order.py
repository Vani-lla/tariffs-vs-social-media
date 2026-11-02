"""Test KOLEJNOŚCI w skali minut: kto pierwszy reaguje na tweet o cłach, rynek czy media (GKG).

Dane dzienne (FNSPID) nie rozstrzygają kolejności, bo artykuł i tweet z tego samego dnia mają tę samą datę.
Tu każde zdarzenie dostaje siatkę przedziałów co `step` (domyślnie 15 min = rozdzielczość GKG) od t - before do t + after:
  rynek:  |log P(koniec)/P(początek)| w przedziale minus średnia z k chwil placebo (ten sam dzień tygodnia i godzina NY,
          ten sam offset od chwili placebo) -> nadwyżka w pb,
  media:  artykuły GKG o cłach na paczkę w przedziale minus poziom bazowy tego zdarzenia z [t - before, t - base_gap)
          -> nadwyżka w artykułach na paczkę; przedział bez paczki = NaN (brak danych, nie zero).
Miary kolejności (CI z bootstrapu klastrowego po dniach NY):
  T50                  po ilu minutach skumulowana nadwyżka osiąga połowę sumy z (t, t + after],
  pierwszy istotny     pierwszy przedział po t, w którym 95% CI nadwyżki leży nad zerem,
  przed tweetem        średnia nadwyżka w ostatniej godzinie PRZED t (czy media albo rynek ruszyły przed Trumpem).
Czas GKG to czas paczki (artykuły zebrane w poprzednich 15 min), więc media mogą wyglądać na spóźnione o 0-15 min.
Dlatego order_test liczy też wariant z mediami cofniętymi o jeden krok (najkorzystniejszy dla mediów).
Media liczone są ze WSZYSTKICH paczek (res["gkg"], res["gkg_files"]), także gęstych z download_gkg_dense.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaning_data.horizons import HorizonTools

NY = "America/New_York"


# =============================================================================== siatka przedziałów
def _grid(before: str, after: str, step: str):
    st, b, a = pd.Timedelta(step), pd.Timedelta(before), pd.Timedelta(after)
    k0, k1 = int(round(b / st)), int(round(a / st))
    off = np.arange(-k0, k1) * int(st.value)                 # początki przedziałów względem t [ns]
    return off, int(st.value)


def _abs_ret_bins(t_ns: np.ndarray, off: np.ndarray, step: int, pi, px, tol: int) -> np.ndarray:
    """|log P(t+o+step) / P(t+o)| dla każdego zdarzenia (wiersz) i przedziału (kolumna). Cena nieświeża -> NaN."""
    if len(t_ns) == 0:
        return np.empty((0, len(off)))
    s = (t_ns[:, None] + off[None, :]).ravel()
    p0 = HorizonTools._last(s, pi, px, tol)
    p1 = HorizonTools._last(s + step, pi, px, tol)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(np.log(p1 / p0)).reshape(len(t_ns), len(off))


def _count_bins(t_ns: np.ndarray, off: np.ndarray, step: int, items: np.ndarray) -> np.ndarray:
    """Liczba elementów (posortowane ns) w [t+o, t+o+step) dla każdego zdarzenia i przedziału."""
    s = t_ns[:, None] + off[None, :]
    return (np.searchsorted(items, s + step, "left") - np.searchsorted(items, s, "left")).astype(float)


# =============================================================================== profile zdarzeń
def event_bins(res: dict, kind: str = "trump_episode", instrument: str = "S&P 500", step: str = "15min",
               before: str = "2h", after: str = "4h", base_gap: str = "1h", k: int = 20, seed: int = 0,
               media_shift: str = "0min") -> dict:
    """Macierze zdarzenie x przedział: nadwyżka |zwrotu| rynku nad placebo i nadwyżka mediów GKG nad poziomem bazowym.
    media_shift='-15min' cofa czas paczek GKG o 15 min (wariant najkorzystniejszy dla mediów)."""
    ev = res["events"]
    ev = ev[ev["kind"] == kind].sort_values("t").reset_index(drop=True)
    if not len(ev):
        raise ValueError(f"Brak zdarzeń rodzaju {kind}.")
    if ev["precision"].iat[0] != "exact":
        raise ValueError(f"{kind} ma tylko datę dzienną: test minutowy wymaga zdarzeń z dokładnym czasem.")
    off, st = _grid(before, after, step)
    t = pd.DatetimeIndex(ev["t"])
    t_ns = HorizonTools._ns(t)
    out = {"events": ev[["event_id", "t"]].assign(day=t.tz_convert(NY).strftime("%Y-%m-%d")),
           "offsets_min": off / 6e10, "step_min": st / 6e10, "kind": kind, "instrument": instrument,
           "before": before, "after": after, "base_gap": base_gap}

    # --- rynek: zdarzenie minus placebo (ten sam offset od chwili placebo)
    price = res["prices"][instrument].dropna()
    pi, px, pstep = HorizonTools._prep_price(price)
    tol = 2 * pstep
    R = _abs_ret_bins(t_ns, off, st, pi, px, tol)
    pl = HorizonTools.matched_placebo(t, price, res["cfg"].period_bounds(), k=k, seed=seed, exclude="2h")
    C = np.full_like(R, np.nan)
    if len(pl):
        Rp = _abs_ret_bins(HorizonTools._ns(pl["time"]), off, st, pi, px, tol)
        pos = pl["event_pos"].to_numpy()
        ok = np.isfinite(Rp)
        S = np.zeros_like(R)
        N = np.zeros_like(R)
        np.add.at(S, pos, np.where(ok, Rp, 0.0))
        np.add.at(N, pos, ok.astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.where(N >= 3, S / N, np.nan)
    out["market"] = 1e4 * (R - C)                                  # pb
    out["market_raw"] = 1e4 * R

    # --- media: GKG o cłach na paczkę minus poziom bazowy zdarzenia
    gkg, files = res.get("gkg"), res.get("gkg_files")
    M = np.full_like(R, np.nan)
    if gkg is not None and files is not None and len(files):
        sh = int(pd.Timedelta(media_shift).value)
        f_ns = np.sort(HorizonTools._ns(pd.Series(files))) + sh
        a_ns = np.sort(HorizonTools._ns(gkg["time_utc"])) + sh
        nf = _count_bins(t_ns, off, st, f_ns)
        na = _count_bins(t_ns, off, st, a_ns)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(nf > 0, na / np.where(nf > 0, nf, 1), np.nan)
        rb = rate[:, off < -int(pd.Timedelta(base_gap).value)]   # poziom bazowy: [t - before, t - base_gap)
        nb = np.isfinite(rb).sum(1)
        base = np.where(nb >= 2, np.nan_to_num(rb).sum(1) / np.maximum(nb, 1), np.nan)
        M = rate - base[:, None]
        out["media_raw"] = rate
        out["media_base"] = base
    out["media"] = M
    return out


def _wmean(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Średnie ważone po wierszach X (zdarzenia), z pominięciem NaN; W: (B, n_zdarzeń)."""
    m = np.isfinite(X).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W @ np.nan_to_num(X)) / (W @ m)


def _t50(c: np.ndarray, step_min: float) -> float:
    """Minuta, w której skumulowana nadwyżka (przedziały po t) osiąga połowę sumy; NaN, gdy suma <= 0."""
    c = np.nan_to_num(np.asarray(c, float))
    tot = c.sum()
    if not tot > 0:
        return np.nan
    cs = np.cumsum(c)
    j = int(np.argmax(cs >= 0.5 * tot))
    prev = cs[j - 1] if j else 0.0
    frac = (0.5 * tot - prev) / c[j] if c[j] > 0 else 1.0
    return float((j + min(max(frac, 0.0), 1.0)) * step_min)


def order_test(res: dict, kind: str = "trump_episode", instrument: str = "S&P 500", step: str = "15min",
               before: str = "2h", after: str = "4h", base_gap: str = "1h", k: int = 20, n_boot: int = 2000,
               seed: int = 0, same_events: bool = True, min_media_cov: float = 0.8) -> dict:
    """Kto pierwszy: rynek czy media. Zwraca dict: curves (profil na przedział z CI), summary (T50, pierwszy istotny
    przedział, ruch przed tweetem, różnica media - rynek z CI), verdict (zdanie), n (liczby zdarzeń).
    same_events=True: oba profile na TYCH SAMYCH zdarzeniach (rynek otwarty w t i GKG pokrywa >= min_media_cov okna)."""
    B = event_bins(res, kind, instrument, step, before, after, base_gap, k, seed)
    Bs = event_bins(res, kind, instrument, step, before, after, base_gap, k, seed, media_shift=f"-{step}")
    off, sm = B["offsets_min"], B["step_min"]
    post, pre1 = off >= 0, (off < 0) & (off >= -60)
    mk, md, mds = B["market"], B["media"], Bs["media"]
    mk_ok = np.isfinite(mk[:, post]).mean(1) >= 0.5
    md_ok = np.isfinite(md[:, post]).mean(1) >= min_media_cov
    if same_events:
        use = mk_ok & md_ok
        mk_use, md_use = use, use
    else:
        mk_use, md_use = mk_ok, md_ok
    n = {"zdarzeń": len(mk), "rynek_dostępny": int(mk_ok.sum()), "GKG_pokrywa_okno": int(md_ok.sum()),
         "w_teście_rynek": int(mk_use.sum()), "w_teście_media": int(md_use.sum())}
    X = {"rynek": np.where(mk_use[:, None], mk, np.nan), "media": np.where(md_use[:, None], md, np.nan),
         "media_-krok": np.where(md_use[:, None], mds, np.nan)}

    days = pd.factorize(B["events"]["day"])[0]
    G = int(days.max()) + 1
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, G, (n_boot, G))
    Wd = np.zeros((n_boot, G))
    np.add.at(Wd, (np.repeat(np.arange(n_boot), G), draw.ravel()), 1.0)
    W = Wd[:, days]                                          # (B, n) wagi zdarzeń w każdym losowaniu
    W1 = np.ones((1, len(days)))

    curves = pd.DataFrame({"min_od_t": off})
    rows, boots = [], {}
    for name, x in X.items():
        enough = np.isfinite(x).any(1).sum() >= 5
        mean = _wmean(x, W1)[0]
        bt = _wmean(x, W)
        lo, hi = np.nanpercentile(bt, 2.5, 0), np.nanpercentile(bt, 97.5, 0)
        if name != "media_-krok":
            curves[f"{name}"], curves[f"{name}_lo"], curves[f"{name}_hi"] = mean, lo, hi
            curves[f"{name}_n"] = np.isfinite(x).sum(0)
        t50 = _t50(mean[post], sm) if enough else np.nan
        t50b = np.array([_t50(b[post], sm) for b in bt]) if enough else np.full(n_boot, np.nan)
        boots[name] = t50b
        sig = np.flatnonzero(post & (lo > 0))
        rows.append({"seria": name, "n_zdarzeń": int(np.isfinite(x).any(1).sum()),
                     "T50_min": t50, "T50_lo": np.nanpercentile(t50b, 2.5) if np.isfinite(t50b).any() else np.nan,
                     "T50_hi": np.nanpercentile(t50b, 97.5) if np.isfinite(t50b).any() else np.nan,
                     "pierwszy_istotny_od_min": float(off[sig[0]]) if len(sig) else np.nan,
                     "szczyt_od_min": float(off[post][np.nanargmax(mean[post])]) if np.isfinite(mean[post]).any() else np.nan,
                     "przed_tweetem_1h": float(np.nanmean(mean[pre1])) if np.isfinite(mean[pre1]).any() else np.nan,
                     "przed_lo": float(np.nanpercentile(np.nanmean(bt[:, pre1], 1), 2.5)) if np.isfinite(bt[:, pre1]).any() else np.nan,
                     "przed_hi": float(np.nanpercentile(np.nanmean(bt[:, pre1], 1), 97.5)) if np.isfinite(bt[:, pre1]).any() else np.nan,
                     "jednostka": "pb" if name == "rynek" else "artykuły/paczkę"})
    for med in ("media", "media_-krok"):
        d = boots[med] - boots["rynek"]
        ok = np.isfinite(d)
        dm = rows[[r["seria"] for r in rows].index(med)]["T50_min"] - rows[0]["T50_min"]
        rows.append({"seria": f"różnica {med} - rynek", "n_zdarzeń": n["w_teście_media"], "T50_min": dm,
                     "T50_lo": np.nanpercentile(d, 2.5) if ok.any() else np.nan,
                     "T50_hi": np.nanpercentile(d, 97.5) if ok.any() else np.nan,
                     "p_boot": float(min(1.0, 2 * min((d[ok] <= 0).mean(), (d[ok] >= 0).mean()))) if ok.sum() > 50 else np.nan,
                     "udział_losowań_z_T50": float(ok.mean()), "jednostka": "min"})
    summary = pd.DataFrame(rows)
    return {"curves": curves, "summary": summary, "verdict": _verdict(summary, n, instrument), "n": n,
            "bins": B, "X": X, "params": {"kind": kind, "instrument": instrument, "step": step, "before": before,
                                  "after": after, "base_gap": base_gap, "k": k, "n_boot": n_boot}}


def _verdict(S: pd.DataFrame, n: dict, inst: str) -> str:
    r = S.set_index("seria")
    if n["w_teście_media"] < 10:
        return (f"Za mało zdarzeń z gęstym GKG ({n['w_teście_media']}): pobierz GKG wokół epizodów "
                "(download_gkg_dense) i przebuduj scalenie.")
    tm, tg = r.loc["rynek", "T50_min"], r.loc["media", "T50_min"]
    if not np.isfinite(tm) or not np.isfinite(tg):
        who = "rynek nie wykazuje" if not np.isfinite(tm) else "media nie wykazują"
        return f"Po tweetach {who} nadwyżki w oknie, więc T50 nie istnieje i kolejności nie da się ustalić."
    d, ds = r.loc["różnica media - rynek"], r.loc["różnica media_-krok - rynek"]
    if d["T50_hi"] < 0 and r.loc["media", "przed_lo"] > 0:
        return (f"Media wyprzedzają rynek: rosną już w godzinie przed tweetem, a połowa ich reakcji przypada na {tg:.0f} min "
                f"wobec {tm:.0f} min dla {inst} (różnica {d['T50_min']:.0f} min, 95% CI {d['T50_lo']:.0f}–{d['T50_hi']:.0f}).")
    if d["T50_lo"] > 0 and ds["T50_lo"] > 0:
        return (f"Rynek reaguje pierwszy: połowa reakcji {inst} po {tm:.0f} min, mediów (GKG) po {tg:.0f} min "
                f"(różnica {d['T50_min']:.0f} min, 95% CI {d['T50_lo']:.0f}–{d['T50_hi']:.0f}; "
                f"także przy mediach cofniętych o 15 min).")
    if d["T50_hi"] < 0:
        return (f"Media reagują przed rynkiem: połowa reakcji mediów po {tg:.0f} min, {inst} po {tm:.0f} min "
                f"(różnica {d['T50_min']:.0f} min, 95% CI {d['T50_lo']:.0f}–{d['T50_hi']:.0f}).")
    if d["T50_lo"] > 0:
        return (f"Rynek wygląda na pierwszy ({tm:.0f} vs {tg:.0f} min), ale po cofnięciu mediów o 15 min (czas paczki GKG) "
                "różnica przestaje być istotna.")
    return (f"Rynek ({tm:.0f} min) i media ({tg:.0f} min) reagują w podobnym czasie; różnica {d['T50_min']:.0f} min "
            f"(95% CI {d['T50_lo']:.0f}–{d['T50_hi']:.0f}) nie jest istotna.")
