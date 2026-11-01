"""SCALANIE wszystkich czystych źródeł w tabele analityczne. Tylko fakty: czas, liczności, ceny.

build_merged(cfg)  -> dict z tabelami (zapisywane do clean_out/merged/):
  events          wszystkie zdarzenia (Trump: posty i epizody; GTA; Federal Register),
  media           wolumeny mediów i kolejnych tweetów w oknach przed/po zdarzeniu,
  market_long     zdarzenie x instrument x horyzont: zwrot, zwrot przed, kontrola placebo, nadwyżka,
  event_table     JEDEN wiersz na zdarzenie: events + media + rynek (szeroko),
  daily           panel dzienny (klucz eff_day: data NY, od 16:00 NY następny dzień),
  trading         ten sam panel na dniach handlowych (weekendy doliczone do poniedziałku) -> PCMCI+ / LP,
  hourly          panel godzinowy UTC (Trump, GKG, rynek).
check_merged(res)  -> tabela testów; test 'BŁĄD' oznacza złamany niezmiennik (np. wyciek z przyszłości).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cleaning_data.base import Config, show
from .events import EventBuilder
from .features import DAY, eff_day, fnspid_daily, media_features
from cleaning_data.horizons import HorizonTools
from .loaders import CleanLoader, tag_of
from .panel import build_daily_panel, build_hourly_panel, event_market_long, to_trading_days

NY = "America/New_York"


def _wide_market(long: pd.DataFrame, values=("ret", "excess_ret", "excess_abs", "pre_ret")) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame(columns=["event_id"])
    w = long.pivot(index="event_id", columns=["instrument", "horizon"], values=list(values))
    w.columns = [f"{tag_of(inst)}_{v}_{h}" for v, inst, h in w.columns]
    return w.reset_index()


def build_merged(cfg: Config, gap: str = "6h", k: int = 20, seed: int = 0, instruments: list | None = None,
                 hourly: bool = True, save: bool = True, relevant_col: str = "relevant",
                 trump: pd.DataFrame | None = None) -> dict:
    """relevant_col: która kolumna wybiera posty-zdarzenia (domyślnie słownik `relevant`; np. 'relevant_strict' albo
    kolumna z predykcji klasyfikatora). trump: gotowa tabela postów (np. trump_clean z doklejonymi etykietami)."""
    L = CleanLoader(cfg)
    tr = L.trump() if trump is None else trump.copy()
    if relevant_col != "relevant":
        if relevant_col not in tr:
            raise KeyError(f"Brak kolumny {relevant_col} w tabeli postów.")
        tr["relevant"] = tr[relevant_col].astype(str).str.lower().isin(["true", "1", "tak"])
    fn, gkg, gta, fr = L.optional("fnspid"), L.optional("gkg"), L.optional("gta"), L.optional("fr")
    gkg_files = L.gkg_files() if gkg is not None else None
    mk = L.market_all()
    if instruments:
        mk = {k_: v for k_, v in mk.items() if k_ in instruments}
    prices = {lab: d["mid_open"].dropna() for lab, d in mk.items()}
    closes = {lab: d["mid_close"].dropna() for lab, d in mk.items()}

    E = EventBuilder(cfg, gap=gap).build(tr, gta, fr)
    ev, posts = E["events"], E["posts"]
    fn_d = fnspid_daily(fn) if fn is not None and len(fn) else None
    media = media_features(ev, fn_d, gkg, gkg_files, posts, cfg.horizons)
    long = event_market_long(ev, prices, cfg.horizons, cfg.period_bounds(), k=k, seed=seed)

    table = ev.merge(media, on="event_id", how="left", validate="1:1") \
              .merge(_wide_market(long), on="event_id", how="left", validate="1:1")
    tr_all = tr[~tr["is_retweet"]]
    daily = build_daily_panel(cfg.period_bounds(), tr_all, posts, fn_d, gkg, gkg_files, ev, closes)
    trading = to_trading_days(daily) if closes else None
    hour = build_hourly_panel(cfg.period_bounds(), posts, gkg, gkg_files, closes, tr_all=tr_all) if hourly else None

    res = {"gap": gap, "events": ev, "posts": posts, "episodes": E["episodes"], "media": media, "market_long": long,
           "event_table": table, "daily": daily, "trading": trading, "hourly": hour,
           "fn_daily": fn_d, "gkg_files": gkg_files, "prices": prices, "closes": closes, "cfg": cfg,
           "sources": {"trump": len(tr), "fnspid": 0 if fn is None else len(fn), "gkg": 0 if gkg is None else len(gkg),
                       "gkg_files": 0 if gkg_files is None else len(gkg_files), "gta": 0 if gta is None else len(gta),
                       "fr": 0 if fr is None else len(fr), "instrumenty": list(prices)}}
    if save:
        out = cfg.out("merged")
        for name in ("events", "media", "market_long", "event_table", "daily", "trading", "hourly"):
            d = res[name]
            if d is not None:
                d.to_csv(out / f"{name}.csv", index=name in ("daily", "trading", "hourly"))
        res["out_dir"] = out
    return res


# =============================================================================== podsumowania
def excess_table(res: dict, kind: str = "trump_episode", metric: str = "excess_abs", n_boot: int = 2000,
                 seed: int = 0) -> pd.DataFrame:
    """Średnia nadwyżka (zdarzenie minus kontrola) per instrument x horyzont, CI 95% z bootstrapu klastrowego po dniach NY."""
    long, ev = res["market_long"], res["events"]
    d = long.merge(ev[["event_id", "kind", "t"]], on="event_id")
    d = d[(d["kind"] == kind) & d[metric].notna()]
    rng = np.random.default_rng(seed)
    rows = []
    for (inst, h), g in d.groupby(["instrument", "horizon"], sort=False):
        v = g[metric].to_numpy(float)
        day = g["t"].dt.tz_convert(NY).dt.strftime("%Y-%m-%d").to_numpy()
        codes = pd.factorize(day)[0]
        G = int(codes.max()) + 1 if len(codes) else 0
        row = {"instrument": inst, "horizon": h, "n": len(v), "n_dni": G, "mean": v.mean() if len(v) else np.nan,
               "ev_abs": g["abs_ret"].mean(), "ctl_abs": g["ctl_abs"].mean()}
        if G >= 5:
            S = np.bincount(codes, weights=v, minlength=G)
            N = np.bincount(codes, minlength=G).astype(float)
            idx = rng.integers(0, G, (n_boot, G))
            b = S[idx].sum(1) / N[idx].sum(1)
            row.update({"lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5),
                        "p_boot": min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))})
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        order = {h: i for i, h in enumerate(res["cfg"].horizons)}
        out = out.sort_values(["instrument", "horizon"], key=lambda s: s.map(order) if s.name == "horizon" else s)
        out["istotne"] = (out.get("lo", np.nan) > 0) | (out.get("hi", np.nan) < 0)
    return out.reset_index(drop=True)


def coverage_table(res: dict) -> pd.DataFrame:
    """Ile zdarzeń każdego rodzaju ma wartość zwrotu (rynek otwarty / dane dostępne) per horyzont."""
    long, ev = res["market_long"], res["events"]
    d = long.merge(ev[["event_id", "kind"]], on="event_id")
    t = d.groupby(["kind", "instrument", "horizon"], sort=False)["ret"].apply(lambda s: round(100 * s.notna().mean(), 1))
    t = t.unstack("horizon")
    return t[[h for h in res["cfg"].horizons if h in t.columns]]


# =============================================================================== testy poprawności
def check_merged(res: dict, n_sample: int = 200, seed: int = 0, strict: bool = True) -> pd.DataFrame:
    """Niezależne testy scalenia. Okna i sumy liczone są tu WPROST (pętla), a nie kodem z features/panel.
    strict=True: przy jakimkolwiek 'BŁĄD' rzuca AssertionError z listą."""
    cfg, ev, long, tab = res["cfg"], res["events"], res["market_long"], res["event_table"]
    hz = cfg.horizons
    rows = []
    add = lambda name, ok, info="", warn=False: rows.append(
        {"test": name, "wynik": "OK" if ok else ("UWAGA" if warn else "BŁĄD"), "szczegóły": info})
    rng = np.random.default_rng(seed)

    # --- zdarzenia
    add("event_id unikalne", not ev["event_id"].duplicated().any(), f"{len(ev):,} zdarzeń")
    lo, hi = min(a for a, _ in cfg.period_bounds()), max(b for _, b in cfg.period_bounds())
    add("wszystkie zdarzenia w okresie badania", bool(((ev["t"] >= lo) & (ev["t"] < hi)).all()),
        f"{ev['t'].min()} .. {ev['t'].max()}")
    add("brak NaT w czasie zdarzeń", bool(ev["t"].notna().all()))
    add("tabela zdarzeń 1:1 po scaleniu", len(tab) == len(ev) and not tab["event_id"].duplicated().any(),
        f"{len(tab):,} wierszy")
    eps = ev[ev["kind"] == "trump_episode"]
    if len(eps) > 1:
        gaps = eps["t"].sort_values().diff().dropna()
        add("epizody rozdzielone przerwą >= gap", bool(gaps.min() >= pd.Timedelta(res.get("gap", "6h"))),
            f"najkrótsza przerwa {gaps.min()}")
    n_posts_ep = int(eps["n_posts"].sum()) if len(eps) else 0
    add("suma postów w epizodach = liczba postów", n_posts_ep == len(res["posts"]), f"{n_posts_ep} vs {len(res['posts'])}")

    # --- rynek
    if len(long):
        n_inst = long["instrument"].nunique()
        add("market_long: zdarzenie x instrument x horyzont", len(long) == len(ev) * n_inst * len(hz),
            f"{len(long):,} = {len(ev):,} x {n_inst} x {len(hz)}")
        short = [h for h, v in hz.items() if pd.Timedelta(v) < DAY]
        dprec = long["event_id"].isin(ev.loc[ev["precision"] == "day", "event_id"])
        add("zdarzenia dzienne: horyzonty < 1d puste", bool(long.loc[dprec & long["horizon"].isin(short), "ret"].isna().all()))
        closed = long["horizon"].isin(short) & ~long["open_at_t"].astype(bool) & ~dprec
        add("rynek zamknięty w t: horyzonty < 1d puste", bool(long.loc[closed, "ret"].isna().all()),
            f"{int(closed.sum()):,} wierszy")
        cu = long.drop_duplicates(["event_id", "instrument"]).merge(ev[["event_id", "kind"]], on="event_id") \
                 .groupby("kind").agg(ctl_n=("ctl_n", "mean"), uniq=("ctl_unique_share", "mean"))
        weak = cu[(cu["uniq"] < 0.5) | (cu["ctl_n"] < 5)]
        add("kontrole placebo: zróżnicowane (>=50% unikalnych) i liczne (>=5/zdarzenie)", weak.empty,
            "; ".join(f"{k_}: {r_.ctl_n:.1f}/zd., unikalne {100 * r_.uniq:.0f}%" for k_, r_ in cu.iterrows()), warn=True)
        r = long["ret"].dropna()
        add("zwroty skończone", bool(np.isfinite(r).all()), f"{len(r):,} wartości")
        big = int((r.abs() > 0.2).sum())
        add("brak zwrotów > 20% (podejrzane ceny)", big == 0, f"{big} takich wierszy", warn=True)
        # niezależne przeliczenie zwrotu dla próbki zdarzeń dokładnych (pętla, bez kodu panel.py)
        bad = 0
        tested = 0
        for inst, price in res["prices"].items():
            p = price.sort_index()
            ii = HorizonTools._ns(p.index)            # zawsze nanosekundy UTC (pandas 2/3: rozdzielczość us/ns)
            step = np.median(np.diff(ii))
            ex = ev[ev["precision"] == "exact"]
            if ex.empty:
                continue
            sm = ex.sample(min(n_sample // max(1, len(res["prices"])), len(ex)), random_state=seed)
            sub = long[(long["instrument"] == inst) & long["event_id"].isin(sm["event_id"])].set_index(["event_id", "horizon"])
            for _, e in sm.iterrows():
                t0 = e["t"].tz_convert("UTC").tz_localize(None).value
                j0 = np.searchsorted(ii, t0, "right") - 1
                if j0 < 0 or t0 - ii[j0] > 2 * step:
                    continue
                for h in ("5m", "1h") if "5m" in hz and "1h" in hz else list(hz)[:1]:
                    t1 = t0 + pd.Timedelta(hz[h]).value
                    j1 = np.searchsorted(ii, t1, "right") - 1
                    if j1 < 0 or t1 - ii[j1] > 2 * step:
                        continue
                    want = np.log(p.iloc[j1] / p.iloc[j0])
                    got = sub.loc[(e["event_id"], h), "ret"]
                    tested += 1
                    bad += int(not np.isclose(want, got, rtol=1e-9, atol=1e-12))
        add("zwroty zgodne z niezależnym przeliczeniem (P(t)=ostatnia świeca <= t)", bad == 0, f"{tested} porównań, {bad} różnic")

    # --- media: niezależne przeliczenie okien tweetów (bez wycieku z przyszłości)
    ex = tab[tab["precision"] == "exact"]
    col = next((c for c in tab.columns if c.startswith("tw_n_post_")), None)
    if col is not None and len(ex):
        lab = col.replace("tw_n_post_", "")
        h = pd.Timedelta(hz[lab])
        pt = np.sort(HorizonTools._ns(res["posts"]["time_utc"]))
        hn = int(h.value)
        sm = ex.sample(min(n_sample, len(ex)), random_state=seed)
        mism = 0
        for _, e in sm.iterrows():
            t = int(e["t"].value)
            post = int(((pt > t) & (pt <= t + hn)).sum())
            pre = int(((pt >= t - hn) & (pt < t)).sum())
            mism += int(post != e[f"tw_n_post_{lab}"] or pre != e[f"tw_n_pre_{lab}"])
        add(f"okna tweetów (t, t+{lab}] i [t-{lab}, t): przeliczenie wprost", mism == 0, f"{len(sm)} zdarzeń, {mism} różnic")
    fcol = next((c for c in tab.columns if c.startswith("fn_n_post_")), None)
    if fcol is not None and res["fn_daily"] is not None:
        lab = fcol.replace("fn_n_post_", "")
        n = int(pd.Timedelta(hz[lab]).days)
        fd = res["fn_daily"]["fn_n"]
        sm = tab.sample(min(n_sample, len(tab)), random_state=seed)
        mism = 0
        for _, e in sm.iterrows():
            if e[f"fn_cover_post_{lab}"] != 1:
                continue
            d0 = pd.Timestamp(e["eff_day"])
            want = float(fd[(fd.index >= d0) & (fd.index < d0 + pd.Timedelta(days=n))].sum())
            mism += int(not np.isclose(want, e[fcol]))
        add(f"okna FNSPID [d, d+{lab}): przeliczenie wprost", mism == 0, f"{len(sm)} zdarzeń, {mism} różnic")
    gcol = [c for c in tab.columns if c.startswith("gkg_rate_post_")]
    if gcol:
        f = tab[[c for c in tab.columns if c.startswith("gkg_files_post_")]]
        add("GKG: okna bez paczek mają NaN (nie 0)",
            bool(all(tab.loc[f[c.replace("rate", "files")] == 0, c].isna().all() for c in gcol)))
        share = float((f.iloc[:, -1] > 0).mean())
        add("GKG: pokrycie zdarzeń paczkami (najdłuższy horyzont)", share > 0.2, f"{100 * share:.1f}% zdarzeń", warn=True)

    # --- panele
    D = res["daily"]
    add("panel dzienny: ciągły kalendarz bez duplikatów",
        D.index.is_unique and D.index.is_monotonic_increasing and len(pd.date_range(D.index[0], D.index[-1])) == len(D),
        f"{len(D)} dni")
    cnt = [c for c in D if c.endswith("_n")]
    add("panel dzienny: liczniki bez NaN i nieujemne", bool(D[cnt].notna().all().all() and (D[cnt] >= 0).all().all()),
        f"{len(cnt)} kolumn")
    add("posty o cłach: panel = lista postów", int(D["tw_topic_n"].sum()) == len(res["posts"]) - int(
        (res["posts"]["eff_day"] >= D.index[-1] + pd.Timedelta(days=1)).sum()))
    T = res["trading"]
    if T is not None:
        last = T.index[-1]
        keep = D.index <= last
        for c in [c for c in cnt if c in T][:6]:
            add(f"dni handlowe: suma '{c}' zachowana", np.isclose(D.loc[keep, c].sum(), T[c].sum()),
                f"{D.loc[keep, c].sum():.0f} vs {T[c].sum():.0f}")
        rc = [c for c in T if c.endswith("_ret")]
        for c in rc[:3]:
            add(f"dni handlowe: suma zwrotów log '{c}' zachowana", np.isclose(D.loc[keep, c].sum(), T[c].sum()))
        add("dni handlowe: same dni Pon-Pt", bool((T.index.dayofweek < 5).all()), f"{len(T)} dni")

    out = pd.DataFrame(rows)
    show(out, "TESTY SCALENIA")
    bad = out[out["wynik"] == "BŁĄD"]
    if strict and len(bad):
        raise AssertionError("Scalenie nie przeszło testów:\n" + bad.to_string())
    return out
