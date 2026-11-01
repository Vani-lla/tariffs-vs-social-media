"""Horyzonty przyczynowości + narzędzia opóźnień.

HorizonTools: rozdzielczość źródeł, zwroty po zdarzeniach, placebo DOPASOWANE (dzień tygodnia x godzina NY),
nadwyżka zwrotu z bootstrapem po dniach. Cena P(t) = ostatnia świeca rozpoczęta <= t (bez informacji z przyszłości).
LagTools: wybielanie AR i korelacja krzyżowa z poprawnym znakiem: corr(x_t, y_{t+k}), k>0 => y PO x.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class HorizonTools:
    NY = "America/New_York"

    # ------------------------------------------------------------------ rozdzielczość
    @staticmethod
    def resolution(ts, regular: bool = False) -> dict:
        """regular=True: rozdzielczość = DOMINUJĄCY krok (nie mediana, bo okna z lukami ją zniekształcają).
        regular=False: ziarno zapisu; >90% znaczników o północy => 'dzień'."""
        t = pd.to_datetime(pd.Series(ts).reset_index(drop=True), utc=True, errors="coerce").dropna().sort_values()
        if t.empty:
            raise ValueError("brak poprawnych znaczników czasu")
        step = t.drop_duplicates().diff().dropna()
        step = step[step > pd.Timedelta(0)]
        mid = float((t.dt.hour.eq(0) & t.dt.minute.eq(0) & t.dt.second.eq(0)).mean())
        sec = float((t.dt.second != 0).mean())
        mnt = float((t.dt.minute != 0).mean())
        if mid > 0.90:
            gran, g = "dzień", pd.Timedelta(days=1)
        elif sec > 0.01:
            gran, g = "sekunda", pd.Timedelta(seconds=1)
        elif mnt > 0.01:
            gran, g = "minuta", pd.Timedelta(minutes=1)
        else:
            gran, g = "godzina", pd.Timedelta(hours=1)
        med = step.median() if len(step) else pd.NaT
        mode = step.mode().iloc[0] if len(step) else pd.NaT
        eff = mode if (regular and pd.notna(mode)) else g
        return {"n": len(t), "ziarno_zapisu": gran, "mediana_kroku": med, "dominujący_krok": mode,
                "efektywna_rozdzielczość": eff, "udział_północy_%": round(100 * mid, 1)}

    @staticmethod
    def resolvable_table(res: dict, horizons: dict) -> pd.DataFrame:
        rows = {}
        for name, r in res.items():
            eff = r["efektywna_rozdzielczość"]
            row = {"rozdzielczość": eff, "ziarno": r["ziarno_zapisu"]}
            for lab, h in horizons.items():
                row[lab] = "✓" if pd.Timedelta(h) >= eff else "✗"
            rows[name] = row
        return pd.DataFrame(rows).T

    # ------------------------------------------------------------------ silnik cen
    @staticmethod
    def _ns(x) -> np.ndarray:
        idx = pd.DatetimeIndex(pd.to_datetime(x, utc=True)).tz_convert("UTC").tz_localize(None)
        return idx.astype("datetime64[ns]").asi8

    @classmethod
    def _prep_price(cls, price: pd.Series):
        price = price.dropna().sort_index()
        price = price[~price.index.duplicated(keep="last")]
        pi, px = cls._ns(price.index), price.to_numpy(float)
        step = int(np.median(np.diff(pi))) if len(pi) > 1 else 0
        return pi, px, step

    @staticmethod
    def _last(t, pi, px, tol):
        """Cena ostatniej świecy rozpoczętej <= t; NaN gdy starsza niż tol (noc, weekend)."""
        j = np.searchsorted(pi, t, side="right") - 1
        jj = np.clip(j, 0, None)
        ok = (j >= 0) & ((t - pi[jj]) <= tol)
        out = np.full(len(t), np.nan)
        out[ok] = px[jj[ok]]
        return out

    @staticmethod
    def _next(t, pi, px, wait):
        """Pierwsza cena >= t, ale nie dalej niż wait (koniec horyzontu wpadł w weekend/przerwę)."""
        j = np.searchsorted(pi, t, side="left")
        jj = np.minimum(j, len(pi) - 1)
        ok = (j < len(pi)) & ((pi[jj] - t) <= wait)
        out = np.full(len(t), np.nan)
        out[ok] = px[jj[ok]]
        return out

    @classmethod
    def returns_at(cls, times, price: pd.Series, horizons: dict, max_stale=None,
                   end_fill: str = "next", max_wait="3D") -> pd.DataFrame:
        """log(P(t+h)/P(t)); kolejność i długość wyniku = kolejność `times` (bez sortowania).
        P(t) musi być świeża (rynek otwarty), inaczej NaN. Koniec horyzontu: świeża cena, a gdy jej brak
        i end_fill='next', pierwsza cena po t+h (do max_wait) — wtedy zwrot obejmuje przerwę weekendową,
        co jest uczciwe, o ile kontrola ma tę samą strukturę (matched_placebo)."""
        pi, px, step = cls._prep_price(price)
        tol = int(pd.Timedelta(max_stale).value) if max_stale is not None else 2 * step
        wait = int(pd.Timedelta(max_wait).value)
        t = pd.DatetimeIndex(pd.to_datetime(pd.Series(times).reset_index(drop=True), utc=True, errors="coerce"))
        out = pd.DataFrame(np.nan, index=t, columns=list(horizons))
        valid = ~t.isna()
        if valid.sum() == 0 or len(pi) == 0:
            return out
        e = cls._ns(t[valid])
        p0 = cls._last(e, pi, px, tol)
        for lab, h in horizons.items():
            hn = int(pd.Timedelta(h).value)
            if hn < step:                       # horyzont krótszy niż świeca: nierozróżnialny
                continue
            p1 = cls._last(e + hn, pi, px, tol)
            if end_fill == "next":
                miss = np.isnan(p1)
                if miss.any():
                    p1[miss] = cls._next(e[miss] + hn, pi, px, wait)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.log(p1 / p0)
            col = np.full(len(t), np.nan)
            col[valid] = r
            out[lab] = col
        return out

    @classmethod
    def forward_returns(cls, events, price, horizons, max_stale=None) -> pd.DataFrame:
        """Zgodność wsteczna: posortowane zdarzenia, koniec horyzontu bez wypełniania."""
        ev = pd.to_datetime(pd.Series(events).reset_index(drop=True), utc=True, errors="coerce").dropna().sort_values()
        return cls.returns_at(ev, price, horizons, max_stale=max_stale, end_fill="last")

    @classmethod
    def open_share(cls, events, price, max_stale=None) -> float:
        """Jaki ułamek zdarzeń przypada na otwarty rynek (ma świeżą cenę P(t))."""
        pi, px, step = cls._prep_price(price)
        tol = int(pd.Timedelta(max_stale).value) if max_stale is not None else 2 * step
        ev = pd.to_datetime(pd.Series(events), utc=True, errors="coerce").dropna()
        if ev.empty:
            return float("nan")
        return float(np.mean(~np.isnan(cls._last(cls._ns(ev), pi, px, tol))))

    # ------------------------------------------------------------------ placebo
    @staticmethod
    def placebo_times(price: pd.Series, bounds, n: int, seed: int = 0) -> pd.DatetimeIndex:
        """Proste placebo (zostawione dla zgodności). Do wniosków używaj matched_placebo."""
        idx = price.dropna().index
        m = np.zeros(len(idx), dtype=bool)
        for a, b in bounds:
            m |= (idx >= a) & (idx < b)
        pool = idx[m]
        if len(pool) == 0:
            return pd.DatetimeIndex([], tz="UTC")
        rng = np.random.default_rng(seed)
        return pool[rng.integers(0, len(pool), n)]

    @classmethod
    def matched_placebo(cls, events, price, bounds, k: int = 20, seed: int = 0, exclude="2h") -> pd.DataFrame:
        """Dla każdego zdarzenia k losowych momentów z tego samego (dzień tygodnia, godzina NY), w okresie
        badania, na otwartym rynku i dalej niż `exclude` od jakiegokolwiek zdarzenia."""
        ev = pd.DatetimeIndex(pd.to_datetime(pd.Series(events).reset_index(drop=True), utc=True,
                                             errors="coerce").dropna()).sort_values()
        idx = price.dropna().index
        m = np.zeros(len(idx), dtype=bool)
        for a, b in bounds:
            m |= (idx >= a) & (idx < b)
        pool = idx[m]
        if len(pool) == 0 or len(ev) == 0:
            return pd.DataFrame({"event_pos": np.array([], dtype=int), "time": pd.DatetimeIndex([], tz="UTC")})
        key = lambda ix: (ix.dayofweek * 24 + ix.hour).to_numpy()
        kp, ke = key(pool.tz_convert(cls.NY)), key(ev.tz_convert(cls.NY))
        by_key = {int(v): np.flatnonzero(kp == v) for v in np.unique(kp)}
        pool_ns, evn, ex = cls._ns(pool), np.sort(cls._ns(ev)), int(pd.Timedelta(exclude).value)
        rng = np.random.default_rng(seed)
        pos, ids = [], []
        for i, kk in enumerate(ke):
            cand = by_key.get(int(kk))
            if cand is None or len(cand) == 0:
                continue
            draw = cand[rng.integers(0, len(cand), min(len(cand), 4 * k))]
            tn = pool_ns[draw]
            j = np.searchsorted(evn, tn)
            d_lo = np.abs(tn - evn[np.maximum(j - 1, 0)])
            d_hi = np.abs(evn[np.minimum(j, len(evn) - 1)] - tn)
            draw = draw[np.minimum(d_lo, d_hi) > ex][:k]
            pos += [i] * len(draw)
            ids += list(draw)
        return pd.DataFrame({"event_pos": np.array(pos, dtype=int), "time": pool[np.array(ids, dtype=int)]})

    @classmethod
    def paired_excess(cls, events, price, horizons, bounds, k: int = 20, seed: int = 0, n_boot: int = 2000,
                      exclude="2h", end_fill: str = "next", max_wait="3D") -> pd.DataFrame:
        """Nadwyżka zwrotu zdarzenia nad średnią jego kontroli (to samo dzień/godzina), z CI 95% z bootstrapu
        klastrowego po dniach (zdarzenia z jednego dnia nie są niezależne). Zdarzenia przy zamkniętym rynku
        wypadają. p_boot to dwustronny p z rozkładu bootstrapu względem zera (orientacyjny)."""
        ev = pd.DatetimeIndex(pd.to_datetime(pd.Series(events).reset_index(drop=True), utc=True,
                                             errors="coerce").dropna()).sort_values()
        if len(ev) == 0:
            return pd.DataFrame()
        r_ev = cls.returns_at(ev, price, horizons, end_fill=end_fill, max_wait=max_wait).reset_index(drop=True)
        pl = cls.matched_placebo(ev, price, bounds, k=k, seed=seed, exclude=exclude)
        r_pl = cls.returns_at(pl["time"], price, horizons, end_fill=end_fill, max_wait=max_wait).reset_index(drop=True)
        r_pl["event_pos"] = pl["event_pos"].to_numpy()
        n = len(ev)
        c_sgn = r_pl.groupby("event_pos").mean().reindex(range(n))
        c_abs = r_pl.drop(columns="event_pos").abs().assign(event_pos=r_pl["event_pos"]) \
                    .groupby("event_pos").mean().reindex(range(n))
        day = np.asarray(ev.tz_convert(cls.NY).strftime("%Y-%m-%d"))
        rng = np.random.default_rng(seed + 1)
        rows = {}
        for h in horizons:
            r, cs, ca = (r_ev[h].to_numpy(float), c_sgn[h].to_numpy(float), c_abs[h].to_numpy(float))
            ok = ~np.isnan(r) & ~np.isnan(cs) & ~np.isnan(ca)
            if ok.sum() < 5:
                rows[h] = {"n": int(ok.sum())}
                continue
            codes = pd.factorize(day[ok])[0]
            G = int(codes.max()) + 1
            N = np.bincount(codes, minlength=G).astype(float)
            out = {"n": int(ok.sum()), "n_dni": G,
                   "śr_|zwrot|_zdarzenia": float(np.abs(r[ok]).mean()), "śr_|zwrot|_kontrola": float(ca[ok].mean())}
            idx = rng.integers(0, G, (n_boot, G))
            for name, v in (("nadwyżka_|zwrot|", np.abs(r[ok]) - ca[ok]), ("nadwyżka_zwrot", r[ok] - cs[ok])):
                S = np.bincount(codes, weights=v, minlength=G)
                b = S[idx].sum(1) / N[idx].sum(1)
                out[name] = float(v.mean())
                out[name + "_CI_dół"], out[name + "_CI_góra"] = [float(x) for x in np.percentile(b, [2.5, 97.5])]
                out[name + "_p_boot"] = float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))
            rows[h] = out
        return pd.DataFrame(rows).T.apply(pd.to_numeric, errors="coerce")

    @staticmethod
    def summarize(r: pd.DataFrame) -> pd.DataFrame:
        rows = {}
        for c in r.columns:
            x = r[c].dropna()
            rows[c] = {"n_ważne": len(x), "pokrycie_%": round(100 * len(x) / max(len(r), 1), 1),
                       "śr_zwrot": x.mean(), "mediana": x.median(), "std": x.std(), "śr_|zwrot|": x.abs().mean()}
        return pd.DataFrame(rows).T


class LagTools:
    @staticmethod
    def prewhiten(x, p: int = 1) -> pd.Series:
        """Reszty z AR(p) (OLS). Usuwa autokorelację, która zawyża surową korelację krzyżową."""
        x = pd.Series(x).astype(float)
        z = x - x.mean()
        if p <= 0:
            return z
        X = np.column_stack([z.shift(i).to_numpy() for i in range(1, p + 1)])
        y = z.to_numpy()
        m = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        beta = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
        return pd.Series(y - X @ beta, index=x.index)

    @staticmethod
    def ccf(x: pd.Series, y: pd.Series, lags) -> pd.DataFrame:
        """corr(x_t, y_{t+k}). k>0: y następuje PO x o k okresów; k<0: y wyprzedza x."""
        rows = []
        for k in lags:
            b = y.shift(-k)
            m = x.notna() & b.notna()
            rows.append({"k": k, "corr": x[m].corr(b[m]) if m.sum() > 3 else np.nan, "n": int(m.sum())})
        df = pd.DataFrame(rows).set_index("k")
        df["ci95"] = 1.96 / np.sqrt(df["n"].clip(lower=1))
        df["istotne"] = df["corr"].abs() > df["ci95"]
        return df