"""Trzy slajdy 16:9 z wyników scalenia (tylko fakty: liczności, czas, zwroty).

Nagłówek każdego slajdu jest ZDANIEM WYLICZONYM z danych (nie wpisanym na sztywno): gdy wynik jest nieistotny,
nagłówek mówi to wprost. Użycie:
    fig1 = slides.slide_data(res); fig2 = slides.slide_market(res); fig3 = slides.slide_media(res)
    slides.save_deck([fig1, fig2, fig3], "slajdy")      # PNG + PDF (+ PPTX, jeśli jest python-pptx)
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .loaders import tag_of
from .merge import excess_table

# ----------------------------------------------------------------------------- styl (paleta referencyjna, tryb jasny)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID, RULE = "#fcfcfb", "#e7e6e1", "#c9c8c2"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
NEG, POS = "#2a78d6", "#eb6834"          # bieguny skali rozbieżnej, środek szary
W, H = 13.333, 7.5
TEAM = "B. Borkowski · I. Kowalczyk · M. Gadomski · P. Korus"

def _pick_font(cands=("Inter", "Segoe UI", "Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans")) -> str:
    """Konkretna czcionka dostępna w systemie. Nazwa ogólna 'sans-serif' byłaby rozwiązywana dopiero przy rysowaniu
    (poza stylem slajdu), więc pomiar szerokości tytułu i rysunek mogłyby użyć różnych czcionek."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((c for c in cands if c in have), "DejaVu Sans")


FONT = _pick_font()

STYLE = {
    "font.family": [FONT],
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK2, "legend.labelcolor": INK2,
    "axes.edgecolor": RULE, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "savefig.edgecolor": SURFACE, "figure.edgecolor": SURFACE, "savefig.transparent": False,
}


def _styled(fn):
    """Każdy slajd rysowany we własnym stylu: ciemny motyw edytora ani styl notebooka (np. dark_background) nie przecieka."""
    import functools

    @functools.wraps(fn)
    def wrap(*a, **kw):
        with plt.rc_context(STYLE):
            return fn(*a, **kw)
    return wrap


def pl(x, nd: int = 1) -> str:
    """Liczba po polsku: przecinek dziesiętny, spacja jako separator tysięcy."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    s = f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")
    return s


def _fit_text(fig, x: float, y: float, text: str, max_w: float, start: int, **kw):
    """Tekst zawijany do faktycznej szerokości (zależy od czcionki systemu); zwraca (artist, wysokość w ułamku figury)."""
    r = fig.canvas.get_renderer()
    fw = fig.get_figwidth() * fig.dpi
    width = start
    while True:
        t = fig.text(x, y, "\n".join(textwrap.wrap(text, width)), va="top", **kw)
        bb = t.get_window_extent(r)
        if bb.width <= max_w * fw or width <= 30:
            return t, bb.height / (fig.get_figheight() * fig.dpi)
        t.remove()
        width -= 4


def _canvas(num: int, kicker: str, title: str, subtitle: str, footer: str):
    fig = plt.figure(figsize=(W, H))
    fig.text(0.045, 0.935, f"{num}/3  ·  {kicker.upper()}", fontsize=10.5, color=MUTED, weight="bold")
    fig.text(0.955, 0.935, TEAM, fontsize=9, color=MUTED, ha="right")
    _, h1 = _fit_text(fig, 0.045, 0.885, title, 0.91, 90, fontsize=21, color=INK, weight="bold", linespacing=1.15)
    y2 = 0.885 - h1 - 0.025
    _, h2 = _fit_text(fig, 0.045, y2, subtitle, 0.91, 170, fontsize=12, color=INK2, linespacing=1.3)
    fig.add_artist(Line2D([0.045, 0.955], [0.07, 0.07], color=GRID, lw=1, transform=fig.transFigure))
    _fit_text(fig, 0.045, 0.058, footer, 0.91, 230, fontsize=8.5, color=MUTED)
    top = y2 - h2 - 0.075
    return fig, top


def _axis_clean(ax, ygrid: bool = True, nd: int | None = None):
    from matplotlib.ticker import FuncFormatter
    ax.tick_params(length=0, labelsize=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: pl(v, nd if nd is not None else (0 if abs(v) >= 10 or v == int(v) else 2))))
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)


def _hz_order(res) -> list:
    return list(res["cfg"].horizons)


# ============================================================================= SLAJD 1: dane
@_styled
def slide_data(res: dict, main: str = "S&P 500"):
    ev, D = res["events"], res["daily"]
    posts, eps = ev[ev.kind == "trump_post"], ev[ev.kind == "trump_episode"]
    off = ev[ev.kind.isin(["gta_announced", "fr"])]
    wk = lambda s: s.resample("W-MON", label="left", closed="left").sum()
    tw_w = wk(D["tw_topic_n"])
    fn_w = wk(D["fn_n"]) if "fn_n" in D else None
    r = float(np.corrcoef(tw_w, fn_w)[0, 1]) if fn_w is not None and tw_w.std() > 0 and fn_w.std() > 0 else np.nan
    n_fn = int(D["fn_n"].sum()) if "fn_n" in D else 0
    title = (f"2018–2019: {pl(len(posts), 0)} tweetów o cłach w {pl(len(eps), 0)} epizodach, "
             f"{pl(n_fn, 0)} historii medialnych i {pl(len(off), 0)} oficjalnych decyzji")
    sub = ("Wszystkie źródła leżą na jednej osi czasu UTC; łączymy je oknami po zdarzeniu, bez kluczy i bez analizy treści. "
           + (f"Tygodnie z większą liczbą tweetów o cłach mają też więcej artykułów FNSPID (r = {pl(r, 2)})." if np.isfinite(r) else ""))
    gk = res["sources"]
    foot = (f"Źródła: Twitter (posty o cłach wg słownika, bez RT), FNSPID (pierwsza wiadomość historii), GTA i Federal Register "
            f"(daty ogłoszeń), ceny M5 {', '.join(res['sources']['instrumenty'])}. GKG: {pl(gk['gkg_files'], 0)} paczek 15-min "
            f"(próbka), więc nie pokazujemy go na osi ciągłej. Epizod = pierwszy post po >= {res.get('gap', '6h')} przerwy.")
    fig, top = _canvas(1, "Dane", title, sub, foot)

    left, right = 0.085, 0.70
    hs = [0.34, 0.24, 0.24, 0.06]
    gap = 0.018
    total = top - 0.10
    unit = (total - gap * 3) / sum(hs)
    y = top
    axes = []
    for h in hs:
        hh = h * unit
        axes.append(fig.add_axes([left, y - hh, right - left, hh]))
        y -= hh + gap
    a_px, a_tw, a_fn, a_off = axes
    if main in res["closes"]:
        c = res["closes"][main].resample("W-MON", label="left", closed="left").last().dropna()
        c = c[(c.index >= D.index[0].tz_localize("UTC")) & (c.index <= D.index[-1].tz_localize("UTC"))]
        a_px.plot(c.index.tz_localize(None), 100 * c / c.iloc[0], color=INK, lw=1.8)
        a_px.set_ylabel(f"{main}\n(1. tydzień = 100)", fontsize=10)
    for ax, s, col, lab in ((a_tw, tw_w, BLUE, "tweety o cłach\n/ tydzień"),
                            (a_fn, fn_w, ORANGE, "historie FNSPID\n/ tydzień")):
        if s is not None:
            ax.bar(s.index, s.to_numpy(), width=5.2, color=col, align="edge", linewidth=0)
            ax.set_ylabel(lab, fontsize=10)
    for k_, (kind, col) in enumerate((("gta_announced", VIOLET), ("fr", AQUA))):
        t = ev.loc[ev.kind == kind, "eff_day"]
        a_off.vlines(t, k_, k_ + 0.8, color=col, lw=1.0, alpha=0.9)
    a_off.set_ylim(-0.1, 1.9)
    a_off.set_yticks([0.4, 1.4], ["GTA", "FR"], fontsize=9)
    a_off.spines["left"].set_visible(False)
    keys = {pd.Timestamp(v): k for k, v in res["cfg"].key_events.items()}
    for ax in axes:
        _axis_clean(ax, ygrid=ax is not a_off)
        ax.set_xlim(D.index[0], D.index[-1])
        for d in keys:
            ax.axvline(d, color=RULE, lw=0.8, ls=(0, (2, 2)), zorder=0)
        if ax is not a_off:
            ax.tick_params(labelbottom=False)
    prev, lvl = None, 0
    for i, d in enumerate(sorted(keys), 1):
        lvl = (1 - lvl) if (prev is not None and (d - prev).days < 25) else 0
        a_px.text(d, 1.02 + 0.09 * lvl, str(i), transform=a_px.get_xaxis_transform(), ha="center", va="bottom",
                  fontsize=8.5, color=INK2, weight="bold")
        prev = d
    a_off.set_yticks([0.4, 1.4], ["GTA", "FR"], fontsize=9)            # po _axis_clean (formatter liczb)
    a_off.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    a_off.xaxis.set_major_formatter(mdates.DateFormatter("%m.%Y"))

    # prawa kolumna: liczby + legenda dat kluczowych
    x0 = 0.735
    tiles = [(pl(len(posts), 0), "tweetów o cłach (bez RT)"), (pl(len(eps), 0), "epizodów (zdarzeń Trumpa)"),
             (pl(n_fn, 0), "historii FNSPID o cłach"), (pl(len(off), 0), "ogłoszeń GTA + dokumentów FR")]
    yy = top - 0.01
    for j, (big, small) in enumerate(tiles):
        xx = x0 + (j % 2) * 0.115
        fig.text(xx, yy, big, fontsize=18, weight="bold", color=INK, va="top")
        fig.text(xx, yy - 0.042, "\n".join(textwrap.wrap(small, 18)), fontsize=9, color=INK2, va="top")
        if j % 2:
            yy -= 0.115
    fig.text(x0, yy - 0.005, "Daty kluczowe", fontsize=10, weight="bold", color=INK2, va="top")
    yy -= 0.036
    for i, d in enumerate(sorted(keys), 1):
        lab = keys[d].split(" ", 1)[1] if " " in keys[d] else keys[d]
        fig.text(x0, yy, f"{i}", fontsize=8.5, weight="bold", color=INK2, va="top")
        fig.text(x0 + 0.016, yy, f"{d:%d.%m.%y}  {lab}"[:40], fontsize=8.5, color=INK2, va="top")
        yy -= 0.0205
        if yy < 0.075:
            break
    return fig


# ============================================================================= SLAJD 2: rynek
@_styled
def slide_market(res: dict, main: str = "S&P 500", n_boot: int = 2000, fdr: bool = True):
    hz = _hz_order(res)
    kinds = [("trump_episode", "epizody tweetów (dokładny czas)", BLUE, "o"),
             ("gta_announced", "ogłoszenia GTA (dzień)", VIOLET, "s"),
             ("gta_implemented", "wejście w życie GTA (dzień)", AQUA, "D")]
    X = {k: excess_table(res, kind=k, n_boot=n_boot) for k, *_ in kinds}
    ep = X["trump_episode"]
    m = ep[ep.instrument == main].set_index("horizon").reindex(hz) if len(ep) else pd.DataFrame(index=hz)
    sig = m[m.get("istotne", pd.Series(False, index=m.index)).fillna(False).astype(bool) & (m.get("mean", 0) > 0)]
    if len(sig):
        hmax = sig["mean"].idxmax()
        span = ", ".join(sig.index[:-1]) + (" i " if len(sig) > 1 else "") + sig.index[-1]
        title = (f"Po tweetach o cłach {main} porusza się mocniej niż w zwykłej chwili: istotnie przy t+{span}, "
                 f"najwięcej +{pl(1e4 * sig.loc[hmax, 'mean'])} pb przy {hmax}")
    else:
        title = f"Ruch {main} po tweetach o cłach nie odróżnia się od zwykłych chwil o tej samej porze (żaden horyzont nie jest istotny)"
    n_ep = int(m["n"].max()) if "n" in m and m["n"].notna().any() else 0
    sub = (f"Nadwyżka |zwrotu| = |log P(t+h)/P(t)| po zdarzeniu minus średnia z 20 losowych chwil o tym samym dniu tygodnia i godzinie NY. "
           f"Kropka pełna = 95% przedział ufności nie obejmuje zera. Epizody: {pl(n_ep, 0)} przy otwartym rynku.")
    foot = ("Przedziały: bootstrap klastrowy po dniach NY (2000 powtórzeń). P(t) = cena otwarcia ostatniej świecy 5-min <= t; przy zamkniętym "
            "rynku horyzonty < 1d są puste. Zdarzenia dzienne (GTA) mają tylko horyzonty >= 1d."
            + (" Tabela: gwiazdka = q < 0,05 po korekcie Benjaminiego-Hochberga na wszystkich polach tabeli." if fdr else ""))
    fig, top = _canvas(2, "Rynek", title, sub, foot)

    ax = fig.add_axes([0.07, 0.13, 0.53, top - 0.19])
    hz = [h for h in hz if any(len(t) and t.loc[t.horizon == h, "mean"].notna().any() for t in X.values())]
    xs = np.arange(len(hz))
    off = {k: (i - 1) * 0.16 for i, (k, *_) in enumerate(kinds)}
    for k, lab, col, mk in kinds:
        t = X[k]
        if t.empty or "lo" not in t:
            continue
        t = t[t.instrument == main].set_index("horizon").reindex(hz)
        ok = t["mean"].notna() & t["lo"].notna()
        for i in np.flatnonzero(ok.to_numpy()):
            r = t.iloc[i]
            x = xs[i] + off[k]
            ax.plot([x, x], [1e4 * r.lo, 1e4 * r.hi], color=col, lw=2, solid_capstyle="round", zorder=2)
            filled = bool(r["istotne"])
            ax.plot(x, 1e4 * r["mean"], marker=mk, ms=8.5, mfc=col if filled else SURFACE, mec=col, mew=2, zorder=3,
                    ls="none")
        ax.plot([], [], marker=mk, color=col, ls="none", ms=8, label=lab)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(xs, [f"t+{h}" for h in hz], fontsize=10.5)
    ax.set_ylabel(f"nadwyżka |zwrotu| {main} [pb]", fontsize=10.5)
    _axis_clean(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, frameon=False, fontsize=10, handletextpad=0.3, labelcolor=INK2,
              columnspacing=1.4, borderaxespad=0)

    # prawy panel: epizody x instrumenty x horyzonty (wartości w pb)
    a2 = fig.add_axes([0.67, 0.13, 0.29, top - 0.19])
    insts = list(res["prices"])
    hz2 = [h for h in hz if h in set(ep["horizon"])] if len(ep) else hz
    M = np.full((len(insts), len(hz2)), np.nan)
    S = np.zeros_like(M, dtype=bool)
    for i, inst in enumerate(insts):
        t = ep[ep.instrument == inst].set_index("horizon").reindex(hz2) if len(ep) else pd.DataFrame(index=hz2)
        if "mean" in t:
            M[i] = 1e4 * t["mean"].to_numpy(float)
            col = "istotne_fdr" if fdr and "istotne_fdr" in t else "istotne"
            S[i] = t.get(col, pd.Series(False, index=t.index)).fillna(False).to_numpy(bool)
    lim = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", [NEG, "#e9e8e4", POS])
    a2.imshow(M, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                a2.text(j, i, pl(M[i, j]) + ("*" if S[i, j] else ""), ha="center", va="center", fontsize=9,
                        color=INK, weight="bold" if S[i, j] else "normal")
    a2.set_xticks(range(len(hz2)), [f"+{h}" for h in hz2], fontsize=9.5)
    a2.set_yticks(range(len(insts)), insts, fontsize=10)
    a2.tick_params(length=0)
    for s in a2.spines.values():
        s.set_visible(False)
    a2.set_xticks(np.arange(-.5, len(hz2)), minor=True)
    a2.set_yticks(np.arange(-.5, len(insts)), minor=True)
    a2.grid(which="minor", color=SURFACE, lw=2)
    a2.tick_params(which="minor", length=0)
    a2.set_title("Epizody, wszystkie instrumenty [pb]; " + ("* istotne po FDR" if fdr else "* istotne"),
                 fontsize=10, color=INK2, loc="left", pad=8)
    return fig


# ============================================================================= SLAJD 3: media vs rynek
def _event_profile(x: pd.Series, days: pd.DatetimeIndex, lags, n_boot: int = 1000, seed: int = 0):
    """Średnia x[d+k] / średnia x z całego okresu (stała, więc nie dzielimy przez małą, losową liczbę).
    Przy dniach otwierających falę dni k < 0 są z definicji spokojniejsze (selekcja), więc NIE są punktem odniesienia;
    pokazujemy je tylko jako tło. CI z bootstrapu po zdarzeniach."""
    pos = x.index.get_indexer(days)
    pos = pos[pos >= 0]
    vals = np.full((len(pos), len(lags)), np.nan)
    arr = x.to_numpy(float)
    for j, k in enumerate(lags):
        p = pos + k
        ok = (p >= 0) & (p < len(arr))
        vals[ok, j] = arr[p[ok]]
    pre = [j for j, k in enumerate(lags) if k < 0]
    base = np.nanmean(arr)
    prof = lambda v: np.nanmean(v, 0) / base
    mean = prof(vals)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pos), (n_boot, len(pos)))
    boots = np.array([prof(vals[i]) for i in idx])
    level = np.nanmean(np.nanmean(vals, 0)[pre]) / np.nanmean(arr)       # poziom przed zdarzeniem vs cały okres
    return mean, np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0), len(pos), level


def first_in_wave(days: pd.DatetimeIndex, quiet: int = 5) -> pd.DatetimeIndex:
    """Dni z epizodem, przed którymi przez `quiet` dni nie było innego epizodu (początek fali)."""
    d = pd.DatetimeIndex(days).sort_values().unique()
    gap = d.to_series().diff().dt.days
    return d[(gap.isna() | (gap > quiet)).to_numpy()]


def _d(k: int) -> str:
    return "0" if k == 0 else f"+{k}"


@_styled
def slide_media(res: dict, main: str = "S&P 500", lags=range(-5, 8)):
    D = res["daily"]
    lags = list(lags)
    ev = res["events"]
    all_days = pd.DatetimeIndex(ev.loc[ev.kind == "trump_episode", "eff_day"].drop_duplicates().sort_values())
    quiet = -min(lags)
    days = first_in_wave(all_days, quiet)
    wave = len(days) >= 15
    if not wave:                                     # za mało izolowanych epizodów: wszystkie dni, z ostrzeżeniem
        days = all_days
    series = []
    tag = tag_of(main)
    if f"{tag}_absret" in D:
        series.append((f"|zwrot dzienny| {main}", D[f"{tag}_absret"], BLUE))
    if "fn_n" in D:
        series.append(("historie FNSPID o cłach / dzień", D["fn_n"], ORANGE))
    prof = {lab: _event_profile(s, days, lags) for lab, s, _ in series}
    peaks = {}
    for lab, (m_, lo, hi, n, lev) in prof.items():
        post = [j for j, k in enumerate(lags) if k >= 0]
        jj = post[int(np.nanargmax(m_[post]))]
        peaks[lab] = (lags[jj], m_[jj], lo[jj] > 1)
    n_days = len(days)
    if len(peaks) == 2:
        (lm, (km, vm, sm)), (lf, (kf, vf, sf)) = list(peaks.items())
        mk, fk = f"|zwrot| {main} ×{pl(vm, 2)} w dniu {_d(km)}", f"artykuły FNSPID ×{pl(vf, 2)} w dniu {_d(kf)}"
        if sm and sf:                      # kolejność ogłaszamy tylko, gdy oba skoki są istotne
            if km < kf:
                title = f"Rynek reaguje pierwszy, media później: {mk}, {fk} (× zwykłego dnia)"
            elif km > kf:
                title = f"Media wyprzedzają rynek: {fk}, {mk} (× zwykłego dnia)"
            else:
                title = f"Rynek i media reagują tego samego dnia ({_d(km)}): |zwrot| {main} ×{pl(vm, 2)}, artykuły FNSPID ×{pl(vf, 2)} zwykłego dnia"
        elif sm:
            title = f"Istotnie reaguje tylko rynek: {mk} (× zwykłego dnia); skok artykułów FNSPID nieistotny"
        elif sf:
            title = f"Istotnie reagują tylko media: {fk} (× zwykłego dnia); skok |zwrotu| {main} nieistotny"
        else:                              # brak istotności: kierunek + co dalej (bez przesady w obie strony)
            if vm > 1 and vf > 1:
                lead = (f"W dniu tweeta rynek (×{pl(vm, 2)}) i media (×{pl(vf, 2)}) są powyżej normy" if km == kf == 0 else
                        f"Po tweecie rynek (×{pl(vm, 2)}, dzień {_d(km)}) i media (×{pl(vf, 2)}, dzień {_d(kf)}) są powyżej normy")
                title = (f"{lead}; przy {pl(n_days, 0)} falach to jeszcze nie jest istotne, "
                         "więc kolejność rozstrzygnie test godzinowy")
            else:
                title = ("Na danych dziennych rynek i media nie odbiegają istotnie od normy wokół fali tweetów; "
                         "kolejność rozstrzygnie test godzinowy")
    else:
        title = "Profil reakcji wokół dni z tweetami o cłach"
    lev = {lab: v[4] for lab, v in prof.items()}
    sel = (f"{pl(n_days, 0)} dni otwierających falę tweetów (brak epizodu w {quiet} dniach wcześniej, z {pl(len(all_days), 0)} dni z epizodem)"
           if wave else f"{pl(n_days, 0)} dni z epizodem (za mało izolowanych; okna się nakładają)")
    sub = (f"Dzień 0 = {sel}. Oś: × średnia z całego okresu (1 = zwykły dzień). Szczyt istotny, gdy 95% CI leży powyżej 1. "
           + "Dni przed falą: " + ", ".join(f"{k.split(' /')[0]} ×{pl(v, 2)}" for k, v in lev.items())
           + (" (spokojniej niż zwykle, bo wybieramy dni bez wcześniejszych tweetów)." if wave and min(lev.values()) < 0.9 else "."))
    foot = ("Klucz dnia: data NY, a od 16:00 NY następny dzień (tweet po zamknięciu sesji liczy się do dnia, w którym rynek może zareagować). "
            "FNSPID ma znaczniki dzienne, więc w dniu 0 kolejność artykułu i tweeta jest nieznana. Pasmo: 95% CI, bootstrap po zdarzeniach.")
    fig, top = _canvas(3, "Media vs rynek", title, sub, foot)
    ax = fig.add_axes([0.07, 0.13, 0.64, top - 0.19])
    for lab, s, col in series:
        m_, lo, hi, _, _ = prof[lab]
        ax.fill_between(lags, lo, hi, color=col, alpha=0.14, lw=0)
        ax.plot(lags, m_, color=col, lw=2.2, marker="o", ms=5.5, mfc=col, mec=SURFACE, mew=1.2, label=lab)
    ax.axhline(1, color=INK2, lw=1)
    ax.axvline(0, color=RULE, lw=1, ls=(0, (2, 2)))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=10, borderaxespad=0, labelcolor=INK2)
    ax.text(0.08, ax.get_ylim()[1], " dzień tweeta", color=MUTED, fontsize=9.5, va="top")
    ax.set_xticks(lags, [f"{k:+d}" if k else "0" for k in lags], fontsize=10)
    ax.set_xlim(lags[0] - 0.3, lags[-1] + 0.3)
    ax.set_xlabel("dni względem epizodu tweetów", fontsize=10.5)
    ax.set_ylabel("× średnia z całego okresu", fontsize=10.5)
    _axis_clean(ax)

    # prawa kolumna: liczby
    x0 = 0.80
    yy = top - 0.01
    for lab, (k_, v_, s_) in peaks.items():
        fig.text(x0, yy, "dzień 0" if k_ == 0 else f"dzień +{k_}", fontsize=21, weight="bold", color=INK, va="top")
        fig.text(x0, yy - 0.055, "\n".join(textwrap.wrap(f"szczyt: {lab}", 26)), fontsize=10, color=INK2, va="top")
        fig.text(x0, yy - 0.115, f"×{pl(v_, 2)} zwykłego dnia\n" + ("istotne (CI > 1)" if s_ else "nieistotne"), fontsize=10, color=MUTED, va="top")
        yy -= 0.20
    ex = res["event_table"]
    if "gkg_files_post_1h" in ex:
        cov = float((ex.loc[ex.kind == "trump_episode", "gkg_files_post_1h"] >= 3).mean())
        msg = (f"GKG co 15 min pokrywa {pl(100 * cov, 0)}% epizodów: kolejność w minutach pokazuje slajd minutowy (sekcja 8)."
               if cov >= 0.3 else
               f"paczki 15-min pokrywają {pl(100 * cov, 0)}% epizodów; do testu w minutach trzeba pobrać GKG wokół zdarzeń (sekcja 8).")
        fig.text(x0, yy, "GKG", fontsize=10, weight="bold", color=INK2, va="top")
        fig.text(x0, yy - 0.03, "\n".join(textwrap.wrap(msg, 30)), fontsize=9.5, color=INK2, va="top")
    return fig


# ============================================================================= SLAJD 3 (minuty): kto pierwszy
@_styled
def slide_order(res: dict, O: dict, num: int = 3, total: int = 3):
    """Kolejność w minutach z order_test: skumulowany udział reakcji rynku i mediów po tweecie (0 -> 1) z 95% CI.
    Linia przerywana = T50 (połowa reakcji). Nagłówek = O['verdict'] (wyliczony z wyniku)."""
    C, S, n, prm = O["curves"], O["summary"].set_index("seria"), O["n"], O["params"]
    inst = prm["instrument"]
    off = C["min_od_t"].to_numpy(float)
    step = float(O["bins"]["step_min"])
    post = off >= 0
    sub = (f"{pl(n['w_teście_media'], 0)} epizodów tweetów z otwartym rynkiem i gęstym GKG (z {pl(n['zdarzeń'], 0)}). "
           f"Rynek: |zwrot| {inst} co {step:.0f} min minus 20 chwil placebo o tej samej porze tygodnia. "
           f"Media: artykuły GKG o cłach na paczkę minus poziom z {prm['before']}–{prm['base_gap']} przed tweetem. "
           "Krzywe: jaka część reakcji z całego okna nastąpiła do danej minuty.")
    foot = ("Czas GKG to czas paczki 15-min (artykuły z poprzednich 15 min), więc media mogą wyglądać na spóźnione do 15 min; "
            "test powtórzono z mediami cofniętymi o 15 min. T50 = minuta, w której skumulowana nadwyżka osiąga połowę sumy z okna "
            f"(0, {prm['after']}]. Pasma i CI: bootstrap klastrowy po dniach NY ({prm['n_boot']} powtórzeń).")
    fig, top = _canvas(num, "Kto pierwszy (minuty)", O["verdict"], sub, foot)
    fig.texts[0].set_text(f"{num}/{total}  ·  KTO PIERWSZY (MINUTY)")
    ax = fig.add_axes([0.07, 0.13, 0.62, top - 0.19])
    B = O["bins"]
    rng = np.random.default_rng(0)
    for name, col, lab in (("rynek", BLUE, f"rynek: |zwrot| {inst}"), ("media", ORANGE, "media: artykuły GKG o cłach")):
        m_ = C[name].to_numpy(float)[post]
        x = np.r_[0.0, off[post] + step]
        tot = np.nansum(m_)
        if not tot > 0:
            continue
        share = np.r_[0.0, np.cumsum(np.nan_to_num(m_)) / tot]
        # CI udziału: bootstrap po dniach (te same wagi co w order_test, tu uproszczony: losowanie zdarzeń w dniach)
        X = O["X"][name]                       # te same zdarzenia co w order_test
        use = np.isfinite(X[:, post]).any(1)
        days = pd.factorize(B["events"]["day"])[0]
        sh = []
        for _ in range(300):
            d = rng.integers(0, days.max() + 1, days.max() + 1)
            w = np.bincount(d, minlength=days.max() + 1)[days] * use
            xm = X[:, post]
            with np.errstate(invalid="ignore", divide="ignore"):
                mb = (w @ np.nan_to_num(xm)) / (w @ np.isfinite(xm))
            tb = np.nansum(mb)
            if tb > 0:
                sh.append(np.r_[0.0, np.cumsum(np.nan_to_num(mb)) / tb])
        if sh:
            sh = np.array(sh)
            ax.fill_between(x, np.percentile(sh, 2.5, 0), np.percentile(sh, 97.5, 0), color=col, alpha=0.14, lw=0)
        ax.plot(x, share, color=col, lw=2.4, label=lab)
        t50 = S.loc[name, "T50_min"]
        if np.isfinite(t50):
            ax.axvline(t50, color=col, lw=1.2, ls=(0, (3, 2)))
            ax.text(t50, 1.03, f"T50 = {pl(t50, 0)} min", color=col, fontsize=9.5, ha="center", va="bottom",
                    transform=ax.get_xaxis_transform())
    ax.axhline(0.5, color=RULE, lw=0.8)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlim(0, off[post][-1] + step)
    ax.set_xlabel("minuty od tweeta", fontsize=10.5)
    ax.set_ylabel("skumulowana część reakcji", fontsize=10.5)
    _axis_clean(ax, nd=1)
    ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=INK2)

    # prawa kolumna: liczby
    x0, yy = 0.73, top - 0.01
    d, ds = S.loc["różnica media - rynek"], S.loc["różnica media_-krok - rynek"]
    tiles = [(f"{pl(S.loc['rynek', 'T50_min'], 0)} min", f"połowa reakcji rynku ({inst})"),
             (f"{pl(S.loc['media', 'T50_min'], 0)} min", "połowa reakcji mediów (GKG)"),
             (f"{'+' if d['T50_min'] > 0 else ''}{pl(d['T50_min'], 0)} min",
              f"media minus rynek, 95% CI {pl(d['T50_lo'], 0)} do {pl(d['T50_hi'], 0)}"),
             (f"{'+' if ds['T50_min'] > 0 else ''}{pl(ds['T50_min'], 0)} min",
              f"to samo, media cofnięte o 15 min: CI {pl(ds['T50_lo'], 0)} do {pl(ds['T50_hi'], 0)}")]
    gap_ = min(0.11, (top - 0.20) / len(tiles))
    for big, small in tiles:
        fig.text(x0, yy, big, fontsize=17, weight="bold", color=INK, va="top")
        fig.text(x0, yy - 0.045, "\n".join(textwrap.wrap(small, 40)), fontsize=9.2, color=INK2, va="top")
        yy -= gap_
    pre = S.loc["media", "przed_tweetem_1h"]
    if np.isfinite(pre):
        lo_, hi_ = S.loc["media", "przed_lo"], S.loc["media", "przed_hi"]
        txt = (f"Godzina przed tweetem: media {'+' if pre > 0 else ''}{pl(pre, 2)} art./paczkę (CI {pl(lo_, 2)} do {pl(hi_, 2)}), "
               + ("czyli ruszyły PRZED Trumpem." if lo_ > 0 else "bez istotnego wyprzedzenia."))
        fig.text(x0, max(yy, 0.17), "\n".join(textwrap.wrap(txt, 44)), fontsize=9.2, color=INK2, va="top")
    return fig


@_styled
def slide_order_profile(res: dict, O: dict):
    """Pomocniczo (do notebooka, nie na slajd): surowe nadwyżki na przedział z CI, także przed tweetem."""
    C = O["curves"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, name, col, unit in ((axes[0], "rynek", BLUE, "pb"), (axes[1], "media", ORANGE, "art./paczkę")):
        ax.fill_between(C["min_od_t"], C[f"{name}_lo"], C[f"{name}_hi"], color=col, alpha=0.15, lw=0, step="post")
        ax.step(C["min_od_t"], C[name], where="post", color=col, lw=2)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvline(0, color=RULE, lw=1, ls="--")
        ax.set_ylabel(f"{name}: nadwyżka [{unit}]")
    axes[1].set_xlabel("minuty od tweeta (początek przedziału)")
    fig.tight_layout()
    return fig


# ============================================================================= zapis
def save_deck(figs, out_dir="slajdy", name: str = "slajdy_dane", dpi: int = 200) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pngs = []
    for i, f in enumerate(figs, 1):
        p = out / f"{name}_{i}.png"
        f.savefig(p, dpi=dpi, facecolor=SURFACE, transparent=False)
        pngs.append(p)
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out / f"{name}.pdf") as pdf:
        for f in figs:
            pdf.savefig(f, facecolor=SURFACE)
    res = {"png": pngs, "pdf": out / f"{name}.pdf"}
    try:
        from pptx import Presentation
        from pptx.util import Inches
        prs = Presentation()
        prs.slide_width, prs.slide_height = Inches(W), Inches(H)
        for p in pngs:
            s = prs.slides.add_slide(prs.slide_layouts[6])
            s.shapes.add_picture(str(p), 0, 0, width=prs.slide_width, height=prs.slide_height)
        prs.save(out / f"{name}.pptx")
        res["pptx"] = out / f"{name}.pptx"
    except ImportError:
        print("python-pptx niedostępny: zapisano PNG i PDF (pip install python-pptx, żeby dostać .pptx).")
    return res
