"""Pobieranie brakujących źródeł (tylko gdy plik/katalog nie istnieje).

NIE pobiera: data.taz ani FNSPID (zostają ręcznie).
Pobiera: Kaggle (tweety, Truth Social), GTA (API), Federal Register (API), GDELT GKG (próbka albo gęsto wokół
zdarzeń: download_gkg_dense), Dukascopy (pip).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .base import Config

GTA_URL = "https://api.globaltradealert.org/api/v2/gta/data/"
FR_URL = "https://www.federalregister.gov/api/v1/documents.json"
GKG_URL = "http://data.gdeltproject.org/gdeltv2/{stamp}.gkg.csv.zip"
FRED_URL = "https://fred.stlouisfed.org/graphdata/fredgraph.csv?id={sid}"


def download_vix(cfg: Config) -> str:
    """VIX dzienny z FRED (VIXCLS), bez klucza API."""
    out = Path(cfg.vix_csv)
    if out.exists():
        return f"jest ({out})"
    from io import StringIO
    r = _session().get(FRED_URL.format(sid="VIXCLS"), timeout=60)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df.columns = ["date", "vix"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["vix"] = pd.to_numeric(df["vix"], errors="coerce")
    lo = pd.Timestamp(min(s for s, _ in cfg.periods.values())) - pd.Timedelta(days=30)
    hi = pd.Timestamp(max(e for _, e in cfg.periods.values())) + pd.Timedelta(days=30)
    df = df.dropna()
    df = df[(df["date"] >= lo) & (df["date"] <= hi)]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return f"zapisano {len(df):,} dni VIX -> {out}"


def _dense_stamps(centers, before="2h", after="4h") -> list[str]:
    """Znaczniki paczek 15-min (RRRRMMDDGGMMSS) pokrywające okna [c - before, c + after] wokół momentów `centers`."""
    b, a = pd.Timedelta(before), pd.Timedelta(after)
    stamps = set()
    for c in pd.to_datetime(pd.Series(list(centers)), utc=True).dropna():
        c = c.tz_convert("UTC").tz_localize(None)
        lo, hi = (c - b).floor("15min"), (c + a).ceil("15min")
        stamps |= {t.strftime("%Y%m%d%H%M%S") for t in pd.date_range(lo, hi, freq="15min")}
    return sorted(stamps)


def plan_gkg_dense(centers, before: str = "2h", after: str = "4h", cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Ile paczek trzeba pobrać do gęstego GKG (przed uruchomieniem pobierania). Paczka GKG 2.1 to ok. 5-25 MB."""
    st = _dense_stamps(centers, before, after)
    have = 0
    if cache_dir is not None and Path(cache_dir).exists():
        names = {p.name[:14] for p in Path(cache_dir).glob("*.gkg.csv.zip")}
        have = sum(x in names for x in st)
    todo = len(st) - have
    return pd.DataFrame([{"zdarzeń": int(pd.Series(list(centers)).notna().sum()), "okno": f"-{before} .. +{after}",
                          "paczek w oknach": len(st), "już na dysku": have, "do pobrania": todo,
                          "transfer [GB] (5-25 MB/paczkę)": f"{todo * 5 / 1024:.0f}-{todo * 25 / 1024:.0f}",
                          "na dysku po odchudzeniu [GB]": f"~{todo * 0.3 / 1024:.1f}"}])


TOPIC_COMMENT = b"topic-only"            # komentarz zipa: paczka odchudzona do artykułów o cłach


def is_thinned(path) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return z.comment == TOPIC_COMMENT
    except (zipfile.BadZipFile, OSError):
        return False


def thin_gkg_zip(path) -> tuple[int, int]:
    """Zostawia w paczce tylko wiersze o cłach (ten sam filtr URL/motywów co GKGExplorer.clean), w tym samym formacie
    i pod tą samą nazwą. Lista paczek (mianownik stawek) się nie zmienia, a plik maleje ok. 30-100 razy.
    Zwraca (wiersze przed, wiersze po)."""
    from .gdelt_gkg import GKGExplorer
    path = Path(path)
    if is_thinned(path):
        return (-1, -1)
    with zipfile.ZipFile(path) as z:
        inner = z.namelist()[0]
        raw = z.read(inner).decode("utf-8", errors="replace")
    lines = [x for x in raw.split("\n") if x]
    keep = []
    for ln in lines:
        f = ln.split("\t")
        url = f[4] if len(f) > 4 else ""
        themes = f[8] if len(f) > 8 else ""
        if GKGExplorer.URL_RE.search(url) or GKGExplorer.THEME_RE.search(themes):
            keep.append(ln)
    tmp = path.with_suffix(".thin")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(inner, "\n".join(keep) + ("\n" if keep else ""))
        z.comment = TOPIC_COMMENT
    tmp.replace(path)
    return len(lines), len(keep)


def download_gkg_dense(cfg: Config, centers, before: str = "2h", after: str = "4h", workers: int = 4,
                       hours: float | None = None, max_files: int | None = None) -> str:
    """CIĄGŁE paczki 15-min w oknach [c - before, c + after] wokół momentów `centers` (np. początków epizodów tweetów).
    Domyślna próbka (plan_gkg_stamps, co ok. 2h15) nie mierzy opóźnień rzędu minut; ta funkcja tak.
    Każda paczka jest pobierana do pliku tymczasowego, odchudzana do artykułów o cłach (thin_gkg_zip, ok. 1-3% rozmiaru)
    i dopiero wtedy dostaje nazwę RRRRMMDDGGMMSS.gkg.csv.zip. Paczka gęsta = zip z komentarzem 'topic-only': po tym
    scalanie odróżnia ją od próbki równomiernej i NIE używa jej w panelach dziennym i godzinowym (patrz build_merged).
    Paczki, które już są na dysku (np. z próbki), zostają bez zmian. Można przerwać i uruchomić ponownie.
    Dziennik: <gkg_cache_dir>/gkg_dense_log.csv. Po pobraniu: run_cleaning(cfg, only=["GKG"]) i ponowne build_merged."""
    import time
    if hours is not None:                                   # zgodność ze starszym wywołaniem (±hours)
        before = after = f"{hours}h"
    d = Path(cfg.gkg_cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    stamps = _dense_stamps(centers, before, after)[:max_files]
    print(plan_gkg_dense(centers, before, after, d).to_string(index=False))
    s = _session()
    rows, t0 = [], time.time()

    def one(st):
        f = d / f"{st}.gkg.csv.zip"
        if f.exists() and f.stat().st_size > 0:             # już jest (gęsta albo z próbki): bez zmian
            return {"stamp": st, "wynik": "cache", "wiersze": None, "o_cłach": None}
        tmp = d / f"{st}.dense.zip"                         # nie pasuje do *.gkg.csv.zip, więc nigdy nie wejdzie do danych
        r = _fetch_gkg(s, st, tmp)
        n0 = n1 = None
        if r == "ok":
            try:
                n0, n1 = thin_gkg_zip(tmp)
                tmp.replace(f)
            except Exception as ex:                          # uszkodzony zip: przy następnym przebiegu pobierze się od nowa
                r = f"błąd odchudzania: {str(ex)[:40]}"
            finally:
                tmp.unlink(missing_ok=True)
        return {"stamp": st, "wynik": r, "wiersze": n0, "o_cłach": n1}

    with ThreadPoolExecutor(workers) as ex:
        for i, row in enumerate(ex.map(one, stamps), 1):
            rows.append(row)
            if i % 100 == 0 or i == len(stamps):
                el = time.time() - t0
                print(f"  GKG gęsto: {i}/{len(stamps)}  ({el / 60:.0f} min, zostało ok. {el / i * (len(stamps) - i) / 60:.0f} min)")
    log = pd.DataFrame(rows)
    lp = d / "gkg_dense_log.csv"
    if lp.exists():
        log = pd.concat([pd.read_csv(lp, dtype={"stamp": str}), log]).drop_duplicates("stamp", keep="last")
    log.to_csv(lp, index=False)
    cnt = log.loc[log["stamp"].isin(stamps), "wynik"].map(lambda x: "błąd" if str(x).startswith("błąd") else x)
    return f"{cnt.value_counts().to_dict()} -> {d} (dziennik: {lp})"


# --------------------------------------------------------------------------- pomocnicze
def ensure_pip(pip_name: str, import_name: str | None = None):
    """Importuje pakiet; jeśli go nie ma, instaluje przez pip."""
    import importlib
    name = import_name or pip_name.replace("-", "_")
    try:
        return importlib.import_module(name)
    except ImportError:
        print(f"Instaluję {pip_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
        importlib.invalidate_caches()
        return importlib.import_module(name)


def _session() -> requests.Session:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    kw = dict(total=4, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504))
    try:
        retry = Retry(allowed_methods=None, **kw)      # ponawiaj także POST
    except TypeError:                                   # starszy urllib3
        retry = Retry(**kw)
    s = requests.Session()
    s.mount("http://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
    s.headers["User-Agent"] = "research-notebook/1.0"
    return s


# --------------------------------------------------------------------------- Kaggle
def download_kaggle(slug: str, dest: str | Path) -> str:
    """Pobiera zbiór przez kagglehub i kopiuje największy plik CSV do `dest`."""
    dest = Path(dest)
    if dest.exists():
        return f"jest ({dest})"
    kh = ensure_pip("kagglehub")
    try:
        folder = Path(kh.dataset_download(slug))
    except Exception as ex:
        raise RuntimeError(f"kagglehub nie pobrał '{slug}' ({ex}). Jeśli prosi o logowanie: ustaw KAGGLE_USERNAME i "
                           "KAGGLE_KEY albo wgraj ~/.kaggle/kaggle.json (kaggle.com -> Settings -> API).")
    allf = [p for p in folder.rglob("*") if p.is_file()]
    csvs = sorted((p for p in allf if p.suffix.lower() == ".csv"), key=lambda p: p.stat().st_size, reverse=True)
    print(f"  {slug}: pliki w zbiorze: {[(p.name, round(p.stat().st_size / 1e6, 2)) for p in allf][:10]}")
    if not csvs:
        raise FileNotFoundError(f"W zbiorze {slug} nie ma CSV (są: {[p.name for p in allf][:10]}). Skopiuj plik ręcznie do {dest}.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(csvs[0], dest)
    note = f" (UWAGA: było {len(csvs)} CSV, wzięto największy)" if len(csvs) > 1 else ""
    return f"pobrano {csvs[0].name} -> {dest}{note}"


# --------------------------------------------------------------------------- GTA
def _first(rec: dict, *names, default=None):
    for n in names:
        if n in rec and rec[n] not in (None, ""):
            return rec[n]
    return default


def _names(v):
    if isinstance(v, list):
        return "; ".join(str(x.get("name", x.get("iso_code", x))) if isinstance(x, dict) else str(x) for x in v)
    if isinstance(v, dict):
        return str(v.get("name", v))
    return v


def download_gta(cfg: Config, api_key: str | None = None) -> str:
    out = Path(cfg.gta_csv)
    if out.exists():
        return f"jest ({out})"
    key = api_key or os.environ.get("GTA_API_KEY")
    if not key:
        return "POMINIĘTO: brak klucza (ustaw zmienną środowiskową GTA_API_KEY albo podaj gta_api_key=...)"
    s = _session()
    hdr = {"Authorization": f"APIKey {key}", "Content-Type": "application/json"}
    impl_keys = ["implementer", "implementing_jurisdictions"]   # nazwa filtra: sprawdzamy obie, pierwsza która nie da 400
    types = list(cfg.gta_types.values())
    rows, shown = [], False
    for pname, (d0, d1) in cfg.periods.items():
        for actor, code in cfg.gta_actors.items():
            offset = 0
            while True:
                r = None
                for ik in list(impl_keys):
                    body = {"request_data": {ik: [code], "intervention_types": types, "announcement_period": [d0, d1]},
                            "limit": 1000, "offset": offset}
                    r = s.post(GTA_URL, headers=hdr, json=body, timeout=180)
                    if r.status_code == 400:
                        print(f"  GTA: filtr '{ik}' odrzucony (400): {r.text[:200]}")
                        impl_keys.remove(ik)
                        continue
                    impl_keys = [ik]                              # ten działa
                    break
                if r is None or r.status_code == 400:
                    raise RuntimeError(f"GTA: żadna nazwa filtra kraju nie przeszła. Ostatnia odpowiedź: {r.text[:300] if r is not None else ''}")
                if r.status_code in (401, 403):
                    raise RuntimeError(f"GTA: {r.status_code} (klucz nieaktywny albo bez dostępu): {r.text[:200]}")
                r.raise_for_status()
                page = r.json()
                if isinstance(page, dict):
                    page = page.get("results", page.get("data", []))
                if page and not shown:
                    print("  GTA: pola w pierwszym rekordzie:", sorted(page[0].keys()))
                    shown = True
                for rec in page:
                    prods = rec.get("affected_products")
                    row = {
                        "intervention_id": _first(rec, "intervention_id", "id"),
                        "actor": actor,
                        "title": _first(rec, "intervention_title", "state_act_title", "title"),
                        "type": _names(_first(rec, "intervention_type", "type")),
                        "evaluation": _names(_first(rec, "gta_evaluation", "evaluation")),
                        "implementing": _names(_first(rec, "implementing_jurisdictions", "implementing_jurisdiction")),
                        "affected": _names(_first(rec, "affected_jurisdictions", "affected_jurisdiction")),
                        "date_announced": _first(rec, "date_announced"),
                        "date_published": _first(rec, "date_published"),
                        "date_implemented": _first(rec, "date_implemented"),
                        "date_removed": _first(rec, "date_removed"),
                    }
                    if prods is not None:
                        row["n_products"] = len(prods) if isinstance(prods, list) else np.nan
                    if "is_in_force" in rec or "intervention_status" in rec:
                        row["is_in_force"] = _first(rec, "is_in_force", "intervention_status")
                    rows.append(row)
                print(f"  GTA {actor} {pname}: offset {offset}, +{len(page)}")
                if len(page) < 1000:
                    break
                offset += 1000
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return f"zapisano {len(df):,} interwencji -> {out}"


# --------------------------------------------------------------------------- Federal Register
def download_fr(cfg: Config) -> str:
    out = Path(cfg.fr_csv)
    if out.exists():
        return f"jest ({out})"
    s = _session()
    fields = ["document_number", "title", "type", "publication_date", "effective_on", "signing_date", "abstract", "html_url"]
    rows = []
    for pname, (d0, d1) in cfg.periods.items():
        quarters = pd.period_range(d0, d1, freq="Q")
        for qname, term in cfg.fr_queries.items():
            for q in quarters:
                a = max(q.start_time, pd.Timestamp(d0)).strftime("%Y-%m-%d")
                b = min(q.end_time.normalize(), pd.Timestamp(d1)).strftime("%Y-%m-%d")
                params = [("conditions[term]", term), ("conditions[publication_date][gte]", a),
                          ("conditions[publication_date][lte]", b), ("per_page", 1000), ("order", "oldest")]
                params += [("conditions[type][]", t) for t in cfg.fr_types]
                params += [("fields[]", f) for f in fields]
                page, got = 1, 0
                while True:
                    r = s.get(FR_URL, params=params + [("page", page)], timeout=90)
                    r.raise_for_status()
                    j = r.json()
                    res = j.get("results", [])
                    rows += [{**d, "query": qname} for d in res]
                    got += len(res)
                    if not res or page >= j.get("total_pages", 1):
                        break
                    page += 1
                print(f"  FR [{qname}] {a}..{b}: {got}")
    df = pd.DataFrame(rows)
    if df.empty:
        return "API zwróciło 0 dokumentów (nie zapisuję pliku)"
    df = df.drop_duplicates(["document_number", "query"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return f"zapisano {len(df):,} dokumentów -> {out}"


# --------------------------------------------------------------------------- GDELT GKG
def plan_gkg_stamps(windows, n_files: int) -> list[str]:
    """Równomiernie rozłożone paczki 15-min wewnątrz okien wokół zdarzeń (limit łącznie n_files)."""
    per = max(1, n_files // max(len(windows), 1))
    stamps = []
    for a, b in windows:
        grid = pd.date_range(pd.Timestamp(a), pd.Timestamp(b), freq="15min")[:-1]
        k = min(per, len(grid))
        idx = np.unique(np.linspace(0, len(grid) - 1, k).round().astype(int))
        stamps += [t.strftime("%Y%m%d%H%M%S") for t in grid[idx]]
    return stamps


def _get_gkg(session, stamp: str, dest_dir: Path):
    f = dest_dir / f"{stamp}.gkg.csv.zip"
    if f.exists() and f.stat().st_size > 0:
        return "cache"
    return _fetch_gkg(session, stamp, f)


def _fetch_gkg(session, stamp: str, f: Path):
    """Pobiera paczkę GKG do pliku f (przez plik .part, żeby przerwanie nie zostawiło uciętego zipa)."""
    dest_dir = f.parent
    tmp = dest_dir / f"{stamp}.part"
    try:
        r = session.get(GKG_URL.format(stamp=stamp), stream=True, timeout=120)
        if r.status_code == 404:
            return "404"
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        if not zipfile.is_zipfile(tmp):
            tmp.unlink(missing_ok=True)
            return "nie-zip"
        tmp.replace(f)
        return "ok"
    except Exception as ex:
        tmp.unlink(missing_ok=True)
        return f"błąd: {str(ex)[:60]}"


def download_gkg(cfg: Config, n_files: int = 300, windows=None, workers: int = 4) -> str:
    d = Path(cfg.gkg_cache_dir)
    if d.exists() and any(d.glob("*.gkg.csv.zip")):
        return f"są pliki w {d}"
    if windows is None:
        from .dukascopy_src import DukascopyExplorer
        windows = list(DukascopyExplorer.DEFAULT_WINDOWS.values())
    d.mkdir(parents=True, exist_ok=True)
    stamps = plan_gkg_stamps(windows, n_files)
    print(f"  GKG: {len(stamps)} plików z {len(windows)} okien wokół zdarzeń (ok. 5-25 MB każdy). "
          "Pełne 2018-2019 to ~70 tys. plików, więc pobieram TYLKO próbkę.")
    s = _session()
    res = {}
    with ThreadPoolExecutor(workers) as ex:
        for i, (st, r) in enumerate(zip(stamps, ex.map(lambda x: _get_gkg(s, x, d), stamps)), 1):
            res[st] = r
            if i % 50 == 0:
                print(f"  GKG: {i}/{len(stamps)}")
    cnt = pd.Series(res).map(lambda x: x if not x.startswith("błąd") else "błąd").value_counts().to_dict()
    return f"{cnt} -> {d}"


# --------------------------------------------------------------------------- Dukascopy
def ensure_dukascopy() -> str:
    ensure_pip("dukascopy-python", "dukascopy_python")
    from .dukascopy_src import DukascopyFetcher
    for frag in ("EUR_USD", "500"):
        try:
            print(f"  Dukascopy: instrumenty pasujące do '{frag}':", DukascopyFetcher.list_instruments(frag)[:8])
        except Exception as ex:
            print("  Dukascopy:", ex)
    return "pakiet zainstalowany (dane pobierze sekcja 7 notebooka, z cache)"


# --------------------------------------------------------------------------- całość
def ensure_all(cfg: Config, gta_api_key: str | None = None, gkg_files: int = 300, gkg_windows=None) -> pd.DataFrame:
    status = []

    def run(name, fn):
        try:
            msg = fn()
        except Exception as ex:
            msg = f"BŁĄD: {type(ex).__name__}: {ex}"
        status.append({"źródło": name, "wynik": msg})
        print(f"[{name}] {msg}")

    for name, slug in cfg.kaggle_datasets.items():
        dest = cfg.trump_sources.get(name)
        if dest:
            run(f"Kaggle: {name}", lambda slug=slug, dest=dest: download_kaggle(slug, dest))
    run("GTA", lambda: download_gta(cfg, gta_api_key))
    run("Federal Register", lambda: download_fr(cfg))
    run("GDELT GKG", lambda: download_gkg(cfg, gkg_files, gkg_windows))
    run("VIX (FRED)", lambda: download_vix(cfg))
    run("Dukascopy (pip)", ensure_dukascopy)
    for nm, p in [("data.taz", cfg.market_archive), ("FNSPID clean", cfg.fnspid_clean_csv)]:
        ok = Path(p).exists()
        status.append({"źródło": nm, "wynik": "jest" if ok else f"BRAK {p} (nie pobieram, dodaj ręcznie)"})
        if not ok:
            print(f"[{nm}] BRAK {p}: nie pobieram, dodaj ręcznie")
    return pd.DataFrame(status)