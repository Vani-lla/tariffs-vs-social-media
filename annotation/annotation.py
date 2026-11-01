"""Adnotacja tweetów Trumpa: próbka -> arkusze CSV -> zgodność -> rozstrzyganie -> gold standard -> walidacja
instrumentu -> połączenie etykiet z danymi scalonymi (clean_out/merged).

Zasady (codebook v0.1.1):
  * jednostka = jeden oryginalny post z Twittera 2018-2019 (bez RT, bez postów będących samym linkiem),
  * adnotacja ŚLEPA: w arkuszu nie ma reakcji rynku, polubień ani liczby artykułów,
  * etykiety: cla (tak/nie), kierunek (ESKALACJA/DEESKALACJA/NEUTRALNE), emocja (opcjonalna), uwagi,
  * pliki CSV: separator ';', kodowanie UTF-8 z BOM (Excel w polskiej wersji otwiera je poprawnie),
  * żadna funkcja nie nadpisuje arkusza, w którym ktoś już coś wpisał (chyba że overwrite=True).
"""
from __future__ import annotations

import itertools
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from cleaning_data.base import Config, show
from merging_data.features import eff_day

CODEBOOK_VERSION = "0.1.2"
FIELDS = {
    "cla": ["tak", "nie"],
    "kierunek": ["ESKALACJA", "DEESKALACJA", "NEUTRALNE"],
    "emocja": ["gniew", "duma", "optymizm", "grozba", "inne", "brak"],
}
MAIN = ["cla", "kierunek"]
ND = "n/d"                                   # kierunek dla cla = nie (nie oznaczamy)
LABEL_COLS = ["cla", "kierunek", "emocja", "uwagi"]
ANNOTATORS = ("BB", "IK", "MG", "PK")          # Borkowski, Kowalczyk, Gadomski, Korus
SEP, ENC = ";", "utf-8-sig"
STAGES = {"pilotaz": "1", "nakladka": "2", "pojedyncze": "2"}

ALIASES = {
    "cla": {"t": "tak", "y": "tak", "yes": "tak", "1": "tak", "true": "tak", "n": "nie", "no": "nie", "0": "nie", "false": "nie"},
    "kierunek": {"e": "ESKALACJA", "esk": "ESKALACJA", "eskalacja": "ESKALACJA", "d": "DEESKALACJA", "deesk": "DEESKALACJA",
                 "deeskalacja": "DEESKALACJA", "n": "NEUTRALNE", "neutr": "NEUTRALNE", "neutralne": "NEUTRALNE",
                 "neutralny": "NEUTRALNE"},
    "emocja": {"grozba": "grozba"},
}


# =============================================================================== wartości i pliki
def _ascii(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
    return re.sub(r"[\s\-]+", "_", s.strip())


def normalize(field: str, x) -> tuple[str, bool]:
    """(wartość kanoniczna, czy poprawna). Pusty napis = brak wpisu (poprawny)."""
    s = _ascii(x)
    if s == "":
        return "", True
    for a in FIELDS[field]:
        if s.lower() == _ascii(a).lower():
            return a, True
    a = ALIASES.get(field, {}).get(s.lower())
    return (a, True) if a else (str(x).strip(), False)


def read_csv_any(path) -> pd.DataFrame:
    """Arkusz .xlsx (karta Adnotacja/Rozstrzyganie) albo CSV (';' lub ','; UTF-8 albo cp1250). Wszystko jako tekst."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        xl = pd.ExcelFile(path)
        sheet = next((n for n in ("Adnotacja", "Rozstrzyganie") if n in xl.sheet_names), xl.sheet_names[0])
        d = pd.read_excel(xl, sheet_name=sheet, dtype=str).fillna("")
        d.columns = [str(c).strip() for c in d.columns]
        if "item_id" in d.columns:
            for c in d.columns:
                d[c] = d[c].astype(str).str.strip().replace({"nan": "", "None": "", "NaN": ""})
            return d[d["item_id"] != ""].reset_index(drop=True)
        raise ValueError(f"{path}: brak kolumny item_id w karcie {sheet}.")
    for enc in (ENC, "cp1250"):                   # cp1250: Excel "CSV (rozdzielany przecinkami)" bez UTF-8
        for sep in (SEP, ",", "\t"):
            try:
                d = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding=enc)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
            if "item_id" in d.columns:
                return d
    raise ValueError(f"{path}: nie znaleziono kolumny item_id (sprawdź separator i nagłówek).")


read_sheet = read_csv_any


def _has_labels(path: Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    try:
        d = read_csv_any(path)
    except Exception:
        return False
    cols = [c for c in ("cla", "kierunek", "FINAL_cla", "FINAL_kierunek") if c in d]
    return bool(len(cols) and (d[cols].apply(lambda s: s.str.strip() != "")).any().any())


def _write(df: pd.DataFrame, path: Path, overwrite: bool, writer=None) -> str:
    path = Path(path)
    if _has_labels(path) and not overwrite:
        return f"POMINIĘTO (ma już wpisy, nie nadpisuję): {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if writer is not None:
        writer(df, path)
    else:
        df.to_csv(path, sep=SEP, index=False, encoding=ENC)
    return f"zapisano: {path} ({len(df)} wierszy)"


# ------------------------------------------------------------------------------- arkusz Excela z listami
XL_FONT = "Arial"
XL_HEAD, XL_GREY, XL_YELLOW, XL_ORANGE = "1F3A5F", "F0EFEC", "FFF6CC", "F9D5C2"
WIDTHS = {"item_id": 9, "etap": 11, "data_utc": 16, "tekst": 90, "cla": 9, "kierunek": 16, "emocja": 12,
          "uwagi": 40, "uzasadnienie": 45}
EXAMPLE = {"item_id": "T0000", "etap": "przykład", "data_utc": "2019-05-05 16:08",
           "tekst": "For 10 months, China has been paying Tariffs to the USA of 25% on 50 Billion Dollars... The 10% will go up to 25% on Friday.",
           "cla": "tak", "kierunek": "ESKALACJA", "emocja": "grozba", "uwagi": "zapowiedź podwyżki stawki = eskalacja"}


def _xl_text(v) -> str:
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip().lower() in ("nan", "none", "<na>", "nat"):
        return ""
    return ILLEGAL_CHARACTERS_RE.sub("", str(v))


def _xlsx_write(df: pd.DataFrame, path: Path, inputs: dict, cand_ids=None, cand_col: str | None = None,
                sheet: str = "Adnotacja", instruction: list[str] | None = None, example: dict | None = None):
    """inputs: kolumna -> pole słownika FIELDS (lista rozwijana) albo None (tekst swobodny).
    cand_ids: lista napisów 'id1|id2|...' per wiersz -> lista rozwijana w kolumnie cand_col (+ 'brak')."""
    from openpyxl import Workbook
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Protection, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = sheet
    cols = list(df.columns)
    L = {c: get_column_letter(i + 1) for i, c in enumerate(cols)}
    thin = Side(style="thin", color="D9D9D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    fill = {k: PatternFill("solid", fgColor=v) for k, v in (("head", XL_HEAD), ("grey", XL_GREY), ("yellow", XL_YELLOW))}
    for j, c in enumerate(cols, 1):
        cell = ws.cell(1, j, c)
        cell.font = Font(name=XL_FONT, bold=True, color="FFFFFF", size=10)
        cell.fill = fill["head"]
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        cell.border = border
        ws.column_dimensions[L[c]].width = WIDTHS.get(c, 26 if c.endswith("_oceny") else 14)
    for i, row in enumerate(df.itertuples(index=False), 2):
        longest = 1
        for j, (c, v) in enumerate(zip(cols, row), 1):
            cell = ws.cell(i, j)
            cell.value = _xl_text(v)
            cell.data_type = "s"                      # tekst zaczynający się od '=' nie stanie się formułą
            edit = c in inputs
            cell.protection = Protection(locked=not edit)
            cell.fill = fill["yellow"] if edit else fill["grey"]
            cell.font = Font(name=XL_FONT, size=10)
            cell.border = border
            wrap = c in ("tekst", "uwagi", "uzasadnienie") or c.endswith("_oceny")
            cell.alignment = Alignment(wrap_text=wrap, vertical="top")
            if wrap and cell.value:
                longest = max(longest, int(np.ceil(len(cell.value) / max(ws.column_dimensions[L[c]].width * 1.1, 1))))
        ws.row_dimensions[i].height = min(15 * longest + 2, 150)
    n = len(df) + 1
    # listy wartości na ukrytej karcie
    ls = wb.create_sheet("Listy")
    for k, (f, vals) in enumerate(FIELDS.items(), 1):
        ls.cell(1, k, f)
        for r, v in enumerate(vals, 2):
            ls.cell(r, k, v)
    ls.sheet_state = "hidden"
    listcol = {f: get_column_letter(k) for k, f in enumerate(FIELDS, 1)}
    for c, f in inputs.items():
        if f in FIELDS and n >= 2:
            dv = DataValidation(type="list", formula1=f"=Listy!${listcol[f]}$2:${listcol[f]}${len(FIELDS[f]) + 1}",
                                allow_blank=True, showErrorMessage=True, errorTitle="Wartość spoza listy",
                                error=f"Wybierz wartość z listy ({', '.join(FIELDS[f])}).")
            dv.add(f"{L[c]}2:{L[c]}{n}")
            ws.add_data_validation(dv)
    if cand_col in cols and cand_ids is not None:
        for i, ids in enumerate(cand_ids, 2):
            opts = ["brak"] + [x for x in str(ids).split("|") if x]
            while len(",".join(opts)) > 250:          # limit Excela dla listy wpisanej wprost
                opts.pop()
            dv = DataValidation(type="list", formula1='"' + ",".join(opts) + '"', allow_blank=True,
                                showErrorMessage=True, errorStyle="warning", errorTitle="Spoza listy kandydatów",
                                error="Tego zdarzenia nie ma wśród kandydatów. Zostawić mimo to?")
            dv.add(f"{L[cand_col]}{i}")
            ws.add_data_validation(dv)
    # brakujący kierunek przy cla = tak -> pomarańczowe tło
    cla = "cla" if "cla" in cols else ("FINAL_cla" if "FINAL_cla" in cols else None)
    if cla and n >= 2:
        for c in [x for x in ("kierunek", "FINAL_kierunek") if x in cols]:
            ws.conditional_formatting.add(f"{L[c]}2:{L[c]}{n}", FormulaRule(
                formula=[f'AND(${L[cla]}2="tak",{L[c]}2="")'], fill=PatternFill("solid", fgColor=XL_ORANGE)))
    first_input = min((cols.index(c) for c in inputs if c in cols), default=1)
    ws.freeze_panes = ws.cell(2, 2)
    ws.protection.sheet = True                    # bez hasła; blokuje edycję szarych kolumn i sortowanie
    ws.protection.formatColumns = False
    ws.protection.formatRows = False
    ws.sheet_view.zoomScale = 110
    if instruction:
        ins = wb.create_sheet("Instrukcja", 1)
        ins.column_dimensions["A"].width = 120
        for r, line in enumerate(instruction, 1):
            cell = ins.cell(r, 1, line)
            cell.font = Font(name=XL_FONT, size=11, bold=line.startswith("#"))
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if line.startswith("#"):
                cell.value = line.lstrip("# ")
        if example:
            r0 = len(instruction) + 2
            ins.cell(r0, 1, "Przykładowy wypełniony wiersz (wzór formatu, nie wpisuj go do arkusza):").font = Font(name=XL_FONT, bold=True)
            for k, (c, v) in enumerate(example.items()):
                a = ins.cell(r0 + 1 + k, 1, f"{c}:  {v}")
                a.font = Font(name=XL_FONT, size=10, color="404040")
    wb.active = 0
    wb.save(path)


# =============================================================================== próbka
class AnnotationSampler:
    """Losowanie warstwowe: miesiąc x trafienie słownika, z nadreprezentacją trafień (hit_share).
    Każdy post dostaje wagę = 1 / prawdopodobieństwo wylosowania (do ważonych miar trafności słownika)."""

    def __init__(self, cfg: Config, n: int = 400, pilot: int = 50, overlap: int = 120, annotators=ANNOTATORS,
                 hit_share: float = 0.5, hit_col: str = "relevant", seed: int = 2026):
        self.cfg, self.n, self.pilot, self.overlap = cfg, n, pilot, overlap
        self.annotators, self.hit_share, self.hit_col, self.seed = tuple(annotators), hit_share, hit_col, seed

    def eligible(self, tr: pd.DataFrame) -> pd.DataFrame:
        lo = min(a for a, _ in self.cfg.period_bounds())
        hi = max(b for _, b in self.cfg.period_bounds())
        d = tr[(tr["time_utc"] >= lo) & (tr["time_utc"] < hi) & ~tr["is_retweet"]].copy()
        if "flag_tylko_link" in d:
            d = d[~d["flag_tylko_link"]]
        d = d[d["text"].str.strip().str.len() >= 15]
        d["hit"] = d[self.hit_col].astype(bool) if self.hit_col in d else False
        d["month"] = d["time_utc"].dt.strftime("%Y-%m")
        return d.reset_index(drop=True)

    @staticmethod
    def _alloc(sizes: pd.Series, n: int) -> pd.Series:
        """Proporcjonalnie do sizes, metodą największych reszt, nie więcej niż dostępne."""
        n = int(min(n, sizes.sum()))
        if n <= 0:
            return sizes * 0
        q = sizes / sizes.sum() * n
        a = np.floor(q).astype(int).clip(upper=sizes)
        rest = n - a.sum()
        order = (q - np.floor(q)).sort_values(ascending=False).index
        for k in itertools.cycle(order):
            if rest <= 0:
                break
            if a[k] < sizes[k]:
                a[k] += 1
                rest -= 1
            if (a >= sizes).all():
                break
        return a

    def draw(self, tr: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        E = self.eligible(tr)
        n_hit = int(min(round(self.n * self.hit_share), E["hit"].sum()))
        n_miss = int(min(self.n - n_hit, (~E["hit"]).sum()))
        parts = []
        for flag, k in ((True, n_hit), (False, n_miss)):
            g = E[E["hit"] == flag]
            sizes = g.groupby("month").size()
            alloc = self._alloc(sizes, k)
            for m, take in alloc.items():
                if take <= 0:
                    continue
                pool = g[g["month"] == m]
                idx = rng.choice(pool.index.to_numpy(), size=int(take), replace=False)
                s = E.loc[idx].copy()
                s["weight"] = len(pool) / take                    # odwrotność prawdopodobieństwa wylosowania
                s["stratum"] = f"{m}|{'trafienie' if flag else 'brak_trafienia'}"
                parts.append(s)
        S = pd.concat(parts).sample(frac=1, random_state=self.seed).reset_index(drop=True)
        S["item_id"] = [f"T{i + 1:04d}" for i in range(len(S))]
        return S

    def assign(self, S: pd.DataFrame) -> pd.DataFrame:
        """Pilotaż: wszyscy; nakładka: rotacja par; reszta: po równo, jedna osoba."""
        A = list(self.annotators)
        pairs = list(itertools.combinations(A, 2))
        roles, who = [], []
        for i in range(len(S)):
            if i < self.pilot:
                roles.append("pilotaz"); who.append("|".join(A))
            elif i < self.pilot + self.overlap:
                roles.append("nakladka"); who.append("|".join(pairs[(i - self.pilot) % len(pairs)]))
            else:
                roles.append("pojedyncze"); who.append(A[(i - self.pilot - self.overlap) % len(A)])
        P = S.copy()
        P["etap"], P["oznaczajacy"] = roles, who
        return P


# =============================================================================== arkusze do wypełnienia
INSTRUKCJA = """# Instrukcja wypełniania arkusza (codebook v{ver})

Pełny codebook: dokument „Codebook adnotacji semantycznej”. Tu najważniejsze zasady.

1. Oznaczasz SAMODZIELNIE i na podstawie treści posta. Nie sprawdzasz reakcji rynku ani tego, co stało się później.
2. Najpierw plik `1_pilotaz_<inicjały>.xlsx`. Po pilotażu jest spotkanie i ewentualna poprawka codebooka, dopiero potem `2_glowna_<inicjały>.xlsx`.
3. Wypełniasz tylko ŻÓŁTE kolumny, wybierając wartości z listy rozwijanej (strzałka przy komórce). Szare kolumny są zablokowane.
4. Wartości:

| kolumna | wartości | kiedy |
|---|---|---|
| cla | tak, nie | zawsze |
| kierunek | ESKALACJA, DEESKALACJA, NEUTRALNE | tylko gdy cla = tak |
| emocja | gniew, duma, optymizm, grozba, inne, brak | opcjonalnie |
| uwagi | dowolny tekst | wątpliwości, przypadki brzegowe |

5. Pomarańczowe pole = przy cla = tak brakuje kierunku.
6. Wątpliwość: kierunek NEUTRALNE + krótka uwaga. Nie zgadujemy.
7. Zapisuj zwykłym Ctrl+S jako .xlsx, bez zmiany nazwy pliku. Plik wraca do folderu `adnotacja/`.
"""

INSTR_LINES = [
    "# Jak wypełniać arkusz (codebook v{ver})",
    "Wypełniasz tylko ŻÓŁTE kolumny: cla, kierunek, emocja, uwagi. Szare kolumny są zablokowane.",
    "Wartości wybierasz z listy rozwijanej (strzałka przy zaznaczonej komórce). Wpisanie czegoś spoza listy Excel odrzuci.",
    "cla: tak / nie (zawsze). Przy cla = nie zostaw kierunek pusty.",
    "kierunek (gdy cla = tak): ESKALACJA, DEESKALACJA, NEUTRALNE. Liczy się kierunek zmiany, nie ton.",
    "emocja (opcjonalnie): gniew, duma, optymizm, grozba, inne, brak.",
    "Pomarańczowe pole = przy cla = tak brakuje kierunku.",
    "Wątpliwość: NEUTRALNE + krótka uwaga. Oznaczasz samodzielnie, bez sprawdzania reakcji rynku.",
    "Zapisuj zwykłym Ctrl+S jako .xlsx, bez zmiany nazwy pliku. Nie sortuj i nie usuwaj wierszy.",
]


def write_sheets(plan: pd.DataFrame, out_dir, annotators=ANNOTATORS, overwrite: bool = False) -> list[str]:
    """Arkusze Excela (.xlsx) z listami rozwijanymi: 1_pilotaz_<INI>.xlsx i 2_glowna_<INI>.xlsx.
    Arkusz z wpisami nie jest nadpisywany. Stare, PUSTE arkusze CSV z wcześniejszej wersji są usuwane."""
    out = Path(out_dir)
    msgs = []
    cols = ["item_id", "etap", "data_utc", "tekst"] + LABEL_COLS
    inputs = {"cla": "cla", "kierunek": "kierunek", "emocja": "emocja", "uwagi": None}
    instr = [x.format(ver=CODEBOOK_VERSION) for x in INSTR_LINES]
    P = plan.copy()
    P["data_utc"] = pd.to_datetime(P["time_utc"], utc=True).dt.strftime("%Y-%m-%d %H:%M")
    P["tekst"] = P["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    for c in LABEL_COLS:
        P[c] = ""
    for a in annotators:
        mine = P[P["oznaczajacy"].str.split("|").map(lambda s: a in s)]
        for stage_file, mask in (("1_pilotaz", mine["etap"] == "pilotaz"), ("2_glowna", mine["etap"] != "pilotaz")):
            part = mine[mask].sort_values("item_id")
            sheet = part[cols]
            w = (lambda df, path: _xlsx_write(df, path, inputs, None, None, "Adnotacja", instr, EXAMPLE))
            msgs.append(_write(sheet, out / f"{stage_file}_{a}.xlsx", overwrite, w))
            old = out / f"{stage_file}_{a}.csv"
            if old.exists() and not _has_labels(old):
                old.unlink()
                msgs.append(f"usunięto pusty arkusz CSV z poprzedniej wersji: {old}")
    key_cols = ["item_id", "post_id", "time_utc", "etap", "oznaczajacy", "stratum", "hit", "weight", "text"]
    msgs.append(_write(plan[[c for c in key_cols if c in plan]], out / "klucz_probki.csv", True))
    (out / "INSTRUKCJA.md").write_text(INSTRUKCJA.format(ver=CODEBOOK_VERSION), encoding="utf-8")
    old_vals = out / "wartosci_dozwolone.csv"
    if old_vals.exists():
        old_vals.unlink()
    msgs.append(f"zapisano: {out / 'INSTRUKCJA.md'}")
    return msgs


def load_plan(folder) -> pd.DataFrame:
    """Plan próbki zapisany przy generowaniu arkuszy (klucz_probki.csv). Używaj go zamiast losować ponownie:
    po rozesłaniu arkuszy próbka nie może się zmienić."""
    P = read_csv_any(Path(folder) / "klucz_probki.csv")
    P["time_utc"] = pd.to_datetime(P["time_utc"], utc=True, format="ISO8601")
    P["hit"] = P["hit"].str.lower().isin(["true", "1"])
    P["weight"] = pd.to_numeric(P["weight"], errors="coerce")
    return P


# =============================================================================== wczytanie i kontrola wypełnienia
def load_sheets(folder, plan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zwraca (etykiety w formacie długim: item_id x oznaczający, raport problemów). Puste arkusze są pomijane."""
    folder = Path(folder)
    rows, probs = [], []
    known = set(plan["item_id"])
    files = sorted(folder.glob("[12]_*_*.xlsx")) + sorted(folder.glob("[12]_*_*.csv"))
    seen = set()
    for p in files:
        m = re.match(r"([12])_(pilotaz|glowna)_(.+)\.(xlsx|csv)$", p.name)
        if not m:
            continue
        if p.stem in seen:                         # ten sam arkusz jako .xlsx i .csv: liczy się .xlsx
            if _has_labels(p):
                probs.append({"plik": p.name, "item_id": "", "pole": "plik", "problem": "pominięto: istnieje wersja .xlsx", "wartosc": ""})
            continue
        seen.add(p.stem)
        who = m.group(3)
        d = read_csv_any(p)
        for c in LABEL_COLS:
            if c not in d:
                d[c] = ""
        filled = d[["cla", "kierunek"]].apply(lambda s: s.str.strip() != "").any(axis=1)
        if not filled.any():
            continue
        for r in d.itertuples(index=False):
            r = r._asdict()
            iid = r["item_id"].strip()
            if iid not in known:
                probs.append({"plik": p.name, "item_id": iid, "pole": "item_id", "problem": "nieznany item_id", "wartosc": iid})
                continue
            rec = {"item_id": iid, "oznaczajacy": who, "plik": p.name, "uwagi": r.get("uwagi", "")}
            for f in FIELDS:
                v, ok = normalize(f, r.get(f, ""))
                if not ok:
                    probs.append({"plik": p.name, "item_id": iid, "pole": f, "problem": "wartość spoza listy", "wartosc": v})
                    v = ""
                rec[f] = v
            # reguły spójności z codebooka
            if rec["cla"] == "":
                if str(r.get("kierunek", "")).strip():
                    probs.append({"plik": p.name, "item_id": iid, "pole": "cla", "problem": "brak cla przy wypełnionych innych polach", "wartosc": ""})
                else:
                    probs.append({"plik": p.name, "item_id": iid, "pole": "cla", "problem": "wiersz niewypełniony", "wartosc": ""})
                continue
            if rec["cla"] == "nie":
                if rec["kierunek"] not in ("", ND):
                    probs.append({"plik": p.name, "item_id": iid, "pole": "kierunek", "problem": "kierunek przy cla=nie (zignorowano)", "wartosc": rec["kierunek"]})
                rec["kierunek"] = ND
            else:
                if rec["kierunek"] == "":
                    probs.append({"plik": p.name, "item_id": iid, "pole": "kierunek", "problem": "brak wartości przy cla=tak", "wartosc": ""})
            rows.append(rec)
    L = pd.DataFrame(rows, columns=["item_id", "oznaczajacy", "plik"] + list(FIELDS) + ["uwagi"])
    if len(L):
        dup = L.duplicated(["item_id", "oznaczajacy"], keep="last")
        for r in L[dup].itertuples():
            probs.append({"plik": r.plik, "item_id": r.item_id, "pole": "item_id", "problem": "ten sam post dwa razy u jednej osoby (wzięto ostatni)", "wartosc": ""})
        L = L[~dup].reset_index(drop=True)
    return L, pd.DataFrame(probs, columns=["plik", "item_id", "pole", "problem", "wartosc"])


def progress(L: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """Ile postów każda osoba ma przydzielone i ile oznaczyła, osobno dla pilotażu i reszty."""
    rows = []
    for a in sorted({x for s in plan["oznaczajacy"] for x in s.split("|")}):
        mine = plan[plan["oznaczajacy"].str.split("|").map(lambda s: a in s)]
        done = set(L.loc[L["oznaczajacy"] == a, "item_id"]) if len(L) else set()
        for st, m in (("pilotaż", mine["etap"] == "pilotaz"), ("główna", mine["etap"] != "pilotaz")):
            ids = set(mine.loc[m, "item_id"])
            rows.append({"oznaczający": a, "część": st, "przydzielone": len(ids), "oznaczone": len(ids & done),
                         "gotowe_%": round(100 * len(ids & done) / max(len(ids), 1), 1)})
    return pd.DataFrame(rows)


# =============================================================================== zgodność
def cohen_kappa(a, b) -> float:
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    m = (a != "") & (b != "")
    a, b = a[m], b[m]
    if len(a) == 0:
        return np.nan
    cats = sorted(set(a) | set(b))
    po = float((a == b).mean())
    pe = float(sum((a == c).mean() * (b == c).mean() for c in cats))
    if pe >= 1:
        return 1.0 if po == 1 else np.nan
    return (po - pe) / (1 - pe)


def krippendorff_alpha(units: pd.DataFrame) -> float:
    """Alfa nominalna. units: wiersz = post, kolumny = oznaczający, '' lub NaN = brak oceny."""
    vals = [[v for v in row if isinstance(v, str) and v != ""] for row in units.to_numpy()]
    vals = [v for v in vals if len(v) >= 2]
    if not vals:
        return np.nan
    cats = sorted({x for v in vals for x in v})
    ix = {c: i for i, c in enumerate(cats)}
    O = np.zeros((len(cats), len(cats)))
    for v in vals:
        m = len(v)
        cnt = np.bincount([ix[x] for x in v], minlength=len(cats)).astype(float)
        O += (np.outer(cnt, cnt) - np.diag(cnt)) / (m - 1)
    n_c = O.sum(1)
    n = n_c.sum()
    if n <= 1:
        return np.nan
    Do = (O.sum() - np.trace(O)) / n
    De = (n * n - (n_c ** 2).sum()) / (n * (n - 1))
    return 1.0 if De == 0 else 1 - Do / De


def _interpret(k: float) -> str:
    if not np.isfinite(k):
        return "brak danych"
    return ("słaba" if k < 0.4 else "umiarkowana" if k < 0.6 else "znaczna (poniżej celu)" if k < 0.7
            else "znaczna (cel osiągnięty)" if k < 0.8 else "prawie pełna")


def agreement(L: pd.DataFrame, plan: pd.DataFrame, stage: str = "nakladka", target: float = 0.70) -> pd.DataFrame:
    """Kappa Cohena dla każdej pary i alfa Krippendorffa, osobno dla cla i kierunku.
    'kierunek | cla=tak' liczony tylko na postach, które OBIE osoby uznały za cła."""
    ids = set(plan.loc[plan["etap"] == stage, "item_id"])
    D = L[L["item_id"].isin(ids)]
    if D.empty:
        return pd.DataFrame()
    rows = []
    specs = [("cla", "cla", False), ("kierunek | cla=tak", "kierunek", True), ("kierunek (wszystkie)", "kierunek", False)]
    for name, f, cond in specs:
        W = D.pivot_table(index="item_id", columns="oznaczajacy", values=f, aggfunc="first").fillna("")
        C = D.pivot_table(index="item_id", columns="oznaczajacy", values="cla", aggfunc="first").fillna("")
        ks, ns, ags = [], [], []
        for a, b in itertools.combinations(W.columns, 2):
            m = (W[a] != "") & (W[b] != "")
            if cond:
                m &= (C[a] == "tak") & (C[b] == "tak")
            if m.sum() >= 5:
                ks.append(cohen_kappa(W.loc[m, a], W.loc[m, b])); ns.append(int(m.sum()))
                ags.append(float((W.loc[m, a] == W.loc[m, b]).mean()))
                rows.append({"etap": stage, "etykieta": name, "para": f"{a}-{b}", "n": int(m.sum()),
                             "zgodność_%": round(100 * float((W.loc[m, a] == W.loc[m, b]).mean()), 1), "kappa": round(ks[-1], 3)})
        WW = W.where(~((C != "tak") & cond), "") if cond else W
        alpha = krippendorff_alpha(WW)
        kw = float(np.average(ks, weights=ns)) if ks else np.nan
        rows.append({"etap": stage, "etykieta": name, "para": "RAZEM", "n": int(sum(ns)),
                     "zgodność_%": round(100 * float(np.average(ags, weights=ns)), 1) if ags else np.nan,
                     "kappa": round(kw, 3) if np.isfinite(kw) else np.nan, "alfa_Krippendorffa": round(alpha, 3) if np.isfinite(alpha) else np.nan,
                     "ocena": _interpret(alpha if np.isfinite(alpha) else kw),
                     "cel_≥_0,70": bool((alpha if np.isfinite(alpha) else kw) >= target)})
    return pd.DataFrame(rows)


def confusion_between(L: pd.DataFrame, plan: pd.DataFrame, field: str = "kierunek", stage: str = "nakladka") -> pd.DataFrame:
    """Gdzie pary się mylą: tabela wartość osoby A x wartość osoby B (sumowana po parach)."""
    ids = set(plan.loc[plan["etap"] == stage, "item_id"])
    W = L[L["item_id"].isin(ids)].pivot_table(index="item_id", columns="oznaczajacy", values=field, aggfunc="first").fillna("")
    pairs = []
    for a, b in itertools.combinations(W.columns, 2):
        m = (W[a] != "") & (W[b] != "")
        pairs.append(pd.DataFrame({"A": W.loc[m, a].to_numpy(), "B": W.loc[m, b].to_numpy()}))
    if not pairs:
        return pd.DataFrame()
    P = pd.concat(pairs, ignore_index=True)
    return pd.crosstab(P["A"], P["B"])


# =============================================================================== rozstrzyganie i gold standard
GOLD_FIELDS = ["cla", "kierunek", "emocja"]


def consensus(L: pd.DataFrame) -> pd.DataFrame:
    """Per post i pole: wartość wspólna albo None. Status: pojedyncze / zgodne / wiekszosc / sporne."""
    out = []
    for iid, g in L.groupby("item_id"):
        rec = {"item_id": iid, "n_oznaczen": len(g)}
        for f in GOLD_FIELDS:
            v = [x for x in g[f] if x != ""]
            if not v:
                rec[f], rec[f + "_status"] = None, "brak"
                continue
            cnt = pd.Series(v).value_counts()
            if len(v) == 1:
                rec[f], st = v[0], "pojedyncze"
            elif len(cnt) == 1:
                rec[f], st = cnt.index[0], "zgodne"
            elif cnt.iloc[0] > len(v) / 2 and len(v) >= 3:
                rec[f], st = cnt.index[0], "wiekszosc"
            else:
                rec[f], st = None, "sporne"
            rec[f + "_status"] = st
            rec[f + "_oceny"] = "; ".join(f"{a}={x}" for a, x in zip(g["oznaczajacy"], g[f]) if x != "")
        out.append(rec)
    return pd.DataFrame(out)


def write_adjudication(L: pd.DataFrame, plan: pd.DataFrame, path, overwrite: bool = False) -> str:
    """Arkusz rozstrzygania: tylko posty ze sporem w cla/kierunku. FINAL_* uzupełnione tam, gdzie
    jest zgoda; puste do wpisania na spotkaniu. Kolumna uzasadnienie obowiązkowa przy każdym sporze."""
    C = consensus(L)
    key = ["cla", "kierunek"]
    disp = C[C[[f + "_status" for f in key]].eq("sporne").any(axis=1)]
    P = plan.set_index("item_id")
    rows = []
    for r in disp.itertuples():
        d = r._asdict()
        rec = {"item_id": d["item_id"], "etap": P.loc[d["item_id"], "etap"], "tekst": P.loc[d["item_id"], "text"]}
        for f in key:
            rec[f + "_oceny"] = d.get(f + "_oceny", "")
            v = d.get(f)
            rec["FINAL_" + f] = "" if v is None or (isinstance(v, float) and np.isnan(v)) else v
        rec["uzasadnienie"] = ""
        rows.append(rec)
    A = pd.DataFrame(rows)
    if A.empty:
        A = pd.DataFrame(columns=["item_id", "etap", "tekst"] +
                         [x for f in key for x in (f + "_oceny", "FINAL_" + f)] + ["uzasadnienie"])
    path = Path(path)
    kept = 0
    if path.exists() and not overwrite:             # scalenie z tym, co już wpisano (np. po pilotażu)
        try:
            old = read_csv_any(path).set_index("item_id")
        except Exception:
            old = pd.DataFrame()
        if len(old) and len(A):
            A = A.set_index("item_id")
            for iid in old.index:
                if iid not in A.index:              # spór już niewidoczny (np. ktoś poprawił arkusz): zostaje w pliku
                    A.loc[iid] = old.loc[iid].reindex(A.columns).fillna("")
                    continue
                for c in [c for c in A.columns if c.startswith("FINAL_")] + ["uzasadnienie"]:
                    v = str(old.loc[iid].get(c, "")).strip()
                    if v:
                        A.loc[iid, c] = v
                        kept += 1
            A = A.reset_index().sort_values("item_id")
        overwrite = True
    note = f" | spornych postów: {len(A)}" + (f" | zachowano wpisów z poprzedniej wersji: {kept}" if kept else "")
    if path.suffix.lower() != ".xlsx":
        return _write(A, path, overwrite) + note
    A = A.rename(columns={"tekst": "tekst"})
    inputs = {"FINAL_cla": "cla", "FINAL_kierunek": "kierunek", "uzasadnienie": None}
    instr = ["# Rozstrzyganie sporów (spotkanie zespołu)",
             "Wiersz = post, przy którym oznaczający się nie zgodzili. Kolumny *_oceny pokazują, kto co wybrał.",
             "Uzupełnij puste żółte pola FINAL_* (listy rozwijane) i ZAWSZE wpisz uzasadnienie decyzji.",
             "FINAL_* już wypełnione = pola, w których była zgoda lub większość; można je zmienić.",
             "Przy FINAL_cla = nie zostaw FINAL_kierunek pusty.",
             "Zapisz Ctrl+S jako .xlsx bez zmiany nazwy; potem uruchom ponownie etap D notebooka."]
    w = lambda df, p: _xlsx_write(df, p, inputs, None, None, "Rozstrzyganie", instr, None)
    try:
        return _write(A, path, overwrite, w) + note
    except PermissionError:
        return f"NIE ZAPISANO {path}: plik jest otwarty w Excelu. Zamknij go i uruchom komórkę ponownie."


def build_gold(L: pd.DataFrame, plan: pd.DataFrame, adjudication_path=None, dev_share: float = 0.4,
               seed: int = 2026, include_pilot: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(gold, nierozstrzygnięte). Wartości ze zgody/większości + FINAL_* z arkusza rozstrzygania.
    Podział dev/test warstwowy po (cla, kierunek), żeby obie części miały wszystkie klasy.
    include_pilot=False: pomija posty z pilotażu (oznaczane wg wcześniejszej wersji codebooka)."""
    if not include_pilot:
        L = L[~L["item_id"].isin(plan.loc[plan["etap"] == "pilotaz", "item_id"])]
    C = consensus(L).set_index("item_id")
    if adjudication_path is not None and Path(adjudication_path).exists():
        A = read_csv_any(adjudication_path).set_index("item_id")
        for f in ["cla", "kierunek"]:
            col = "FINAL_" + f
            if col not in A:
                continue
            for iid, v in A[col].items():
                v = str(v).strip()
                if v == "" or iid not in C.index:
                    continue
                if f in FIELDS:
                    v, ok = normalize(f, v)
                    if not ok:
                        continue
                if C.loc[iid, f + "_status"] in ("sporne", "brak") or C.loc[iid, f] != v:
                    C.loc[iid, f + "_status"] = "rozstrzygniete"
                C.loc[iid, f] = v
        if "uzasadnienie" in A:
            C["uzasadnienie"] = A["uzasadnienie"].reindex(C.index).fillna("")
    C.loc[C["cla"] == "nie", "kierunek"] = ND
    need = C["cla"].isna() | ((C["cla"] == "tak") & C["kierunek"].isna())
    open_ = C[need].reset_index()
    G = C[~need].reset_index()
    keep = [c for c in ["post_id", "time_utc", "text", "etap", "stratum", "hit", "weight"] if c in plan]
    G = G.merge(plan[["item_id"] + keep], on="item_id", how="left")
    G["zrodlo_etykiety"] = G[[f + "_status" for f in MAIN]].apply(
        lambda r: "rozstrzygniete" if "rozstrzygniete" in set(r) else ("wiekszosc" if "wiekszosc" in set(r) else
                  ("pojedyncze" if "pojedyncze" in set(r) else "zgodne")), axis=1)
    rng = np.random.default_rng(seed)
    G["split"] = "test"
    for _, g in G.groupby(G["cla"] + "|" + G["kierunek"]):
        idx = rng.permutation(g.index.to_numpy())
        G.loc[idx[: int(round(dev_share * len(idx)))], "split"] = "dev"
    G["codebook"] = CODEBOOK_VERSION
    return G, open_


# =============================================================================== walidacja instrumentu
def dictionary_validity(gold: pd.DataFrame, tr: pd.DataFrame, cols=("relevant", "relevant_strict")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trafność słownika ceł względem gold (cla). Miary ważone wagą próbkowania = szacunek dla CAŁEGO korpusu
    (próbka nadreprezentuje trafienia słownika). Zwraca (miary, lista pomyłek do przejrzenia)."""
    T = tr[["post_id"] + [c for c in cols if c in tr]].copy()
    G = gold.merge(T, on="post_id", how="left")
    y = G["cla"].eq("tak").to_numpy()
    w = pd.to_numeric(G.get("weight", 1), errors="coerce").fillna(1).to_numpy()
    rows, errs = [], []
    for c in cols:
        if c not in G:
            continue
        p = G[c].fillna(False).astype(bool).to_numpy()
        for lab, ww in (("ważone (cały korpus)", w), ("nieważone (próbka)", np.ones_like(w))):
            tp, fp = (ww * (p & y)).sum(), (ww * (p & ~y)).sum()
            fn, tn = (ww * (~p & y)).sum(), (ww * (~p & ~y)).sum()
            prec = tp / (tp + fp) if tp + fp else np.nan
            rec = tp / (tp + fn) if tp + fn else np.nan
            rows.append({"słownik": c, "miara": lab, "precyzja": prec, "czułość": rec,
                         "F1": 2 * prec * rec / (prec + rec) if prec and rec else np.nan,
                         "trafność": (tp + tn) / (tp + fp + fn + tn), "n": int(len(G))})
        errs.append(G.loc[p & ~y].assign(błąd=f"{c}: fałszywy alarm (słownik tak, gold nie)")[["item_id", "błąd", "text"]])
        errs.append(G.loc[~p & y].assign(błąd=f"{c}: pominięcie (słownik nie, gold tak)")[["item_id", "błąd", "text"]])
    return pd.DataFrame(rows).round(3), pd.concat(errs, ignore_index=True) if errs else pd.DataFrame()


def label_distribution(gold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in MAIN:
        vc = gold[f].value_counts()
        for v, n in vc.items():
            rows.append({"etykieta": f, "wartość": v, "n": int(n), "udział_%": round(100 * n / len(gold), 1),
                         "dev": int(((gold[f] == v) & (gold["split"] == "dev")).sum()),
                         "test": int(((gold[f] == v) & (gold["split"] == "test")).sum())})
    return pd.DataFrame(rows)


# =============================================================================== ocena klasyfikatora
def evaluate(gold: pd.DataFrame, pred: pd.DataFrame, split: str = "test", fields=MAIN) -> dict:
    """pred: item_id + kolumny etykiet. Kierunek oceniany na postach, które w gold są o cłach.
    Zwraca dict: pole -> {accuracy, macro_f1, n, per_class (DataFrame), confusion (DataFrame)}."""
    G = gold[gold["split"] == split] if split else gold
    M = G.merge(pred[["item_id"] + [f for f in fields if f in pred]], on="item_id", suffixes=("", "_pred"))
    out = {}
    for f in fields:
        if f + "_pred" not in M:
            continue
        d = M[M["cla"] == "tak"] if f == "kierunek" else M
        y, p = d[f].astype(str), d[f + "_pred"].astype(str)
        labs = [l for l in FIELDS[f] if l in set(y) | set(p)]
        per = []
        for l in labs:
            tp = int(((y == l) & (p == l)).sum()); fp = int(((y != l) & (p == l)).sum()); fn = int(((y == l) & (p != l)).sum())
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            per.append({"klasa": l, "precyzja": pr, "czułość": rc, "F1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
                        "wsparcie": int((y == l).sum())})
        per = pd.DataFrame(per)
        out[f] = {"n": len(d), "accuracy": float((y == p).mean()) if len(d) else np.nan,
                  "macro_f1": float(per["F1"].mean()) if len(per) else np.nan, "per_class": per.round(3),
                  "confusion": pd.crosstab(y.rename("gold"), p.rename("predykcja"))}
    return out


def evaluation_table(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{"etykieta": f, "n": r["n"], "accuracy": round(r["accuracy"], 3), "macro_F1": round(r["macro_f1"], 3)}
                         for f, r in res.items()])


class KeywordBaseline:
    """Punkt odniesienia dla klasyfikatora: reguły słownikowe z codebooka (wskaźniki w tekście).
    Klasyfikator (LLM / zero-shot) jest wart użycia tylko, jeśli na części testowej wyraźnie go pokonuje."""
    ESC = r"\btariffs? (?:will|on|of)\b|\bimpos|\braise|\bincreas|retaliat|punish|\bwill pay\b|\bthreat|\bsanction|\bban\b|\bboycott"
    DEESC = r"\bdeal\b|agreement|\btalks?\b|negotiat|\bdelay|postpon|\bsuspend|\bremov|\blift|\bprogress|\bmeeting|\bmet with\b|truce"
    TRADE = r"tariff|\btrade\b|\bduties\b|\bimport|\bexport|deficit|sanction|\bnafta\b|\busmca\b|\bwto\b"

    def predict(self, items: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        t = items[text_col].fillna("").str.lower()
        cla = np.where(t.str.contains(self.TRADE), "tak", "nie")
        e, d = t.str.count(self.ESC), t.str.count(self.DEESC)
        kier = np.select([e > d, d > e], ["ESKALACJA", "DEESKALACJA"], "NEUTRALNE")
        out = pd.DataFrame({"item_id": items["item_id"].to_numpy(), "cla": cla, "kierunek": kier})
        out.loc[out["cla"] == "nie", "kierunek"] = ND
        return out


def majority_baseline(gold: pd.DataFrame) -> pd.DataFrame:
    dev = gold[gold["split"] == "dev"]
    mode = {f: dev.loc[dev["cla"] == "tak", f].mode().iat[0] if f != "cla" else dev["cla"].mode().iat[0] for f in MAIN}
    out = pd.DataFrame({"item_id": gold["item_id"]})
    for f, v in mode.items():
        out[f] = v
    return out


# =============================================================================== prompt dla klasyfikatora LLM
PROMPT = """Jesteś anotatorem. Oznacz post Donalda Trumpa zgodnie z codebookiem (v{ver}).
Etykiety:
- cla: tak (główny przekaz dotyczy ceł, sankcji, handlu, umów lub negocjacji handlowych) / nie.
- kierunek (tylko gdy cla=tak): ESKALACJA (nowe cło, groźba, odwet, podwyżka), DEESKALACJA (porozumienie, odroczenie,
  obniżka, postęp rozmów), NEUTRALNE (opis, ocena przeszłości, brak sygnału zmiany). Liczy się kierunek zmiany, nie ton.
Reguły: groźba warunkowa = ESKALACJA; warunkowa obietnica = DEESKALACJA;
przy kilku wątkach liczy się główny przekaz; w razie wątpliwości NEUTRALNE.
{examples}Post: \"\"\"{text}\"\"\"
Odpowiedz wyłącznie JSON: {{"cla": "...", "kierunek": "..."}}"""


def build_prompts(items: pd.DataFrame, gold_dev: pd.DataFrame | None = None, k_per_class: int = 1, seed: int = 0) -> pd.DataFrame:
    ex = ""
    if gold_dev is not None and len(gold_dev):
        picks = pd.concat([g.sample(min(k_per_class, len(g)), random_state=seed) for _, g in gold_dev.groupby("kierunek")])
        ex = "Przykłady:\n" + "".join(
            f'Post: """{r.text}"""\n{{"cla": "{r.cla}", "kierunek": "{r.kierunek}"}}\n' for r in picks.itertuples()) + "\n"
    return pd.DataFrame({"item_id": items["item_id"].to_numpy(),
                         "prompt": [PROMPT.format(ver=CODEBOOK_VERSION, examples=ex, text=t) for t in items["text"]]})


def parse_llm(answer: str) -> dict:
    """Odpowiedź modelu -> etykiety kanoniczne; niepoprawne pola = ''."""
    m = re.search(r"\{.*\}", str(answer), re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    out = {f: normalize(f, d.get(f, ""))[0] if normalize(f, d.get(f, ""))[1] else "" for f in MAIN}
    if out["cla"] == "nie":
        out["kierunek"] = ND
    return out


# =============================================================================== połączenie z danymi scalonymi
def link_to_merged(labels: pd.DataFrame, merged_dir, label_cols=("cla", "kierunek")) -> dict:
    """labels: post_id + etykiety (gold albo predykcje klasyfikatora dla całego korpusu).
    Łączy przez event_id = 'tw_<post_id>' (zdarzenia-posty) i przypisuje posty do epizodów (merge_asof po czasie).
    Zwraca dict: posts (zdarzenia-posty z etykietami), episodes (etykiety zagregowane), market (market_long z etykietami),
    coverage (ile etykiet trafiło do zdarzeń; posty spoza słownika nie są zdarzeniami w merged)."""
    md = Path(merged_dir)
    E = pd.read_csv(md / "events.csv", dtype={"ref_id": str}, parse_dates=["t", "eff_day"])
    lab = labels[["post_id"] + list(label_cols)].copy()
    lab["post_id"] = lab["post_id"].astype(str)
    lab["event_id"] = "tw_" + lab["post_id"]
    P = E[E["kind"] == "trump_post"].merge(lab.drop(columns="post_id"), on="event_id", how="inner")
    EP = E[E["kind"] == "trump_episode"][["event_id", "t"]].rename(columns={"event_id": "episode_id", "t": "t_ep"}).sort_values("t_ep")
    P = pd.merge_asof(P.sort_values("t"), EP, left_on="t", right_on="t_ep", direction="backward")
    agg = {"n_oznaczonych": ("event_id", "size")}
    if "cla" in label_cols:
        agg["udzial_cla_tak"] = ("cla", lambda s: float((s == "tak").mean()))
    if "kierunek" in label_cols:
        agg["udzial_eskalacja"] = ("kierunek", lambda s: float((s == "ESKALACJA").mean()))
        agg["udzial_deeskalacja"] = ("kierunek", lambda s: float((s == "DEESKALACJA").mean()))
        agg["kierunek_pierwszego"] = ("kierunek", "first")
    EPL = P.sort_values("t").groupby("episode_id").agg(**agg).reset_index().rename(columns={"episode_id": "event_id"})
    out = {"posts": P, "episodes": EPL,
           "coverage": {"etykiet": len(lab), "w zdarzeniach-postach": len(P),
                        "poza zdarzeniami (słownik ich nie złapał)": len(lab) - len(P),
                        "epizodów z etykietą": len(EPL)}}
    mp = md / "market_long.csv"
    if mp.exists():
        M = pd.read_csv(mp)
        out["market"] = pd.concat([
            M.merge(P[["event_id"] + list(label_cols)], on="event_id", how="inner").assign(poziom="post"),
            M.merge(EPL, on="event_id", how="inner").assign(poziom="epizod")], ignore_index=True)
    return out


def market_by_label(market: pd.DataFrame, label: str = "kierunek", instrument: str = "S&P 500", level: str = "post",
                    metric: str = "excess_abs") -> pd.DataFrame:
    """Opisowo: średnia nadwyżka w pb per wartość etykiety i horyzont (na gold to mała próbka, bez wniosków)."""
    d = market[(market["poziom"] == level) & (market["instrument"] == instrument) & market[metric].notna()]
    if label not in d:
        return pd.DataFrame()
    d = d[~d[label].isin(["", ND])]
    t = d.groupby([label, "horizon"])[metric].agg(["mean", "count"])
    t["mean"] = 1e4 * t["mean"]
    t = t.rename(columns={"mean": "śr_nadwyżka_pb", "count": "n"}).unstack("horizon")
    order = [h for h in ("1m", "5m", "1h", "3h", "1d", "3d", "7d") if h in t.columns.get_level_values(1)]
    return t.reindex(columns=pd.MultiIndex.from_product([["śr_nadwyżka_pb", "n"], order]))


def add_label_counts(daily: pd.DataFrame, labeled_posts: pd.DataFrame, label_cols=("kierunek",)) -> pd.DataFrame:
    """Dzienne liczniki postów per wartość etykiety (klucz eff_day = daily.day), np. tw_kierunek_ESKALACJA_n.
    Kolumny kończą się na _n, więc to_trading_days(daily) zsumuje je poprawnie do dni handlowych."""
    D = daily.copy()
    t = pd.to_datetime(labeled_posts["time_utc"], utc=True)
    day = pd.Series(eff_day(t), index=labeled_posts.index)
    for c in label_cols:
        for v in sorted(labeled_posts[c].dropna().unique()):
            if v in ("", ND):
                continue
            cnt = day[labeled_posts[c] == v].value_counts()
            D[f"tw_{c}_{re.sub(r'[^0-9A-Za-z]+', '_', v)}_n"] = cnt.reindex(D.index).fillna(0.0).to_numpy()
    return D


# =============================================================================== próba generalna (TYLKO TEST)
def simulate_filled(folder, out_folder, plan: pd.DataFrame, seed: int = 0, noise: float = 0.15) -> list[str]:
    """TYLKO DO PRÓBY GENERALNEJ procesu: wypełnia kopie arkuszy regułami KeywordBaseline + losowy szum per osoba.
    Wyniki z symulacji NIE są danymi do pracy."""
    from openpyxl import load_workbook
    rng = np.random.default_rng(seed)
    out = Path(out_folder)
    out.mkdir(parents=True, exist_ok=True)
    truth = KeywordBaseline().predict(plan).set_index("item_id")
    msgs = []

    def labels(iid):
        t = truth.loc[iid]
        vals = {f: t[f] for f in MAIN}
        for f in MAIN:
            if rng.random() < noise:
                vals[f] = rng.choice(FIELDS[f])
        if vals["cla"] == "nie":
            vals["kierunek"] = ""
        elif vals["kierunek"] == ND:
            vals["kierunek"] = "NEUTRALNE"
        return vals

    for p in sorted(Path(folder).glob("[12]_*_*.xlsx")):
        wb = load_workbook(p)
        ws = wb["Adnotacja"]
        hdr = {c.value: c.column for c in ws[1]}
        for r in range(2, ws.max_row + 1):
            iid = ws.cell(r, hdr["item_id"]).value
            if not iid:
                continue
            for f, v in labels(iid).items():
                ws.cell(r, hdr[f]).value = v
        wb.save(out / p.name)
        msgs.append(f"symulacja: {out / p.name}")
    for p in sorted(Path(folder).glob("[12]_*_*.csv")):
        d = read_csv_any(p)
        for i, r in d.iterrows():
            for f, v in labels(r["item_id"]).items():
                d.at[i, f] = v
        d.to_csv(out / p.name, sep=SEP, index=False, encoding=ENC)
        msgs.append(f"symulacja: {out / p.name}")
    return msgs