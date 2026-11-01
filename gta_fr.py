"""GTA (interwencje) i Federal Register (akty prawne): wczytanie z cache'u CSV, eksploracja, czyszczenie.
Klucz API GTA nie jest tu potrzebny: pracujemy na plikach zapisanych przez notebook pobierający dane.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .base import CleaningLog, Config, TableProfiler, show


class GTAExplorer:
    DATE_COLS = ["date_announced", "date_published", "date_implemented", "date_removed"]

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.out_dir = cfg.out("gta")
        self.df = pd.DataFrame()

    def load(self) -> pd.DataFrame:
        raw = pd.read_csv(self.cfg.gta_csv)
        for c in self.DATE_COLS:
            if c in raw:
                raw[c] = pd.to_datetime(raw[c], errors="coerce")
        self.df = raw
        return raw

    def explore(self):
        d = self.df if len(self.df) else self.load()
        show(d.head(5), "GTA: head")
        show(TableProfiler.column_overview(d), "GTA: przegląd kolumn")
        for c in self.DATE_COLS:
            if c in d:
                show(TableProfiler.datetime_summary(d[c]), f"GTA: {c}")

        a, i = d.get("date_announced"), d.get("date_implemented")
        checks = {
            "wiersze": len(d),
            "duplikaty (intervention_id, actor)": int(d.duplicated(["intervention_id", "actor"]).sum()),
            "brak date_announced": int(a.isna().sum()),
            "brak date_implemented": int(i.isna().sum()),
            "wdrożone PRZED ogłoszeniem": int((i < a).sum()),
            "wdrożone > 2 lata po ogłoszeniu": int(((i - a).dt.days > 730).sum()),
            "usunięte PRZED wdrożeniem": int((d["date_removed"] < i).sum()) if "date_removed" in d else None,
            "n_products == 0": int((d["n_products"] == 0).sum()) if "n_products" in d else None,
        }
        show(checks, "GTA: kontrole logiczne")
        show(d.assign(rok_ogłoszenia=a.dt.year).groupby(["actor", "rok_ogłoszenia"]).size().unstack(0).fillna(0).astype(int),
             "GTA: interwencje per rok ogłoszenia i aktor")
        for col in ("type", "evaluation", "is_in_force"):
            if col in d:
                show(d[col].value_counts(dropna=False).head(10), f"GTA: {col}")
        ev = d.dropna(subset=["date_announced", "date_implemented"]).groupby(
            ["actor", "date_announced", "date_implemented"]).size()
        show(ev.describe().rename("interwencji na zdarzenie"), "GTA: ile interwencji w jednym zdarzeniu (grupowanie)")
        show(ev.sort_values(ascending=False).head(8).rename("interwencji"), "GTA: największe zdarzenia")

    def events(self, min_products: int = 20, drop_calendar: bool = True) -> pd.DataFrame:
        """Zdarzenia z wagą (liczba produktów), bez rutynowych zmian 1 stycznia; znak wg ocen GTA."""
        d = (self.df if len(self.df) else self.load()).dropna(subset=["date_announced"]).copy()
        if drop_calendar:
            d = d[~((d["date_announced"].dt.month == 1) & (d["date_announced"].dt.day == 1))]
        ev = d["evaluation"].astype(str).str.lower()
        d["_red"], d["_green"] = (ev == "red").astype(int), (ev == "green").astype(int)
        g = d.groupby(["date_announced", "actor"]).agg(interwencji=("intervention_id", "size"),
                                                       produkty=("n_products", "sum"),
                                                       red=("_red", "sum"), green=("_green", "sum")).reset_index()
        g["znak"] = np.where(g["green"] > g["red"], "liberalizacja", "restrykcja")
        return g[g["produkty"] >= min_products].sort_values("date_announced").reset_index(drop=True)

    def clean(self) -> pd.DataFrame:
        d = (self.df if len(self.df) else self.load()).copy()
        log = CleaningLog("GTA")
        n = len(d); d = d.drop_duplicates(["intervention_id", "actor"]); log.add("duplikaty (intervention_id, actor)", n, len(d))
        for c in ("title", "type", "evaluation", "implementing", "affected"):
            if c in d:
                d[c] = d[c].astype("string").str.strip()
        if "evaluation" in d:
            d["evaluation"] = d["evaluation"].str.title()
        # flagi zamiast usuwania: o losie anomalii decyduje badacz
        d["has_implemented"] = d["date_implemented"].notna()
        d["flag_impl_przed_ogłoszeniem"] = d["date_implemented"] < d["date_announced"]
        if "date_removed" in d:
            d["flag_usunięte_przed_wdrożeniem"] = d["date_removed"] < d["date_implemented"]
        d["flag_brak_produktów"] = d.get("n_products", pd.Series(1, index=d.index)).fillna(0) == 0
        for f in [c for c in d.columns if c.startswith("flag_")] + ["has_implemented"]:
            log.add_count(f"FLAGA {f} = True" if f != "has_implemented" else "ma date_implemented",
                          int(d[f].sum()), "nie usunięto")
        d.to_csv(self.out_dir / "gta_clean.csv", index=False)
        log.save(self.out_dir / "gta_cleaning_log.csv")
        return log.to_frame()


class FederalRegisterExplorer:
    DATE_COLS = ["publication_date", "effective_on", "signing_date"]
    REL_RE = re.compile(r"tariff|duty|duties|section 232|section 301|ieepa|reciprocal|import", re.I)
    STRICT_RE = re.compile(r"tariff|duty|duties|section 232|section 301|ieepa|reciprocal", re.I)
    NOISE_RE = re.compile(r"magnitsky|african growth|foreign-trade zone|information collection|anti-?dumping|"
                          r"countervailing|five-year|sunset review|scope ruling|administrative review", re.I)

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.out_dir = cfg.out("federal_register")
        self.df = pd.DataFrame()

    def load(self) -> pd.DataFrame:
        raw = pd.read_csv(self.cfg.fr_csv)
        for c in self.DATE_COLS:
            if c in raw:
                raw[c] = pd.to_datetime(raw[c], errors="coerce")
        if "query" in raw:
            raw["query"] = raw["query"].astype(str)
        self.df = raw
        return raw

    def explore(self):
        d = self.df if len(self.df) else self.load()
        show(d.drop(columns=["abstract"], errors="ignore").head(5), "FR: head")
        show(TableProfiler.column_overview(d), "FR: przegląd kolumn")
        pub, eff, sgn = d["publication_date"], d["effective_on"], d.get("signing_date")
        checks = {
            "wiersze": len(d),
            "duplikaty document_number": int(d.duplicated("document_number").sum()),
            "publication_date min / max": (pub.min(), pub.max()),
            "effective_on PRZED publikacją": int((eff < pub).sum()),
            "effective_on > 1 rok po publikacji": int(((eff - pub).dt.days > 365).sum()),
            "signing_date PO publikacji": int((sgn > pub).sum()) if sgn is not None else None,
        }
        show(checks, "FR: kontrole logiczne")
        show(d.assign(rok=pub.dt.year).groupby(["rok", "type"]).size().unstack(1).fillna(0).astype(int), "FR: dokumenty per rok i typ")
        miss = d.groupby("type")[[c for c in ("effective_on", "signing_date") if c in d]].apply(lambda g: g.isna().mean().round(2))
        show(miss, "FR: udział braków effective_on / signing_date per typ (dokumenty prezydenckie często bez effective_on)")
        if "query" in d:
            show(d["query"].value_counts(), "FR: dokumenty per zapytanie")
        text = (d["title"].fillna("") + " " + d.get("abstract", pd.Series("", index=d.index)).fillna(""))
        rel = text.str.contains(self.REL_RE)
        print(f"\nDokumenty, w których tytuł/abstrakt zawiera słowa kluczowe ceł: {rel.mean():.1%} (reszta to szum pełnotekstowego wyszukiwania)")
        show(d.loc[~rel, ["document_number", "type", "title"]].head(8), "FR: przykładowy szum")

    def events(self) -> pd.DataFrame:
        """Dokumenty rdzeniowe: ścisły słownik, bez szumu, bez duplikatów; data zdarzenia = podpis, jeśli jest."""
        d = (self.df if len(self.df) else self.load()).copy()
        text = d["title"].fillna("") + " " + d.get("abstract", pd.Series("", index=d.index)).fillna("")
        d["core"] = text.str.contains(self.STRICT_RE) & ~text.str.contains(self.NOISE_RE)
        q = d.groupby("document_number")["query"].agg(lambda s: ";".join(sorted(set(s)))) if "query" in d else None
        d = d[d["core"]].drop_duplicates("document_number")
        if q is not None:
            d["queries"] = d["document_number"].map(q)
        d["event_date"] = d["signing_date"].fillna(d["publication_date"]) if "signing_date" in d else d["publication_date"]
        return d.sort_values("event_date")[["event_date", "publication_date", "type", "queries", "title"]].reset_index(drop=True)

    def clean(self) -> pd.DataFrame:
        d = (self.df if len(self.df) else self.load()).copy()
        log = CleaningLog("FR")
        n = len(d)
        d = d.dropna(subset=["document_number", "publication_date"]); log.add("brak numeru lub daty publikacji", n, len(d))
        if "query" in d:
            qmap = d.groupby("document_number")["query"].agg(lambda s: ";".join(sorted(set(s))))
        n = len(d); d = d.drop_duplicates("document_number"); log.add("duplikaty document_number", n, len(d))
        if "query" in d:
            d["queries"] = d["document_number"].map(qmap)
        n = len(d); d = d.drop_duplicates("document_number"); log.add("duplikaty document_number", n, len(d))
        d["title"] = d["title"].astype("string").str.strip()
        text = d["title"].fillna("") + " " + d.get("abstract", pd.Series("", index=d.index)).fillna("")
        d["relevant"] = text.str.contains(self.REL_RE)
        d["flag_eff_przed_pub"] = d["effective_on"] < d["publication_date"]
        d["lag_sign_to_pub_d"] = (d["publication_date"] - d.get("signing_date")).dt.days if "signing_date" in d else np.nan
        log.add_count("FLAGA relevant = False (szum, zostaje w pliku)", int((~d["relevant"]).sum()), "filtr słownikowy, sprawdź ręcznie")
        log.add_count("FLAGA effective_on przed publikacją", int(d["flag_eff_przed_pub"].sum()))
        d.to_csv(self.out_dir / "fr_clean.csv", index=False)
        log.save(self.out_dir / "fr_cleaning_log.csv")
        return log.to_frame()
