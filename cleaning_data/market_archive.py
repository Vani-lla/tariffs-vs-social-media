"""Archiwum `data.taz` z danymi giełdowymi: rozpoznanie formatu, rozpakowanie, eksploracja, czyszczenie."""
from __future__ import annotations

import bz2
import gzip
import lzma
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

from .base import TABULAR_SUFFIXES, CleaningLog, Config, read_table, show
from .ohlcv import OHLCVTools


class ArchiveInspector:
    """.taz bywa tar.gz (alias .tgz) albo tar.Z (stary `compress`). Format wykrywamy po bajtach, nie po rozszerzeniu."""
    MAGIC = [(b"\x1f\x8b", "gzip"), (b"\x1f\x9d", "compress"), (b"BZh", "bzip2"),
             (b"\xfd7zXZ\x00", "xz"), (b"PK\x03\x04", "zip")]

    def __init__(self, path: str | Path, work_dir: str | Path):
        self.path = Path(path)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def detect(self) -> str:
        if self.path.is_dir():
            return "katalog"
        with open(self.path, "rb") as f:
            head = f.read(512)
        for magic, name in self.MAGIC:
            if head.startswith(magic):
                return name
        if head[257:262] == b"ustar":
            return "tar"
        return "nieznany/zwykły plik"
    # ------------------------------------------------------------------ pomocnicze
    def _decompress_Z(self) -> Path:
        out = self.work_dir / (self.path.stem + ".decompressed")
        if out.exists() and out.stat().st_size > 0:
            return out
        try:
            import unlzw3
            out.write_bytes(unlzw3.unlzw(self.path.read_bytes()))
            return out
        except ImportError:
            pass
        try:  # gzip potrafi czytać format .Z
            with open(out, "wb") as f:
                subprocess.run(["gzip", "-dc", str(self.path)], stdout=f, check=True)
            return out
        except Exception as ex:
            raise RuntimeError("Plik jest w formacie .Z (compress). Zainstaluj: pip install unlzw3 "
                               f"(lub użyj systemowego gzip). Szczegóły: {ex}")

    def _open_tar(self):
        fmt = self.detect()
        src = self._decompress_Z() if fmt == "compress" else self.path
        try:
            return tarfile.open(src, "r:*")
        except tarfile.ReadError:
            return None  # to nie tar: pojedynczy skompresowany plik

    # ------------------------------------------------------------------ API
    def list_members(self) -> pd.DataFrame:
        fmt = self.detect()
        if fmt == "katalog":
            return pd.DataFrame([{"nazwa": str(p.relative_to(self.path)), "rozmiar_MB": round(p.stat().st_size / 1e6, 3)}
                                 for p in self.path.rglob("*") if p.is_file()])
        if fmt == "zip":
            with zipfile.ZipFile(self.path) as z:
                return pd.DataFrame([{"nazwa": i.filename, "rozmiar_MB": round(i.file_size / 1e6, 3)}
                                     for i in z.infolist() if not i.is_dir()])
        t = self._open_tar()
        if t is not None:
            with t:
                return pd.DataFrame([{"nazwa": m.name, "rozmiar_MB": round(m.size / 1e6, 3)}
                                     for m in t.getmembers() if m.isfile()])
        return pd.DataFrame([{"nazwa": self.path.name, "rozmiar_MB": round(self.path.stat().st_size / 1e6, 3)}])

    def extract(self) -> list[Path]:
        fmt = self.detect()
        if fmt == "katalog":
            return sorted(p for p in self.path.rglob("*") if p.is_file())
        dest = self.work_dir.resolve()
        if fmt == "zip":
            with zipfile.ZipFile(self.path) as z:
                for i in z.infolist():
                    target = (dest / i.filename).resolve()
                    if not str(target).startswith(str(dest)):
                        raise RuntimeError(f"Podejrzana ścieżka w archiwum: {i.filename}")
                z.extractall(dest)
        else:
            t = self._open_tar()
            if t is not None:
                with t:
                    for m in t.getmembers():
                        target = (dest / m.name).resolve()
                        if not str(target).startswith(str(dest)) or m.issym() or m.islnk():
                            print("Pominięto podejrzany element archiwum:", m.name)
                            continue
                        try:
                            t.extract(m, dest, filter="data")
                        except TypeError:  # starsze Pythony nie mają argumentu filter
                            t.extract(m, dest)
            else:
                opener = {"gzip": gzip.open, "bzip2": bz2.open, "xz": lzma.open}.get(fmt)
                if opener is None:
                    shutil.copy(self.path, dest / self.path.name)
                else:
                    with opener(self.path, "rb") as fi, open(dest / self.path.stem, "wb") as fo:
                        shutil.copyfileobj(fi, fo)
        return sorted(p for p in dest.rglob("*") if p.is_file() and not p.name.endswith(".decompressed"))


class MarketArchiveExplorer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.inspector = ArchiveInspector(cfg.market_archive, cfg.market_work_dir)
        self.out_dir = cfg.out("market")
        self.files: list[Path] = []
        self.results: dict = {}

    def inspect(self) -> dict:
        p = self.inspector.path
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
        info = {"plik": str(p), "wykryty_format": self.inspector.detect(), "rozmiar_MB": round(size / 1e6, 2)}
        show(info, "Archiwum")
        members = self.inspector.list_members()
        show(members.head(50), f"Zawartość ({len(members)} plików, pokazano max 50)")
        return info

    def extract(self) -> pd.DataFrame:
        self.files = self.inspector.extract()
        tab = [p for p in self.files if p.suffix.lower() in TABULAR_SUFFIXES]
        base = (self.inspector.path if self.inspector.path.is_dir() else self.inspector.work_dir).resolve()
        df = pd.DataFrame({"plik": [str(p.resolve().relative_to(base)) for p in self.files],
                           "tabelaryczny": [p in tab for p in self.files],
                           "MB": [round(p.stat().st_size / 1e6, 3) for p in self.files]})
        show(df.head(60), "Rozpakowane pliki")
        return df

    def _tables(self) -> list[Path]:
        if not self.files:
            root = self.inspector.path if self.inspector.path.is_dir() else Path(self.cfg.market_work_dir)
            self.files = sorted(p for p in root.rglob("*") if p.is_file())
        tabs = [p for p in self.files if p.suffix.lower() in TABULAR_SUFFIXES]
        return sorted(tabs, key=lambda p: ("instrument" not in str(p).lower(), str(p)))

    def peek(self, n_files: int = 5, n: int = 3):
        for p in self._tables()[:n_files]:
            try:
                show(read_table(p, nrows=n), f"HEAD: {p.name}")
            except Exception as ex:
                print(f"{p.name}: nie udało się wczytać ({ex})")

    def explore(self, max_files: int = 10, nrows: int | None = None):
        """Eksploracja każdego pliku tabelarycznego. nrows ogranicza wczytanie dla bardzo dużych plików."""
        for p in self._tables()[:max_files]:
            try:
                df = read_table(p, nrows=nrows)
            except Exception as ex:
                print(f"{p.name}: nie udało się wczytać ({ex})")
                continue
            res = OHLCVTools.explore(df, assume_tz=self.cfg.market_assume_tz)
            self.results[p.name] = res
            print("\n" + "#" * 80 + f"\n# {p.name}  ({len(df):,} wierszy)\n" + "#" * 80)
            OHLCVTools.print_report(p.name, res, show)

    def clean(self, max_files: int | None = None) -> pd.DataFrame:
        logs = []
        for p in self._tables()[:max_files]:
            try:
                df = read_table(p)
            except Exception as ex:
                print(f"{p.name}: pominięto ({ex})")
                continue
            d, log = OHLCVTools.clean(df, p.name, assume_tz=self.cfg.market_assume_tz)
            if d.empty:
                print(f"{p.name}: po czyszczeniu pusty, pomijam zapis")
            else:
                d.to_csv(self.out_dir / f"{p.stem}_clean.csv")
            lf = log.to_frame(); lf.insert(0, "plik", p.name)
            logs.append(lf)
        full = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
        full.to_csv(self.out_dir / "market_cleaning_log.csv", index=False)
        return full

    def find(self, fragment: str, side: str = "bid") -> Path:
        frag, sd = fragment.lower(), side.lower()
        hits = [p for p in self._tables() if p.name.lower().startswith(frag + "-") and f"-{sd}-" in p.name.lower()]
        if not hits:
            raise FileNotFoundError(f"Brak pliku '{fragment}' ({side}). Dostępne: {[p.name for p in self._tables()][:10]}...")
        return hits[0]

    def _ohlc(self, p: Path, nrows: int | None = None) -> pd.DataFrame:
        d, _ = OHLCVTools.prepare(read_table(p, nrows=nrows), self.cfg.market_assume_tz)
        d = d.sort_index()
        return d[~d.index.duplicated(keep="last")]

    def load_price(self, fragment: str, side: str = "mid", nrows: int | None = None) -> pd.Series:
        """Cena na początku świecy 5-min. side='mid' = średnia bid i ask."""
        sides = ["bid", "ask"] if side == "mid" else [side]
        ps = [OHLCVTools.start_price(self._ohlc(self.find(fragment, s), nrows)) for s in sides]
        if len(ps) == 1:
            return ps[0]
        a, b = ps[0].align(ps[1], join="inner")
        return ((a + b) / 2).rename(fragment)

    def check_tz(self, fragment: str, side: str = "bid") -> dict:
        d = self._ohlc(self.find(fragment, side))
        res = OHLCVTools.tz_check(d)
        show(res, f"{fragment}: kontrola strefy czasowej (market_assume_tz={self.cfg.market_assume_tz})")
        return res