"""Czyszczenie źródeł (notebook 01_cleaning). Nie zależy od merging_data ani annotation."""
from .base import Config, CleaningLog, TableProfiler, show, read_table, parse_utc
from .ohlcv import OHLCVTools
from .fnspid import FNSPIDExplorer, FNSPIDTopics
from .market_archive import ArchiveInspector, MarketArchiveExplorer
from .dukascopy_src import DukascopyFetcher, DukascopyExplorer
from .gdelt_gkg import GKGExplorer
from .gta_fr import GTAExplorer, FederalRegisterExplorer
from .trump_truth import TrumpExplorer
from .horizons import HorizonTools, LagTools
from .cleaning import FNSPIDCleaner, FRCleaner, MarketCleaner, CleanAudit, run_cleaning

try:  # pobieranie wymaga `requests`; jego brak nie może blokować reszty
    from .downloaders import (ensure_all, ensure_pip, download_vix, download_gkg_dense, plan_gkg_dense,
                              thin_gkg_zip)
    _DL = ["ensure_all", "ensure_pip", "download_vix", "download_gkg_dense", "plan_gkg_dense", "thin_gkg_zip"]
except ImportError as _ex:  # pragma: no cover
    print(f"UWAGA: moduł downloaders niedostępny ({_ex}).")
    _DL = []

__all__ = ["Config", "CleaningLog", "TableProfiler", "show", "read_table", "parse_utc", "OHLCVTools",
           "FNSPIDExplorer", "FNSPIDTopics", "ArchiveInspector", "MarketArchiveExplorer", "DukascopyFetcher",
           "DukascopyExplorer", "GKGExplorer", "GTAExplorer", "FederalRegisterExplorer", "TrumpExplorer",
           "HorizonTools", "LagTools", "FNSPIDCleaner", "FRCleaner", "MarketCleaner", "CleanAudit",
           "run_cleaning"] + _DL
