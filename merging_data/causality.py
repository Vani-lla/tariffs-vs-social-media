from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STANDARD_COLUMNS: tuple[str, ...] = (
    "media_framing",
    "delta_vix",
    "sp500_return",
    "delta_yield_10y",
)


@dataclass(frozen=True)
class PCMCIConfig:
    """Konfiguracja analizy przyczynowości PCMCI/PCMCI+."""

    tau_max: int = 7
    pc_alpha: float = 0.05
    alpha_level: float = 0.05
    use_pcmciplus: bool = True
    tau_min: int = 0

    def validate(self) -> None:
        """Waliduje zakres parametrów konfiguracji."""
        if self.tau_max < 1:
            raise ValueError("tau_max musi byc >= 1.")
        if not (0.0 < self.pc_alpha < 1.0):
            raise ValueError("pc_alpha musi byc w przedziale (0, 1).")
        if not (0.0 < self.alpha_level < 1.0):
            raise ValueError("alpha_level musi byc w przedziale (0, 1).")
        if self.use_pcmciplus:
            if self.tau_min != 0:
                raise ValueError("Dla PCMCI+ tau_min musi wynosic 0.")
        elif self.tau_min != 1:
            raise ValueError("Dla klasycznego PCMCI tau_min musi wynosic 1.")


class DailyColumnAdapter:
    """Adapter mapujacy kolumny `daily.csv` na nazwy standaryzowane dla PCMCI."""

    DEFAULT_MAPPING: dict[str, str] = {
        "media_framing": "gkg_topic_n",
        "delta_vix": "S_P_500_rv",
        "sp500_return": "S_P_500_ret",
        "delta_yield_10y": "TLT_ret",
    }

    @classmethod
    def to_standard(
        cls,
        df: pd.DataFrame,
        mapping: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """
        Zwraca DataFrame ze standaryzowanymi kolumnami dla modulu przyczynowosci.

        Domyslnie uzywa jawnego mapowania kolumn z `clean_out/merged/daily.csv`.
        Dla `delta_vix` i `delta_yield_10y` domyslne mapowanie korzysta z proxy,
        gdyz te serie nie zawsze wystepuja bezposrednio w panelu dziennym.
        """
        if df.empty:
            raise ValueError("Wejsciowy DataFrame jest pusty.")

        selected_mapping = dict(cls.DEFAULT_MAPPING if mapping is None else mapping)
        missing_targets = [col for col in STANDARD_COLUMNS if col not in selected_mapping]
        if missing_targets:
            raise ValueError(
                f"Brak mapowania dla docelowych kolumn: {missing_targets}."
            )

        missing_sources = [
            source_col
            for source_col in selected_mapping.values()
            if source_col not in df.columns
        ]
        if missing_sources:
            raise KeyError(
                "Brak kolumn zrodlowych w danych wejsciowych: "
                f"{missing_sources}."
            )

        standardized = pd.DataFrame(index=df.index)
        for target_col in STANDARD_COLUMNS:
            standardized[target_col] = pd.to_numeric(
                df[selected_mapping[target_col]], errors="coerce"
            )

        if standardized.isna().any().any():
            bad = standardized.columns[standardized.isna().any()].tolist()
            raise ValueError(
                "Po mapowaniu pojawily sie NaN. Sprawdz dane lub mapowanie dla: "
                f"{bad}."
            )

        return standardized


class PCMCIAnalyzer:
    """Uruchamia analize przyczynowosci Tigramite i zwraca istotne relacje."""

    def __init__(self, config: PCMCIConfig) -> None:
        self.config = config
        self.config.validate()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Uruchamia PCMCI/PCMCI+ i zwraca istotne relacje przyczynowe.

        Args:
            df: DataFrame ze standaryzowanymi kolumnami i bez NaN.

        Returns:
            Tabela relacji z kolumnami:
            ['cause_var', 'effect_var', 'lag_tau', 'mci_val', 'p_value'].
        """
        self._validate_input(df)
        pp, pcmci_cls, parcorr_cls, _ = self._load_tigramite()

        tig_df = pp.DataFrame(df.values, var_names=df.columns.tolist())
        cond_ind_test = parcorr_cls(significance="analytic")
        pcmci = pcmci_cls(dataframe=tig_df, cond_ind_test=cond_ind_test)

        if self.config.use_pcmciplus:
            results = pcmci.run_pcmciplus(
                tau_min=self.config.tau_min,
                tau_max=self.config.tau_max,
                pc_alpha=self.config.pc_alpha,
            )
        else:
            results = pcmci.run_pcmci(
                tau_min=self.config.tau_min,
                tau_max=self.config.tau_max,
                pc_alpha=self.config.pc_alpha,
            )

        return self.parse_significant_links(
            val_matrix=results["val_matrix"],
            p_matrix=results["p_matrix"],
            var_names=df.columns.tolist(),
            alpha_level=self.config.alpha_level,
            tau_min=self.config.tau_min,
            tau_max=self.config.tau_max,
        )

    @staticmethod
    def parse_significant_links(
        *,
        val_matrix: np.ndarray,
        p_matrix: np.ndarray,
        var_names: list[str],
        alpha_level: float,
        tau_min: int,
        tau_max: int,
    ) -> pd.DataFrame:
        """Konwertuje macierze Tigramite do tabeli istotnych relacji."""
        if val_matrix.shape != p_matrix.shape:
            raise ValueError("val_matrix i p_matrix musza miec identyczny ksztalt.")

        n_vars = len(var_names)
        if val_matrix.shape[0] != n_vars or val_matrix.shape[1] != n_vars:
            raise ValueError("Ksztalt macierzy nie zgadza sie z liczba zmiennych.")

        rows: list[dict[str, Any]] = []
        for cause_idx in range(n_vars):
            for effect_idx in range(n_vars):
                for tau in range(tau_min, tau_max + 1):
                    p_val = float(p_matrix[cause_idx, effect_idx, tau])
                    mci_val = float(val_matrix[cause_idx, effect_idx, tau])
                    if np.isfinite(p_val) and np.isfinite(mci_val) and p_val <= alpha_level:
                        rows.append(
                            {
                                "cause_var": var_names[cause_idx],
                                "effect_var": var_names[effect_idx],
                                "lag_tau": int(tau),
                                "mci_val": mci_val,
                                "p_value": p_val,
                            }
                        )

        if not rows:
            return pd.DataFrame(
                columns=["cause_var", "effect_var", "lag_tau", "mci_val", "p_value"]
            )

        out = pd.DataFrame(rows)
        out = out.reindex(out["mci_val"].abs().sort_values(ascending=False).index)
        return out.reset_index(drop=True)

    def plot_causal_graph(
        self,
        *,
        results: dict[str, Any],
        var_names: list[str],
        output_png: str | Path,
    ) -> Path:
        """Rysuje i zapisuje graf przyczynowy do pliku PNG."""
        _, _, _, tp = self._load_tigramite()
        output_path = Path(output_png)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        graph = results.get("graph")
        if graph is None:
            raise KeyError("Brak `graph` w wynikach Tigramite - nie mozna narysowac grafu.")

        plot_obj = tp.plot_graph(
            graph=graph,
            val_matrix=results["val_matrix"],
            var_names=var_names,
            link_colorbar_label="MCI",
        )

        fig = plot_obj[0] if isinstance(plot_obj, tuple) else plot_obj
        if not hasattr(fig, "savefig"):
            raise RuntimeError("Tigramite zwrocilo nieoczekiwany obiekt wykresu.")

        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        return output_path

    @staticmethod
    def _load_tigramite() -> tuple[Any, Any, Any, Any]:
        try:
            from tigramite import data_processing as pp
            from tigramite import plotting as tp
            from tigramite.independence_tests.parcorr import ParCorr
            from tigramite.pcmci import PCMCI
        except ImportError as exc:
            raise ImportError(
                "Brak biblioteki tigramite. Zainstaluj zaleznosci z requirements.txt."
            ) from exc

        return pp, PCMCI, ParCorr, tp

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Wejsciowy DataFrame jest pusty.")
        if df.isna().any().any():
            raise ValueError("Wejsciowy DataFrame nie moze zawierac NaN.")
        if df.shape[1] < 2:
            raise ValueError("Do analizy przyczynowosci potrzebne sa co najmniej 2 zmienne.")


def run_pcmci_from_daily(
    daily_df: pd.DataFrame,
    config: PCMCIConfig,
    mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Adapter + analiza PCMCI na panelu `daily.csv`."""
    standardized = DailyColumnAdapter.to_standard(daily_df, mapping=mapping)
    analyzer = PCMCIAnalyzer(config)
    return analyzer.run(standardized)
