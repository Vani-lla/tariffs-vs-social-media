from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cleaning_data.base import Config
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


OUT_DIR = REPO_ROOT / "clean_out" / "pcmci_trump_fnspid"
VIX_PATH = REPO_ROOT / "data" / "vix_daily.csv"


def ensure_vix_data() -> pd.DataFrame:
    """Zwraca szereg VIX w formacie date -> vix; jeśli nie ma pliku, pobiera lokalnie."""
    if VIX_PATH.exists():
        vix = pd.read_csv(VIX_PATH, parse_dates=["date"])
    else:
        cfg = Config()
        try:
            from cleaning_data.downloaders import download_vix

            download_vix(cfg)
            vix = pd.read_csv(VIX_PATH, parse_dates=["date"])
        except Exception:
            try:
                import yfinance as yf
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "Nie udało się pobrać VIX z FRED ani Yahoo Finance. "
                    "Dodaj lokalny plik data/vix_daily.csv lub sprawdź dostęp do internetu."
                ) from exc

            VIX_PATH.parent.mkdir(parents=True, exist_ok=True)
            raw = yf.download("^VIX", start="2018-01-01", end="2020-01-01", progress=False, auto_adjust=False)
            if raw.empty:
                raise RuntimeError("Yahoo Finance nie zwrócił danych VIX.")
            close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            vix = close.reset_index().rename(columns={"Date": "date", "index": "date"})
            if "date" not in vix.columns:
                vix = close.reset_index()
            vix.columns = ["date", "vix"]
            vix["date"] = pd.to_datetime(vix["date"])
            vix = vix[["date", "vix"]].dropna().copy()
            vix.to_csv(VIX_PATH, index=False)

    vix = vix[["date", "vix"]].dropna().copy()
    vix = vix.rename(columns={"date": "day"})
    vix["day"] = pd.to_datetime(vix["day"])
    vix = vix.set_index("day").sort_index()
    return vix


def build_experiment_frame() -> pd.DataFrame:
    """Buduje DataFrame dla Eksperymentu B: Trump -> FNSPID -> VIX."""
    trading = pd.read_csv(REPO_ROOT / "clean_out" / "merged" / "trading.csv", parse_dates=["day"])
    trading = trading.set_index("day").sort_index()

    vix = ensure_vix_data()
    trading_days = trading.index
    vix_reindexed = vix.reindex(trading_days, method="ffill")
    d_vix = vix_reindexed["vix"].diff().fillna(0.0)

    cols = ["tw_topic_n", "fn_n", "gta_ann_n", "fr_n", "S_P_500_ret", "d_vix"]
    frame = trading[cols[:-1]].copy()
    frame["d_vix"] = d_vix

    for col in ["tw_topic_n", "fn_n", "gta_ann_n", "fr_n"]:
        if col in frame.columns:
            frame[col] = np.log1p(frame[col].astype(float))

    frame = frame.fillna(frame.mean())
    frame = frame.loc[:, cols]
    return frame


def parse_pcmci_links(
    val_matrix: np.ndarray,
    p_matrix: np.ndarray,
    var_names: list[str],
    alpha_level: float,
    tau_min: int,
    tau_max: int,
) -> pd.DataFrame:
    """Konwertuje macierze PCMCI do tabeli istotnych relacji."""
    rows = []
    for i, cause in enumerate(var_names):
        for j, effect in enumerate(var_names):
            for lag in range(tau_min, tau_max + 1):
                p_val = float(p_matrix[i, j, lag])
                mci_val = float(val_matrix[i, j, lag])
                if np.isfinite(p_val) and np.isfinite(mci_val) and p_val <= alpha_level:
                    rows.append(
                        {
                            "cause_var": cause,
                            "effect_var": effect,
                            "lag_tau": int(lag),
                            "mci_val": mci_val,
                            "p_value": p_val,
                        }
                    )

    if not rows:
        return pd.DataFrame(columns=["cause_var", "effect_var", "lag_tau", "mci_val", "p_value"])

    out = pd.DataFrame(rows)
    out = out.reindex(out["mci_val"].abs().sort_values(ascending=False).index)
    return out.reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_experiment_frame()
    cols = ["tw_topic_n", "fn_n", "gta_ann_n", "fr_n", "S_P_500_ret", "d_vix"]

    data = pp.DataFrame(df[cols].values, var_names=cols)
    cond_ind_test = ParCorr(significance="analytic")
    pcmci = PCMCI(dataframe=data, cond_ind_test=cond_ind_test)

    results = pcmci.run_pcmciplus(
        tau_min=0,
        tau_max=5,
        pc_alpha=0.05,
    )

    links = parse_pcmci_links(
        val_matrix=results["val_matrix"],
        p_matrix=results["p_matrix"],
        var_names=cols,
        alpha_level=0.05,
        tau_min=0,
        tau_max=5,
    )

    out_csv = OUT_DIR / "pcmci_significant_links.csv"
    links.to_csv(out_csv, index=False)

    plot_obj = tp.plot_graph(
        graph=results["graph"],
        val_matrix=results["val_matrix"],
        var_names=cols,
        link_colorbar_label="MCI",
    )
    fig = plot_obj[0] if isinstance(plot_obj, tuple) else plot_obj
    fig.savefig(OUT_DIR / "pcmci_graph.png", dpi=200, bbox_inches="tight")

    print(f"Zapisano: {out_csv}")
    print(f"Zapisano: {OUT_DIR / 'pcmci_graph.png'}")
    print(links.to_string(index=False))


if __name__ == "__main__":
    main()

