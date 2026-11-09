from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Pozwala uruchamiac skrypt bez instalowania pakietu projektu.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from merging_data.causality import PCMCIAnalyzer, PCMCIConfig, DailyColumnAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uruchamia analize PCMCI/PCMCI+ na daily.csv.")
    parser.add_argument(
        "--daily-csv",
        type=Path,
        default=Path("clean_out/merged/daily.csv"),
        help="Sciezka do clean_out/merged/daily.csv",
    )
    parser.add_argument("--tau-max", type=int, default=7)
    parser.add_argument("--pc-alpha", type=float, default=0.05)
    parser.add_argument("--alpha-level", type=float, default=0.05)
    parser.add_argument(
        "--classic-pcmci",
        action="store_true",
        help="Wylacza PCMCI+ i uruchamia klasyczne PCMCI (tau_min=1).",
    )
    parser.add_argument(
        "--graph-out",
        type=Path,
        default=None,
        help="Opcjonalna sciezka do pliku PNG z grafem przyczynowym.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("clean_out/merged/pcmci_significant_links.csv"),
        help="Sciezka wyjsciowa CSV z istotnymi relacjami.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    daily_df = pd.read_csv(args.daily_csv)

    source_cols = list(DailyColumnAdapter.DEFAULT_MAPPING.values())
    missing = [col for col in source_cols if col not in daily_df.columns]
    if missing:
        raise KeyError(f"Brak wymaganych kolumn w daily.csv: {missing}")

    cleaned_daily = daily_df[source_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if cleaned_daily.empty:
        raise ValueError("Po odrzuceniu NaN nie zostaly zadne wiersze do analizy PCMCI.")

    standardized = DailyColumnAdapter.to_standard(cleaned_daily)
    config = PCMCIConfig(
        tau_max=args.tau_max,
        pc_alpha=args.pc_alpha,
        alpha_level=args.alpha_level,
        use_pcmciplus=not args.classic_pcmci,
        tau_min=0 if not args.classic_pcmci else 1,
    )

    analyzer = PCMCIAnalyzer(config)
    links = analyzer.run(standardized)

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    links.to_csv(args.csv_out, index=False)
    print(f"Zapisano relacje: {args.csv_out} (wiersze: {len(links)})")

    if args.graph_out is not None:
        pp, pcmci_cls, parcorr_cls, _ = analyzer._load_tigramite()
        tig_df = pp.DataFrame(standardized.values, var_names=standardized.columns.tolist())
        test = parcorr_cls(significance="analytic")
        pcmci = pcmci_cls(dataframe=tig_df, cond_ind_test=test)

        if config.use_pcmciplus:
            results = pcmci.run_pcmciplus(
                tau_min=config.tau_min,
                tau_max=config.tau_max,
                pc_alpha=config.pc_alpha,
            )
        else:
            results = pcmci.run_pcmci(
                tau_min=config.tau_min,
                tau_max=config.tau_max,
                pc_alpha=config.pc_alpha,
            )

        out_path = analyzer.plot_causal_graph(
            results=results,
            var_names=standardized.columns.tolist(),
            output_png=args.graph_out,
        )
        print(f"Zapisano graf: {out_path}")


if __name__ == "__main__":
    main()
