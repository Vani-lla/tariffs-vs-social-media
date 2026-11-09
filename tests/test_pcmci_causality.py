from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from merging_data.causality import DailyColumnAdapter, PCMCIAnalyzer, PCMCIConfig


class TestDailyColumnAdapter(unittest.TestCase):
    def test_to_standard_uses_default_mapping(self) -> None:
        df = pd.DataFrame(
            {
                "gkg_topic_n": [1.0, 2.0],
                "S_P_500_rv": [0.2, 0.3],
                "S_P_500_ret": [0.01, -0.02],
                "TLT_ret": [0.001, -0.004],
            }
        )
        out = DailyColumnAdapter.to_standard(df)

        self.assertListEqual(
            out.columns.tolist(),
            ["media_framing", "delta_vix", "sp500_return", "delta_yield_10y"],
        )
        self.assertEqual(out.shape, (2, 4))

    def test_to_standard_raises_for_missing_source(self) -> None:
        with self.assertRaises(KeyError):
            DailyColumnAdapter.to_standard(pd.DataFrame({"S_P_500_ret": [0.1]}))


class TestParseSignificantLinks(unittest.TestCase):
    def test_parse_returns_sorted_significant_links(self) -> None:
        var_names = ["a", "b"]
        val = np.zeros((2, 2, 3), dtype=float)
        p = np.ones((2, 2, 3), dtype=float)

        val[0, 1, 0] = 0.10
        p[0, 1, 0] = 0.04
        val[1, 0, 2] = -0.30
        p[1, 0, 2] = 0.01

        out = PCMCIAnalyzer.parse_significant_links(
            val_matrix=val,
            p_matrix=p,
            var_names=var_names,
            alpha_level=0.05,
            tau_min=0,
            tau_max=2,
        )

        self.assertEqual(len(out), 2)
        self.assertEqual(out.iloc[0]["cause_var"], "b")
        self.assertEqual(out.iloc[0]["effect_var"], "a")
        self.assertEqual(int(out.iloc[0]["lag_tau"]), 2)

    def test_parse_returns_empty_dataframe_when_no_links(self) -> None:
        var_names = ["a", "b"]
        val = np.zeros((2, 2, 2), dtype=float)
        p = np.ones((2, 2, 2), dtype=float)

        out = PCMCIAnalyzer.parse_significant_links(
            val_matrix=val,
            p_matrix=p,
            var_names=var_names,
            alpha_level=0.05,
            tau_min=0,
            tau_max=1,
        )
        self.assertTrue(out.empty)


class TestPCMCIConfig(unittest.TestCase):
    def test_pcmciplus_default_is_valid(self) -> None:
        config = PCMCIConfig()
        config.validate()

    def test_classic_requires_tau_min_one(self) -> None:
        with self.assertRaises(ValueError):
            PCMCIConfig(use_pcmciplus=False, tau_min=0).validate()


if __name__ == "__main__":
    unittest.main()

