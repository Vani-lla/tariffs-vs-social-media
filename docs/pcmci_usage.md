# PCMCI / PCMCI+ - szybkie wywolanie

Ten projekt udostepnia modul `merging_data/causality.py` oraz skrypt `scripts/run_pcmci.py`.

## 1) Uruchomienie przez CLI

```bash
python3 scripts/run_pcmci.py \
  --tau-max 5 \
  --pc-alpha 0.05 \
  --alpha-level 0.05 \
  --csv-out clean_out/merged/pcmci_significant_links.csv
```

Tryb klasyczny PCMCI (bez relacji rownoczesnych `tau=0`):

```bash
python3 scripts/run_pcmci.py \
  --classic-pcmci \
  --tau-max 5 \
  --pc-alpha 0.05 \
  --alpha-level 0.05
```

## 2) Wywolanie z kodu Python

```python
import pandas as pd
from merging_data.causality import PCMCIConfig, DailyColumnAdapter, PCMCIAnalyzer

# 1. Wczytanie danych dziennych
raw = pd.read_csv("clean_out/merged/daily.csv")

# 2. Mapowanie kolumn do standardu przyczynowego
x = DailyColumnAdapter.to_standard(raw)

# 3. Konfiguracja (domyslnie PCMCI+ z tau_min=0)
cfg = PCMCIConfig(
    tau_max=5,
    pc_alpha=0.05,
    alpha_level=0.05,
    use_pcmciplus=True,
    tau_min=0,
)

# 4. Analiza i tabela istotnych relacji
analyzer = PCMCIAnalyzer(cfg)
links = analyzer.run(x)
print(links.head())
```

## 3) Opcjonalny graf PNG

```bash
python3 scripts/run_pcmci.py \
  --tau-max 5 \
  --pc-alpha 0.05 \
  --alpha-level 0.05 \
  --graph-out clean_out/merged/pcmci_graph.png
```


## 4) Eksperyment B: Trump -> FNSPID -> VIX

```bash
python3 scripts/run_pcmci_trump_fnspid.py
```

Ten skrypt działa na danych z `clean_out/merged/trading.csv`, dołącza VIX z `data/vix_daily.csv`, tworzy zmienne:

```python
["tw_topic_n", "fn_n", "gta_ann_n", "fr_n", "S_P_500_ret", "d_vix"]
```

oraz stosuje:
- `np.log1p()` dla liczności (`tw_topic_n`, `fn_n`, `gta_ann_n`, `fr_n`)
- `fillna(mean())` dla braków
- `PCMCI+` z `tau_min=0`, `tau_max=5`, `pc_alpha=0.05`, `alpha_level=0.05`

Wynik zapisuje do:

```text
clean_out/pcmci_trump_fnspid/pcmci_significant_links.csv
```

## 5) Generowanie grafów PNG

Dla eksperymentu bazowego:

```bash
python3 scripts/run_pcmci.py \
  --tau-max 5 \
  --pc-alpha 0.05 \
  --alpha-level 0.05 \
  --graph-out clean_out/merged/pcmci_graph.png
```

Dla eksperymentu Trump/FNSPID/VIX:

```bash
python3 scripts/run_pcmci_trump_fnspid.py
```

Skrypt eksportuje graf do:

```text
clean_out/pcmci_trump_fnspid/pcmci_graph.png
```

Warto zauważyć, że bazowy skrypt `scripts/run_pcmci.py` zapisuje do `clean_out/merged/`, a eksperyment B do osobnego katalogu `clean_out/pcmci_trump_fnspid/`, żeby zachować niezależność wyników.
