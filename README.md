# Czy ramowanie ceł i sankcji w mediach wyprzedza nastroje rynkowe?
Autorzy: Bartek Borkowski, Igor Kowalczyk, Mateusz Gadomski, Paulina Korus

## Analiza przyczynowosci (PCMCI / PCMCI+)

Dodano modul `merging_data/causality.py` do analizy przyczynowosci dla szeregow dziennych.

- Domyslnie uzywany jest PCMCI+ (`use_pcmciplus=True`, `tau_min=0`) dla relacji rownoczesnych (`tau=0`).
- Tryb klasyczny PCMCI jest dostepny po ustawieniu `use_pcmciplus=False` i `tau_min=1`.
- Adapter `DailyColumnAdapter` mapuje kolumny z `clean_out/merged/daily.csv` do:
  `media_framing`, `delta_vix`, `sp500_return`, `delta_yield_10y`.

Szybkie uruchomienie:

```bash
pip install -r requirements.txt
python scripts/run_pcmci.py --tau-max 7 --pc-alpha 0.05 --alpha-level 0.05
```
