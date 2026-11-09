# Proposal

## Why
Potrzebujemy produkcyjnego, powtarzalnego modulu do badania zaleznosci przyczynowych miedzy framingiem medialnym i zmiennymi rynkowymi. Aktualnie brakuje jednego API, ktore laczy mapowanie danych dziennych, estymacje PCMCI/PCMCI+ i czytelny wynik analityczny.

## What Changes
- Dodanie modulu Python do analizy przyczynowosci z Tigramite, z pelnym typowaniem i walidacja wejscia.
- Domyslny tryb obliczen: PCMCI+ (`use_pcmciplus=True`, `tau_min=0`) dla relacji rownoczesnych `tau=0`.
- Zachowanie opcji klasycznego PCMCI (`use_pcmciplus=False`, `tau_min=1`).
- Dodanie jawnego adaptera mapowania kolumn z `clean_out/merged/daily.csv` na zestaw standaryzowany.
- Dodanie zaleznosci runtime (`tigramite`, `statsmodels`, `scipy`, `pandas`, `numpy`, `matplotlib`).
- Dodanie opcji wizualizacji grafu przyczynowego i zapisu do `.png`.

## Capabilities

### New Capabilities
- `causal-discovery`: Analiza przyczynowosci czasowej (PCMCI/PCMCI+) z tabelarycznym wynikiem istotnych relacji i opcjonalnym grafem.

### Modified Capabilities
- _Brak_

## Impact
- Kod: `merging_data/causality.py`, eksporty w `merging_data/__init__.py`, runner `scripts/run_pcmci.py`.
- Dane: adapter oparty o kolumny z `clean_out/merged/daily.csv`.
- Zaleznosci: nowy `requirements.txt`.
- Dokumentacja: aktualizacja `README.md`.

