# Tasks

## 1. Konfiguracja i API PCMCI

- [x] 1.1 Dodac `PCMCIConfig` z domyslnym trybem PCMCI+ (`use_pcmciplus=True`, `tau_min=0`) i zweryfikowac walidacje przez `python -m unittest tests/test_pcmci_causality.py`.
- [x] 1.2 Dodac przelacznik trybu klasycznego PCMCI (`use_pcmciplus=False`, `tau_min=1`) i zweryfikowac scenariusz negatywny testem konfiguracji.

## 2. Adapter danych dziennych

- [x] 2.1 Zaimplementowac jawne mapowanie kolumn z `clean_out/merged/daily.csv` do nazw standaryzowanych i zweryfikowac testem mapowania adaptera.
- [x] 2.2 Dodac walidacje brakujacych kolumn i NaN po mapowaniu, a nastepnie zweryfikowac testem bledu `KeyError`/`ValueError`.

## 3. Silnik analizy przyczynowosci

- [x] 3.1 Zaimplementowac uruchamianie `run_pcmciplus` lub `run_pcmci` wraz z `ParCorr(significance='analytic')` i zweryfikowac recznie uruchamiajac `python scripts/run_pcmci.py --help`.
- [x] 3.2 Dodac parser `val_matrix`/`p_matrix` do tabeli wynikowej i zweryfikowac sortowanie po `|mci_val|` testem jednostkowym.
- [x] 3.3 Dodac obsluge pustej macierzy istotnych krawedzi (pusty DataFrame) i zweryfikowac testem jednostkowym.

## 4. Wizualizacja i uruchamianie

- [x] 4.1 Dodac opcjonalna funkcje zapisu grafu przyczynowego do `.png` przez `tp.plot_graph` i zweryfikowac, ze plik jest tworzony dla przykładowego uruchomienia.
- [x] 4.2 Dodac runner CLI `scripts/run_pcmci.py` i zweryfikowac generowanie CSV z relacjami na danych `clean_out/merged/daily.csv`.

## 5. Zaleznosci i dokumentacja

- [x] 5.1 Utworzyc `requirements.txt` z `tigramite`, `statsmodels`, `scipy`, `pandas`, `numpy`, `matplotlib` i zweryfikowac instalacje przez `pip install -r requirements.txt`.
- [x] 5.2 Zaktualizowac `README.md` o instrukcje uruchomienia i zweryfikowac komendy z dokumentacji.
