# Porównanie eksperymentów PCMCI

## 1. Cel i założenie

Projekt porównuje dwa różne podejścia do analizy przyczynowości:

- Eksperyment A (Baseline): szereg dzienny z `daily.csv` i szerokim media-market proxy.
- Eksperyment B (Trump -> FNSPID -> VIX): szereg handlowy z `trading.csv` i zmienne z bezpośredniego łańcucha informacji politycznych i mediów.

Oba eksperymenty są prowadzone niezależnie i mają osobne katalogi wynikowe, aby zachować pełną zgodność z oryginalnym baseline.

## 2. Eksperyment A: Baseline GDELT

### Źródło danych
- Plik wejściowy: `clean_out/merged/daily.csv`
- Wersja modelowa: dzienne szeregi czasowe, bez braków po preprocessingie

### Zmienne modelowe
- `media_framing = gkg_topic_n`
- `delta_vix = S_P_500_rv`
- `sp500_return = S_P_500_ret`
- `delta_yield_10y = TLT_ret`

### Metodologia
- `tau_min = 0`
- `tau_max = 5`
- `pc_alpha = 0.05`
- `alpha_level = 0.05`
- test niezależności: `ParCorr(significance='analytic')`
- tryb: `PCMCI+`
- dane są z `daily.csv`, czyli nie ma rolowania weekendów i nie ma jawnie odseparowanej zmiennej `d_vix`

### Prawie najważniejsze krawędzie (z `clean_out/merged/pcmci_significant_links.csv`)
- `media_framing -> media_framing` (lag 1, mci=0.681, p≈1.26e-67)
- `sp500_return -> delta_vix` (lag 1, mci=-0.379, p≈4.06e-18)
- `delta_vix -> delta_vix` (lag 1, mci=0.356, p≈5.51e-16)
- `sp500_return <-> delta_yield_10y` (lag 0, mci≈-0.340, p≈1.07e-14)
- `delta_vix <-> sp500_return` (lag 0, mci≈-0.207, p≈4.20e-06)
- `sp500_return -> media_framing` (lag 1, mci=-0.162, p≈3.45e-04)

Interpretacja:
- Baseline pokazuje silną, głównie rynkową zależność między zwrotami i zmiennością,
- a `media_framing` zachowuje wyraźną autoregresyjną strukturę (samopodobieństwo czasowe) i tylko słabszy transfer do rynku.

## 3. Eksperyment B: Trump -> FNSPID -> VIX

### Źródło danych
- Plik wejściowy: `clean_out/merged/trading.csv`
- Dodatkowy plik VIX: `data/vix_daily.csv`
- Dni handlowe z rolowaniem weekendów, zgodnie z dokumentacją projektu

### Zmienne modelowe
- `tw_topic_n` — liczba tematów o cłach / tweetów Trumpa z danym dniem
- `fn_n` — liczba artykułów FNSPID z danym dniem
- `gta_ann_n` — ogłoszenia GTA
- `fr_n` — ogłoszenia Federal Register
- `S_P_500_ret` — zwrot S&P 500
- `d_vix = vix.diff()` — pierwsza różnica dziennego VIX

### Przekształcenia wstępne
- `np.log1p()` dla liczności: `tw_topic_n`, `fn_n`, `gta_ann_n`, `fr_n`
- `Z = Z.fillna(Z.mean())` dla braków
- dane z `trading.csv` są indeksowane po `day`
- `vix` jest zreindeksowany do dni handlowych i przekształcony do `d_vix`

### Prawie najważniejsze krawędzie (z `clean_out/pcmci_trump_fnspid/pcmci_significant_links.csv`)
- `d_vix <-> S_P_500_ret` (lag 0, mci=-0.818, p≈2.92e-123)
- `fn_n -> fn_n` (lag 1, mci=0.280, p≈1.47e-10)
- `tw_topic_n -> tw_topic_n` (lag 1, mci=0.162, p≈2.64e-04)
- `tw_topic_n <-> fn_n` (lag 0, mci=0.157, p≈4.07e-04)
- `gta_ann_n -> tw_topic_n` (lag 2, mci=0.131, p≈3.13e-03)
- `fr_n -> tw_topic_n` (lag 3, mci=0.128, p≈4.14e-03)
- `gta_ann_n -> fr_n` (lag 4, mci=0.121, p≈6.47e-03)
- `gta_ann_n <-> fn_n` (lag 0, mci=0.091, p≈4.01e-02)

Interpretacja:
- W Experymencie B pojawia się spójny łańcuch informacyjny: `GTA/FR -> Twitter -> FNSPID` i dalej do rynku.
- Silna relacja `d_vix <-> S_P_500_ret` jest oczekiwana i sugeruje, że zmienność rynku i zwroty są jednocześnie silnie skorelowane w modelu.
- Wzorce między tweetami Trumpa, artykułami FNSPID i ogłoszeniami politycznymi są znacznie bardziej sensowne niż w baseline i pokazują, że ten eksperyment lepiej wychwytuje kanały komunikacji + propagację informacji.

## 4. Kluczowe różnice metodologiczne

### A. Baseline GDELT
- źródło medialne: `gkg_topic_n` (GDELT)
- źródło rynku: `S_P_500_rv`, `S_P_500_ret`, `TLT_ret`
- dane: `daily.csv`
- model: media-market proxy, mniej bezpośredni kanał informacyjny
- siła główna: sprzężenie rynku i zmienności, autoregresja `media_framing`

### B. Trump -> FNSPID -> VIX
- źródło medialne: `tw_topic_n`, `fn_n`, `gta_ann_n`, `fr_n`
- źródło rynku: `S_P_500_ret`, `d_vix`
- dane: `trading.csv` + `vix_daily.csv`
- model: bezpośredni łańcuch informacyjny i propagacja przez politykę / media / rynek
- siła główna: `GTA/FR -> Twitter -> FNSPID` plus obecność silnej relacji z VIX

## 5. Wniosek

Eksperyment A jest dobrym baselinem do zrozumienia ogólnej struktury rynku i szerokiego odczucia medialnego (GDELT) bez bezpośredniego przypisania zdań do spraw taryfowych.

Eksperyment B jest bardziej zbieżny z dokumentacją projektu i z logiką badania: łańcuch informacji narasta od źródeł politycznych / administracyjnych (`GTA`, `FR`) do tweetów Trumpa i dalej do FNSPID, a następnie do reakcji rynku (VIX / S&P 500).

Dla dalszej analizy warto traktować:
- Eksperyment A jako „kontrolę / baseline”
- Eksperyment B jako „kanał przyczynowy z polityką i mediami”

