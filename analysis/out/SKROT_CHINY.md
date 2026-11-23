# Skrót: tweety o cłach vs chińskie ETF-y notowane w USA (2018–2019)

Pełne wyniki: `china_summary.txt`. Wykresy: `fig_china_zwroty_wg_etykiety.png`, `fig_china_indeks.png`.
Skrypt: `analysis/china_etf_study.py`, dane: `data/china_etfs_daily.csv` (yfinance, ceny skorygowane).

## Konstrukcja

- **Koszyk ustalony z góry:** KWEB (chiński internet), CQQQ (chińska technologia), FXI, MCHI (szeroki rynek Chin), ASHR (akcje A).
  Nie dobieramy spółek po fakcie, więc nie ma survivorship ani look-ahead bias. Wszystkie 5 ETF-ów ma pełne notowania 2017–2020.
- **ETF-y handlują na NYSE, w godzinach tweetów.** Hang Seng zamyka się wcześniej, więc tam reakcja przesuwa się na następną sesję.
- **Miara: zwrot nadzwyczajny AR = r_ETF − β·r_SPY.** β pochodzi z 2017 r. (KWEB 1,33, CQQQ 1,35, FXI 1,15, MCHI 1,24, ASHR 0,42).
  AR odfiltrowuje ogólny ruch rynku USA i zostawia reakcję specyficzną dla Chin.
- **Etykiety, testy i klasyfikator są te same co w analizie S&P 500** (gold MG_pierwszy / PK_pierwszy / zgodne, korekta FDR).

## Wyniki (gold MG_pierwszy, pb = 0,01%)

| Typ posta | n | AR dnia sesji: śr. | % dni z AR > 0 | AR 5 sesji: śr. | AR 5 sesji PRZED postem |
|---|---|---|---|---|---|
| ESKALACJA | 35 | **−27** | 40% | −74 | **−132** |
| DEESKALACJA | 12 | **+39** | 75% | −14 | +56 |
| NEUTRALNE | 15 | +5 | 67% | −2 | −231 |
| nie o cłach (tło) | 103 | +1 | 50% | −27 | −58 |

1. **Inaczej niż dla S&P 500, kierunek jest zgodny z intuicją.** Po eskalacji chińskie ETF-y tracą względem SPY, po deeskalacji zyskują.
   Różnica w dniu sesji to ok. **−66 pb** (KWEB+CQQQ: −84 pb). Znak jest taki sam we wszystkich trzech wariantach etykiet:
   p_perm = 0,07 (MG), 0,03 (PK), 0,01 (zgodne). Korelacja Spearmana kierunek vs AR wynosi od −0,26 do −0,44.
2. **Efekt koncentruje się w postach o Chinach.** Dla nich: ESK −77 pb, DEESK +91 pb w dniu sesji (p = 0,02), a w 5 sesjach −134 vs +135 pb (p = 0,03).
   W postach o cłach wobec innych krajów (UE, Meksyk, Kanada, Indie) nie ma różnicy (p = 0,89). Taki wzorzec jest spójny z reakcją na treść, a nie na samo tło rynkowe.
   Uwaga: bardzo małe n (11 vs 5).
3. **Efekt dotyczy dnia publikacji i nie utrwala się wyraźnie.** Następna sesja i okna 3 i 5 sesji dają różnice tego samego znaku, ale nieistotne.
4. **Te same zastrzeżenia co dla S&P:**
   - **Odwrotna przyczynowość / klastrowanie.** Przed postami eskalacyjnymi chińskie ETF-y już traciły (−132 pb w 5 sesjach przed, vs +56 przed deeskalacją; p = 0,01).
     Eskalacja pojawia się więc w okresach, gdy Chiny już słabną, i różnica w dniu 0 może częściowo być kontynuacją trendu.
   - **Po korekcie FDR żaden test nie jest istotny** (648 testów, 67 z p < 0,05 przy ~32 oczekiwanych przypadkowo; nadwyżka sugeruje jednak, że nie wszystko jest szumem).
   - **Jeden dzień sesji = jedna obserwacja:** różnica maleje (MG: −51 pb, p = 0,25; zgodne: −92 pb, p = 0,05).
   - **Moc:** przy n = 35 vs 12 wykrywalny efekt to ok. 95 pb dziennie, a obserwowany wynosi ok. 66 pb.
5. **Zmienność nie rośnie.** |AR| w dni postów o cłach jest taki sam jak w pozostałe dni.
6. **Na całym korpusie (dzienny indeks z klasyfikatora) nie ma zależności.** Korelacje z opóźnieniem −5…+5 i Granger są nieistotne w obie strony,
   ale klasyfikator słabo rozpoznaje kierunek.
7. **Kontrola sensu działa.** W znanych dniach eskalacji AR koszyka jest wyraźnie ujemny: 22.03.2018 −195 pb, 06.05.2019 −308 pb, 01.08.2019 −198 pb.

## Wniosek w jednym zdaniu

Chińskie ETF-y reagują na tweety o cłach w oczekiwanym kierunku (eskalacja −, deeskalacja +), mocniej niż S&P 500 i głównie przy postach o Chinach.
Wynik jest spójny między anotatorami, ale nie przechodzi korekty na wielokrotne testowanie i jest częściowo zakłócony tym, że Trump eskalował w dni, gdy Chiny już traciły.
