# Skrót: czy eskalacja / deeskalacja ceł wiąże się ze wzrostem S&P 500?

Pełne wyniki: `summary.txt`. Wykres: `fig_zwroty_wg_etykiety.png`.
Dane: 167 postów Trumpa (2018–2019) oznaczonych przez MG i PK; etykiety wg MG, a gdy MG nie oznaczył – wg PK.
pb = punkt bazowy = 0,01%.

## Co dzieje się z S&P 500 po poście?

| Typ posta | n | 1h po: śr. | 24h po: śr. | 24h po: % wzrostów | 5 sesji: śr. | 5 sesji: % wzrostów |
|---|---|---|---|---|---|---|
| ESKALACJA | 32–35 | +1 pb | **+13 pb** | 53% | **+40 pb** | 71% |
| DEESKALACJA | 10–12 | +1 pb | **−54 pb** | 27% | **−184 pb** | 36% |
| NEUTRALNE (o cłach) | 11–15 | −7 pb | +11 pb | 64% | +56 pb | 73% |
| Posty nie o cłach (tło) | 75–103 | +2 pb | +6 pb | 56% | +18 pb | 64% |

## Najciekawsze wyniki

1. **Eskalacja NIE wiąże się ze spadkiem S&P.** Po postach eskalacyjnych rynek zachowuje się jak po zwykłych postach (lekko na plus). Brak efektu w każdym oknie: 1h, 3h, 24h, dzień, 3 i 5 sesji.
2. **Deeskalacja wiąże się ze SPADKAMI, nie ze wzrostem.** To odwrotność intuicji. Różnica eskalacja vs deeskalacja w 5 sesjach ≈ +224 pb (p = 0,003); przy etykietach zgodnych +279 pb (p = 0,02).
3. **Ale ten wynik jest kruchy.** Napędzają go 3 złe dni: 2018-12-04 (−324 pb, wątek „Tariff Man”), 2018-12-14 (−187 pb) i 2019-03-22 (−194 pb). Po usunięciu 2 skrajnych obserwacji z każdej grupy różnica znika (p = 0,35). Po korekcie FDR (594 testy) różnica w 5 sesjach ma q = 0,10, czyli nie jest istotna na poziomie 5%.
   Korektę na poziomie 5% przechodzą tylko 3 testy (q = 0,044) i wszystkie mówią to samo: po postach deeskalacyjnych rynek spada w ciągu kilku godzin
   (S&P 500 w 6 h: −69 pb wobec postów nie o cłach; Nasdaq w 3 h: −60 pb; posty o cłach ogółem wobec pozostałych, Nasdaq w 3 h: −41 pb).
4. **Najbardziej prawdopodobna interpretacja: to rynek wpływa na Trumpa, a nie odwrotnie.** Uspokajające posty („negocjacje idą świetnie”, „umowa blisko”) pojawiały się w dni, kiedy rynek już spadał – próba uspokojenia rynku.
5. **W ciągu 1 godziny nie ma żadnej reakcji** – różnice ≈ 0 pb dla wszystkich typów postów.
6. **Posty o cłach (dowolny kierunek) idą w parze z większymi ruchami rynku:** średni ruch 24h ≈ 78–100 pb vs 52 pb dla innych postów. Wynika to głównie z tego, że posty o cłach skupiają się w niespokojnych okresach (IV kw. 2018, maj i sierpień 2019).
7. **Na całym korpusie (7 890 tweetów, dzienny indeks z klasyfikatora) nie ma korelacji z S&P** – ani tego samego dnia, ani z opóźnieniem od −5 do +5 dni; test Grangera nieistotny w obie strony. Uwaga: klasyfikator słabo rozpoznaje kierunek, więc to wynik orientacyjny.
8. **Wynik jest podobny dla innych rynków.** Nasdaq zachowuje się jak S&P. Hang Seng i obligacje (TLT) nie wykazują wyraźnego wzorca. EURUSD lekko rośnie po eskalacji (nieistotne po korekcie).
9. **Posty o Chinach:** ten sam wzór, silniejszy (deeskalacja wobec Chin: −139 pb w dniu sesji, eskalacja ≈ 0). Bardzo małe n (5 vs 11).

## Wniosek w jednym zdaniu

Nie ma dowodu, że eskalacja ceł w tweetach obniża (ani podnosi) S&P 500. Jedyny widoczny wzorzec – deeskalacja w dni spadków – sugeruje, że Trump reagował na rynek, ale przy ~12 postach deeskalacyjnych to tylko hipoteza.

## Ograniczenia

- Próba za mała: z tą liczbą postów da się wykryć dopiero różnicę ≈ 90 pb dziennie; realistyczne efekty to 10–30 pb.
- Wyniki zależą od tego, czyje etykiety bierzemy: wg PK wzór jest ten sam, ale słabszy.
