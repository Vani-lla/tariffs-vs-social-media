# Rynek reaguje na pierwszą wiarygodną informację o cłach, niezależnie od kanału

> **Hipoteza (przeformułowana):** rynek reaguje na *pierwszą wiarygodną informację* o zmianie ceł, bez względu na to,
> czy podaje ją rząd (USTR, proklamacja, Federal Register), czy Trump na Twitterze. Późniejsze potwierdzenie
> i samo wejście ceł w życie są już wycenione.
>
> **Hipoteza wyjściowa** brzmiała: „rynek reaguje na oficjalne cła, a Trump je tylko nagłaśnia”. Dane potwierdzają ją
> tylko częściowo (patrz *Wniosek*), stąd nowe sformułowanie.

Skrypt: `analysis/hypothesis_official_vs_trump.py`. Pełne wyniki: `hip_summary.txt`, tabele `hip_*.csv`.
Rynki: S&P 500 (SPY) i chińskie ETF-y (koszyk 5 ETF, zwrot nadzwyczajny względem SPY). Dane dzienne, sesje NYSE 2018–2019.

## Dwa podejścia

**A. Wszystkie 503 sesje.** Każdy dzień ma typ:
- **oficjalne ogłoszenie:** GTA (duże interwencje albo USA–Chiny), merytoryczne dokumenty Federal Register, kronika;
- **wejście ceł w życie;**
- **intensywny dzień Trumpa:** ≥ 3 tweety o cłach, górne 10% dni. Tweet o cłach pojawia się w 44% sesji, więc pojedynczy tweet nie jest „zdarzeniem”;
- **spokojny dzień.**

Miara: wielkość ruchu |r|, bo kierunek większości zdarzeń z GTA i FR nie jest oznaczony.

**B. Kronika 22 kluczowych epizodów USA–Chiny.** Dla każdego epizodu trzy daty: sygnał Trumpa, oficjalne ogłoszenie i wejście w życie.
Miara: zwrot w 2 sesjach, ze znakiem względem kierunku epizodu (eskalacja powinna dawać spadek, deeskalacja wzrost).
Daty tweetów sprawdziłem w danych. Część godzin oficjalnych komunikatów jest przybliżona (kolumna `uwagi` w `hip_03_kronika.csv`), dlatego okno ma 2 sesje.

## Wyniki

![](fig_hip_1_typy_dni.png)

1. **S&P 500 rusza się mocniej w dni oficjalnych ogłoszeń:** 78 pb vs 62 pb w spokojny dzień (p = 0,03). W regresji efekt wynosi +21 pb (p = 0,09).
   **Intensywne tweety Trumpa nie zwiększają ruchu S&P** (−1 pb, p = 0,88). W tym zakresie dane **wspierają hipotezę**.
2. **Na chińskich ETF-ach żaden typ dnia nie różni się od spokojnego** (wszystkie p > 0,4).
3. **Wejście ceł w życie nie porusza żadnego rynku** (S&P +9 pb, p = 0,39; Chiny +6 pb, p = 0,71). Rynek wycenia cła z wyprzedzeniem, a dzień, w którym cła zaczynają obowiązywać, nie wnosi nowej informacji.

![](fig_hip_2_regresja.png)

![](fig_hip_3_epizody.png)

![](fig_hip_4_etapy.png)

4. **W kluczowych epizodach kierunek ma znaczenie, ale tylko na chińskich ETF-ach:**
   - **Sygnał Trumpa:** +90 pb w oczekiwanym kierunku (p = 0,05, trafnie w 67% epizodów).
   - **Oficjalne ogłoszenie:** +28 pb (p = 0,48).
   - **Wejście w życie:** +11 pb (p = 0,85).
   - **Ogłoszenie, gdy Trump powiedział to już wcześniej:** −23 pb (p = 0,69).
5. **W 2019 r. Trump często był pierwszy, a nie tylko „nagłaśniał”.** Tak było 5.05.2019 (podwyżka do 25%: Chiny −450 pb, S&P −210 pb), 1.08.2019 (10% na 300 mld: −225 / −163 pb) i 24.02.2019 (odroczenie: +190 pb).
   Oficjalny dokument przychodził 4–19 dni później i rynek już go nie zauważał.
   W 2018 r. pierwsze były zwykle działania USTR (np. memorandum 22.03.2018: S&P −469 pb, Chiny −196 pb), ale ich kierunek był mniej spójny.
6. **Na S&P 500 w kronice żadna grupa nie jest istotna** (wszystkie p > 0,3). Ruch indeksu w tych dniach zależy też od innych czynników (Fed, wyniki spółek).

![](fig_hip_5_os_czasu.png)

## Wniosek

Hipoteza **„rynek reaguje na oficjalne cła, Trump tylko nagłaśnia” jest prawdziwa tylko częściowo:**
- **Prawdziwa** dla rutynowego strumienia tweetów: zwykłe, nawet częste tweety o cłach nie poruszają rynku, a oficjalne ogłoszenia lekko zwiększają ruch S&P 500.
- **Wejście ceł w życie nie ma znaczenia.** Liczy się zapowiedź, a nie wdrożenie.
- **Nieprawdziwa** dla tweetów, które *same były zapowiedzią decyzji* (2019). Wtedy to tweet był nową informacją, chińskie ETF-y reagowały na niego, a późniejszy dokument był już wyceniony.

Przyjmujemy więc hipotezę przeformułowaną: **rynek reaguje na pierwszą wiarygodną informację o zmianie ceł, niezależnie od kanału.**
W 2018 r. był nią zwykle komunikat USTR, w 2019 r. często tweet Trumpa. Dane są z nią spójne:
- ruch rynku pojawia się przy pierwszej informacji, a nie przy potwierdzeniu;
- tweety, które nie niosą nowej decyzji (zdecydowana większość), rynku nie ruszają;
- wejście ceł w życie jest neutralne, bo zostało wycenione wcześniej.

Kolejny krok to test tej hipotezy wprost: oznaczyć w kronice, która informacja była „nowa i wiarygodna” (zapowiedź konkretnej decyzji z datą i stawką), i porównać ją z resztą komunikatów z obu kanałów.

## Ograniczenia

- Mała próba w kronice (7–17 epizodów na grupę), szerokie przedziały ufności. Wynik +90 pb jest na granicy istotności (p = 0,05, CI od +1 do +185 pb).
- Dane dzienne: nie da się rozdzielić reakcji na tweet i ogłoszenie, gdy oba wypadają w tej samej sesji (2 epizody).
- Kronikę ułożyłem ręcznie. Daty tweetów są sprawdzone w danych, ale godziny części oficjalnych komunikatów są przybliżone, a daty z Federal Register to daty publikacji (mogą być późniejsze niż komunikat USTR).
- Lista oficjalnych zdarzeń z GTA i FR jest szeroka i zawiera też zdarzenia mało istotne, co może zaniżać efekt w analizie A.
