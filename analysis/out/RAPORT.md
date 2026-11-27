# Tweety Trumpa o cłach a rynek akcji (2018–2019): raport z badania

Raport opisuje, jak szukaliśmy związku między tym, co Donald Trump pisał o cłach na Twitterze, a zachowaniem rynków akcji, jak po drodze zmieniała się nasza hipoteza i co z tego ostatecznie wynika. Pokazujemy tylko wyniki istotne statystycznie. Tam, gdzie efektu nie było, piszemy o tym wprost, bo brak efektu też jest informacją.

## Definicje

- **Eskalacja, deeskalacja, neutralne.** Post jest eskalacyjny, gdy zapowiada lub zaostrza cła albo grozi nimi. Jest deeskalacyjny, gdy łagodzi konflikt, na przykład odracza cła albo mówi o postępach w negocjacjach. Neutralny to post o cłach bez wyraźnego kierunku.
- **Adnotacja i gold standard.** Dwie osoby (MG i PK) niezależnie oznaczyły 167 postów: czy dotyczą ceł, w jakim kierunku i jakiego kraju. Gold standard to ostateczna etykieta posta: bierzemy etykietę MG, a gdy jej brak, etykietę PK. Wariant „zgodne” zawiera tylko posty, co do których obie osoby się zgodziły.
- **Kappa Cohena (κ).** Miara zgodności dwóch osób oznaczających, z odjętą zgodnością przypadkową: 1 to pełna zgodność, 0 to zgodność jak przy losowaniu, a poniżej 0,4 zgodność jest słaba.
- **Sesja i punkt bazowy (pb).** Sesja to jeden dzień handlu na giełdzie w Nowym Jorku. Punkt bazowy to jedna setna punktu procentowego, więc „−66 pb” oznacza zwrot niższy o 0,66 punktu procentowego.
- **Zwrot nadzwyczajny (AR) i beta.** Chińskie ETF-y notowane w USA poruszają się razem z całym rynkiem amerykańskim. Żeby zobaczyć reakcję specyficzną dla Chin, odejmujemy część zwrotu tłumaczoną ruchem S&P 500: AR = zwrot ETF − beta × zwrot S&P 500. Beta, oszacowana na danych z 2017 roku, mówi, o ile zwykle zmienia się ETF, gdy S&P 500 zmienia się o 1%.
- **Wartość p.** Prawdopodobieństwo, że bez prawdziwego efektu dostalibyśmy przez przypadek różnicę co najmniej tak dużą jak w danych. Wynik jest istotny, gdy p < 0,05.
- **Korekta na wielokrotne testowanie (FDR).** Przy setkach testów część da p < 0,05 czysto przypadkiem, więc korekta podnosi poprzeczkę. Wynik po korekcie nazywamy wartością q.
- **Moc testu.** Zdolność testu do wykrycia efektu, który naprawdę istnieje. Przy małej liczbie obserwacji test wykrywa tylko bardzo duże efekty.
- **Odwrotna przyczynowość.** Sytuacja, w której to nie post wpływa na rynek, tylko rynek wpływa na to, co i kiedy zostaje napisane.

## Historia badania

**Etap 0: pytanie wyjściowe i dane.** Projekt wystartował od pytania, czy sposób mówienia o cłach w mediach wyprzedza nastroje na rynku. Zawęziliśmy je do tweetów Trumpa z lat 2018–2019. Najpierw sprawdziliśmy, czy umiemy rzetelnie oznaczać posty. Zgodność była bardzo dobra co do tego, czy post dotyczy ceł (κ = 0,80) i jakiego kraju (κ = 0,85), ale słaba co do kierunku (κ = 0,30). Nawet ludziom trudno więc ocenić, czy post zaostrza, czy łagodzi konflikt, i ta niepewność obciąża każdą analizę opartą na kierunku.

**Etap 1. Hipoteza: „eskalacja w tweetach obniża S&P 500, a deeskalacja go podnosi”.** Porównaliśmy zachowanie S&P 500 po postach różnego typu, w oknach od 1 godziny do 5 sesji. Hipoteza się nie potwierdziła. Po postach eskalacyjnych rynek zachowywał się tak samo jak po postach niezwiązanych z cłami, a w ciągu pierwszej godziny po żadnym typie posta nie było widać reakcji. Wynik przeciwny do intuicji był taki, że to po postach deeskalacyjnych S&P 500 spadał. W ciągu 6 godzin po takich postach indeks był o około 69 pb niżej niż po postach niezwiązanych z cłami. To jeden z nielicznych wyników, które przechodzą korektę FDR (q = 0,04). Podobnie zachowywał się Nasdaq. W oknie 5 sesji różnica między eskalacją a deeskalacją wynosiła około 220 pb (p = 0,003), ale znikała po usunięciu dwóch najbardziej skrajnych dni z każdej grupy. Najprostsze wyjaśnienie to odwrotna przyczynowość: uspokajające posty („negocjacje idą świetnie”) pojawiały się w dni, kiedy rynek już spadał. Dodatkowo na całym korpusie 7 890 tweetów, ocenionych automatycznym klasyfikatorem, nie znaleźliśmy żadnej korelacji z S&P 500, ani tego samego dnia, ani z opóźnieniem do 5 dni.

**Etap 2. Hipoteza: „efekt jest, ale S&P 500 jest na niego za mało wrażliwy”.** Cła na Chiny powinny bardziej uderzać w spółki chińskie niż w cały rynek amerykański. Zbudowaliśmy więc z góry ustalony koszyk pięciu chińskich ETF-ów notowanych w Nowym Jorku i mierzyliśmy ich zwrot nadzwyczajny (AR). Dla postów dotyczących Chin kierunek okazał się zgodny z intuicją: w dniu posta eskalacyjnego AR wynosił średnio −77 pb, a deeskalacyjnego +91 pb (p = 0,02; dla etykiet zgodnych p = 0,01). Przy postach o cłach na inne kraje takiej różnicy nie było. Ten sam problem co wcześniej wrócił jednak w innej postaci. Już w pięciu sesjach przed postem eskalacyjnym chińskie ETF-y traciły średnio 132 pb, a przed deeskalacyjnym zyskiwały 56 pb (p = 0,01). Trump eskalował więc w okresach, gdy Chiny już słabły, i część „reakcji” może być po prostu kontynuacją trendu. Żaden z 648 testów w tej części nie przeszedł korekty FDR.

**Etap 3. Hipoteza: „rynek reaguje na oficjalne decyzje o cłach, a Trump je tylko nagłaśnia”.** Skoro zwykłe tweety nie ruszają rynku, sprawdziliśmy oficjalne źródła: bazę interwencji handlowych Global Trade Alert, dokumenty Federal Register i ręcznie ułożoną kronikę 22 kluczowych epizodów konfliktu USA–Chiny. W dni oficjalnych ogłoszeń S&P 500 poruszał się mocniej niż w spokojne dni (średnio 78 pb wobec 62 pb, p = 0,03). Dni, w których cła faktycznie wchodziły w życie, nie różniły się niczym od zwykłych dni, co oznacza, że rynek wyceniał cła już w chwili zapowiedzi. Kronika pokazała jednak, że w 2019 roku to Trump często był pierwszy, a dokument pojawiał się kilka do kilkunastu dni po tweecie. Na chińskich ETF-ach takie tweety dawały ruch w oczekiwanym kierunku średnio o 90 pb (p = 0,05, czyli na samej granicy istotności). Hipoteza okazała się więc prawdziwa tylko w części.

**Etap 4. Hipoteza: „rynek reaguje na pierwszą wiarygodną informację o cłach, niezależnie od tego, czy podaje ją rząd, czy Trump”.** Przetestowaliśmy ją wprost, dzieląc komunikaty na cztery grupy według tego, czy niosły nową informację i czy pochodziły od Trumpa, czy z oficjalnego źródła. Na S&P 500 wzór był dokładnie taki, jak przewiduje hipoteza: nowa informacja poruszała rynek w oczekiwanym kierunku w obu kanałach, a informacja już znana w żadnym. Istotny okazał się jednak tylko jeden wariant. Przy etykietach zgodnych efekt nowości wyniósł 97 pb (p = 0,04), a w wariancie głównym nie był istotny. Na chińskich ETF-ach nowa informacja od Trumpa działała mocniej niż oficjalna, ale bez istotności. Po korekcie FDR żaden z tych wyników nie jest istotny.

## Wnioski

Najważniejszy wniosek jest negatywny. Na danych z lat 2018–2019 nie da się wykazać, że tweety Trumpa o cłach w przewidywalny sposób zmieniają kurs S&P 500. Zdecydowana większość takich postów nie niesie nowej informacji, a rynek je ignoruje.

Jedyny wynik, który przetrwał korektę na wielokrotne testowanie, jest przeciwny do pierwotnej intuicji: po postach deeskalacyjnych rynek amerykański spadał w ciągu kilku godzin. Razem z tym, że eskalacja wobec Chin pojawiała się w okresach słabości chińskich spółek, wskazuje to raczej na wpływ rynku na Trumpa niż odwrotnie.

Hipoteza końcowa, że liczy się pierwsza wiarygodna informacja, a nie kanał, tłumaczy wszystkie obserwacje: dlaczego zwykłe tweety nie działają, dlaczego wejście ceł w życie jest obojętne i dlaczego niektóre tweety z 2019 roku poruszały rynek. Nie udało się jej jednak potwierdzić na poziomie istotności, więc jest to najlepsze dostępne wyjaśnienie, a nie udowodniony fakt.

## Ograniczenia

- **Za mała próba.** Mieliśmy około 35 postów eskalacyjnych i 12 deeskalacyjnych, a w kronice od 7 do 25 obserwacji na grupę. Test wykrywa więc dopiero różnice rzędu 90–200 pb dziennie, a realistyczne efekty są kilkukrotnie mniejsze. Brak istotności oznacza głównie brak danych, a nie dowód, że efektu nie ma.
- **Wielokrotne testowanie.** Na ponad tysiąc testów wyników z p < 0,05 było dwa do trzech razy więcej, niż daje przypadek. Nie wszystko jest więc szumem, ale pojedyncze wyniki trzeba traktować ostrożnie.
- **Słaba zgodność co do kierunku posta**, przez co wyniki zależą od tego, czyje etykiety przyjmiemy.
- **Dane dzienne, ręcznie ułożona kronika i stała beta z 2017 roku.** Tweeta i komunikatu z tego samego dnia nie da się rozdzielić, a godziny części komunikatów są przybliżone.

## Co dalej

Żeby rozstrzygnąć hipotezę o pierwszej wiarygodnej informacji, potrzeba więcej epizodów, na przykład postów z Truth Social z lat 2025–2026, oraz danych śróddziennych, które oddzielą reakcję na tweet od reakcji na komunikat.

Szczegóły: `SKROT.md` (S&P 500), `SKROT_CHINY.md` (chińskie ETF-y), `SKROT_HIPOTEZA.md` (oficjalne cła i test 2×2) oraz skrypty w katalogu `analysis/`.
