# Überraschende Befunde: Was das Tweet-Korpus tatsächlich enthält

Zusammenfassung der explorativen Untersuchungen vom 19. August 2026, die im Anschluss an das
Evaluationsaudit durchgeführt wurden (siehe `evaluation-diagnosis.de.md`). Das Audit hat
festgestellt, was die publizierten Modelle nicht messen. Dieses Dokument hält fest, was die Daten
stattdessen enthalten - Befunde, nach denen in der Dissertation nicht gefragt wurde und die die
meisten Leser nicht erwarten würden.

Die Zahlen stammen aus den Skripten in `tweetsCompanyNumbersPrediction/src/probes/`, den gespeicherten
Ergebnisartefakten und den am 21. August 2026 ergänzten Kontrollläufen auf den archivierten
gelabelten Dataframes. Die Probes bilden Gruppen aus zehn aufeinanderfolgenden Tweets ohne Labels
zur Gruppenbildung und verwenden, sofern nicht anders angegeben, TF-IDF-Merkmale und einen linearen
Klassifikator. Dieses Dokument trennt reproduzierte Messwerte, plausible Mechanismen und noch zu
prüfende Hypothesen ausdrücklich voneinander.

**Hinweis zur Erstellung:** Die ursprünglichen Probes und die erste Zusammenfassung wurden mit
Claude Fable 5 erstellt. ChatGPT 5.6 Sol prüfte anschließend Dissertation, Code und Messwerte,
reproduzierte die Kernexperimente und ergänzte strengere Cross-Company-Kontrollen. Alle lokalen
Analysen liefen direkt auf den Repository-Daten.

---

## Warum diese Befunde überraschend sind, in einfachen Worten

**Tweets sind eine Uhr.** Zehn Beiträge tragen genug kleine Spuren - einen Produktnamen, einen Monat,
einen Kurs, ein Ereignis, den Absender - um sie in einem gemischten Split, der Beispiele aus jeder
bekannten Woche in Training und Test enthält, in zwei von drei Fällen der richtigen von 262 Wochen
zuzuordnen. Das ist keine Datierung einer noch nie gesehenen Zukunftswoche, aber ein ungewöhnlich
starker zeitlicher Fingerabdruck.

**Die Uhr funktioniert über Unternehmen hinweg.** Ein Teil des ursprünglichen Transfers stammt von
Tweets, die in mehreren Unternehmensdateien identisch vorkommen. Nach Entfernung aller solcher
Tweet-IDs und mit einem nur auf dem Quellunternehmen angepassten Vokabular erreicht Apple auf Amazon
aber weiterhin 51,0% Quartalsgenauigkeit. Das "Wann" ist damit nicht nur eine Eigenschaft eines
Unternehmens, sondern auch des gemeinsamen Markt- und Plattformdiskurses.

**Das Korpus enthält ein gemeinsames Informationsbroker-Netz.** Ein Prozent der Accounts schreibt
mehr als die Hälfte aller Beiträge. Nach Entfernung identischer Cross-Company-Tweets stammen noch
60,3% der Apple-, 86,4% der Amazon- und 68,8% der Tesla-Beiträge von 14.143 Autoren, die über alle
drei Unternehmen schreiben. Viele hochaktive Quellen wirken wie Nachrichten-, Trading- oder
Feed-Accounts; eine formale Bot-Klassifikation wurde jedoch nicht durchgeführt.

**Der Kalender dominiert ehrliche Zukunftstests.** Auf demselben Q+1-Walk-forward-Test erreicht eine
Regel ohne Text für Apple 83,2% Accuracy/MCC 0,745 und für Amazon 79,9%/0,735; das lineare Textmodell
erreicht nur 40,2%/0,001 beziehungsweise 23,9%/-0,106. Diese Zahlen sind wegen des anderen Protokolls
nicht direkt mit den historischen 87%/0,77 gleichzusetzen, zeigen aber die Stärke der Saisonbaseline.

**Ein lineares Vollvokabular-Modell gewinnt selbst verkündete Ergebnisse nicht stabil zurück.** Bei
Apple steigt die Accuracy für das Vorquartalslabel um die Ergebniswoche herum sichtbar an, bleibt
aber insgesamt schwach. Das zeigt eine Informationsspur, nicht ihre Abwesenheit. Stärkere
Repräsentationen, bessere zeitliche Ausrichtung und gezielte Zahlenextraktion bleiben offen.

**Bei Apple schlagen zweiundvierzig Kalenderwörter das Vollvokabular.** Monats- und Saisonwörter
erreichen im Walk-forward-Test MCC +0,291, während das Vollvokabular negativ bleibt. Bei Amazon gilt
dies nur für Gruppen, die tatsächlich ein Kalenderwort enthalten; insgesamt liegt der MCC dort fast
bei null. Der Befund ist stark, aber nicht universell.

**Falsch mit System.** Außerhalb der Stichprobe waren die Modelle nicht bloß nutzlos, sie waren
schlechter als Raten. Der Grund ist fast mechanisch: Ein Modell, das ein Quartal nie gesehen hat,
greift zum ähnlichsten, und das ist das unmittelbar vorangehende - aber die Finanzlabels kippen
von einem Quartal zum nächsten, also ist "wie beim letzten Mal" zuverlässig falsch. Eine Münze
hätte es besser gemacht.

**Das beste Experiment kann auf vier Zeitfenster kollabiert sein.** Wurde der im Repository sichtbare
`EqualClassSampler` im publizierten Lauf wie dokumentiert benutzt, reduzierte er den Apple-Pool fast
vollständig auf vier Fenster in 2015 und 2016. Ohne historisches Run-Manifest ist dieser Mechanismus
sehr plausibel, aber nicht endgültig beweisbar.

**Der spannendste bisher gefundene Zukunftskanal sind typisierte Zahlen.** Bei Tesla tragen in Tweets
zitierte Auslieferungsschätzungen einen explorativen Zusatznutzen. Das ist noch nicht bestätigt
(gepaarter exakter Test p = 0,125), weist aber auf eine konkrete Architektur hin: Kennzahl, Einheit,
Bezugszeitraum, Schätzung/Istwert und Veröffentlichungszeitpunkt statt bloßer Wortvektoren.

---

## 1. Tweets datieren sich selbst - bis auf die Woche genau

Zehn aufeinanderfolgende Tweets über Apple lassen sich ohne jede Labelinformation ihrem Zeitraum
zuordnen:

| Auflösung | Klassen | Accuracy | Zufall |
| --- | ---: | ---: | ---: |
| Jahr | 5 | 0,943 | 0,200 |
| Quartal | 20 | 0,860 | 0,050 |
| Monat | 60 | 0,807 | 0,017 |
| ISO-Woche | 262 | **0,666** | 0,004 |

In zwei von drei Fällen lassen sich zehn Tweets der richtigen von 262 bereits im Training
repräsentierten Wochen zuordnen. Das belegt Periodenerkennung, nicht Generalisierung auf eine neue
Woche. Sie ist ein plausibler Hauptmechanismus hoher gemischter Ergebnisse: Das Finanzlabel ist
innerhalb eines Quartals konstant, der Text identifiziert den Zeitraum, und die publizierte
Auswertung teilte Quartale zwischen Training und Test. Welcher Anteil eines konkreten historischen
Scores daraus stammt, ist ohne dessen Einzelvorhersagen nicht bestimmbar.

Die Uhr steckt nicht nur in den aktivsten Quellen und Wiederholungen. Nach Entfernung des obersten
Prozents der Accounts und Deduplizierung - gemeinsam werden rund 60% aller Tweets entfernt - liegt
die Accuracy immer noch bei 0,817 für das Quartal und 0,648 für die Woche. Da diese Kontrolle
Aktivität, Autoridentität und Duplikate gleichzeitig verändert, isoliert sie keinen reinen Botanteil.

## 2. Die Uhr überträgt sich zwischen Unternehmen

Ein Quartalsklassifikator, der nur auf Apple-Tweets trainiert wurde, datiert Amazon-Tweets:

| Trainiert auf | Apple | Amazon | Tesla |
| --- | ---: | ---: | ---: |
| Apple | 0,856 | **0,664** | 0,463 |
| Amazon | 0,604 | 0,889 | 0,477 |
| Tesla | 0,494 | 0,561 | 0,818 |

Zufall bei gleichverteilten Klassen ist 0,05. Apple auf Amazon: exaktes Quartal 66%, innerhalb eines
Quartals 82%. Dieser erste Test passte das gemeinsame Vokabular allerdings auf alle Unternehmen an,
und 8,6-17,0% der Tweet-IDs überschneiden sich je Unternehmenspaar. Er belegt daher noch keinen
sauberen Transfer.

Eine strengere Kontrolle entfernte jede Tweet-ID, die in mehr als einer Unternehmensdatei vorkommt,
und passte das Vokabular ausschließlich auf dem Quellunternehmen an:

| Trainiert auf | Apple | Amazon | Tesla |
| --- | ---: | ---: | ---: |
| Apple | - | **0,510** | 0,311 |
| Amazon | **0,489** | - | 0,344 |
| Tesla | 0,360 | **0,423** | - |

Damit bleibt ein starker gemeinsamer Plattform- und Marktzustand bestehen. Nach Entfernung der
Cross-Company-Duplikate stammen 60,3% der Apple-, 86,4% der Amazon- und 68,8% der Tesla-Tweets von
14.143 Autoren, die über alle drei Unternehmen schreiben. Entfernt man zusätzlich jeden Autor, der
in mehreren Unternehmensmengen vorkommt, bleiben einzelne Richtungen über der jeweiligen
Mehrheitsbaseline: Apple auf Amazon 0,204 statt 0,083, Amazon auf Apple 0,166 statt 0,112 und Tesla
auf Amazon 0,206 statt 0,083. Andere Richtungen liegen näher an der Baseline. Das spricht für zwei
Schichten: ein dominantes gemeinsames Informationsbroker-Netz und eine schwächere allgemeine
Epochensprache.

Was die Uhr im Rohtext trägt: explizite Daten und Monatsnamen, Aktienkursniveaus (`114`, `117`,
`172`, `210` - der Kurs ist ein Zeitstempel), Produkteinführungen (`iphone6s`, `pixel`,
`homepod`, `iphone11`), Ereignisse (`blackmonday`, `election`, `trump`) und mitgenannte Ticker, die
in einer bestimmten Saison in Mode waren (`nflx`, `bynd`, `roku`). Eine historische
Quartalserkennungsrate von 91,8% wird berichtet, ist mit den vorhandenen Artefakten aber nicht
reproduzierbar. Das aktuelle Diagnoseprogramm, das sein TF-IDF-Vokabular nur auf Trainingsgruppen
anpasst, erreicht mit Seed 1337 eine Accuracy von 0,730. Die Uhr bleibt stark; 91,8% dürfen nicht als
aktuell bestätigter Wert erscheinen.

## 3. Das Korpus wird von wenigen professionellen Quellen dominiert

| Unternehmen | Autoren | Top-10-Accounts schreiben | Oberstes Prozent schreibt |
| --- | ---: | ---: | ---: |
| Apple | 89.120 | 22,9% | **67,3%** |
| Amazon | 42.512 | 14,1% | 54,4% |
| Tesla | 46.563 | 9,7% | 54,4% |

Zu den Top-Accounts gehören `_peripherals` und `computer_hware` (je rund 91.000 Tweets),
`MacHashNews`, `PortfolioBuzz`, `retail_Dbt`, `ExactOptionPick` und `TradingGuru`; mehrere posten über
verschiedene Unternehmen. Die Dissertation stellt selbst fest, dass ihre zehn häufigsten Quellen
Finanznachrichten-Accounts und keine typischen Einzelpersonen sind. Die neue Erkenntnis ist daher
nicht deren bloße Existenz, sondern die extreme Konzentration und das unternehmensübergreifende
Netz. Ob einzelne Accounts Bots, teilautomatisierte Feeds oder manuell betriebene professionelle
Quellen sind, wurde nicht klassifiziert.

Die eigenen Interpretationsergebnisse der Dissertation zeigen das bereits. Zu den für Apple
berichteten "wichtigsten Wörtern" gehören `cultofmac`, `DeidreZune` und `TechCrunch` -
Account-Handles. Integrated Gradients zeigt damit, dass die Modellentscheidung auf Quellenmarker
reagieren kann. Die Attribution erklärt, worauf das Modell reagiert hat; sie beweist weder, dass der
Account selbst die Finanzänderung erklärt, noch dass die anschließend formulierte ökonomische
Erzählung kausal ist.

## 4. Ein lineares Textmodell gewinnt selbst verkündete Ergebnisse nicht stabil zurück

Eine Kennzahl wird typischerweise nach Ende ihres wirtschaftlichen Quartals verkündet. Tweets aus
Quartal Q können daher das Ergebnis von Q-1 diskutieren. Als Negativkontrolle wurde geprüft, ob ein
lineares TF-IDF-Modell dieses bereits öffentliche Label im Walk-forward-Setting zurückgewinnt:

| Ziel | Apple | Amazon | Tesla |
| --- | ---: | ---: | ---: |
| Eigenes Quartal (die Aufgabe der Dissertation) | -0,042 | -0,123 | -0,011 |
| Vorquartal - bereits verkündet und diskutiert | **-0,134** | **-0,083** | **-0,044** |
| Folgequartal (echte Prognose) | -0,004 | -0,111 | -0,018 |

Werte sind MCC. Das geprüfte Bag-of-Words-Modell gewinnt das öffentliche Vorquartalslabel nicht
stabil zurück. Eine Spur bleibt: Apples Accuracy auf dem
Vorquartalslabel steigt von 0,21 in Woche 1 des Quartals auf 0,31 in Woche 5 - Apples
Ergebniswoche - und auf etwa 0,34 spät im Quartal, während die Accuracy auf dem eigenen Quartal
flach bleibt. Die Verkündung gelangt also in den Text, wird durch diese Repräsentation aber nur
schwach extrahiert. Da der TF-IDF-Vektorisierer vor den Folds auf dem gesamten Zeitraum angepasst
wurde und nur eine Modellfamilie geprüft ist, darf daraus keine allgemeine Unmöglichkeit abgeleitet
werden.

## 5. Warum die geprüften Werte außerhalb des Zeitraums negativ werden

Die Vorhersagen des Modells mit vollem Vokabular stimmen außerhalb des Trainingszeitraums in 72%
der Fälle (Apple) bzw. 71% (Amazon) mit dem Label des **Vorquartals** überein, mit der Wahrheit
aber nur in 38% bzw. 21%. Außerhalb des Zeitraums ist das Modell ein Persistenzprognostiker:
Ungesehene Quartale ähneln am meisten dem unmittelbar vorangehenden Quartal, also liefert das
Modell dessen Label.

Die Labels alternieren jedoch annähernd: Apple `0 0 2 3 | 0 0 2 3 | ...`, Amazon
`3 0 1 1 | 3 0 1 1 | ...`.
Benachbarte Quartale teilen ihr Label nur in 32% (Apple) bzw. 21% (Amazon) der Fälle. Ein
Persistenzprognostiker auf einem alternierenden Ziel liegt systematisch falsch: MCC -0,146 und
-0,176, schlechter als Zufall. Das liefert einen konkreten Mechanismus für die negativen Werte der
geprüften linearen Modelle. Als allgemeine Hypothese folgt daraus: Auf saisonal alternierenden
Zielen kann ein auf driftendem Text trainierter Epochenähnlichkeitsklassifikator außerhalb der
Stichprobe anti-prädiktiv werden.

## 6. Kalenderwörter sind der stabilste geprüfte übertragbare Textinhalt

Vokabular-Ablation im Walk-forward-Setting:

| Vokabular (Apple) | MCC außerhalb des Zeitraums |
| --- | ---: |
| Voll (rund 60.000 Tokens) | -0,03 bis -0,14 |
| Voll ohne Kalender-Tokens | -0,13 bis -0,17 |
| Nur Kalender-Tokens | **+0,20** |

Das Entfernen der Kalenderwörter macht das Apple-Modell schlechter; Kalenderwörter allein liefern
dort den einzigen positiven Wert der geprüften Ablation. Ein festes Vokabular aus nur 42 Monats-
und Saisonwörtern erreicht bei Apple insgesamt MCC +0,29 und **+0,43 auf den 45% der Gruppen, die
ein solches Wort nennen**. Bei Amazon beträgt der Gesamt-MCC nur +0,005; auf Gruppen mit
Kalenderwort steigt er auf +0,34, auf den übrigen fällt er auf -0,37. Für Apple schlagen 42 feste
Wörter damit das rund 60.000 Tokens große Vollvokabular. Selbst dort erreicht das Modell nur einen
Teil der reinen Saisonbaseline (MCC 0,76), weil 55% der Gruppen kein Kalenderwort nennen.

## 7. Auf Zentroid-Ebene driftet Sprache gleichmäßig ohne saisonale Spitze

Kosinus-Ähnlichkeit zwischen den quartalsweisen TF-IDF-Zentroiden:

| Abstand (Quartale) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Apple | 0,894 | 0,834 | 0,789 | 0,755 | 0,729 | 0,711 | 0,693 | 0,678 |
| Amazon | 0,904 | 0,852 | 0,815 | 0,782 | 0,754 | 0,735 | 0,721 | 0,716 |
| Tesla | 0,924 | 0,882 | 0,859 | 0,841 | 0,814 | 0,794 | 0,780 | 0,768 |

Der Abfall ist monoton, etwa 2-3 Punkte pro Quartal, bei Tesla am langsamsten. Sprache aus demselben
Kalenderquartal ein Jahr später (Abstand 4) ist **nicht** ähnlicher als Sprache drei Quartale
entfernt: Es gibt auf Zentroid-Ebene keine zusätzliche saisonale Spitze. Die Uhr ist in dieser
groben Darstellung überwiegend monoton: Text verrät stark *wann*; eine Wiederkehr der gesamten
Themensprache nach vier Quartalen ist in diesem Test nicht sichtbar. Das schließt schwächere
saisonale Teilthemen nicht aus.

## 8. Die dokumentierte Balancierung kann das Vorzeigeexperiment auf vier Fenster reduzieren

`EqualClassSampler` behält die ersten n Tweets je Klasse, wobei n die Größe der kleinsten Klasse
ist. Auf die zeitlich sortierten Apple-4-Klassen-Daten angewendet (n = 93.686):

| Klasse | Inhalt des balancierten Pools |
| --- | --- |
| 0 "decrease" | 100% Tweets aus 2015Q1 |
| 1 "small increase" | 100% aus 2015Q3 |
| 2 "moderate increase" | 100% aus 2016Q3 |
| 3 "strong increase" | 72% aus 2015Q4, 28% aus 2016Q4 |

Falls der publizierte Apple-Lauf diesen Sampler wie auf Seite 114 beschrieben verwendete, maß das
beste Ergebnis (Apple@10, vier Klassen, Accuracy 0,85, MCC 0,80) im Wesentlichen die Aufgabe
"unterscheide 2015Q1, 2015Q3, 2016Q3 und die Weihnachtssaisons 2015/2016"; der Pool enthielt dann
praktisch keinen Tweet nach 2016Q4. Das Repository enthält keine andere sichtbare
Balancierungsimplementierung. Ohne historisches Run-Manifest, exakten Commit und damaligen
Zwischenartefakt bleibt die Zuordnung sehr plausibel, aber bedingt.

## 9. Das neue Hybridmodell, zerlegt: Kalender plus zitierte Zahlen

Das Hybridmodell auf diesem Branch (`trainNumericTextSignalQuarterModel.py`) berichtet 80,56%
Accuracy und MCC 0,7387 auf 36 späteren Unternehmensquartalen. Der Lauf reproduziert sich auf vier
Nachkommastellen. Je Unternehmen richtig von 12:

| Zweig | Amazon | Apple | Tesla | Gesamt |
| --- | ---: | ---: | ---: | ---: |
| Nur saisonaler Prior | 10 | 10 | 2 | 22/36 |
| + numerischer Text (50/50) | 10 | 11 | 4 | 25/36 |
| + Tesla-Forward-Level-Zweig | 10 | 11 | 6 | 27/36 |
| + Tesla-Konfliktgate (post hoc) | 10 | 11 | **8** | **29/36** |
| + Gate, mit gemischtem Text | 10 | 11 | 4 | 25/36 |

Von den sieben gegenüber dem Kalender gewonnenen Quartalen sind sechs Tesla, zwei davon stammen
aus einem Gate, das nach Sichtung der Testjahre eingestellt wurde, und Amazon gewinnt keines. In der
Richtung (Anstieg gegen Rückgang) erreicht der saisonale Prior allein 0,9167 / 0,8003 - identisch
mit dem vollen Hybrid. Der allgemeine Hybrid mit gemischtem Text (MCC 0,5473) schlägt knapp
denselben Hybrid mit echtem Text (0,5466).

Die ehrliche Kurzbeschreibung lautet: der Kalender für Apple und Amazon, in Tweets zitierte
Auslieferungszahlen für Tesla, plus ein von Hand eingestelltes Gate. Das Mischen der Textmerkmale
senkt Tesla von 8 auf 4 richtige Quartale; gegenüber der gepaarten Mischkontrolle gibt es vier nur
vom echten Text korrekt gelöste Fälle und keinen umgekehrten Fall. Der exakte zweiseitige Test ergibt
jedoch p = 0,125, und das Gate wurde nach Sichtung der Testjahre entworfen. Das Tesla-Ergebnis ist
deshalb ein vielversprechender, aber unbestätigter Kandidat.

Seine Natur ist wissenschaftlich besonders interessant: Tesla-Auslieferungen sind eine Kennzahl,
die vor der Veröffentlichung numerisch geschätzt wird. Tweets können Analystenschätzungen, Guidance
oder andere vorab zirkulierende Werte weitertragen. Der Text ist dann nicht bloß Stimmung, sondern
ein Transportkanal strukturierter numerischer Information. Ob dieser Kanal über einen eingefrorenen
Holdout hinweg zusätzlichen Nutzen gegenüber dem Analystenkonsens liefert, ist die entscheidende
offene Frage.

### Zielzeitpunkt: Amazon ist anders ausgerichtet

Die Amazon-Finanz-CSV ist gegenüber dem wirtschaftlichen Berichtsquartal um ein Quartal nach vorn
verschoben: 29,33 Mrd. USD aus Amazons Q4/2014 stehen im Datensatz als 2015Q1; 22,72 Mrd. USD aus
Q1/2015 als 2015Q2. Die Dissertation beschreibt entsprechend, dass die als Q1 geführten Höchstwerte
aus dem Weihnachtsgeschäft des Vorquartals stammen. Dadurch können Amazon-Tweets im sogenannten
Zielquartal die Veröffentlichung des Zielwerts bereits enthalten. Diese Aufgabe ist teilweise
Report-Reaktion beziehungsweise Rekonstruktion, nicht ausschließlich Prognose.

Künftige Auswertungen müssen daher je Ziel drei Zeiten getrennt speichern: wirtschaftliches
Berichtsquartal, Veröffentlichungszeitpunkt und erlaubter Forecast-Cutoff. Das Target kann weiterhin
die Quartalszahl bleiben; nur die Informationsmenge vor der Vorhersage wird korrekt begrenzt.

## 10. Was die ursprünglichen Modelle aus den Wortvektoren gelernt haben

Die Korrelation zwischen Embedding-Inhalt und Label war im gemischten Split real und stark. Als
Mechanismusskizze, nicht als mathematische Korrelationsidentität, lässt sie sich so darstellen:

```text
Embedding-Inhalt -> Zeitraum -> im Datensatz festes Quartalslabel
```

Die erste Verbindung ist groß - Text stempelt sich innerhalb bekannter Perioden fein. Die zweite
ist innerhalb der Stichprobe deterministisch - ein Quartal, ein Label. Die trainierbare
Embedding-Tabelle (63 Millionen
Parameter bei Apple, gegenüber 3,8 Millionen im LSTM) diente als Nachschlagewerk von
Zeitraummarkern zu Labels, mit der bereits in den vortrainierten Vektoren latenten Epochenstruktur
als möglichem Vorsprung. Außerhalb des Zeitraums kann diese Zuordnung brechen; die beobachtete
Annäherung an die vorige Epoche und die alternierenden Labels erklären dann die Anti-Prädiktion der
linearen Kontrollen. Zusätzlich wurden die historischen Word2Vec-Repräsentationen vor der
Kreuzvalidierung auf dem Vollkorpus trainiert und waren damit transduktiv.

### Was der dritte Dissertationsschritt tatsächlich leisten kann

Die Topic-/Important-Word-Stufe ist wissenschaftlich wertvoll, aber anders als ursprünglich
gedeutet. Viele manuelle Topics - Brexit, Wahl, COVID-19 oder Hongkong-Proteste - sind einmalige
Zeitanker. Ein Topic-zu-Klasse-Zusammenhang kann deshalb erneut `Topic -> Zeitraum -> Quartalslabel`
statt einen wirtschaftlichen Wirkungskanal abbilden. Integrated Gradients erklärt zudem die
Vorhersage des Modells, nicht die Ursache der realen Kennzahl.

Dokument und Implementierung stimmen auch bei der Auswahl nicht vollständig überein: Die
Dissertation beschreibt zehn besonders ähnliche Tweetgruppen; der sichtbare Code sucht zehn
ähnliche gelernte Topics, übernimmt alle ihnen zugeordneten Dokumente und bildet danach
labelabhängige Gruppen. Einzelne narrative Erklärungen sind rückblickend konstruiert; besonders
deutlich ist die Deutung von Tweets aus dem bis 2020 reichenden Korpus über die erst 2021 entstandene
`#AppleToo`-Bewegung.

Positiv neu gerahmt kann dieser Schritt einen dynamischen Ereignis- und Quellenatlas liefern:
past-only Topics je Quartal, Topic-Prävalenz über die Zeit, getrennte professionelle und individuelle
Quellen sowie Important Words als Beschreibung der Modellreaktion. Erst eine anschließend auf
unberührten Quartalen geprüfte Änderung der Topic-Prävalenz darf als Forecast-Kandidat gelten.

## 11. Was nicht behauptet wird

- Nicht, dass Tweets keine Information über Unternehmen enthalten. Belegt ist nur, dass die
  geprüften linearen Vollvokabular-Modelle unter Walk-forward-Evaluation keine stabile finanzielle
  Richtung extrahierten. Die Apple-Ergebniswochenspur und das Tesla-Zahlensignal sprechen gerade
  gegen eine allgemeine Informationslosigkeit.
- Nicht, dass das Tesla-Signal bestätigt ist. Es beruht auf 4-6 Quartalen, p = 0,125 gegen die
  Mischkontrolle, mit einem auf den Testjahren entworfenen Gate.
- Nicht, dass die Datierungsgenauigkeiten Obergrenzen sind. Stärkere Modelle würden genauer
  datieren.
- Nicht, dass die aktivsten Accounts nachweislich Bots sind. Belegt sind Konzentration,
  Quellenüberschneidung, Nachrichten-/Feed-Charakter vieler Top-Accounts und Duplikate.
- Mehrere TF-IDF-Probes passten Vokabular und IDF vorab auf dem Vollzeitraum an. Die 42 festen
  Kalenderwörter sind davon nicht betroffen; exakte Vollvokabular-Walk-forward-Zahlen sind jedoch
  transduktiv und sollten mit foldweisem Fit wiederholt werden.
- Die historische Quartalserkennungsrate von 91,8% ist berichtet, aber nicht reproduziert. Das
  aktuelle trainingsseitig angepasste Diagnoseprogramm erreicht 73,0% bei Seed 1337.

## 12. Wozu diese Befunde taugen

1. **Ein methodisches Ergebnis für das Feld.** Social-Media-Text stempelt sich selbst so fein, dass
   ein zeitveränderliches Label bei zufälligen zeitgemischten Splits aus Text vorhersagbar
   erscheinen kann, selbst wenn der Text keinen kausalen Inhalt zum Ziel enthält. Zufällige
   Splits über die Zeit lassen die Zeit durchsickern. Dieses Projekt ist eine ungewöhnlich saubere,
   vollständig quantifizierte Fallstudie.
2. **Eine Neurahmung des Interpretationskapitels.** Die Important-Word- und Topic-Pipeline ist eine
   diskriminative Schlüsselwortanalyse nach Zeitraum. Umbenannt in "für jedes Fenster
   charakteristische Begriffe", quellenstratifiziert und je Quartal statt je Klasse
   berechnet, überleben ihre qualitativen Beobachtungen; als "Treiber finanzieller Veränderung"
   nicht.
3. **Eine Richtung für eine konfirmatorische Studie.** Der bisher explorativ positive Kanal -
   öffentlich vor der Veröffentlichung zitierte Zahlen - verweist auf Kennzahlen, die öffentlich
   numerisch geschätzt werden. Das ist eine prüfbare, aber noch unbestätigte Hypothese mit
   eingefrorenen Merkmalen und einem unberührten Holdout.
4. **Ein neuer Forschungsgegenstand: das Informationsbroker-Netz.** Die hohe Autorüberschneidung
   zeigt, dass das Korpus eine gemeinsame marktweite Informationsinfrastruktur abbildet. Daraus
   lassen sich Quellenzuverlässigkeit, Lead-Lag-Beziehungen, Nachrichtenweitergabe und
   unternehmensübergreifende Regime untersuchen - unabhängig davon, ob ein Quartalsforecast gelingt.
5. **Ein positives Architekturprinzip.** Der starke Zeitencoder muss nicht verworfen werden. Er kann
   als expliziter Störgrößenzweig modelliert werden, während ein zweiter, gegenüber Zeit und Quelle
   möglichst invariantes Residuum die Quartalszahl vorhersagt. Erst dessen Zusatznutzen wäre ein
   glaubwürdiges finanzielles Textsignal.

## 13. Vorgeschlagene Richtungen für weitere Studien

Jede Richtung folgt aus einem der obigen Befunde. Zeit-, Quellen-, Topic- und
Repräsentationskontrollen sind mit den vorhandenen Daten möglich. Exakte Release-Zeitpunkte,
Analystenkonsens oder zusätzliche Unternehmen erfordern ergänzende Metadaten beziehungsweise neue
Finanzreihen; das Target bleibt in allen Forecast-Varianten eine Quartalskennzahl.

1. **Das Zeit-Leakage-Audit als publizierte Methode.** Die Datierungsuntersuchungen zu einem
   allgemeinen Test machen: Bevor einer Studie "Text sagt X vorher" geglaubt wird, sind zu
   berichten (a) wie genau sich der Text selbst datiert, (b) wie gut eine reine Datums- oder
   Saisonregel X vorhersagt und (c) das Ergebnis unter zeitraumgruppierten Splits. Rückwirkend auf
   publizierte Social-Media-Vorhersagestudien mit zufälligen Splits anwenden. Dieses Projekt
   liefert das durchgerechnete Beispiel und die Zahlen. (Befunde 1, 4, 5, 6)

2. **Eine quellenstratifizierte Replikation.** Accounts als automatisiert, professionell oder
   individuell klassifizieren (Postvolumen, Duplikatrate, Vorlagenregelmäßigkeit, Regelmäßigkeit der
   Postzeiten) und dann jede Analyse allein auf der menschlichen Teilmenge wiederholen. Zwei Fragen:
   Erscheint irgendein Textsignal, sobald die Feeds weg sind, und wie viel von dem, was die
   Literatur "öffentliche Stimmung" über Unternehmen nennt, ist Feed-Ausgabe? Der Korpusanteil des
   obersten Prozents der Accounts sollte in jeder Arbeit berichtet werden, die diesen Datensatz
   nutzt. (Befund 3)

3. **Das Informationsbroker-Netz modellieren.** Die 14.143 über Apple, Amazon und Tesla aktiven
   Autoren als Graph untersuchen: Wer veröffentlicht eine Kennzahl oder Schätzung zuerst, wer
   übernimmt sie, wie stabil ist die Quelle, und wie schnell wandert Information zwischen
   Unternehmen? Identische Tweet-IDs müssen dabei vor jedem Cross-Company-Test entfernt werden.
   Ein Autor-Holdout prüft, ob ein Signal auf neue Quellen überträgt. (Befunde 2, 3)

4. **Die Quartale sauber beschreiben.** Den neuronalen Important-Word-Pfad durch eine
   Schlüsselwortstatistik je Quartal ersetzen (zum Beispiel gewichtete Log-Odds mit Prior) auf der
   menschlichen Teilmenge, plus Topic-Prävalenz über die zwanzig Quartale. Das liefert die
   Beschreibung "was wurde wann besprochen", die das Interpretationskapitel anstrebte, ohne den
   Umweg über 67 Millionen Parameter, und lässt sich als ausdrücklich explorative Analyse mit
   n = 20 mit den Kennzahlen korrelieren. (Befunde 3, 10)

5. **Die Zahlen-als-Träger-Hypothese konfirmatorisch prüfen.** Das Tesla-Ergebnis legt nahe, dass
   Social Media dort nützlich ist, wo die Öffentlichkeit eine Kennzahl vor ihrer Veröffentlichung
   numerisch zitiert. Solche Kennzahlen vorab wählen - Fahrzeugauslieferungen, Schätzungen von
   Stückzahlen, Abonnentenzuwächse, Kinokassen- oder Spieleverkaufszahlen, App-Downloads -, die
   Regexes, Merkmale, Fusionsgewichte und das Gate einfrieren und einmalig auf einem unberührten
   Holdout auswerten. Direkt gegen den Analystenkonsens vergleichen, um zu erfahren, ob der Text
   über den Konsens hinaus, den er weitergibt, irgendetwas beiträgt. (Befund 9)

6. **Eine Informationskurve relativ zur Veröffentlichung messen.** Tweets nicht nur nach
   Kalenderquartal, sondern nach Abstand zum Release gruppieren: 30 und 7 Tage vorher,
   Veröffentlichungstag und danach. Dadurch werden Erwartung, Bekanntgabe und Reaktion getrennt.
   Für den Forecast dürfen ausschließlich Vorveröffentlichungstexte verwendet werden. Der
   Amazon-Zeitversatz macht diese Kontrolle zwingend. (Befunde 4, 9)

7. **Ein Ziel mit mehr als zwanzig Werten.** Zwanzig Quartale je Unternehmen können höchstens
   zwanzig Fälle entscheiden. Entweder viele Unternehmen bündeln (dieselbe Pipeline über einige
   hundert Ticker ergibt Tausende Unternehmensquartale) oder zu einem feiner abgetasteten Ziel
   wechseln, etwa der Ergebnisüberraschung oder der Kursreaktion in den Tagen nach jeder
   Verkündung. Die schwache Spur in Apples Ergebniswoche ist genau die Stelle, an der ein
   Ereignisfenster-Design zuerst nachsehen würde. (Befunde 4, 8)

8. **Die Abweichung vorhersagen, nicht das Niveau.** Weil die Labels saisonal alternieren, lautet
   die informative Frage nicht "geht es hoch", sondern "geht es stärker hoch als in diesem
   Quartal üblich". Das Ziel als Residuum gegenüber der saisonalen Erwartung definieren; ein
   Textsignal muss dann die Null schlagen statt den Kalender, und die Persistenzfalle verschwindet
   konstruktionsbedingt. (Befunde 5, 6)

9. **Die Uhr als Forschungsgegenstand.** Driftraten und unternehmensübergreifende Übertragung auf
   anderen Plattformen, Themen und Sprachen messen; die Beiträge von Ereignissen,
   Plattformvokabular und automatisierten Accounts trennen; prüfen, wie genau ein Modell einen
   undatierten Beitrag datieren kann. Anwendungen: Herkunftsprüfung, Erkennung rückdatierter oder
   synthetischer Inhalte, Altersschätzung archivierter Texte. (Befunde 1, 2, 7)

10. **Zeit-/Quellenrepräsentation und Finanzresiduum trennen.** Einen gemeinsamen Encoder auf allen
    fünf Unternehmen Zeit, Ereignisse und Quelle lernen lassen. Ein adversarieller oder
    orthogonalisierter zweiter Zweig soll daraus Zeit und Autor entfernen und nur den Zusatzbeitrag
    zur saisonalen Quartalsbaseline vorhersagen. Google- und Microsoft-Tweets können dafür als
    ungelabelte Kontrolldomänen dienen; das Target der drei Forecast-Unternehmen bleibt ihre
    Quartalszahl. (Befunde 1–3, 7)

11. **Die ursprüngliche Hypothese mit einer fairen Repräsentation erneut prüfen.** Die
   Bag-of-Words- und trainierbaren Embedding-Modelle sind möglicherweise schlicht die falschen
   Instrumente. Die zeitraumgruppierte Walk-forward-Auswertung mit einem modernen Satz-Encoder auf
   der menschlichen Teilmenge und dem Residualziel wiederholen. Wenn Semantik jenseits des
   Zeitraums existiert, ist dies das Setting, in dem sie sich endlich zeigen könnte; wenn nicht,
   wird das negative Ergebnis deutlich stärker. (Befund 10)

12. **Die Aggregationsskala selbst untersuchen.** Dass N = 10 über viele Architekturen und
    Unternehmen häufig besser als 5 oder 20 ist, kann ein echtes Verhältnis von Kontext zu Rauschen
    anzeigen. Gruppen nach Tweetanzahl gegen feste Zeitfenster und längengleiche Kontrollen testen;
    erst dann ist klar, ob N = 10 eine sprachliche, zeitliche oder rein technische Eigenschaft ist.

## Reproduktion

Aus `tweetsCompanyNumbersPrediction/src`, mit dem Repository im Pfad:

| Skript | Erzeugt |
| --- | --- |
| `probes/exp_unexpected.py` | Abschnitte 1-3, die Q+1-Prognose, Tweet-Volumen |
| `probes/exp_unexpected_control.py` | die Kontrolle ohne das aktivste Prozent und ohne Duplikate sowie die Träger-Tokens der Uhr |
| `probes/exp_cross_company_network.py` | Cross-Company-ID-, Autoren- und quellenexklusive Transferkontrollen aus Abschnitt 2 |
| `probes/exp_deeper.py` | Abschnitt 4 (verzögertes Label, Zuwachs je Woche), Abschnitt 7, das Vokabular des Verkündungsmodells |
| `probes/exp_calendar.py` | Abschnitt 6, Vokabular-Ablation |
| `probes/exp_calendar2.py` | Abschnitt 6 (42-Wörter-Modell) und Abschnitt 5 (Persistenz-Echo) |
| `trainNumericTextSignalQuarterModel.py` | Abschnitt 9 (`output/numeric_text_signal_quarter_results.json`) |

Die Probes laufen auf der CPU; die großen TF-IDF-Kontrollen können mehrere Minuten benötigen. Unter
Windows sollte für `exp_deeper.py` UTF-8-Ausgabe aktiviert werden, weil einzelne Tokens nicht in
CP1252 darstellbar sind. Nur die Tweet-Volumenanalyse in `exp_unexpected.py` lädt die Finanz-CSVs;
Datierungs-, Quellen- und Vokabularbefunde verwenden sie nicht. Die Forecast-Ziele der übrigen
Probes stammen aus den bereits gelabelten Dataframes.
