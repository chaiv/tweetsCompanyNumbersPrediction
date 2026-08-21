# Ueberraschende Befunde: Was das Tweet-Korpus tatsaechlich enthaelt

Zusammenfassung der explorativen Untersuchungen vom 19. August 2026, die im Anschluss an das
Evaluationsaudit durchgefuehrt wurden (siehe `evaluation-diagnosis.md`). Das Audit hat
festgestellt, was die publizierten Modelle nicht messen. Dieses Dokument haelt fest, was die Daten
stattdessen enthalten - Befunde, nach denen in der Dissertation nicht gefragt wurde und die die
meisten Leser nicht erwarten wuerden.

Jede Zahl unten stammt aus den Skripten in `tweetsCompanyNumbersPrediction/src/probes/`, ausgefuehrt
auf den archivierten gelabelten Dataframes (Gruppen aus 10 aufeinanderfolgenden Tweets, keine Labels
zur Gruppenbildung, TF-IDF-Merkmale und ein linearer Klassifikator, sofern nicht anders angegeben).
Nichts davon ist die Interpretation eines Modells; alles ist eine Messung.

**Hinweis zur Erstellung:** Die Untersuchungen und diese Zusammenfassung wurden mit Claude Fable 5
erstellt; die Analysen liefen direkt auf den Repository-Daten.

---

## Warum diese Befunde ueberraschend sind, in einfachen Worten

**Tweets sind eine Uhr.** Die meisten Menschen nehmen an, dass eine Handvoll kurzer Beitraege ueber
ein Unternehmen aus fast jeder Zeit stammen koennte. Das stimmt nicht. Zehn Beitraege tragen genug
kleine Spuren - einen Produktnamen, einen Monat, einen Kurs, ein Ereignis, den Absender - um sie in
zwei von drei Faellen der richtigen Woche zuzuordnen. Niemand schreibt das Datum absichtlich in seine
Tweets; es steht trotzdem darin.

**Die Uhr funktioniert ueber Unternehmen hinweg.** Man wuerde erwarten, dass Tweets ueber Apple und
Tweets ueber Amazon wenig gemeinsam haben. Doch ein Modell, das nur Apple-Tweets gesehen hat, kann
trotzdem sagen, aus welchem Quartal ein Amazon-Tweet stammt. Die Nachrichten des Tages, der Slang
des Jahres und dieselben automatisierten Accounts tauchen ueberall auf; das "Wann" ist eine
Eigenschaft der ganzen Plattform, nicht eines Unternehmens.

**Die Menge besteht ueberwiegend aus Maschinen.** Der Datensatz wird als oeffentliche Meinung ueber
fuenf Unternehmen beschrieben. Tatsaechlich hat ein Prozent der Accounts mehr als die Haelfte aller
Beitraege geschrieben, und die aktivsten Accounts sind automatisierte Nachrichten- und
Boersenalarm-Feeds. Was die Modelle ueber "die Oeffentlichkeit" gelernt haben, haben sie zum
grossen Teil von Software gelernt.

**Die publizierten Zahlen waren genau so hoch wie der Kalender allein.** Eine Regel, die gar keinen
Text liest - "dieses Quartal im Jahr laeuft meistens so" - erreicht dieselbe Accuracy und
Korrelation, die die Dissertation fuer ihre besten Modelle berichtet. Das ist kein Zufall: Die
Modelle hatten den Kalender im Text gefunden, weil der Kalender das Lauteste darin ist.

**Das Modell konnte nicht einmal Ergebnisse lesen, die schon oeffentlich waren.** Man wuerde
erwarten, dass ein Modell aus Tweets zumindest erkennen kann, dass Apple gerade ein gutes Quartal
gemeldet hat, denn genau darueber wird getwittert. Es kann es nicht, jedenfalls nicht mit dieser Art
von Modell. Das beantwortet die Frage, ob die Methode oder das Material das Problem war: Das
Material traegt die Botschaft nicht in einer Form, die ein solches Modell lesen kann.

**Zweiundvierzig Woerter schlagen 150.000.** Behaelt man nur die Namen von Monaten und Jahreszeiten,
erhaelt man fuer ungesehene Quartale einen besseren Prognostiker als mit dem gesamten Vokabular.
Alles andere im Text hilft ausserhalb des Zeitraums, in dem es gelernt wurde, nicht - es schadet
sogar, weil es das Modell am falschen Zeitraum verankert.

**Falsch mit System.** Ausserhalb der Stichprobe waren die Modelle nicht bloss nutzlos, sie waren
schlechter als Raten. Der Grund ist fast mechanisch: Ein Modell, das ein Quartal nie gesehen hat,
greift zum aehnlichsten, und das ist das unmittelbar vorangehende - aber die Finanzlabels kippen
von einem Quartal zum naechsten, also ist "wie beim letzten Mal" zuverlaessig falsch. Eine Muenze
haette es besser gemacht.

**Das beste Experiment war ein Spiel mit vier Zeitfenstern.** Der Balancierungsschritt hat das
Vorzeigeexperiment stillschweigend auf vier Fenster in 2015 und 2016 reduziert. Das Modell mit 85
Prozent wurde im Grunde gefragt: "Stammt das aus dem Fruehjahr 2015, dem Herbst 2015, dem Herbst
2016 oder aus der Weihnachtszeit?" - eine Frage, die der Text muehelos beantwortet.

**Das Einzige, was funktioniert, ist gar keine Sprache.** Das einzige echte Signal im ganzen Projekt
sind Zahlen, die Menschen in Tweets ueber Tesla-Auslieferungen zitieren - Schaetzungen und
durchgesickerte Werte, die schon existierten, bevor jemand sie twitterte. Der Text ist ein Bote, der
eine Zahl ueberbringt, keine Menge, die etwas spuert.

---

## 1. Tweets datieren sich selbst - bis auf die Woche genau

Zehn aufeinanderfolgende Tweets ueber Apple lassen sich ohne jede Labelinformation ihrem Zeitraum
zuordnen:

| Aufloesung | Klassen | Accuracy | Zufall |
| --- | ---: | ---: | ---: |
| Jahr | 5 | 0,943 | 0,200 |
| Quartal | 20 | 0,860 | 0,050 |
| Monat | 60 | 0,807 | 0,017 |
| ISO-Woche | 262 | **0,666** | 0,004 |

In zwei von drei Faellen lassen sich zehn Tweets der richtigen von 262 Wochen zuordnen. Das ist der
Mechanismus hinter jedem hohen Wert der Dissertation: Das Finanzlabel ist innerhalb eines Quartals
konstant, der Text identifiziert das Quartal, und die publizierte Auswertung teilte Quartale
zwischen Training und Test.

Die Uhr steckt in der Sprache selbst, nicht nur in automatisierten Accounts. Nach Entfernung des
obersten Prozents der Accounts und aller exakten Duplikate - was rund 60% aller Tweets entfernt -
liegt die Accuracy immer noch bei 0,817 fuer das Quartal und 0,648 fuer die Woche.

## 2. Die Uhr uebertraegt sich zwischen Unternehmen

Ein Quartalsklassifikator, der nur auf Apple-Tweets trainiert wurde, datiert Amazon-Tweets:

| Trainiert auf | Apple | Amazon | Tesla |
| --- | ---: | ---: | ---: |
| Apple | 0,856 | **0,664** | 0,463 |
| Amazon | 0,604 | 0,889 | 0,477 |
| Tesla | 0,494 | 0,561 | 0,818 |

Zufall ist 0,05. Apple auf Amazon: exaktes Quartal 66%, innerhalb eines Quartals 82%. Nach
Entfernung der Top-Accounts und Duplikate sinkt die Uebertragung auf etwa die Haelfte (Apple auf
Amazon 0,53, Amazon auf Apple 0,48), bleibt aber zehnmal ueber Zufall. Etwa die Haelfte der
unternehmensuebergreifenden Uhr ist gemeinsame Sprache; die andere Haelfte sind dieselben
automatisierten Accounts, die ueber mehrere Unternehmen posten.

Was die Uhr im Rohtext traegt: explizite Daten und Monatsnamen, Aktienkursniveaus (`114`, `117`,
`172`, `210` - der Kurs ist ein Zeitstempel), Produkteinfuehrungen (`iphone6s`, `pixel`,
`homepod`, `iphone11`), Ereignisse (`blackmonday`, `election`, `trump`) und mitgenannte Ticker, die
in einer bestimmten Saison in Mode waren (`nflx`, `bynd`, `roku`). Die Pipeline der Dissertation
entfernt Ziffern und erreicht trotzdem 91,8% Quartalsgenauigkeit; die Uhr ueberlebt also auch ohne
Kurse.

## 3. Das Korpus besteht ueberwiegend aus sprechenden Maschinen

| Unternehmen | Autoren | Top-10-Accounts schreiben | Oberstes Prozent schreibt |
| --- | ---: | ---: | ---: |
| Apple | 89.120 | 22,9% | **67,3%** |
| Amazon | 42.512 | 14,1% | 54,4% |
| Tesla | 46.563 | 9,7% | 54,4% |

Die Top-Accounts sind automatisierte Feeds: `_peripherals` und `computer_hware` (je rund 91.000
Tweets), `MacHashNews`, `PortfolioBuzz`, `retail_Dbt`, `ExactOptionPick`, `TradingGuru` - und die
letzten drei posten ueber mehrere Unternehmen. Der Datensatz wird in der Dissertation und auf Kaggle
als oeffentliche Meinung beschrieben; der Grossteil ist algorithmische Feed-Ausgabe.

Die eigenen Interpretationsergebnisse der Dissertation zeigen das bereits. Zu den fuer Apple
berichteten "wichtigsten Woertern" gehoeren `cultofmac`, `DeidreZune` und `TechCrunch` -
Account-Handles. Integrated Gradients hat korrekt berichtet, dass die Evidenz des Modells darin
bestand, wer wann gepostet hat; die narrative Deutung ("Apple-zentrierte Medien, die Innovation
betonen") wurde darauf gesetzt.

## 4. Selbst bereits verkuendete Ergebnisse lassen sich nicht aus dem Text gewinnen

Die Kennzahl eines Quartals wird im Folgequartal verkuendet (Apple und Amazon in Woche 4-5,
Tesla-Auslieferungen in den ersten Tagen). Tweets aus Quartal Q diskutieren also das Ergebnis von
Q-1. Wenn die Methode ueberhaupt Finanzinhalte extrahieren koennte, muesste sie das wiederfinden.
Walk-forward, ausschliesslich auf ungesehenen Quartalen:

| Ziel | Apple | Amazon | Tesla |
| --- | ---: | ---: | ---: |
| Eigenes Quartal (die Aufgabe der Dissertation) | -0,042 | -0,123 | -0,011 |
| Vorquartal - bereits verkuendet und diskutiert | **-0,134** | **-0,083** | **-0,044** |
| Folgequartal (echte Prognose) | -0,004 | -0,111 | -0,018 |

Werte sind MCC. Nicht einmal die oeffentliche, bereits verkuendete Zahl laesst sich mit einem
Bag-of-Words-Modell aus diesem Korpus gewinnen. Die einzige Spur: Apples Accuracy auf dem
Vorquartalslabel steigt von 0,21 in Woche 1 des Quartals auf 0,31 in Woche 5 - Apples
Ergebniswoche - und auf etwa 0,34 spaet im Quartal, waehrend die Accuracy auf dem eigenen Quartal
flach bleibt. Die Verkuendung gelangt in den Text. Sie ist viel zu schwach, um nutzbar zu sein.

## 5. Warum jeder Wert ausserhalb des Zeitraums negativ ist, nicht null

Die Vorhersagen des Modells mit vollem Vokabular stimmen ausserhalb des Trainingszeitraums in 72%
der Faelle (Apple) bzw. 71% (Amazon) mit dem Label des **Vorquartals** ueberein, mit der Wahrheit
aber nur in 38% bzw. 21%. Ausserhalb des Zeitraums ist das Modell ein Persistenzprognostiker:
Ungesehene Quartale aehneln am meisten dem unmittelbar vorangehenden Quartal, also liefert das
Modell dessen Label.

Die Labels alternieren jedoch: Apple `0 0 2 3 | 0 0 2 3 | ...`, Amazon `3 0 1 1 | 3 0 1 1 | ...`.
Benachbarte Quartale teilen ihr Label nur in 32% (Apple) bzw. 21% (Amazon) der Faelle. Ein
Persistenzprognostiker auf einem alternierenden Ziel liegt systematisch falsch: MCC -0,146 und
-0,176, schlechter als Zufall. Das erklaert jede negative Zahl des Audits und geht ueber dieses
Projekt hinaus: Auf saisonal alternierenden Zielen ist maschinelles Lernen auf driftendem Text
ausserhalb der Stichprobe nicht neutral - es ist anti-praediktiv.

## 6. Der einzige uebertragbare Inhalt des Textes ist der Kalender selbst

Vokabular-Ablation im Walk-forward-Setting:

| Vokabular (Apple) | MCC ausserhalb des Zeitraums |
| --- | ---: |
| Voll (rund 60.000 Tokens) | -0,03 bis -0,14 |
| Voll ohne Kalender-Tokens | -0,13 bis -0,17 |
| Nur Kalender-Tokens | **+0,20** |

Das Entfernen der Kalenderwoerter macht das Modell schlechter; Kalenderwoerter allein liefern den
einzigen positiven Wert. Ein Vokabular aus nur 42 Monats- und Saisonwoertern erreicht bei Apple
insgesamt MCC +0,29 und **+0,43 auf den 45% der Gruppen, die einen Monat nennen** (Amazon: +0,34
auf diesen Gruppen, -0,37 auf den uebrigen). Zweiundvierzig Woerter schlagen 150.000. Das beste
ehrliche Textmodell ist ein Monatsnamenleser - und es erreicht nur die Haelfte der saisonalen
Baseline (0,76), weil 55% der Gruppen gar keinen Monat nennen.

## 7. Sprache driftet gleichmaessig und kehrt nicht mit den Jahreszeiten wieder

Kosinus-Aehnlichkeit zwischen den quartalsweisen TF-IDF-Zentroiden:

| Abstand (Quartale) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Apple | 0,894 | 0,834 | 0,789 | 0,755 | 0,729 | 0,711 | 0,693 | 0,678 |
| Amazon | 0,904 | 0,852 | 0,815 | 0,782 | 0,754 | 0,735 | 0,721 | 0,716 |
| Tesla | 0,924 | 0,882 | 0,859 | 0,841 | 0,814 | 0,794 | 0,780 | 0,768 |

Der Abfall ist monoton, etwa 2-3 Punkte pro Quartal, bei Tesla am langsamsten. Sprache aus demselben
Kalenderquartal ein Jahr spaeter (Abstand 4) ist **nicht** aehnlicher als Sprache drei Quartale
entfernt: Es gibt auf Zentroid-Ebene keine saisonale Wiederkehr der Sprache. Weihnachtsgerede ist zu
schwach, um das saisonale Label zu tragen; das tun nur explizite Kalenderwoerter. Die Uhr ist
monoton: Text verraet *wann*, aber aus dem Thema allein nie *welche Jahreszeit*.

## 8. Die dokumentierte Balancierung hat das Vorzeigeexperiment auf vier Kalenderfenster kollabiert

`EqualClassSampler` behaelt die ersten n Tweets je Klasse, wobei n die Groesse der kleinsten Klasse
ist. Auf die zeitlich sortierten Apple-4-Klassen-Daten angewendet (n = 93.686):

| Klasse | Inhalt des balancierten Pools |
| --- | --- |
| 0 "decrease" | 100% Tweets aus 2015Q1 |
| 1 "small increase" | 100% aus 2015Q3 |
| 2 "moderate increase" | 100% aus 2016Q3 |
| 3 "strong increase" | 72% aus 2015Q4, 28% aus 2016Q4 |

Das beste publizierte Ergebnis (Apple@10, vier Klassen, Accuracy 0,85, MCC 0,80) hat also die
Aufgabe "unterscheide 2015Q1, 2015Q3, 2016Q3 und die Weihnachtssaisons 2015/2016" gemessen, und das
Modell hat praktisch keinen Tweet nach 2016Q4 gesehen. Die Dissertation gibt auf Seite 114 an, dass
die Balancierung vor dem Split erfolgte; das Repository enthaelt keine andere
Balancierungsimplementierung.

## 9. Das neue Hybridmodell, zerlegt: Kalender plus zitierte Zahlen

Das Hybridmodell auf diesem Branch (`trainNumericTextSignalQuarterModel.py`) berichtet 80,56%
Accuracy und MCC 0,7387 auf 36 spaeteren Unternehmensquartalen. Der Lauf reproduziert sich auf vier
Nachkommastellen. Je Unternehmen richtig von 12:

| Zweig | Amazon | Apple | Tesla | Gesamt |
| --- | ---: | ---: | ---: | ---: |
| Nur saisonaler Prior | 10 | 10 | 2 | 22/36 |
| + numerischer Text (50/50) | 10 | 11 | 4 | 25/36 |
| + Tesla-Forward-Level-Zweig | 10 | 11 | 6 | 27/36 |
| + Tesla-Konfliktgate (post hoc) | 10 | 11 | **8** | **29/36** |
| + Gate, mit gemischtem Text | 10 | 11 | 4 | 25/36 |

Von den sieben gegenueber dem Kalender gewonnenen Quartalen sind sechs Tesla, zwei davon stammen
aus einem Gate, das nach Sichtung der Testjahre eingestellt wurde, und Amazon gewinnt keines. In der
Richtung (Anstieg gegen Rueckgang) erreicht der saisonale Prior allein 0,9167 / 0,8003 - identisch
mit dem vollen Hybrid. Der allgemeine Hybrid mit gemischtem Text (MCC 0,5473) schlaegt knapp
denselben Hybrid mit echtem Text (0,5466).

Die ehrliche Beschreibung in einem Satz: der Kalender fuer Apple und Amazon, in Tweets zitierte
Auslieferungszahlen fuer Tesla, plus ein von Hand eingestelltes Gate. Der Tesla-Anteil ist real
(Mischen senkt ihn von 8 auf 4) und das einzige echte textbasierte Signal des Projekts. Seine Natur
ist entscheidend: Tesla-Auslieferungen sind eine Kennzahl, die die Oeffentlichkeit vor der
Veroeffentlichung numerisch zitiert - Analystenschaetzungen, Guidance, durchgesickerte Zahlen - und
die Tweets geben diese Zahlen weiter. Der Text wirkt als Traeger von Zahlen, die jemand bereits
kannte, nicht als Quelle von Schwarmwissen. EPS und Umsatz werden nicht als Niveaus getwittert, und
fuer sie gibt es nichts.

## 10. Was die urspruenglichen Modelle aus den Wortvektoren gelernt haben

Die Korrelation zwischen Embedding-Inhalt und Label war real und stark. Sie lief ueber die Zeit:

```text
corr(Embedding-Inhalt, Label) = corr(Embedding-Inhalt, Zeitraum) x Identitaet(Zeitraum -> Label)
```

Der erste Faktor ist gross - Text stempelt sich selbst auf die Woche genau. Der zweite ist innerhalb
der Stichprobe exakt - ein Quartal, ein Label. Die trainierbare Embedding-Tabelle (63 Millionen
Parameter bei Apple, gegenueber 3,8 Millionen im LSTM) diente als Nachschlagewerk von
Zeitraummarkern zu Labels, mit der bereits in den vortrainierten Vektoren latenten Epochenstruktur
als Vorsprung. Ausserhalb des Zeitraums bricht der zweite Faktor weg, das Modell faellt auf die
naechstliegende Epoche zurueck, und die alternierenden Labels machen daraus Anti-Praediktion.

## 11. Was nicht behauptet wird

- Nicht, dass Tweets keine Information ueber Unternehmen enthalten. Die Aussage ist, dass dieses
  Korpus, diese Labels und diese Repraesentation keine extrahierbare finanzielle Richtung
  enthalten - nicht fuer das laufende Quartal, nicht fuer das naechste und nicht einmal fuer das
  bereits verkuendete vorige.
- Nicht, dass das Tesla-Signal bestaetigt ist. Es beruht auf 4-6 Quartalen, p = 0,125 gegen die
  Mischkontrolle, mit einem auf den Testjahren entworfenen Gate.
- Nicht, dass die Datierungsgenauigkeiten Obergrenzen sind. Staerkere Modelle wuerden genauer
  datieren.
- Die Untersuchungen nutzten Roh-Tweettexte mit Ziffern; die Pipeline der Dissertation entfernt
  Ziffern. Der Quartalswert mit der Dissertationspipeline (91,8%) ist die vergleichbare Zahl.

## 12. Wozu diese Befunde taugen

1. **Ein methodisches Ergebnis fuer das Feld.** Social-Media-Text stempelt sich selbst so fein, dass
   jedes zeitveraenderliche Label ohne kausalen Inhalt aus dem Text vorhersagbar ist. Zufaellige
   Splits ueber die Zeit lassen die Zeit durchsickern. Dieses Projekt ist eine ungewoehnlich saubere,
   vollstaendig quantifizierte Fallstudie.
2. **Eine Neurahmung des Interpretationskapitels.** Die Important-Word- und Topic-Pipeline ist eine
   diskriminative Schluesselwortanalyse nach Zeitraum. Umbenannt in "fuer jedes Fenster
   charakteristische Begriffe", bereinigt um automatisierte Accounts und je Quartal statt je Klasse
   berechnet, ueberleben ihre qualitativen Beobachtungen; als "Treiber finanzieller Veraenderung"
   nicht.
3. **Eine Richtung fuer eine konfirmatorische Studie.** Der eine Kanal, der funktioniert hat -
   oeffentlich vor der Veroeffentlichung zitierte Zahlen - verweist auf Kennzahlen, die oeffentlich
   numerisch geschaetzt werden. Das ist eine pruefbare Hypothese mit eingefrorenen Merkmalen und
   einem unberuehrten Holdout.

## 13. Vorgeschlagene Richtungen fuer weitere Studien

Jede Richtung folgt aus einem der obigen Befunde und ist mit den vorhandenen Daten und der
vorhandenen Pipeline machbar; die ersten drei brauchen keinerlei neue Daten.

1. **Das Zeit-Leakage-Audit als publizierte Methode.** Die Datierungsuntersuchungen zu einem
   allgemeinen Test machen: Bevor einer Studie "Text sagt X vorher" geglaubt wird, sind zu
   berichten (a) wie genau sich der Text selbst datiert, (b) wie gut eine reine Datums- oder
   Saisonregel X vorhersagt und (c) das Ergebnis unter zeitraumgruppierten Splits. Rueckwirkend auf
   publizierte Social-Media-Vorhersagestudien mit zufaelligen Splits anwenden. Dieses Projekt
   liefert das durchgerechnete Beispiel und die Zahlen. (Befunde 1, 4, 5, 6)

2. **Eine Replikation nur mit menschlichen Accounts.** Accounts als automatisiert oder menschlich
   klassifizieren (Postvolumen, Duplikatrate, Vorlagenregelmaessigkeit, Regelmaessigkeit der
   Postzeiten) und dann jede Analyse allein auf der menschlichen Teilmenge wiederholen. Zwei Fragen:
   Erscheint irgendein Textsignal, sobald die Feeds weg sind, und wie viel von dem, was die
   Literatur "oeffentliche Stimmung" ueber Unternehmen nennt, ist Feed-Ausgabe? Der Korpusanteil des
   obersten Prozents der Accounts sollte in jeder Arbeit berichtet werden, die diesen Datensatz
   nutzt. (Befund 3)

3. **Die Quartale sauber beschreiben.** Den neuronalen Important-Word-Pfad durch eine
   Schluesselwortstatistik je Quartal ersetzen (zum Beispiel gewichtete Log-Odds mit Prior) auf der
   menschlichen Teilmenge, plus Topic-Praevalenz ueber die zwanzig Quartale. Das liefert die
   Beschreibung "was wurde wann besprochen", die das Interpretationskapitel anstrebte, ohne den
   Umweg ueber 67 Millionen Parameter, und laesst sich als ausdruecklich explorative Analyse mit
   n = 20 mit den Kennzahlen korrelieren. (Befunde 3, 10)

4. **Die Zahlen-als-Traeger-Hypothese konfirmatorisch pruefen.** Das Tesla-Ergebnis legt nahe, dass
   Social Media dort nuetzlich ist, wo die Oeffentlichkeit eine Kennzahl vor ihrer Veroeffentlichung
   numerisch zitiert. Solche Kennzahlen vorab waehlen - Fahrzeugauslieferungen, Schaetzungen von
   Stueckzahlen, Abonnentenzuwaechse, Kinokassen- oder Spieleverkaufszahlen, App-Downloads -, die
   Regexes, Merkmale, Fusionsgewichte und das Gate einfrieren und einmalig auf einem unberuehrten
   Holdout auswerten. Direkt gegen den Analystenkonsens vergleichen, um zu erfahren, ob der Text
   ueber den Konsens hinaus, den er weitergibt, irgendetwas beitraegt. (Befund 9)

5. **Ein Ziel mit mehr als zwanzig Werten.** Zwanzig Quartale je Unternehmen koennen hoechstens
   zwanzig Faelle entscheiden. Entweder viele Unternehmen buendeln (dieselbe Pipeline ueber einige
   hundert Ticker ergibt Tausende Unternehmensquartale) oder zu einem feiner abgetasteten Ziel
   wechseln, etwa der Ergebnisueberraschung oder der Kursreaktion in den Tagen nach jeder
   Verkuendung. Die schwache Spur in Apples Ergebniswoche ist genau die Stelle, an der ein
   Ereignisfenster-Design zuerst nachsehen wuerde. (Befunde 4, 8)

6. **Die Abweichung vorhersagen, nicht das Niveau.** Weil die Labels saisonal alternieren, lautet
   die informative Frage nicht "geht es hoch", sondern "geht es staerker hoch als in diesem
   Quartal ueblich". Das Ziel als Residuum gegenueber der saisonalen Erwartung definieren; ein
   Textsignal muss dann die Null schlagen statt den Kalender, und die Persistenzfalle verschwindet
   konstruktionsbedingt. (Befunde 5, 6)

7. **Die Uhr als Forschungsgegenstand.** Driftraten und unternehmensuebergreifende Uebertragung auf
   anderen Plattformen, Themen und Sprachen messen; die Beitraege von Ereignissen,
   Plattformvokabular und automatisierten Accounts trennen; pruefen, wie genau ein Modell einen
   undatierten Beitrag datieren kann. Anwendungen: Herkunftspruefung, Erkennung rueckdatierter oder
   synthetischer Inhalte, Altersschaetzung archivierter Texte. (Befunde 1, 2, 7)

8. **Die urspruengliche Hypothese mit einer fairen Repraesentation erneut pruefen.** Die
   Bag-of-Words- und trainierbaren Embedding-Modelle sind moeglicherweise schlicht die falschen
   Instrumente. Die zeitraumgruppierte Walk-forward-Auswertung mit einem modernen Satz-Encoder auf
   der menschlichen Teilmenge und dem Residualziel wiederholen. Wenn Semantik jenseits des
   Zeitraums existiert, ist dies das Setting, in dem sie sich endlich zeigen koennte; wenn nicht,
   wird das negative Ergebnis deutlich staerker. (Befund 10)

## Reproduktion

Aus `tweetsCompanyNumbersPrediction/src`, mit dem Repository im Pfad:

| Skript | Erzeugt |
| --- | --- |
| `probes/exp_unexpected.py` | Abschnitte 1-3, die Q+1-Prognose, Tweet-Volumen |
| `probes/exp_unexpected_control.py` | die Kontrolle ohne Bots und Duplikate sowie die Traeger-Tokens der Uhr |
| `probes/exp_deeper.py` | Abschnitt 4 (verzoegertes Label, Zuwachs je Woche), Abschnitt 7, das Vokabular des Verkuendungsmodells |
| `probes/exp_calendar.py` | Abschnitt 6, Vokabular-Ablation |
| `probes/exp_calendar2.py` | Abschnitt 6 (42-Woerter-Modell) und Abschnitt 5 (Persistenz-Echo) |
| `trainNumericTextSignalQuarterModel.py` | Abschnitt 9 (`output/numeric_text_signal_quarter_results.json`) |

Alle Untersuchungen laufen in Minuten auf der CPU; keine nutzt die Finanz-CSVs als Eingabe.
