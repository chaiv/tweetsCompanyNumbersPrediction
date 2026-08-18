# Was die alten Modelle gelernt haben - und worauf die aktuellen Ergebnisse wirklich beruhen

Umfassende, laienverstaendliche Evaluation der alten `main`-Implementierung, der frueheren automatisierten Analyse, der aktuellen Quartalsmodelle sowie der Topic- und Important-Word-Analyse.

Stand: 18. August 2026

**Hinweis zur Erstellung:** Die Diagnose, die Codepruefung und das weitere Modelltraining wurden automatisiert mit ChatGPT 5.6 Sol sowie Claude Fable / Opus 5 durchgefuehrt.

## Kurzfassung

Dieses Projekt untersucht, ob sich aus oeffentlichen Texten ueber Unternehmen die Entwicklung einer Quartalskennzahl ableiten laesst. Beispiele fuer solche Kennzahlen sind Amazon-Umsatz, Apple-EPS und Tesla-Auslieferungen. Ausserdem soll erklaert werden, welche Woerter und Themen mit den Modellentscheidungen zusammenhaengen.

Die wichtigsten Ergebnisse sind:

1. **Die alte Implementierung ist wissenschaftlich interessant.** Sie bildet ein ungewoehnlich breites Informationssystem ab: Tweets werden mit Berichtsperioden verbunden, zu Gruppen zusammengefasst, mit einem LSTM klassifiziert und anschliessend bis zu Woertern und Topics interpretiert. Besonders interessant ist der starke zeitliche und ereignisbezogene Fingerabdruck der Sprache.
2. **Die alte Accuracy von etwa 0,87 ist kein sauberer Beleg fuer echte Zukunftsprognose.** Viele Tweetgruppen desselben Quartals wurden wie voneinander unabhaengige Testfaelle behandelt. Ausserdem konnte die Hauptauswertung fuer einen fruehen Testblock auch spaetere Textbloecke zum Training verwenden. Das Modell konnte deshalb Perioden und Quellen wiedererkennen, ohne eine unbekannte spaetere Quartalszahl vorhersagen zu muessen.
3. **Die fruehere automatisierte Analyse hat die zentrale Evaluationsschwaeche im Kern richtig erkannt.** Einige Aussagen waren jedoch zu absolut oder sachlich ungenau. So gibt es nicht 20 verschiedene Klassenlabels, sondern 20 Quartalsergebnisse mit nur zwei oder vier moeglichen Klassen. Auch der gesamte Interpretationspfad war nicht automatisch korrekt.
4. **Das aktuelle beste Ergebnis betraegt 80,56 % Accuracy und MCC 0,7387 auf 36 spaeteren Company-Quarters.** Dieses Ergebnis gehoert zu einem Hybrid aus Saisonprior, numerischen Textsignalen und einem Tesla-Sonderzweig. Es ist kein reines Textmodell.
5. **Der isolierte numerische Textzweig erreicht 50,00 % Accuracy und MCC 0,3224.** Die transparente Variante ohne den nachtraeglich entworfenen Tesla-Konflikt-Gate erreicht 75,00 % Accuracy und MCC 0,6633.
6. **Die 80,56 % sind explorativ.** Der Tesla-Konflikt-Gate wurde nach Betrachtung der Fehler aus denselben Testjahren 2017 bis 2019 entworfen. Er muss auf neuen, vorher unberuehrten Quartalen bestaetigt werden.
7. **Topics und Important Words sind jetzt zeitlich sauberer angebunden.** Exakte Modellattributionen werden von deskriptiven Topic-Erklaerungen getrennt. Topicmodelle und Wortlexika werden nur auf frueheren Quartalen trainiert und danach auf Zukunftsquartale angewandt.
8. **Der Branch ist noch nicht vollstaendig tweet-content-frei.** Grosse Rohdatensaetze sind nicht eingecheckt, aber einige Demo- und Testdateien enthalten noch vollstaendige oder tweetartige Texte.

Die wissenschaftlich korrekte Gesamtaussage lautet daher:

> Die alte Arbeit zeigt eine breite, erklaerbare Forschungsplattform und starke zeitliche Sprachsignaturen. Das aktuelle System zeigt explorativ, dass lokale Textsignale in einem leakage-bewussten Quartalshybrid nuetzlich sein koennen. Weder die alte 87-%-Accuracy noch die aktuellen 80,56 % belegen bereits ein bestaetigtes reines Textmodell fuer unbekannte zukuenftige Quartale.

---

## Inhaltsverzeichnis

1. [Die Forschungsfrage in einfachen Worten](#1-die-forschungsfrage-in-einfachen-worten)
2. [Wichtige Begriffe](#2-wichtige-begriffe)
3. [Pruefumfang und Beweisstandard](#3-pruefumfang-und-beweisstandard)
4. [Die alte main-Pipeline Schritt fuer Schritt](#4-die-alte-main-pipeline-schritt-fuer-schritt)
5. [Warum die alte Auswertung hohe Werte liefern konnte](#5-warum-die-alte-auswertung-hohe-werte-liefern-konnte)
6. [Was an der alten Implementierung wissenschaftlich interessant ist](#6-was-an-der-alten-implementierung-wissenschaftlich-interessant-ist)
7. [Schwache und fehlerhafte Stellen in main](#7-schwache-und-fehlerhafte-stellen-in-main)
8. [Pruefung der frueheren automatisierten Analyse](#8-pruefung-der-frueheren-automatisierten-analyse)
9. [Das aktuelle Quartalsmodell](#9-das-aktuelle-quartalsmodell)
10. [Training und Zukunftstest](#10-training-und-zukunftstest)
11. [Ergebnisse und statistische Einordnung](#11-ergebnisse-und-statistische-einordnung)
12. [Topics und Important Words](#12-topics-und-important-words)
13. [Konkrete Erklaerungsbeispiele](#13-konkrete-erklaerungsbeispiele)
14. [Claim Ladder](#14-claim-ladder)
15. [Datenschutz- und Tweet-Inhaltsaudit](#15-datenschutz--und-tweet-inhaltsaudit)
16. [Empfohlene Forschungsroadmap](#16-empfohlene-forschungsroadmap)
17. [Reproduktion und Codeevidenz](#17-reproduktion-und-codeevidenz)
18. [Abschliessendes Urteil](#18-abschliessendes-urteil)

---

## 1. Die Forschungsfrage in einfachen Worten

### 1.1 Was soll vorhergesagt werden?

Das Ziel bleibt ausschliesslich eine **Quartalskennzahl** beziehungsweise deren Veraenderung. Es wird kein Aktienkurs als Ersatz-Ziel verwendet.

Im aktuellen Mehrfirmenexperiment sind dies:

| Unternehmen | Kennzahl |
| --- | --- |
| Amazon | Umsatz beziehungsweise Net Sales |
| Apple | Earnings per Share, kurz EPS |
| Tesla | Fahrzeugauslieferungen beziehungsweise Car Sales |

Die prozentuale Veraenderung wird in vier Klassen eingeteilt:

| Klasse | Bedeutung |
| ---: | --- |
| 0 | Rueckgang |
| 1 | schwacher Anstieg von 0 bis 15 % |
| 2 | moderater Anstieg von ueber 15 bis 30 % |
| 3 | starker Anstieg von ueber 30 % |

Die Richtung Rueckgang gegen Anstieg kann zusaetzlich ausgewertet werden. Sie bleibt aber nur eine einfachere Zusatzdiagnostik und ersetzt nicht das Vierklassenziel.

### 1.2 Welche Informationen darf das Modell verwenden?

Das aktuelle Textsystem verwendet nur lokal vorhandene Texte und daraus berechnete Aggregationen. Die Finanz-CSV und der zu prognostizierende aktuelle Quartalswert werden nicht als Eingabefeatures verwendet.

Vergangene Zielklassen duerfen fuer das Training und fuer einen Saisonprior verwendet werden. Das ist wichtig: Ein Modell ohne aktuelle Finanzwerte kann trotzdem historische Target-Labels kennen. Deshalb ist die korrekte Bezeichnung des besten Systems **no-finance hybrid** und nicht **pure text**.

### 1.3 Ist das ein Forecast des naechsten Quartals?

Nein, noch nicht im strengen Sinn von `Q -> Q+1`.

Die Texte eines Quartals werden verwendet, um die Kennzahl desselben Berichtsquartals einzuschaetzen. Weil die Kennzahl typischerweise erst nach dem Ende des Quartals berichtet wird, ist das ein **Current-Quarter-Nowcast**.

Gleichzeitig wird zeitlich sauberer getestet: Das Modell wird auf frueheren Jahren trainiert und auf spaeteren Jahren ausgewertet. Somit gilt:

- **Zeitlich zukuenftiger Test:** ja.
- **Ziel des naechsten Quartals Q+1:** nein.
- **Ziel des Textquartals selbst:** ja.

Ein echtes Q+1-Experiment waere moeglich, muesste aber die Targets um ein Quartal verschieben und wuerde eine andere Forschungsfrage beantworten.

---

## 2. Wichtige Begriffe

### Company-Quarter

Eine Kombination aus Unternehmen und Quartal, zum Beispiel `Amazon 2018Q2`. Alle Texte dieser Kombination beziehen sich auf dieselbe Zielrealisierung. Deshalb ist ein Company-Quarter die primaere unabhaengige Auswertungseinheit.

### Training, Validation und Test

- **Training:** Daraus lernt das Modell seine Parameter.
- **Validation:** Damit werden Modellvariante und Hyperparameter ausgewaehlt.
- **Test:** Diese Daten duerfen erst nach der Auswahl fuer die abschliessende Bewertung verwendet werden.

### Leakage

Leakage bedeutet, dass Informationen aus dem Testfall direkt oder indirekt in das Training oder in die Modellauswahl gelangen. Das kann auch ohne identische Texte passieren. Wenn andere Texte desselben Quartals im Training liegen, kennt das Modell bereits viele Merkmale genau der Periode, die es angeblich vorhersagen soll.

### Pseudoreplikation

Ein Quartal hat nur ein Finanzziel. Werden tausend Tweetgruppen dieses Quartals als tausend unabhaengige Testfaelle gezaehlt, wird dieselbe Zielrealisierung tausendfach wiederholt. Die Stichprobe wirkt dadurch groesser und sicherer, als sie wirklich ist.

### Accuracy

Der Anteil korrekt vorhergesagter Klassen. 80,56 % bedeuten hier 29 richtige Entscheidungen bei 36 Company-Quarters.

### MCC

Der Matthews Correlation Coefficient beruecksichtigt alle Teile der Konfusionsmatrix und ist bei ungleich verteilten Klassen aussagekraeftiger als Accuracy allein. Ein Wert von 1 ist perfekt, 0 entspricht grob keinem Zusammenhang und negative Werte sprechen fuer systematisch falsche Entscheidungen.

### Log Loss

Log Loss bewertet nicht nur die gewaehlte Klasse, sondern auch, wie sicher das Modell war. Eine selbstbewusst falsche Vorhersage wird staerker bestraft.

### Baseline

Eine einfache Vergleichsregel. Ein komplexes Textmodell muss beispielsweise besser sein als eine Saisonregel, die nur fruehere gleiche Kalenderquartale betrachtet.

### Shuffle-Kontrolle

Textsignale werden absichtlich zwischen Quartalen vertauscht, waehrend Targets und restliche Modellteile gleich bleiben. Faellt die Leistung nicht, war der Text vermutlich nicht ausschlaggebend.

### Attribution

Eine Attribution beschreibt, wie stark ein konkretes Eingabefeature zu einer Modellentscheidung beitraegt. Sie erklaert das Modell, beweist aber keine oekonomische Ursache.

### Topic

Ein Topic ist eine Gruppe haeufig gemeinsam auftretender Begriffe. Ein Topic fasst Textkontext zusammen. Es ist nicht automatisch ein kausaler Grund fuer eine Quartalsaenderung.

---

## 3. Pruefumfang und Beweisstandard

### 3.1 Gepruefte Staende

| Pruefobjekt | Referenz |
| --- | --- |
| Alte Implementierung | `main` am Commit `23a2fbb5ee1820b2ec8840816133ab823ef84bb6` |
| Aktueller Branch | `baselines`, HEAD `0e708b1bc5a4a58c75c27f5f6ccb40e8a2f3e9bf` plus aktueller Working Tree |
| Primaeres Ergebnis | `output/numeric_text_signal_quarter_results.json` |
| Topic-/Wortergebnis | `output/numeric_text_topics_important_words.json` |
| Testzeitraum | rollende Tests fuer 2017, 2018 und 2019 |

Der `main`-Branch wurde direkt aus den Git-Objekten gelesen. Ein Checkout war nicht noetig, sodass der bereits veraenderte Arbeitsbaum unangetastet blieb.

### 3.2 Drei Evidenzstufen

| Stufe | Bedeutung | Beispiel |
| --- | --- | --- |
| Codefakt | Direkt im referenzierten Code sichtbar | `pd.to_datetime(post_date)` wird ohne `unit='s'` aufgerufen. |
| Reproduziertes Ergebnis | Targets, Wahrscheinlichkeiten und Metriken liegen lokal vor und wurden erneut berechnet | 29 von 36 richtigen Company-Quarters, Accuracy 0,8056, MCC 0,7387. |
| Berichteter Altbefund | Nur in der frueheren Analyse oder Dissertation berichtet | Die exakte alte Quartalserkennungsrate kann ohne damaliges Ergebnisartefakt nicht vollstaendig reproduziert werden. |

### 3.3 Warum es nur 36 primaere Testfaelle gibt

Es gibt drei Unternehmen, vier Quartale pro Jahr und drei Testjahre:

```text
3 Unternehmen x 4 Quartale x 3 Testjahre = 36 Company-Quarters
```

Ob intern 1.000 oder 1.000.000 Texte verarbeitet werden, aendert diese Zahl nicht. Mehr Texte koennen eine Quartalsrepraesentation verbessern, erzeugen aber keine neuen unabhaengigen Finanzereignisse.

### 3.4 Grenze dieses Audits

Der Audit bewertet den sichtbaren Repository-Stand. Er rekonstruiert nicht die exakte Hardware, Datenversion und Checkpointauswahl aller publizierten historischen Laeufe. Deshalb kann nicht sicher bestimmt werden, welcher alte Checkpoint jede publizierte Zahl erzeugt hat.

---

## 4. Die alte main-Pipeline Schritt fuer Schritt

### 4.1 Daten und Experimentregistry

`PredictionModelPath.py` definiert Experimente fuer mehrere Unternehmen, Kennzahlen, Gruppengroessen und Zielvarianten. Im Code finden sich unter anderem:

- Amazon Revenue,
- Apple EPS,
- Tesla Car Sales,
- Google Search Engine Market Share,
- binaere Klassen,
- vier Klassen,
- Gruppengroessen 5, 10 und 20.

Das ist wissenschaftlich interessant, weil dieselbe Gesamtidee auf unterschiedliche Unternehmen und Kennzahltypen angewandt werden kann.

### 4.2 Verbindung von Tweets und Finanzzahlen

`TweetNumbersConnector` sucht die Finanzzeile, deren Zeitintervall den Zeitstempel eines Tweets umfasst. Dabei gelten zwei gute Integritaetsregeln:

- Fehlt eine passende Finanzzeile, bricht der Prozess ab.
- Passen mehrere Finanzzeilen, bricht der Prozess ebenfalls ab.

Der Tweet erhaelt damit den Wert seines eigenen Berichtszeitraums. Die alte README beschrieb dagegen teilweise die naechste gemeldete Zahl. Code und Dokumentation meinten somit nicht denselben Prognosehorizont.

### 4.3 Diskretisierung des Targets

Der vierstufige Pfad verwendet Rueckgang, schwachen, moderaten und starken Anstieg. Im alten Klassifikator gibt es dabei technische Randprobleme:

- Die Intervalle ueberlappen an 15 und 30.
- Wegen der ersten passenden Regel fallen exakt 15 und 30 in die niedrigere Klasse.
- Zwischen -0,01 und 0 bleibt eine kleine Luecke.
- Der binaere Pfad und der Vierklassenpfad verwenden nicht immer dieselbe Skala: einmal Verhaeltnis, einmal Prozentwert.

Das Target-Schema sollte daher explizit versioniert und mit Grenzwerttests abgesichert werden.

### 4.4 Bildung der Tweetgruppen

`DataframeSplitter.getSplitIds` arbeitet vereinfacht so:

1. Alle Zeilen einer Klasse werden ausgewaehlt.
2. Diese Zeilen werden in fortlaufende Bloecke der Groesse 5, 10 oder 20 geschnitten.
3. Jeder Block wird zu einem Trainingssample.
4. Alle Texte im Block tragen dasselbe Klassenlabel.

Die Gruppenbildung kennt jedoch keine explizite Quartalsgrenze. Dadurch koennen an einer Periodengrenze Texte verschiedener Quartale in einer Gruppe landen, wenn sie dieselbe Klasse haben.

Noch wichtiger ist die Wiederholung des Targets: Ein einzelnes Quartal kann sehr viele Gruppen erzeugen, obwohl alle Gruppen dieselbe Finanzrealisierung teilen.

### 4.5 Repraesentation einer Gruppe

`TweetGroup` tokenisiert jeden Post und verbindet die Tokenfolgen mit einem `<SEP>`-Token. Das ist eine sinnvolle Idee, weil Tweetgrenzen nicht vollstaendig verschwinden.

Die Gruppe wird anschliessend als eine lange Sequenz an das LSTM gegeben. Eine echte Hierarchie `Token -> Tweet -> Quartal` existiert im alten Modell noch nicht.

### 4.6 Top2Vec und Wortvektoren

Top2Vec wird auf dem lokalen Textkorpus trainiert. Seine Wortvektoren werden als 300-dimensionale Initialisierung fuer das LSTM verwendet. Damit dient derselbe semantische Raum sowohl der Vorhersage als auch der Topicinterpretation.

Das ist konzeptionell elegant, fuer einen strikten Zukunftstest aber problematisch: Wenn Top2Vec vor dem Split auf dem gesamten Korpus trainiert wird, beeinflussen spaetere Testtexte bereits Vokabular und Vektoren. Das ist kein direktes Label-Leakage, aber eine transduktive Nutzung der Zukunftstexte.

### 4.7 Das alte LSTM

Das Hauptmodell besitzt:

- ein trainierbares Embedding,
- ein zweischichtiges LSTM mit Hidden Size 512,
- zwei verborgene lineare Schichten,
- eine finale Klassenausgabe.

Der letzte Hidden State wird als Repraesentation der gesamten gepaddeten Sequenz verwendet. Das waere bei korrekt gepackten Sequenzen grundsaetzlich moeglich. Im alten Code fehlen aber echte Sequenzlaengen, Packing und Maskierung. Dadurch kann der Endzustand viele PAD-Schritte durchlaufen.

### 4.8 Training

Der Lightning-Trainer verwendet:

- CUDA standardmaessig,
- Mixed Precision,
- TensorBoard-Logging,
- Checkpointing nach Validation Loss,
- Early Stopping,
- optional Class Weights.

Diese Infrastruktur ist eine Staerke. Einige Details sind jedoch problematisch: Die Early-Stopping-Patience ist genauso gross wie die maximale Epochenzahl, und manuell rekonstruierte Checkpointpfade koennen eine alte unversionierte Datei laden.

### 4.9 Alte Hauptauswertung

Die Hauptauswertung teilt die Gruppen jeder Klasse in zehn chronologische Bloecke. Fuer Fold `k` gilt:

- Block `k` wird Test.
- Alle anderen Bloecke werden Training und Validation.

Damit koennen fuer einen fruehen Testblock auch spaetere Bloecke im Training liegen. Ausserdem koennen andere Gruppen desselben Quartals auf beiden Seiten vorkommen.

Die Auswertung ist daher kein strenger Forecast spaeterer unbekannter Quartale.

### 4.10 Weitere Splitideen

Die Codebasis enthaelt auch:

- einen globalen 80/20-Zeitsplit,
- eine klassenweise Expanding-Window-Variante,
- eine stratified-temporal Variante,
- eine Subsequent-Variante fuer Interpretation.

Dass mehrere Protokolle implementiert wurden, zeigt das richtige wissenschaftliche Anliegen: Ergebnisse sollen gegen unterschiedliche Zeitannahmen geprueft werden. Die konkrete alte Umsetzung behebt jedoch nicht automatisch Quartalspseudoreplikation, Targetstratifizierung und die fehlerhafte Zeitkonvertierung.

### 4.11 Alte Interpretationspipeline

Die alte Idee war:

1. Integrated Gradients berechnet Tokenattributionen.
2. Token werden wieder ihren Tweets zugeordnet.
3. Originalwort und Part-of-Speech-Tag werden hinzugefuegt.
4. Top2Vec oder BERTopic ordnet Dokumente Topics zu.
5. Important Words und Topics werden gemeinsam analysiert.
6. Manuelle und LLM-generierte Topics koennen mit Modelltopics verglichen werden.

Diese Forschungslogik ist stark. Mehrere Implementierungsdetails waren jedoch fehlerhaft; sie werden in Abschnitt 7 beschrieben.

---

## 5. Warum die alte Auswertung hohe Werte liefern konnte

### 5.1 Ein synthetisches Beispiel

Angenommen, Tesla 2018Q4 hat Klasse 3 und es gibt 10.000 Texte. Bei Gruppengroesse 10 entstehen etwa 1.000 Gruppen mit demselben Label.

Wenn ein Split 100 Gruppen testet und 900 Gruppen desselben Quartals trainiert, sind die Tweet-IDs zwar verschieden. Das Modell sieht aber sehr viele andere Texte aus genau derselben Periode.

Es kann dadurch lernen:

| Textmerkmal | Was es verraten kann |
| --- | --- |
| damaliger Produktname | Quartal oder Produktzyklus |
| damalige Kampagne | Zeitraum |
| bestimmte Nachrichtenquelle | Quelle und Erscheinungsphase |
| zeittypische Marktwoerter | Marktregime |
| wiederkehrendes Template | Quelle oder Zeitraum |

Da das Quartal im vorbereiteten Datensatz ein festes Label besitzt, reicht eine Periodenerkennung oft schon fuer eine scheinbar gute Finanzklassifikation.

### 5.2 Was die hohe Accuracy dann wirklich misst

Die alte Accuracy kann eine Mischung messen aus:

- echter Finanzinformation im Text,
- Erkennung des Quartals,
- Erkennung des Jahres,
- Saisonmustern,
- Quellen- und Autorenmustern,
- Produkt- und Ereignisnamen,
- wiederkehrenden Texttemplates.

Ohne passende Baselines und einen globalen Zukunftssplit kann nicht bestimmt werden, welcher Anteil echte zukunftsrelevante Textinformation ist.

### 5.3 Warum „20 distinct labels“ falsch ist

Bei 2015Q1 bis 2019Q4 existieren 20 Quartalsrealisierungen pro Unternehmen. Es gibt aber nur vier moegliche Klassenwerte oder beim binaeren Ziel zwei.

Korrekt ist:

> Viele Tweetgruppen teilen sich nur 20 unabhaengige Quartalsergebnisse.

Falsch ist:

> Es gibt 20 verschiedene Klassenlabels.

### 5.4 Die Saisonbaseline

Viele Unternehmenskennzahlen haben wiederkehrende Quartalsmuster. Beispielsweise kann Q4 regelmaessig anders aussehen als Q1.

Eine Saisonbaseline fragt fuer ein neues Q2 nur: Welche Klassen hatten fruehere Q2 desselben Unternehmens?

Wenn diese einfache Regel bereits stark ist, muss ein Textmodell zeigen, welchen zusaetzlichen Nutzen der Text liefert. Ein Vergleich nur mit dem globalen Mehrheitslabel reicht nicht.

---

## 6. Was an der alten Implementierung wissenschaftlich interessant ist

Die methodischen Schwaechen machen die alte Arbeit nicht wertlos. Sie veraendern, welche Schlussfolgerung erlaubt ist.

### 6.1 Ein vollstaendiges Informationssystem

Die Codebasis implementiert nicht nur einen LSTM-Klassifikator. Sie umfasst:

- Datenintegration,
- Targetkonstruktion,
- Textbereinigung,
- Near-Duplicate-Erkennung,
- Gruppierung,
- Wortvektoren,
- neuronale Klassifikation,
- mehrere Metriken,
- lokale Attribution,
- Topic Modelling,
- manuelle Analyse,
- LLM-Vergleich.

Damit untersucht die Dissertation einen durchgaengigen Erkenntnisprozess von oeffentlichem Text bis zu erklaerbaren Unternehmenskennzahlen.

### 6.2 Mehrere Unternehmen und Kennzahlen

Die gemeinsame Architektur wird auf Umsatz, EPS, Fahrzeugzahlen und Suchmaschinenmarktanteil angewandt. Das ist wertvoll, weil ein Verfahren, das nur fuer eine einzelne Kennzahl funktioniert, weniger allgemein ist.

Die alte Codebasis beweist noch keine saubere firmenuebergreifende Generalisierung. Sie schafft aber eine sinnvolle vergleichende Versuchsmatrix.

### 6.3 Binaere und vierstufige Ziele

Die Trennung zwischen Richtung und Staerke ist wissenschaftlich sinnvoll:

- Binaer beantwortet: Rueckgang oder Anstieg?
- Vierstufig beantwortet: Wie stark ist die Veraenderung?

Die vier Klassen haben eine natuerliche Ordnung. Das alte Modell behandelt sie nominal; spaetere ordinale Modelle koennen diese Struktur explizit nutzen.

### 6.4 Multi-Scale-Aggregation

Die Gruppengroessen 5, 10 und 20 sind mehr als nur Hyperparameter. Sie bilden eine Forschungsfrage ab:

> Wie viel kollektiver oeffentlicher Diskurs wird benoetigt, damit aus schwachen Einzeltexten ein stabiles Signal entsteht?

Die alte Auswertung beantwortet diese Frage wegen der Pseudoreplikation noch nicht belastbar. Die Idee einer vorab definierten Evidenzbudget-Ablation bleibt aber stark.

### 6.5 Der Temporal-Fingerprint als eigener Befund

Die vielleicht wichtigste wissenschaftliche Erkenntnis der alten hohen Werte ist nicht die behauptete Zukunftsprognose, sondern ein starker **Zeit- und Regimefingerabdruck** in der Sprache.

Wortwahl kann Berichtsperioden kodieren durch:

- Ereignisse,
- Produktzyklen,
- Kampagnen,
- Nachrichtenquellen,
- Marktstimmung,
- saisonale Diskussionen.

Das motiviert eigene Forschungsfragen:

- Welche Topics unterscheiden Quartale?
- Welche Sprachmuster wiederholen sich saisonal?
- Welche Signale erscheinen vor einer Ergebnisveroeffentlichung?
- Welche erscheinen erst danach?
- Welcher Textbeitrag bleibt nach Kontrolle fuer Saison und Quelle uebrig?

### 6.6 Gemeinsamer Raum fuer Vorhersage und Topics

Top2Vec liefert sowohl Wortvektoren fuer den LSTM als auch Topics. Damit kann untersucht werden, ob semantische Achsen gleichzeitig fuer Klassifikation und Interpretation relevant sind.

Diese Verbindung ist originell, muss aber pro Fold sauber trainiert oder als feste externe Repraesentation behandelt werden.

### 6.7 Token -> Tweet -> Topic

Die Kombination von Integrated Gradients, Originalwort, POS-Tag und Dokumenttopic ist ein sinnvoller Versuch, lokale Modellentscheidungen in eine hoeherstufige sozialwissenschaftliche Interpretation zu ueberfuehren.

Heute sollte dieser Pfad so umgesetzt werden:

1. Nur echte Zukunftstestfaelle erklaeren.
2. PAD- und SEP-Tokens ausschliessen.
3. Signed und absolute Attribution getrennt speichern.
4. Pro Tweet korrekt aggregieren.
5. Erst danach nach Topics zusammenfassen.
6. Stabilitaet ueber Folds und Seeds pruefen.

### 6.8 Mehrere Topicmodelle und Qualitaetsdimensionen

Eine gemeinsame Extractor-Schnittstelle unterstuetzt Top2Vec und BERTopic. Topicqualitaet wird ueber Coherence, Diversity und Silhouette betrachtet.

Das ist wissenschaftlich besser als die Annahme, eine einzige Topiczerlegung sei die Wahrheit. Zusaetzlich sollten zeitliche Stabilitaet und held-out Generalisierung gemessen werden.

### 6.9 Mensch-Maschine- und LLM-Vergleich

`ManualTopicAnalyzer` und `LLMTopicsCompare` vergleichen manuelle oder LLM-generierte Begriffe mit Modelltopics, sowohl direkt als auch im Embeddingraum.

Das ist als Triangulationsdesign interessant. Fuer eine belastbare Studie braucht es:

- verblindete Rater,
- vorab definierte Prompts,
- feste Aehnlichkeitsschwellen,
- Inter-Rater-Reliabilitaet,
- dieselben held-out Dokumente fuer alle Systeme.

### 6.10 Gute Daten- und Evaluationsideen

Weitere erhaltenswerte Bausteine sind:

- genau eine Finanzzeile pro Zeitintervall,
- SimHash-basierte Near-Duplicate-Erkennung,
- Class Weights und EqualClassSampler,
- Precision, Recall, F1, Accuracy und MCC,
- gespeicherte Testindizes,
- mehrere zeitliche Splitvarianten,
- TensorBoard und Checkpointing.

Diese Elemente zeigen methodisches Problembewusstsein. Sie sind jedoch kein automatischer Beleg dafuer, dass jeder resultierende Lauf valide war.

---

## 7. Schwache und fehlerhafte Stellen in main

### 7.1 Evaluationsdesign

| Problem | Auswirkung |
| --- | --- |
| Gruppen statt Company-Quarters werden bewertet | Ein Quartalsziel wird sehr oft gezaehlt. |
| Andere Gruppen desselben Quartals koennen in Train und Test liegen | Periodenwiedererkennung wird moeglich. |
| Haupt-KFold trainiert auf allen anderen Bloecken | Fuer fruehe Tests koennen spaetere Texte im Training liegen. |
| Validation wird innerhalb des Trainingspools zufaellig stratifiziert | Training und Validation koennen dieselben Perioden teilen. |
| Saisonbaseline fehlt im Hauptlauf | Textskill und Quartalssaison werden verwechselt. |
| Balancing erfolgt teilweise vor dem Split | Zeitabdeckung und Klassenhaeufigkeiten werden veraendert. |

### 7.2 Transduktive Top2Vec-Nutzung

Top2Vec wird auf dem kompletten Textkorpus trainiert, bevor der Forecast-Split feststeht. Spaetere Testtexte beeinflussen dadurch:

- Vokabular,
- semantische Nachbarschaften,
- Startvektoren des LSTM,
- Topicstruktur.

Fuer einen strikten Zukunftstest muss das Topic-/Embeddingmodell nur auf vergangenen Texten trainiert oder als klar externe, zeitlich fixe Ressource deklariert werden.

### 7.3 Target- und Metrikdefinition

- Der Connector liefert den Wert desselben Intervalls, die alte README beschrieb teilweise Q+1.
- Verhaeltnis und Prozentwert tragen aehnliche Namen.
- Die Vierklassenintervalle haben Ueberlappungen und eine kleine Luecke.
- Tesla-Produktion, Auslieferung und Absatz duerfen nicht ohne dokumentierte Provenienz gleichgesetzt werden.

### 7.4 Padding und letzter Hidden State

Die Batches werden auf die laengste Sequenz gepaddet. Das LSTM erhaelt aber keine echten Laengen und keine Maske. Der verwendete letzte Hidden State kann deshalb einen grossen Anteil PAD-Verarbeitung enthalten.

Die korrekte Aussage ist nicht, dass ein letzter Hidden State immer falsch sei. Er ist nur in dieser unmaskierten Kombination problematisch.

Moegliche Reparaturen:

- `padding_idx` im Embedding setzen,
- `pack_padded_sequence` verwenden,
- maskiertes Mean Pooling,
- Attention mit Paddingmaske,
- hierarchische Token-/Tweet-/Quartalsaggregation.

### 7.5 Checkpointrisiko

Lightning kann bei vorhandenen Dateinamen versionierte Dateien wie `model-v1.ckpt` schreiben. Einige Skripte bauen danach den unversionierten Pfad manuell zusammen. Dadurch kann ein alter Checkpoint geladen werden.

Der Trainer selbst testet mit `ckpt_path='best'`. Daher ist die pauschale Aussage, jedes Skript werte zwingend einen alten Checkpoint aus, zu breit. Das Risiko in den manuellen Reload-Pfaden ist dennoch real.

### 7.6 Datumsfehler

Die Trainingsskripte verwenden `pd.to_datetime(post_date)` ohne `unit='s'`. Epoch-Sekunden werden dadurch als Nanosekunden interpretiert und landen im Jahr 1970.

Die numerische Reihenfolge kann dabei erhalten bleiben, Kalenderjahr und Quartal sind jedoch falsch. Die fruehere Diagnose ist hier korrekt.

### 7.7 Reproduzierbarkeit und Portabilitaet

- Seeds werden nicht durchgaengig fuer Torch und NumPy gesetzt.
- `loadModel` ist teilweise hart an `cuda:0` gebunden.
- `map_location` fehlt beim Laden.
- `strict=False` kann inkompatible State-Dict-Keys verdecken.
- Early Stopping kann bei zehn Epochen und Patience zehn kaum frueh stoppen.

### 7.8 Fehler im alten Topic- und Important-Word-Pfad

| Befund | Bedeutung |
| --- | --- |
| `extractMostImportantWords.py` verwendet `df.head(50000)` und nicht die gespeicherten Testindizes | Trainingstexte koennen in der Erklaerung landen. |
| Integrated Gradients wird pro Sample durch sein eigenes Maximum geteilt | Rangfolgen innerhalb eines Samples bleiben, Groessen zwischen Samples werden unvergleichbar. |
| Kein Schutz gegen Division durch null | Nullattribution kann NaN erzeugen. |
| Eine Gruppensumme wird beim Flattening fuer mehrere Tweets wiederholt | Tweet-Level-Werte sind falsch aggregiert. |
| Das Captum-Konvergenzdelta wird verworfen | Attributionsqualitaet wird nicht kontrolliert. |
| `findMostImportantTopicTweets.py` referenziert eine nicht definierte Variable | Das Skript kann in diesem Pfad abbrechen. |

Die Forschungsfrage bleibt relevant. Die alte konkrete Ausgabe darf aber nicht pauschal als mechanisch korrekt bezeichnet werden.

---

## 8. Pruefung der frueheren automatisierten Analyse

### 8.1 Richtig oder im Kern richtig

| Aussage | Urteil | Begruendung |
| --- | --- | --- |
| Das Ziel ist innerhalb eines Quartals konstant. | richtig | Alle Texte eines Company-Quarters teilen dieselbe Zielrealisierung. |
| Quarter-sharing Splits erlauben Periodenerkennung. | richtig | Gruppen, nicht Quartale, werden getrennt. |
| Die Hauptauswertung verwendet spaetere Bloecke im Training. | richtig | Testblock `k`, Training alle anderen Bloecke. |
| Eine Saisonbaseline ist notwendig. | richtig | Vergangene gleiche Kalenderquartale tragen ein starkes Signal. |
| Das Datum wird als Nanosekunden interpretiert. | richtig | `unit='s'` fehlt. |
| Fixe Checkpointnamen koennen einen alten Lauf auswerten. | richtig fuer bestimmte Pfade | Manueller unversionierter Reload nach Lightning-Checkpointing. |
| Die Daten beweisen nicht, dass Tweets keine Finanzinformation enthalten. | richtig | Die Diagnose betrifft die Evaluation, nicht die Existenz eines Signals. |

### 8.2 Teilweise richtig, aber zu stark formuliert

| Aussage | Korrektur |
| --- | --- |
| „20 distinct labels“ | Es sind 20 Quartals-Outcomes, aber nur zwei oder vier Klassenwerte. |
| „Das LSTM lernt in keiner Konfiguration“ | Bestimmte Laeufe kollabierten; nicht jede denkbare LSTM-Konfiguration wurde getestet. |
| „Der letzte Hidden State ist die Ursache“ | Das konkrete Problem ist vor allem unmaskiertes Padding plus Endzustand. |
| „Jedes Trainingsskript laedt einen alten Checkpoint“ | Vier manuelle Reload-Pfade sind gefaehrdet; der Trainer verwendet `best`. |
| „91,8 % Quartalserkennung“ | Methodisch plausibler Altbefund, aber ohne eingechecktes Resultat hier nicht exakt reproduziert. |
| „Alle Out-of-period-Protokolle zeigen keinen positiven Zusammenhang“ | In der alten Analyse berichtet, aber nicht fuer alle Datensaetze mit lokalen Rohartefakten erneut verifiziert. |

### 8.3 Falsch, unbelegt oder fuer das Projekt nicht zwingend

| Aussage | Urteil |
| --- | --- |
| Die Interpretation-Pipeline sei mechanisch sauber. | falsch; mehrere konkrete Codefehler widersprechen dem. |
| Die publizierten 0,87/0,77 seien niemals aus Text entstanden. | nicht beweisbar; der konkrete Mechanismus des historischen Checkpoints ist ohne Artefakt offen. |
| Mehr als 20 Ziele erforderten einen nichtquartalsweisen Target. | nicht noetig; mehr Jahre, Unternehmen oder externe Holdouts erhoehen ebenfalls die Zahl unabhaengiger Faelle. |
| Das Target muesse zwingend auf Q+1 verschoben werden. | nur fuer einen literal next-quarter Forecast; nicht fuer einen Current-Quarter-Nowcast. |
| Die negative Prognosediagnose mache alte Topics automatisch korrekt. | falsch; der Interpretationspfad muss separat repariert werden. |

### 8.4 Gesamturteil zur frueheren Analyse

Die Hauptkritik ist nicht halluziniert: Split und Zielstruktur erzeugen eine reale Abkuerzung. Zu weit gehen absolute Aussagen ueber alle LSTMs, alle Trainingspfade und die gesamte Interpretation.

Die praezise Einordnung lautet:

> Starke Diagnose des Evaluationsdesigns, aber zu breite Aussagen ueber Architektur, Experimente und Codequalitaet.

---

## 9. Das aktuelle Quartalsmodell

### 9.1 Ziel und Daten

Das Ziel bleibt die vierstufige Veraenderung der Quartalskennzahl. Verwendet werden Amazon, Apple und Tesla.

Nicht als aktuelle Eingabefeatures verwendet werden:

- Finanz-CSV,
- der aktuelle Quartalswert,
- die aktuelle prozentuale Zielveraenderung,
- Word Embeddings,
- externe Daten.

Vergangene Zielklassen werden fuer Modelltraining und Saisonprior verwendet.

### 9.2 Zielnahe Textselektion

Ein Text geht in die numerische Aggregation ein, wenn er sowohl einen Unternehmensmarker als auch einen Kennzahlmarker enthaelt.

Beispiele:

| Unternehmen | Unternehmensmarker | Kennzahlmarker |
| --- | --- | --- |
| Amazon | Amazon, AMZN | revenue, net sales, AWS sales |
| Apple | Apple, AAPL | EPS, earnings per share |
| Tesla | Tesla, TSLA | deliveries, delivery, production |

Das Ergebnis speichert keine vollstaendigen Texte. Es speichert nur aggregierte Features.

### 9.3 Sechs Textansichten

| Ansicht | Inhalt | Hypothese |
| --- | --- | --- |
| `all` | alle relevanten Texte des Quartals | Grundpegel |
| `late_third` | letztes Drittel | spaetere Information ist naeher am Periodenende |
| `reported` | reported, actual, announced | bereits berichteter oder rueckblickender Stand |
| `forward_estimate` | estimate, consensus, guidance, future | Erwartungssprache |
| `early_reported` | Reportingsprache im ersten Drittel | Proxy fuer alten Referenzstand |
| `late_forward_estimate` | Schaetzungen im letzten Drittel | Proxy fuer erwarteten neuen Stand |

### 9.4 Featurefamilien

Aus jeder Ansicht werden unter anderem berechnet:

- Gesamtzahl und Zahl relevanter Texte,
- Anteil relevanter Texte,
- Zahl der Prozentangaben,
- positive und negative Prozentwerte,
- Median und Quartile,
- direkte Verteilung der Prozentwerte auf vier Klassen,
- Anteil von reported-, estimate-, guidance-, beat-, miss- und future-Sprache,
- robuste absolute Kennzahllevel,
- Differenzen zwischen fruehen, spaeten, berichteten und erwarteten Levels.

### 9.5 Synthetisches Beispiel

Angenommen, fruehe Texte nennen einen berichteten Lieferstand von 70.000 Einheiten. Spaete Schaetzungen nennen 81.000.

```text
estimated_change = (81.000 / 70.000 - 1) x 100 = 15,7 %
```

15,7 % faellt in Klasse 2. Das ist Feature Engineering aus Text, nicht das Auslesen des echten Testtargets.

### 9.6 Numerischer Textklassifikator

Die Quartalsfeatures werden:

1. nur auf Trainingsquartalen standardisiert,
2. auf den Bereich `[-8, 8]` begrenzt,
3. mit regularisierter logistischer Regression klassifiziert.

Validation waehlt:

- Regularisierung,
- aktuelle Features allein oder mit zeitlichen Differenzen,
- optionale Firmenidentitaet.

### 9.7 Saisonprior

Fuer ein neues Q2 betrachtet der Saisonprior nur fruehere Q2-Labels desselben Unternehmens und erzeugt daraus eine geglaettete Klassenverteilung.

Er verwendet keine Finanzwerte als Input, aber historische Targets. Darum ist er kein Textfeature.

### 9.8 Fusion

Der allgemeine Hybrid mischt Saison- und Textwahrscheinlichkeiten mit festem Gewicht.

Bei Tesla kommt ein Forward-Level-Signal hinzu. Es leitet aus spaeten Schaetzleveln und fruehen berichteten Levels eine erwartete Veraenderung ab.

### 9.9 Tesla-Konflikt-Gate

Der Gate erkennt zwei spezielle Konfliktmuster zwischen Basismodell und numerischem Textmodell. In diesen Faellen ersetzt er die Vorhersage durch die numerische Textverteilung.

Der Gate verbessert 75,00 % auf 80,56 %. Seine Schwellen wurden jedoch nach Betrachtung der Fehler von 2017 bis 2019 entworfen. Deshalb ist er explorativ und nicht bestaetigend.

### 9.10 Warum kein CUDA benoetigt wird

Das aktuelle beste Modell ist kein LSTM. Regex-Aggregation, Skalierung und logistische Regression laufen auf CPU. CUDA war fuer die frueheren neuronalen Modelle relevant, nicht fuer den aktuellen 80,56-%-Lauf.

---

## 10. Training und Zukunftstest

### 10.1 Rolling-Origin-Schema

| Testjahr | Training | Validation | Test |
| ---: | --- | --- | --- |
| 2017 | 2015 | 2016 | 2017 |
| 2018 | 2015-2016 | 2017 | 2018 |
| 2019 | 2015-2017 | 2018 | 2019 |

Nach der Auswahl wird auf Training plus Validation neu gefittet und genau das folgende Testjahr bewertet.

Drei Seeds werden verwendet:

- 1337,
- 101337,
- 201337.

Die Wahrscheinlichkeiten der drei Laeufe werden pro Company-Quarter gemittelt.

### 10.2 Was im aktuellen Protokoll nicht leakt

- Kein Testlabel geht in Fit, Skalierung oder Auswahl ein.
- Trainingsjahre liegen global vor dem Validierungsjahr.
- Das Validierungsjahr liegt global vor dem Testjahr.
- Jede Company-Quarter-Kombination wird genau einmal gezaehlt.
- Topic- und Wortmodelle werden nur auf frueheren Quartalen gefittet.

### 10.3 Verbleibende Einschraenkungen

- Es wird Text aus dem gesamten zu bewertenden Quartal aggregiert.
- Ein Echtzeit-Cutoff bei 25 %, 50 % oder 75 % des Quartals ist noch kein primaerer Test.
- Es gibt nur 36 unabhaengige Testfaelle.
- Der Tesla-Gate ist post hoc.
- Amazon und Apple werden stark von Saisonmustern getragen.
- Der lokale 2020-Bestand ist nicht dicht genug fuer einen gleichartigen vollstaendigen neuen Holdout.

---

## 11. Ergebnisse und statistische Einordnung

### 11.1 Vierklassenmetriken

| Modell | Accuracy | MCC | Log Loss | Einordnung |
| --- | ---: | ---: | ---: | --- |
| Numerischer Text allein | 0,5000 | 0,3224 | 1,3078 | reines numerisches Textsignal |
| Saisonprior ohne Finanzfeatures | 0,6111 | 0,4743 | 0,9519 | nur fruehere gleiche Kalenderquartale |
| Saison + numerischer Text, fest 50/50 | 0,6944 | 0,5854 | 1,0298 | allgemeiner Hybrid |
| Saison + Tesla Forward | 0,7500 | 0,6633 | 0,9460 | transparente Variante ohne Konflikt-Gate |
| Saison + Tesla Konflikt-Gate | **0,8056** | **0,7387** | 0,9173 | primaeres exploratives Ergebnis |
| Primaerer Bundle-Shuffle | 0,6944 | 0,5850 | 1,1094 | Textbundle innerhalb der Firma verschoben |

### 11.2 Ergebnis je Unternehmen

| Unternehmen | Richtig | Accuracy | MCC |
| --- | ---: | ---: | ---: |
| Amazon | 10 / 12 | 0,8333 | 0,7828 |
| Apple | 11 / 12 | 0,9167 | 0,8765 |
| Tesla | 8 / 12 | 0,6667 | 0,5664 |
| Gesamt | 29 / 36 | 0,8056 | 0,7387 |

### 11.3 Richtungsdiagnostik

Werden dieselben Wahrscheinlichkeiten nur zu Rueckgang gegen Anstieg zusammengefasst, entstehen:

- Accuracy 0,9167,
- MCC 0,8003.

Das ist ein einfacheres Ziel und darf die primaere Vierklassenauswertung nicht ersetzen.

### 11.4 Unsicherheit

29 richtige Entscheidungen bei 36 Faellen ergeben fuer Accuracy ein Wilson-95-%-Intervall von ungefaehr:

```text
0,650 bis 0,902
```

Das Intervall ist breit. Die wahre Leistung kann deutlich unter oder ueber dem Punktwert liegen.

### 11.5 Vergleich mit der Shuffle-Kontrolle

Das primaere Modell ist in vier Quartalen korrekt, in denen die Bundle-Shuffle-Kontrolle falsch liegt. Der umgekehrte Fall tritt nicht auf.

Der gepaarte zweiseitige exakte Test ergibt:

```text
p = 0,125
```

Das ist eine positive Richtung, aber kein statistisch signifikanter Unterschied auf dem 5-%-Niveau.

### 11.6 Was behauptet werden darf

- Ein leakage-bewusster no-finance Hybrid erreicht explorativ 80,56 % und MCC 0,7387.
- Numerische Text- und Erwartungsfeatures helfen bei bestimmten Tesla-Entscheidungen.
- Der isolierte numerische Textzweig enthaelt ein positives, aber begrenztes Signal.
- Amazon und Apple besitzen starke saisonale Targetmuster.

### 11.7 Was nicht behauptet werden darf

- Nicht: Ein reines Textmodell erreicht 80 bis 90 %.
- Nicht: Der zusaetzliche Textbeitrag ist statistisch bestaetigt.
- Nicht: 2017 bis 2019 seien nach Entwicklung des Gates weiterhin ein unberuehrter finaler Holdout.
- Nicht: Das Modell prognostiziere bereits das naechste Quartal Q+1.
- Nicht: Topics verursachten eine Finanzveraenderung.

### 11.8 Naechster bestaetigender Test

Regex, Features, Hyperparameter, Fusionsgewichte und Gate-Schwellen muessen eingefroren werden. Danach ist ein vollstaendig neuer Holdout erforderlich.

Erst ein solcher Test trennt echte Generalisierung von nachtraeglicher Anpassung.

---

## 12. Topics und Important Words

### 12.1 Warum Integrated Gradients hier nicht die richtige Hauptmethode ist

Der aktuelle numerische Textzweig besteht aus aggregierten Regexfeatures und logistischer Regression. Er hat keine Token-Embedding-Schicht, auf die Integrated Gradients sinnvoll angewandt werden koennte.

Fuer diesen Zweig ist die exakte lineare Attribution:

```text
Beitrag eines Features = standardisierter Featurewert x Klassenkoeffizient
```

Die Summe dieser Beitraege plus Intercept rekonstruiert den Entscheidungsscore des numerischen Textmodells.

### 12.2 Vier Erklaerungsebenen

| Ebene | Berechnung | Aussagekraft |
| --- | --- | --- |
| Exakte Textfeature-Attribution | standardisierter Wert mal OVR-Koeffizient | exakt fuer den numerischen Textzweig |
| Modellnahe Cue-Woerter | Featurebeitraege werden Sprachfamilien und vorkommenden Cues zugeordnet | Featurefamilie exakt, Verteilung auf Einzelwoerter deskriptiv |
| Past-only Important Words | quartalsstabile Klassen-Log-Odds nur aus frueheren Quartalen | zeitlich saubere Klassenassoziation, nicht kausal |
| Past-only Topics | TF-IDF plus NMF nur auf frueheren Quartalen; Testtexte werden transformiert | Kontextbeschreibung, keine additive Modellattribution |

### 12.3 Modellnahe Cues

Featurefamilien werden mit konkreten Textmustern verbunden:

- reported,
- estimate,
- guidance,
- beat,
- miss,
- future,
- positive und negative Richtung,
- Prozentmarker,
- absolute Zahlenmarker,
- Unternehmens- und Kennzahlbegriffe.

Bei Medianen, Quantilen und Quartalsaggregation ist die Verteilung auf einzelne Woerter nicht mehr mathematisch exakt. Deshalb wird sie ausdruecklich als deskriptive Bruecke bezeichnet.

### 12.4 Past-only Important Words

Das Wortlexikon lernt, welche Begriffe in frueheren Quartalen stabil mit einer Klasse verbunden waren.

Wichtig ist die Reihenfolge:

1. Nur Training plus Validation werden verwendet.
2. Woerter muessen ueber mehrere Quartale wiederkehren.
3. Erst danach wird geprueft, welche dieser Woerter im Zukunftsquartal vorkommen.
4. Das Testlabel wird nicht zum Wortfit verwendet.

Damit wird ein einmaliger Begriff aus einem einzigen Quartal weniger leicht zu einem angeblich wichtigen Wort.

### 12.5 Past-only NMF-Topics

Fuer jedes Unternehmen und Testjahr wird ein kleines Topicmodell auf den bis dahin vergangenen Quartalen trainiert.

Verwendet werden:

- TF-IDF zur Textrepraesentation,
- NMF zur Topiczerlegung,
- maximal 250 relevante Dokumente pro Quartal,
- deterministische, quartalsbalancierte Auswahl.

Das Zukunftsquartal wird nur in das bereits gelernte Topicmodell projiziert.

### 12.6 Vollstaendiger Entscheidungsweg

Jede Erklaerung enthaelt getrennt:

- Saisonwahrscheinlichkeiten,
- numerische Textwahrscheinlichkeiten,
- Forward-Level-Wahrscheinlichkeiten,
- Wahrscheinlichkeiten vor dem Konflikt-Gate,
- finale Wahrscheinlichkeiten,
- Gate-Aktivierungsanteil,
- exakte Textfeaturebeitraege,
- Cue-Woerter,
- past-only Important Words,
- past-only Topics.

So wird sichtbar, ob ein Topic nur Kontext beschreibt oder ob der Textzweig die finale Klasse tatsaechlich beeinflusst hat.

### 12.7 Reproduktionsschutz

Das Erklaerungsskript spielt die gespeicherten Foldentscheidungen erneut ab. Es bricht ab, wenn Accuracy, MCC oder finale Klassen nicht mit dem primaeren Ergebnis uebereinstimmen.

Der erfolgreiche Replay reproduzierte:

- Hybrid Accuracy 0,8056,
- Hybrid MCC 0,7387,
- numerische Text-Accuracy 0,5000,
- numerischen Text-MCC 0,3224.

---

## 13. Konkrete Erklaerungsbeispiele

Alle Beispiele enthalten nur aggregierte Begriffe und Wahrscheinlichkeiten, keine vollstaendigen Originaltexte.

### 13.1 Amazon 2017Q1

| Groesse | Ergebnis |
| --- | --- |
| Wahre Klasse | 3 |
| Numerische Textklasse | 3 |
| Finale Klasse | 3 |
| Saisonwahrscheinlichkeit fuer Klasse 3 | 0,750 |
| Numerische Textwahrscheinlichkeit fuer Klasse 3 | 0,379 |
| Finale Wahrscheinlichkeit fuer Klasse 3 | 0,564 |

Der starke Saisonpfad wird durch den Text bestaetigt, nicht ersetzt.

Groesste positive Textfeaturebeitraege fuer Klasse 3:

- `all__miss_tweet_fraction`: +0,206,
- `forward_estimate__miss_tweet_fraction`: +0,192,
- `early_reported__log_percent_mentions`: +0,137.

Groesste negative Beitraege:

- `reported__signed_percent_negative_fraction`: -0,153,
- `reported__percent_class_0_fraction`: -0,153,
- `late_third__miss_tweet_fraction`: -0,142.

Modellnahe Marker und Begriffe umfassen unter anderem:

- `percentage_value`,
- `numeric_value`,
- actual,
- misses,
- estimates,
- revenue,
- miss.

Past-only Important Words fuer die finale Klasse beginnen mit:

- business,
- misses,
- miss,
- earnings,
- sales.

Dominante Topicbereiche:

- revenue, Amazon, AWS, cloud,
- growth, revenue growth, year-over-year.

### 13.2 Tesla 2018Q1

Hier veraendert der Konflikt-Gate die Entscheidung.

| Zweig | Wahrscheinlichkeiten `[0, 1, 2, 3]` | Argmax |
| --- | --- | ---: |
| Saisonprior | `[0,313; 0,563; 0,063; 0,063]` | 1 |
| Numerischer Text | `[0,143; 0,456; 0,208; 0,193]` | 1 |
| Forward-Level | `[0,042; 0,042; 0,042; 0,875]` | 3 |
| Vor Konflikt-Gate | `[0,135; 0,275; 0,088; 0,501]` | 3 |
| Final nach Gate | `[0,143; 0,456; 0,208; 0,193]` | 1, korrekt |

Groesste exakte Beitraege zur numerischen Textklasse 1:

- `late_forward_estimate__metric_tweet_fraction`: +0,491,
- `all__log_percent_mentions`: +0,243,
- `early_reported__metric_tweet_fraction`: +0,228,
- `reported__metric_tweet_fraction`: +0,216.

Nach den abstrakten Prozent- und Zahlenmarkern folgen Begriffe wie:

- production,
- report,
- estimates,
- expect,
- miss.

Past-only Important Words beginnen mit:

- model,
- model production,
- cars,
- quarter.

Die dominanten Topics betreffen Produktion, Model-Produktion und Auslieferungen.

Dieses Beispiel zeigt Nutzen und Risiko zugleich: Der Textzweig korrigiert die Forward-Fusion korrekt. Die Entscheidung erfolgt aber ueber einen nachtraeglich entwickelten Tesla-Gate. Topicwoerter erklaeren den Kontext; sie validieren nicht die Gate-Schwellen.

---

## 14. Claim Ladder

Die Claim Ladder trennt technische Machbarkeit, empirische Beobachtung, Hypothese und unzulaessige Schlussfolgerung.

| Aussage | Status | Wissenschaftlich korrekte Formulierung |
| --- | --- | --- |
| Die Gesamtpipeline ist technisch realisierbar. | belegt | Lokale Texte koennen mit Berichtsperioden verbunden, gruppiert, klassifiziert und bis zu Woertern und Topics zurueckverfolgt werden. |
| Textgruppen enthalten starke Klassenkorrelationen. | im alten Split belegt | Unter den damaligen Gruppierungs- und Splitbedingungen ist die Zielklasse stark separierbar. |
| Sprache kodiert Zeit, Ereignisse und Marktregime. | starke Hypothese | Alte hohe Werte sowie Saison- und Shuffle-Diagnosen motivieren einen Temporal-Fingerprint-Test. |
| Bestimmte Woerter und Topics sind interpretativ relevant. | Kandidaten | Stabilitaet muss auf past-only Fits und Zukunftstests erneut gemessen werden. |
| Text sagt unbekannte spaetere Quartale mit 87 % voraus. | nicht belegt | Die alte 0,87 darf nicht als strikte Zukunftsgeneralisierung interpretiert werden. |
| Das aktuelle reine Textmodell erreicht 80,56 %. | falsch | 80,56 % gehoeren zum Hybrid; numerischer Text allein erreicht 50,00 %. |
| Ein Topic verursacht eine Quartalsaenderung. | nicht belegt | Attribution und Topiczuordnung zeigen Modellassoziation, keine oekonomische Kausalitaet. |

### Der erhaltenswerte Dissertationsbeitrag

Der wissenschaftliche Kern liegt in vier Punkten:

1. Entwurf eines modularen Informationssystems, das Social-Media-Text mit Unternehmenskennzahlen verbindet.
2. Empirischer Hinweis auf starke zeitliche und ereignisbezogene Sprachsignaturen.
3. Multi-Scale-Aggregation als Antwort auf schwache Einzeltweets.
4. Mehrstufige Interpretation von Tokenattribution bis Topic- und LLM-Vergleich.

Diese Beitraege bleiben bestehen, auch wenn die alte Forecast-Accuracy neu eingeordnet werden muss.

---

## 15. Datenschutz- und Tweet-Inhaltsaudit

### 15.1 Ergebnis

Der Audit des sichtbaren Branchbaums lautet **FAIL** fuer eine strikte Null-Tweet-Content-Anforderung.

Es gibt keinen grossen eingecheckten Rohdatensatz. Einige Dateien enthalten jedoch vollstaendige oder tweetartige Inhalte.

### 15.2 Hohe Prioritaet

| Datei | Befund | Empfohlene Aktion |
| --- | --- | --- |
| `tweetsCompanyNumbersPrediction/src/tests/companyTweetsDummy.csv` | mehrere vollstaendige Textzeilen mit Handles oder Links | durch deterministische synthetische Texte ersetzen |
| `tweetsCompanyNumbersPrediction/src/predictSingleTweetGroup.py` | fest codierte laengere Tweetgruppe und kommentierte Beispiele | synthetisieren oder per CLI laden |
| `tweetsCompanyNumbersPrediction/src/tests/TestNearDuplicateDetector.py` | realistisch wirkende Marktposts | neutral synthetisieren |
| `tweetsCompanyNumbersPrediction/src/tests/TweetSentimentAnalysisTest.py` | Apple-/AAPL-artige Posttexte | neutral synthetisieren |

### 15.3 Niedrigere Prioritaet

Offensichtlich synthetische Textfixtures finden sich unter anderem in:

- `PipelineTest.py`,
- `TweetTextFilterTransformerTest.py`,
- `nlpvectorstest.py`,
- `HyperlinkRemoverTest.py`.

Sie sind keine Bulk-Rohdaten. Bei einer absolut strikten Null-Content-Policy sollten aber auch diese Beispiele ohne Handles, reale Links oder marktnahe Formulierungen auskommen.

### 15.4 Was bereits sauber ist

- Keine grossen CompanyTweets-Gesamtdaten im Repositorybaum.
- Keine eingecheckten trainierten Checkpoints oder exportierten Tweetgruppen.
- `tokenizer.json` enthaelt Vokabular, keine zusammenhaengenden Posts.
- Die neuen Ergebnis-JSONs enthalten Metriken, Features, Begriffe und Topicaggregate, aber keine vollstaendigen Texte.
- `numeric_text_topics_important_words.json` enthaelt keine Autoren, Handles, URLs oder Tweet-IDs.

### 15.5 Bereinigungsplan

1. `companyTweetsDummy.csv` durch eindeutig synthetische Kunstsaetze ersetzen.
2. Fest codierte Demo-Texte aus `predictSingleTweetGroup.py` entfernen oder ueber Eingabe laden.
3. Near-Duplicate- und Sentimentfixtures neutral neu formulieren.
4. Einen Pre-Commit-Scanner fuer `body`-CSV-Schemata, `t.co`, Handles, Cashtags und lange Textliterale hinzufuegen.
5. Alle Tests erneut ausfuehren.
6. Falls fuer eine Publikation erforderlich, die Git-Historie separat auf historische Blobs pruefen. Eine Bereinigung des aktuellen Baums entfernt keine alten Commits.

### 15.6 Minimaler Freigabestandard

- Kein Hochrisikotreffer im Branch-Tree-Scan.
- Keine CSV- oder JSON-Datei mit mehreren vollstaendigen Postsaetzen.
- Keine Autoren, Handles, Tweet-URLs oder Plattform-IDs in Fixtures.
- Forschungsartefakte speichern nur Aggregationen und Metriken.

---

## 16. Empfohlene Forschungsroadmap

### Prioritaet P0

#### Gate einfrieren und neu testen

Der Tesla-Konflikt-Gate darf nicht weiter anhand von 2017 bis 2019 veraendert werden. Architektur und Schwellen muessen auf neuen Quartalen eingefroren geprueft werden.

#### Baselines immer gemeinsam berichten

Jeder Lauf sollte mindestens enthalten:

- globale Mehrheitsbaseline,
- Saisonbaseline,
- Persistenzbaseline,
- reinen Textzweig,
- Hybrid,
- Text-Shuffle-Kontrolle.

#### Company-Quarter als primaere Einheit

Keine primaere Metrik darf einzelne Tweets oder Tweetgruppen wie unabhaengige Finanzereignisse zaehlen.

### Prioritaet P1

#### Fruehe Nowcast-Cutoffs

Features sollten nach festen Anteilen des Quartals ausgewertet werden:

- 25 %,
- 50 %,
- 75 %,
- 100 %.

So wird sichtbar, ob das Signal frueh verfuegbar ist oder erst durch nachtraegliche Berichts- und Zusammenfassungstexte entsteht.

#### Hierarchisches Firmenmodell

Ein gemeinsames Modell kann allgemeine Sprachmuster lernen und zugleich firmenspezifische Koeffizienten oder Adapter besitzen.

#### Interpretationsstabilitaet

Important Words und Topics sollten ueber Seeds, Folds und Unternehmen verglichen werden. Ein Begriff, der nur in einem Lauf erscheint, ist ein schwacher Befund.

#### Target-Provenienz

Pro Unternehmen sollten dokumentiert werden:

- Datenquelle,
- Kennzahlname,
- Einheit,
- Berichtsdatum,
- Reportingperioden,
- Definition der prozentualen Veraenderung,
- Klassengrenzen.

### Prioritaet P2

#### Fairer LSTM-Vergleich

Ein neuer LSTM- oder Attention-Zweig sollte:

- Vokabular und trainierbare Repraesentation nur aus vergangenen Perioden beziehen,
- korrekt maskieren oder packen,
- `padding_idx` setzen,
- Token -> Tweet -> Quartal hierarchisch modellieren,
- Textbeitrag separat ausgeben,
- gegen identische Saison- und Shuffle-Kontrollen antreten.

#### LLM-Vergleich

LSTM, Topicmodell und LLM sollten dieselben held-out Company-Quarters erhalten. Prompts, Temperatureinstellungen und Bewertung muessen vorab feststehen.

### Ziel bleibt Quartalszahl

Mehr unabhaengige Daten muessen nicht durch einen taeglichen Aktienkurs erzeugt werden. Geeignete Wege sind:

- mehr abgeschlossene Quartale,
- weitere sauber definierte Unternehmen,
- ein externer Holdout,
- mehrere Kennzahlen mit klarer Provenienz.

---

## 17. Reproduktion und Codeevidenz

### 17.1 Aktuelles numerisches Quartalsmodell ausfuehren

Aus dem Verzeichnis `tweetsCompanyNumbersPrediction/src`:

```bash
python trainNumericTextSignalQuarterModel.py
```

Ergebnis:

```text
output/numeric_text_signal_quarter_results.json
```

### 17.2 Topics und Important Words erzeugen

```bash
python extractNumericQuarterTopicsAndImportantWords.py
```

Ergebnis:

```text
output/numeric_text_topics_important_words.json
```

Der Lauf verarbeitete insgesamt:

- 5.170 relevante Amazon-Texte,
- 4.271 relevante Apple-Texte,
- 34.657 relevante Tesla-Texte,
- jeweils 20 vorhandene Quartale pro Unternehmen.

Fuer Topic- und Wortmodelle gilt die quartalsbalancierte Obergrenze von 250 Dokumenten. Die Vorhersagefeatures selbst bleiben unveraendert.

### 17.3 Tests

```bash
python -m unittest tests.alltestsuite
```

Beim letzten vollstaendigen Lauf bestanden:

- 111 registrierte Tests,
- 19 zusaetzliche Tests.

### 17.4 Wichtige aktuelle Dateien

| Thema | Datei |
| --- | --- |
| numerische Textmerkmale | [`NumericQuarterTextFeatures.py`](tweetsCompanyNumbersPrediction/src/classifier/NumericQuarterTextFeatures.py) |
| Training, Saison und Tesla-Gate | [`trainNumericTextSignalQuarterModel.py`](tweetsCompanyNumbersPrediction/src/trainNumericTextSignalQuarterModel.py) |
| exakte Attribution und past-only Topics | [`NumericQuarterTextExplanations.py`](tweetsCompanyNumbersPrediction/src/featureinterpretation/NumericQuarterTextExplanations.py) |
| Replay und Erklaerungsausgabe | [`extractNumericQuarterTopicsAndImportantWords.py`](tweetsCompanyNumbersPrediction/src/extractNumericQuarterTopicsAndImportantWords.py) |
| Tests der Erklaerung | [`NumericQuarterTextExplanationsTest.py`](tweetsCompanyNumbersPrediction/src/tests/NumericQuarterTextExplanationsTest.py) |

### 17.5 Zentrale main-Codeevidenz

Die Zeilenangaben beziehen sich auf `main`-Commit `23a2fbb5ee1820b2ec8840816133ab823ef84bb6`.

| Thema | Datei und Zeilen |
| --- | --- |
| Same-period Join | `tweetnumbersconnector/tweetnumbersconnector.py:22-48` |
| Klassenweise Gruppen | `nlpvectors/DataframeSplitter.py:40-85` |
| Tweetgruppe und SEP | `nlpvectors/TweetGroup.py:27-50` |
| LSTM-Endzustand | `classifier/LSTMNN.py:9-30` |
| Haupt-KFold | `trainNumbersPredictionModelStratifiedKFoldTemporalPerClass.py:71-117` |
| Globaler 80/20-Split | `trainNumbersPredictionModelTemporalSplit.py:60-78` |
| Expanding Window pro Klasse | `trainNumbersPredictionModelStratifiedExpandingWindowPerClass.py:76-113` |
| Trainer und Checkpointing | `classifier/Trainer.py:20-39` |
| Metriken | `classifier/ClassificationMetrics.py:12-34` |
| Topic-Backends | `topicmodelling/TopicExtractor.py:31-179` |
| Topicqualitaet | `topicmodelling/TopicEvaluation.py:24-48` |
| Wort, POS und Topic Mapping | `featureinterpretation/InterpretationDataframeUtil.py:8-31` |
| LLM-Topicvergleich | `topicmodelling/llmcomparison/LLMTopicsCompare.py:43-99` |
| Near-Duplicate-Erkennung | `tweetpreprocess/nearduplicates/NearDuplicateDetector.py:34-66` |

### 17.6 Neue Ergebnisartefakte und Datenschutz

Die neue Erklaerungsdatei speichert:

- Targets und Vorhersagen auf Company-Quarter-Ebene,
- Wahrscheinlichkeiten je Modellzweig,
- Featurewerte und Koeffizientenbeitraege,
- einzelne aggregierte Cue-Terme,
- past-only Important Words,
- Topicwoerter und Topicgewichte.

Sie speichert nicht:

- vollstaendige Texte,
- Autoren,
- Handles,
- URLs,
- Tweet-IDs,
- Dokument-IDs.

Ein rekursiver Key-Audit blockiert typische Rohtext- und Identifikatorfelder.

---

## 18. Abschliessendes Urteil

### Zur alten Implementierung

Die alte `main`-Codebasis ist wissenschaftlich interessant, weil sie:

- ein komplettes Informationssystem realisiert,
- mehrere Unternehmen und Kennzahlen abbildet,
- schwache Einzeltexte aggregiert,
- Vorhersage und Interpretation verbindet,
- Topicqualitaet quantitativ betrachtet,
- manuelle, neuronale und LLM-Perspektiven zusammenfuehrt,
- einen starken zeitlichen Fingerabdruck der Sprache sichtbar macht.

Sie ist jedoch kein ueberzeugender Beleg fuer 87 % echte Zukunftsprognose. Hauptgruende sind Quartalspseudoreplikation, Training auf spaeteren Bloecken, fehlende Saisonbaseline, transduktive Embeddings und konkrete Codeprobleme.

### Zur frueheren automatisierten Analyse

Die fruehere automatisierte Analyse hat die zentrale Schwachstelle des Evaluationsdesigns richtig erkannt. Sie wurde unzuverlaessig, wo sie aus einem berechtigten Befund universelle Aussagen ueber alle LSTMs, alle Checkpoints oder die gesamte Interpretationspipeline ableitete.

### Zum aktuellen Modell

Das aktuelle System verbessert die zeitliche Evaluation deutlich und erreicht explorativ 80,56 % Accuracy sowie MCC 0,7387. Dieses Ergebnis gehoert zu einem Hybrid. Der reine numerische Textzweig erreicht 50,00 % Accuracy und MCC 0,3224.

Der Tesla-Konflikt-Gate ist post hoc. Deshalb bleibt 75,00 % Accuracy und MCC 0,6633 die transparentere ungated Referenz, bis neue Quartale den Gate bestaetigen.

### Zu Topics und Important Words

Die Topic- und Wortanalyse ist nun enger mit dem tatsaechlichen Entscheidungsweg verbunden. Exakte Textfeaturebeitraege, modellnahe Cues, past-only Important Words und past-only Topics werden getrennt ausgewiesen.

Die wichtigste Grenze bleibt:

> Topics beschreiben den Kontext eines Modells. Sie beweisen weder eine kausale Wirkung auf die Finanzkennzahl noch einen additiven Beitrag zur Saisonprior oder zum Tesla-Gate.

### Endfazit

Das Projekt hat zwei gleichzeitig wahre Ergebnisse:

1. Die alte hohe Accuracy darf nicht als sauberer Zukunftsforecast interpretiert werden.
2. Die alte Forschungsplattform, die zeitlichen Sprachsignaturen und die Verbindung von Vorhersage mit interpretierbaren Topics sind wissenschaftlich wertvoll und verdienen eine leakage-saubere Fortsetzung.
