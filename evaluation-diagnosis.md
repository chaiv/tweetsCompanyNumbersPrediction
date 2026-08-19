# Was die alten Modelle gelernt haben - und worauf die aktuellen Ergebnisse wirklich beruhen

Umfassende, laienverständliche Evaluation der alten `main`-Implementierung, der direkten Code- und Ergebnisprüfung, der aktuellen Quartalsmodelle sowie der Topic- und Important-Word-Analyse.

Stand: 20. August 2026

**Hinweis zur Erstellung:** Die Codeprüfung, die experimentelle Rekonstruktion und das weitere Modelltraining wurden automatisiert mit ChatGPT 5.6 Sol sowie Claude Fable / Opus 5 durchgeführt.

## Kurzfassung

Dieses Projekt untersucht, ob sich aus öffentlichen Texten über Unternehmen die Entwicklung einer Quartalskennzahl ableiten lässt. Beispiele für solche Kennzahlen sind Amazon-Umsatz, Apple-EPS und Tesla-Auslieferungen. Außerdem soll erklärt werden, welche Wörter und Themen mit den Modellentscheidungen zusammenhängen.

Die wichtigsten Ergebnisse sind:

1. **Die alte Implementierung ist wissenschaftlich interessant.** Sie bildet ein ungewöhnlich breites Informationssystem ab: Tweets werden mit Berichtsperioden verbunden, zu Gruppen zusammengefasst, mit einem LSTM klassifiziert und anschließend bis zu Wörtern und Topics interpretiert. Besonders interessant ist der starke zeitliche und ereignisbezogene Fingerabdruck der Sprache.
2. **Die alte Accuracy von etwa 0,87 ist kein sauberer Beleg für echte Zukunftsprognose.** Viele Tweetgruppen desselben Quartals wurden wie voneinander unabhängige Testfälle behandelt. Außerdem konnte die Hauptauswertung für einen frühen Testblock auch spätere Textblöcke zum Training verwenden. Das Modell konnte deshalb Perioden und Quellen wiedererkennen, ohne eine unbekannte spätere Quartalszahl vorhersagen zu müssen.
3. **Die direkte Codeprüfung zeigt eine zentrale Evaluationsschwäche.** Es gibt 20 Quartalsergebnisse pro Unternehmen, aber nur zwei oder vier mögliche Klassen. Weil Gruppen statt ganzer Quartale getrennt wurden und für frühe Testblöcke spätere Blöcke im Training liegen konnten, misst die alte Auswertung nicht sauber die Vorhersage unbekannter Zukunftsquartale. Auch der Interpretationspfad enthält konkrete technische Fehler.
4. **Das aktuelle beste Ergebnis beträgt 80,56 % Accuracy und MCC 0,7387 auf 36 späteren Company-Quarters.** Dieses Ergebnis gehört zu einem Hybrid aus Saisonprior, numerischen Textsignalen und einem Tesla-Sonderzweig. Der Aufbau übernimmt Aggregation und Interpretierbarkeit aus dem alten System, trennt Saison, Textzahlen, Tesla-Level und Topics aber in kontrollierbare Zweige. Es ist kein reines Textmodell.
5. **Der isolierte numerische Textzweig erreicht 50,00 % Accuracy und MCC 0,3224.** Die transparente Variante ohne den nachträglich entworfenen Tesla-Konflikt-Gate erreicht 75,00 % Accuracy und MCC 0,6633.
6. **Die 80,56 % sind explorativ.** Der Tesla-Konflikt-Gate wurde nach Betrachtung der Fehler aus denselben Testjahren 2017 bis 2019 entworfen. Er muss auf neuen, vorher unberührten Quartalen bestätigt werden.
7. **Topics und Important Words sind jetzt zeitlich sauberer angebunden.** Exakte Modellattributionen werden von deskriptiven Topic-Erklärungen getrennt. Topicmodelle und Wortlexika werden nur auf früheren Quartalen trainiert und danach auf Zukunftsquartale angewandt.
8. **Der Branch ist noch nicht vollständig tweet-content-frei.** Große Rohdatensätze sind nicht eingecheckt, aber einige Demo- und Testdateien enthalten noch vollständige oder tweetartige Texte.

Die wissenschaftlich korrekte Gesamtaussage lautet daher:

> Die alte Arbeit zeigt eine breite, erklärbare Forschungsplattform und starke zeitliche Sprachsignaturen. Das aktuelle System zeigt explorativ, dass lokale Textsignale in einem leakage-bewussten Quartalshybrid nützlich sein können. Weder die alte 87-%-Accuracy noch die aktuellen 80,56 % belegen bereits ein bestätigtes reines Textmodell für unbekannte zukünftige Quartale.

---

## Inhaltsverzeichnis

1. [Die Forschungsfrage in einfachen Worten](#1-die-forschungsfrage-in-einfachen-worten)
2. [Wichtige Begriffe](#2-wichtige-begriffe)
3. [Prüfumfang und Beweisstandard](#3-prüfumfang-und-beweisstandard)
4. [Die alte main-Pipeline Schritt für Schritt](#4-die-alte-main-pipeline-schritt-für-schritt)
5. [Warum die alte Auswertung hohe Werte liefern konnte](#5-warum-die-alte-auswertung-hohe-werte-liefern-konnte)
6. [Was an der alten Implementierung wissenschaftlich interessant ist](#6-was-an-der-alten-implementierung-wissenschaftlich-interessant-ist)
7. [Schwache und fehlerhafte Stellen in main](#7-schwache-und-fehlerhafte-stellen-in-main)
8. [Direkte Code- und Ergebnisprüfung](#8-direkte-code--und-ergebnisprüfung)
9. [Herleitung und Aufbau des aktuellen Quartalsmodells](#9-herleitung-und-aufbau-des-aktuellen-quartalsmodells)
10. [Training und Zukunftstest](#10-training-und-zukunftstest)
11. [Ergebnisse und statistische Einordnung](#11-ergebnisse-und-statistische-einordnung)
12. [Topics und Important Words](#12-topics-und-important-words)
13. [Konkrete Erklärungsbeispiele](#13-konkrete-erklärungsbeispiele)
14. [Claim Ladder](#14-claim-ladder)
15. [Datenschutz- und Tweet-Inhaltsaudit](#15-datenschutz--und-tweet-inhaltsaudit)
16. [Empfohlene Forschungsroadmap](#16-empfohlene-forschungsroadmap)
17. [Reproduktion und Codeevidenz](#17-reproduktion-und-codeevidenz)
18. [Abschließendes Urteil](#18-abschließendes-urteil)

---

## 1. Die Forschungsfrage in einfachen Worten

### 1.1 Was soll vorhergesagt werden?

Das Ziel bleibt ausschließlich eine **Quartalskennzahl** beziehungsweise deren Veränderung. Es wird kein Aktienkurs als Ersatz-Ziel verwendet.

Im aktuellen Mehrfirmenexperiment sind dies:

| Unternehmen | Kennzahl |
| --- | --- |
| Amazon | Umsatz beziehungsweise Net Sales |
| Apple | Earnings per Share, kurz EPS |
| Tesla | Fahrzeugauslieferungen beziehungsweise Car Sales |

Die prozentuale Veränderung wird in vier Klassen eingeteilt:

| Klasse | Bedeutung |
| ---: | --- |
| 0 | Rückgang |
| 1 | schwacher Anstieg von 0 bis 15 % |
| 2 | moderater Anstieg von über 15 bis 30 % |
| 3 | starker Anstieg von über 30 % |

Die Richtung Rückgang gegen Anstieg kann zusätzlich ausgewertet werden. Sie bleibt aber nur eine einfachere Zusatzdiagnostik und ersetzt nicht das Vierklassenziel.

### 1.2 Welche Informationen darf das Modell verwenden?

Das aktuelle Textsystem verwendet nur lokal vorhandene Texte und daraus berechnete Aggregationen. Die Finanz-CSV und der zu prognostizierende aktuelle Quartalswert werden nicht als Eingabefeatures verwendet.

Vergangene Zielklassen dürfen für das Training und für einen Saisonprior verwendet werden. Das ist wichtig: Ein Modell ohne aktuelle Finanzwerte kann trotzdem historische Target-Labels kennen. Deshalb ist die korrekte Bezeichnung des besten Systems **no-finance hybrid** und nicht **pure text**.

### 1.3 Ist das ein Forecast des nächsten Quartals?

Nein, noch nicht im strengen Sinn von `Q -> Q+1`.

Die Texte eines Quartals werden verwendet, um die Kennzahl desselben Berichtsquartals einzuschätzen. Weil die Kennzahl typischerweise erst nach dem Ende des Quartals berichtet wird, ist das ein **Current-Quarter-Nowcast**.

Gleichzeitig wird zeitlich sauberer getestet: Das Modell wird auf früheren Jahren trainiert und auf späteren Jahren ausgewertet. Somit gilt:

- **Zeitlich zukünftiger Test:** ja.
- **Ziel des nächsten Quartals Q+1:** nein.
- **Ziel des Textquartals selbst:** ja.

Ein echtes Q+1-Experiment wäre möglich, müsste aber die Targets um ein Quartal verschieben und würde eine andere Forschungsfrage beantworten.

---

## 2. Wichtige Begriffe

### Company-Quarter

Eine Kombination aus Unternehmen und Quartal, zum Beispiel `Amazon 2018Q2`. Alle Texte dieser Kombination beziehen sich auf dieselbe Zielrealisierung. Deshalb ist ein Company-Quarter die primäre unabhängige Auswertungseinheit.

### Training, Validation und Test

- **Training:** Daraus lernt das Modell seine Parameter.
- **Validation:** Damit werden Modellvariante und Hyperparameter ausgewählt.
- **Test:** Diese Daten dürfen erst nach der Auswahl für die abschließende Bewertung verwendet werden.

### Leakage

Leakage bedeutet, dass Informationen aus dem Testfall direkt oder indirekt in das Training oder in die Modellauswahl gelangen. Das kann auch ohne identische Texte passieren. Wenn andere Texte desselben Quartals im Training liegen, kennt das Modell bereits viele Merkmale genau der Periode, die es angeblich vorhersagen soll.

### Pseudoreplikation

Ein Quartal hat nur ein Finanzziel. Werden tausend Tweetgruppen dieses Quartals als tausend unabhängige Testfälle gezählt, wird dieselbe Zielrealisierung tausendfach wiederholt. Die Stichprobe wirkt dadurch größer und sicherer, als sie wirklich ist.

### Accuracy

Der Anteil korrekt vorhergesagter Klassen. 80,56 % bedeuten hier 29 richtige Entscheidungen bei 36 Company-Quarters.

### MCC

Der Matthews Correlation Coefficient berücksichtigt alle Teile der Konfusionsmatrix und ist bei ungleich verteilten Klassen aussagekräftiger als Accuracy allein. Ein Wert von 1 ist perfekt, 0 entspricht grob keinem Zusammenhang und negative Werte sprechen für systematisch falsche Entscheidungen.

### Log Loss

Log Loss bewertet nicht nur die gewählte Klasse, sondern auch, wie sicher das Modell war. Eine selbstbewusst falsche Vorhersage wird stärker bestraft.

### Baseline

Eine einfache Vergleichsregel. Ein komplexes Textmodell muss beispielsweise besser sein als eine Saisonregel, die nur frühere gleiche Kalenderquartale betrachtet.

### Shuffle-Kontrolle

Textsignale werden absichtlich zwischen Quartalen vertauscht, während Targets und restliche Modellteile gleich bleiben. Fällt die Leistung nicht, war der Text vermutlich nicht ausschlaggebend.

### Attribution

Eine Attribution beschreibt, wie stark ein konkretes Eingabefeature zu einer Modellentscheidung beiträgt. Sie erklärt das Modell, beweist aber keine ökonomische Ursache.

### Topic

Ein Topic ist eine Gruppe häufig gemeinsam auftretender Begriffe. Ein Topic fasst Textkontext zusammen. Es ist nicht automatisch ein kausaler Grund für eine Quartalsänderung.

---

## 3. Prüfumfang und Beweisstandard

### 3.1 Geprüfte Stände

| Prüfobjekt | Referenz |
| --- | --- |
| Alte Implementierung | `main` am Commit `23a2fbb5ee1820b2ec8840816133ab823ef84bb6` |
| Aktueller Branch | `baselines`, HEAD `0e708b1bc5a4a58c75c27f5f6ccb40e8a2f3e9bf` plus aktueller Working Tree |
| Primäres Ergebnis | `output/numeric_text_signal_quarter_results.json` |
| Topic-/Wortergebnis | `output/numeric_text_topics_important_words.json` |
| Testzeitraum | rollende Tests für 2017, 2018 und 2019 |

Der `main`-Branch wurde direkt aus den Git-Objekten gelesen. Ein Checkout war nicht nötig, sodass der bereits veränderte Arbeitsbaum unangetastet blieb.

### 3.2 Drei Evidenzstufen

| Stufe | Bedeutung | Beispiel |
| --- | --- | --- |
| Codefakt | Direkt im referenzierten Code sichtbar | `pd.to_datetime(post_date)` wird ohne `unit='s'` aufgerufen. |
| Reproduziertes Ergebnis | Targets, Wahrscheinlichkeiten und Metriken liegen lokal vor und wurden erneut berechnet | 29 von 36 richtigen Company-Quarters, Accuracy 0,8056, MCC 0,7387. |
| Berichteter Altbefund | Nur in der Dissertation oder einem historischen Ergebnisbericht dokumentiert | Die exakte alte Quartalserkennungsrate kann ohne damaliges Ergebnisartefakt nicht vollständig reproduziert werden. |

### 3.3 Warum es nur 36 primäre Testfälle gibt

Es gibt drei Unternehmen, vier Quartale pro Jahr und drei Testjahre:

```text
3 Unternehmen x 4 Quartale x 3 Testjahre = 36 Company-Quarters
```

Ob intern 1.000 oder 1.000.000 Texte verarbeitet werden, ändert diese Zahl nicht. Mehr Texte können eine Quartalsrepräsentation verbessern, erzeugen aber keine neuen unabhängigen Finanzereignisse.

### 3.4 Grenze dieses Audits

Der Audit bewertet den sichtbaren Repository-Stand. Er rekonstruiert nicht die exakte Hardware, Datenversion und Checkpointauswahl aller publizierten historischen Läufe. Deshalb kann nicht sicher bestimmt werden, welcher alte Checkpoint jede publizierte Zahl erzeugt hat.

---

## 4. Die alte main-Pipeline Schritt für Schritt

### 4.1 Daten und Experimentregistry

`PredictionModelPath.py` definiert Experimente für mehrere Unternehmen, Kennzahlen, Gruppengrößen und Zielvarianten. Im Code finden sich unter anderem:

- Amazon Revenue,
- Apple EPS,
- Tesla Car Sales,
- Google Search Engine Market Share,
- binäre Klassen,
- vier Klassen,
- Gruppengrößen 5, 10 und 20.

Das ist wissenschaftlich interessant, weil dieselbe Gesamtidee auf unterschiedliche Unternehmen und Kennzahltypen angewandt werden kann.

### 4.2 Verbindung von Tweets und Finanzzahlen

`TweetNumbersConnector` sucht die Finanzzeile, deren Zeitintervall den Zeitstempel eines Tweets umfasst. Dabei gelten zwei gute Integritätsregeln:

- Fehlt eine passende Finanzzeile, bricht der Prozess ab.
- Passen mehrere Finanzzeilen, bricht der Prozess ebenfalls ab.

Der Tweet erhält damit den Wert seines eigenen Berichtszeitraums. Die alte README beschrieb dagegen teilweise die nächste gemeldete Zahl. Code und Dokumentation meinten somit nicht denselben Prognosehorizont.

### 4.3 Diskretisierung des Targets

Der vierstufige Pfad verwendet Rückgang, schwachen, moderaten und starken Anstieg. Im alten Klassifikator gibt es dabei technische Randprobleme:

- Die Intervalle überlappen an 15 und 30.
- Wegen der ersten passenden Regel fallen exakt 15 und 30 in die niedrigere Klasse.
- Zwischen -0,01 und 0 bleibt eine kleine Lücke.
- Der binäre Pfad und der Vierklassenpfad verwenden nicht immer dieselbe Skala: einmal Verhältnis, einmal Prozentwert.

Das Target-Schema sollte daher explizit versioniert und mit Grenzwerttests abgesichert werden.

### 4.4 Bildung der Tweetgruppen

`DataframeSplitter.getSplitIds` arbeitet vereinfacht so:

1. Alle Zeilen einer Klasse werden ausgewählt.
2. Diese Zeilen werden in fortlaufende Blöcke der Größe 5, 10 oder 20 geschnitten.
3. Jeder Block wird zu einem Trainingssample.
4. Alle Texte im Block tragen dasselbe Klassenlabel.

Die Gruppenbildung kennt jedoch keine explizite Quartalsgrenze. Dadurch können an einer Periodengrenze Texte verschiedener Quartale in einer Gruppe landen, wenn sie dieselbe Klasse haben.

Noch wichtiger ist die Wiederholung des Targets: Ein einzelnes Quartal kann sehr viele Gruppen erzeugen, obwohl alle Gruppen dieselbe Finanzrealisierung teilen.

### 4.5 Repräsentation einer Gruppe

`TweetGroup` tokenisiert jeden Post und verbindet die Tokenfolgen mit einem `<SEP>`-Token. Das ist eine sinnvolle Idee, weil Tweetgrenzen nicht vollständig verschwinden.

Die Gruppe wird anschließend als eine lange Sequenz an das LSTM gegeben. Eine echte Hierarchie `Token -> Tweet -> Quartal` existiert im alten Modell noch nicht.

### 4.6 Top2Vec und Wortvektoren

Top2Vec wird auf dem lokalen Textkorpus trainiert. Seine Wortvektoren werden als 300-dimensionale Initialisierung für das LSTM verwendet. Damit dient derselbe semantische Raum sowohl der Vorhersage als auch der Topicinterpretation.

Das ist konzeptionell elegant, für einen strikten Zukunftstest aber problematisch: Wenn Top2Vec vor dem Split auf dem gesamten Korpus trainiert wird, beeinflussen spätere Testtexte bereits Vokabular und Vektoren. Das ist kein direktes Label-Leakage, aber eine transduktive Nutzung der Zukunftstexte.

### 4.7 Das alte LSTM

Das Hauptmodell besitzt:

- ein trainierbares Embedding,
- ein zweischichtiges LSTM mit Hidden Size 512,
- zwei verborgene lineare Schichten,
- eine finale Klassenausgabe.

Der letzte Hidden State wird als Repräsentation der gesamten gepaddeten Sequenz verwendet. Das wäre bei korrekt gepackten Sequenzen grundsätzlich möglich. Im alten Code fehlen aber echte Sequenzlängen, Packing und Maskierung. Dadurch kann der Endzustand viele PAD-Schritte durchlaufen.

### 4.8 Training

Der Lightning-Trainer verwendet:

- CUDA standardmäßig,
- Mixed Precision,
- TensorBoard-Logging,
- Checkpointing nach Validation Loss,
- Early Stopping,
- optional Class Weights.

Diese Infrastruktur ist eine Stärke. Einige Details sind jedoch problematisch: Die Early-Stopping-Patience ist genauso groß wie die maximale Epochenzahl, und manuell rekonstruierte Checkpointpfade können eine alte unversionierte Datei laden.

### 4.9 Alte Hauptauswertung

Die Hauptauswertung teilt die Gruppen jeder Klasse in zehn chronologische Blöcke. Für Fold `k` gilt:

- Block `k` wird Test.
- Alle anderen Blöcke werden Training und Validation.

Damit können für einen frühen Testblock auch spätere Blöcke im Training liegen. Außerdem können andere Gruppen desselben Quartals auf beiden Seiten vorkommen.

Die Auswertung ist daher kein strenger Forecast späterer unbekannter Quartale.

### 4.10 Weitere Splitideen

Die Codebasis enthält auch:

- einen globalen 80/20-Zeitsplit,
- eine klassenweise Expanding-Window-Variante,
- eine stratified-temporal Variante,
- eine Subsequent-Variante für Interpretation.

Dass mehrere Protokolle implementiert wurden, zeigt das richtige wissenschaftliche Anliegen: Ergebnisse sollen gegen unterschiedliche Zeitannahmen geprüft werden. Die konkrete alte Umsetzung behebt jedoch nicht automatisch Quartalspseudoreplikation, Targetstratifizierung und die fehlerhafte Zeitkonvertierung.

### 4.11 Alte Interpretationspipeline

Die alte Idee war:

1. Integrated Gradients berechnet Tokenattributionen.
2. Token werden wieder ihren Tweets zugeordnet.
3. Originalwort und Part-of-Speech-Tag werden hinzugefügt.
4. Top2Vec oder BERTopic ordnet Dokumente Topics zu.
5. Important Words und Topics werden gemeinsam analysiert.
6. Manuelle und LLM-generierte Topics können mit Modelltopics verglichen werden.

Diese Forschungslogik ist stark. Mehrere Implementierungsdetails waren jedoch fehlerhaft; sie werden in Abschnitt 7 beschrieben.

---

## 5. Warum die alte Auswertung hohe Werte liefern konnte

### 5.1 Ein synthetisches Beispiel

Angenommen, Tesla 2018Q4 hat Klasse 3 und es gibt 10.000 Texte. Bei Gruppengröße 10 entstehen etwa 1.000 Gruppen mit demselben Label.

Wenn ein Split 100 Gruppen testet und 900 Gruppen desselben Quartals trainiert, sind die Tweet-IDs zwar verschieden. Das Modell sieht aber sehr viele andere Texte aus genau derselben Periode.

Es kann dadurch lernen:

| Textmerkmal | Was es verraten kann |
| --- | --- |
| damaliger Produktname | Quartal oder Produktzyklus |
| damalige Kampagne | Zeitraum |
| bestimmte Nachrichtenquelle | Quelle und Erscheinungsphase |
| zeittypische Marktwörter | Marktregime |
| wiederkehrendes Template | Quelle oder Zeitraum |

Da das Quartal im vorbereiteten Datensatz ein festes Label besitzt, reicht eine Periodenerkennung oft schon für eine scheinbar gute Finanzklassifikation.

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

Bei 2015Q1 bis 2019Q4 existieren 20 Quartalsrealisierungen pro Unternehmen. Es gibt aber nur vier mögliche Klassenwerte oder beim binären Ziel zwei.

Korrekt ist:

> Viele Tweetgruppen teilen sich nur 20 unabhängige Quartalsergebnisse.

Falsch ist:

> Es gibt 20 verschiedene Klassenlabels.

### 5.4 Die Saisonbaseline

Viele Unternehmenskennzahlen haben wiederkehrende Quartalsmuster. Beispielsweise kann Q4 regelmäßig anders aussehen als Q1.

Eine Saisonbaseline fragt für ein neues Q2 nur: Welche Klassen hatten frühere Q2 desselben Unternehmens?

Wenn diese einfache Regel bereits stark ist, muss ein Textmodell zeigen, welchen zusätzlichen Nutzen der Text liefert. Ein Vergleich nur mit dem globalen Mehrheitslabel reicht nicht.

---

## 6. Was an der alten Implementierung wissenschaftlich interessant ist

Die methodischen Schwächen machen die alte Arbeit nicht wertlos. Sie verändern, welche Schlussfolgerung erlaubt ist.

### 6.1 Ein vollständiges Informationssystem

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

Damit untersucht die Dissertation einen durchgängigen Erkenntnisprozess von öffentlichem Text bis zu erklärbaren Unternehmenskennzahlen.

### 6.2 Mehrere Unternehmen und Kennzahlen

Die gemeinsame Architektur wird auf Umsatz, EPS, Fahrzeugzahlen und Suchmaschinenmarktanteil angewandt. Das ist wertvoll, weil ein Verfahren, das nur für eine einzelne Kennzahl funktioniert, weniger allgemein ist.

Die alte Codebasis beweist noch keine saubere firmenübergreifende Generalisierung. Sie schafft aber eine sinnvolle vergleichende Versuchsmatrix.

### 6.3 Binäre und vierstufige Ziele

Die Trennung zwischen Richtung und Stärke ist wissenschaftlich sinnvoll:

- Binär beantwortet: Rückgang oder Anstieg?
- Vierstufig beantwortet: Wie stark ist die Veränderung?

Die vier Klassen haben eine natürliche Ordnung. Das alte Modell behandelt sie nominal; spätere ordinale Modelle können diese Struktur explizit nutzen.

### 6.4 Multi-Scale-Aggregation

Die Gruppengrößen 5, 10 und 20 sind mehr als nur Hyperparameter. Sie bilden eine Forschungsfrage ab:

> Wie viel kollektiver öffentlicher Diskurs wird benötigt, damit aus schwachen Einzeltexten ein stabiles Signal entsteht?

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
- Welche Signale erscheinen vor einer Ergebnisveröffentlichung?
- Welche erscheinen erst danach?
- Welcher Textbeitrag bleibt nach Kontrolle für Saison und Quelle übrig?

### 6.6 Gemeinsamer Raum für Vorhersage und Topics

Top2Vec liefert sowohl Wortvektoren für den LSTM als auch Topics. Damit kann untersucht werden, ob semantische Achsen gleichzeitig für Klassifikation und Interpretation relevant sind.

Diese Verbindung ist originell, muss aber pro Fold sauber trainiert oder als feste externe Repräsentation behandelt werden.

### 6.7 Token -> Tweet -> Topic

Die Kombination von Integrated Gradients, Originalwort, POS-Tag und Dokumenttopic ist ein sinnvoller Versuch, lokale Modellentscheidungen in eine höherstufige sozialwissenschaftliche Interpretation zu überführen.

Heute sollte dieser Pfad so umgesetzt werden:

1. Nur echte Zukunftstestfälle erklären.
2. PAD- und SEP-Tokens ausschließen.
3. Signed und absolute Attribution getrennt speichern.
4. Pro Tweet korrekt aggregieren.
5. Erst danach nach Topics zusammenfassen.
6. Stabilität über Folds und Seeds prüfen.

### 6.8 Mehrere Topicmodelle und Qualitätsdimensionen

Eine gemeinsame Extractor-Schnittstelle unterstützt Top2Vec und BERTopic. Topicqualität wird über Coherence, Diversity und Silhouette betrachtet.

Das ist wissenschaftlich besser als die Annahme, eine einzige Topiczerlegung sei die Wahrheit. Zusätzlich sollten zeitliche Stabilität und held-out Generalisierung gemessen werden.

### 6.9 Mensch-Maschine- und LLM-Vergleich

`ManualTopicAnalyzer` und `LLMTopicsCompare` vergleichen manuelle oder LLM-generierte Begriffe mit Modelltopics, sowohl direkt als auch im Embeddingraum.

Das ist als Triangulationsdesign interessant. Für eine belastbare Studie braucht es:

- verblindete Rater,
- vorab definierte Prompts,
- feste Ähnlichkeitsschwellen,
- Inter-Rater-Reliabilität,
- dieselben held-out Dokumente für alle Systeme.

### 6.10 Gute Daten- und Evaluationsideen

Weitere erhaltenswerte Bausteine sind:

- genau eine Finanzzeile pro Zeitintervall,
- SimHash-basierte Near-Duplicate-Erkennung,
- Class Weights und EqualClassSampler,
- Precision, Recall, F1, Accuracy und MCC,
- gespeicherte Testindizes,
- mehrere zeitliche Splitvarianten,
- TensorBoard und Checkpointing.

Diese Elemente zeigen methodisches Problembewusstsein. Sie sind jedoch kein automatischer Beleg dafür, dass jeder resultierende Lauf valide war.

---

## 7. Schwache und fehlerhafte Stellen in main

### 7.1 Evaluationsdesign

| Problem | Auswirkung |
| --- | --- |
| Gruppen statt Company-Quarters werden bewertet | Ein Quartalsziel wird sehr oft gezählt. |
| Andere Gruppen desselben Quartals können in Train und Test liegen | Periodenwiedererkennung wird möglich. |
| Haupt-KFold trainiert auf allen anderen Blöcken | Für frühe Tests können spätere Texte im Training liegen. |
| Validation wird innerhalb des Trainingspools zufällig stratifiziert | Training und Validation können dieselben Perioden teilen. |
| Saisonbaseline fehlt im Hauptlauf | Textskill und Quartalssaison werden verwechselt. |
| Balancing erfolgt teilweise vor dem Split | Zeitabdeckung und Klassenhäufigkeiten werden verändert. |

### 7.2 Transduktive Top2Vec-Nutzung

Top2Vec wird auf dem kompletten Textkorpus trainiert, bevor der Forecast-Split feststeht. Spätere Testtexte beeinflussen dadurch:

- Vokabular,
- semantische Nachbarschaften,
- Startvektoren des LSTM,
- Topicstruktur.

Für einen strikten Zukunftstest muss das Topic-/Embeddingmodell nur auf vergangenen Texten trainiert oder als klar externe, zeitlich fixe Ressource deklariert werden.

### 7.3 Target- und Metrikdefinition

- Der Connector liefert den Wert desselben Intervalls, die alte README beschrieb teilweise Q+1.
- Verhältnis und Prozentwert tragen ähnliche Namen.
- Die Vierklassenintervalle haben Überlappungen und eine kleine Lücke.
- Tesla-Produktion, Auslieferung und Absatz dürfen nicht ohne dokumentierte Provenienz gleichgesetzt werden.

### 7.4 Padding und letzter Hidden State

Die Batches werden auf die längste Sequenz gepaddet. Das LSTM erhält aber keine echten Längen und keine Maske. Der verwendete letzte Hidden State kann deshalb einen großen Anteil PAD-Verarbeitung enthalten.

Die korrekte Aussage ist nicht, dass ein letzter Hidden State immer falsch sei. Er ist nur in dieser unmaskierten Kombination problematisch.

Mögliche Reparaturen:

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

Die numerische Reihenfolge kann dabei erhalten bleiben, Kalenderjahr und Quartal sind jedoch falsch. Das folgt direkt aus der Einheit des gespeicherten Zeitstempels und dem Verhalten von `pd.to_datetime`.

### 7.7 Reproduzierbarkeit und Portabilität

- Seeds werden nicht durchgängig für Torch und NumPy gesetzt.
- `loadModel` ist teilweise hart an `cuda:0` gebunden.
- `map_location` fehlt beim Laden.
- `strict=False` kann inkompatible State-Dict-Keys verdecken.
- Early Stopping kann bei zehn Epochen und Patience zehn kaum früh stoppen.

### 7.8 Fehler im alten Topic- und Important-Word-Pfad

| Befund | Bedeutung |
| --- | --- |
| `extractMostImportantWords.py` verwendet `df.head(50000)` und nicht die gespeicherten Testindizes | Trainingstexte können in der Erklärung landen. |
| Integrated Gradients wird pro Sample durch sein eigenes Maximum geteilt | Rangfolgen innerhalb eines Samples bleiben, Größen zwischen Samples werden unvergleichbar. |
| Kein Schutz gegen Division durch null | Nullattribution kann NaN erzeugen. |
| Eine Gruppensumme wird beim Flattening für mehrere Tweets wiederholt | Tweet-Level-Werte sind falsch aggregiert. |
| Das Captum-Konvergenzdelta wird verworfen | Attributionsqualität wird nicht kontrolliert. |
| `findMostImportantTopicTweets.py` referenziert eine nicht definierte Variable | Das Skript kann in diesem Pfad abbrechen. |

Die Forschungsfrage bleibt relevant. Die alte konkrete Ausgabe darf aber nicht pauschal als mechanisch korrekt bezeichnet werden.

---

## 8. Direkte Code- und Ergebnisprüfung

### 8.1 Wie die Prüfung durchgeführt wurde

Die Prüfung wurde direkt aus dem Repository und den lokalen Ergebnisartefakten rekonstruiert:

1. Der zu prüfende `main`-Stand wurde auf Commit `23a2fbb5ee1820b2ec8840816133ab823ef84bb6` fixiert.
2. Der Targetfluss wurde vom Finanz-Join über die Klassenbildung bis zum Samplelabel verfolgt.
3. Für jedes Sample wurde bestimmt, welche reale Beobachtung unabhängig ist: nicht die Tweetgruppe, sondern das Company-Quarter.
4. Die Splitskripte wurden daraufhin geprüft, ob ein Testquartal vollständig außerhalb von Training und Validation bleibt und ob alle Trainingsdaten zeitlich davor liegen.
5. Unüberwachte Vorverarbeitung wie Top2Vec wurde darauf geprüft, ob Testtexte vor dem Split in Vokabular oder Vektoren eingehen.
6. LSTM-Batching, Padding, Datumsumwandlung und Checkpointladen wurden entlang der tatsächlichen Aufrufpfade untersucht.
7. Der Interpretationspfad wurde von Integrated Gradients über das Token-/Tweet-Mapping bis zur Topiczuordnung verfolgt.
8. Für das neue System wurden gespeicherte Targets und Wahrscheinlichkeiten erneut geladen; Accuracy, MCC, Log Loss, Firmenmetriken, Wilson-Intervall und gepaarte Shuffle-Kontrolle wurden nachgerechnet.

Damit werden drei Dinge getrennt: direkt sichtbare Codefakten, lokal reproduzierbare Ergebnisse und historische Angaben, für die das ursprüngliche Ergebnisartefakt fehlt.

### 8.2 Direkt aus dem Code belegte Befunde

| Befund | Codeevidenz | Konsequenz |
| --- | --- | --- |
| Das Ziel ist innerhalb eines Company-Quarters konstant. | Der Connector ordnet allen Texten desselben Berichtsintervalls denselben Finanzwert zu. | Tweetgruppen desselben Quartals sind keine unabhängigen Finanzfälle. |
| Die alte Gruppenbildung trennt Klassenblöcke, nicht ganze Quartale. | `DataframeSplitter.getSplitIds` schneidet fortlaufende Zeilen einer Klasse in Gruppen. | Andere Texte derselben Periode können das Periodenmuster bereits im Training zeigen. |
| Das Haupt-KFold trainiert für Testblock `k` auf allen anderen Blöcken. | Das Splitskript schließt nur den aktuellen Block aus. | Für frühe Testblöcke können spätere Texte im Training liegen. |
| Top2Vec wird vor dem Forecast-Split auf dem Vollkorpus trainiert. | Das Topictraining liest das gesamte Dataframe. | Zukunftstexte beeinflussen Vokabular, Nachbarschaften und LSTM-Startvektoren transduktiv. |
| Epoch-Sekunden werden ohne `unit='s'` konvertiert. | Direkter Aufruf von `pd.to_datetime(post_date)`. | Kalenderjahr und Quartal werden falsch als 1970 interpretiert. |
| Bestimmte Reload-Pfade bauen einen unversionierten Checkpointnamen zusammen. | Manuelles Laden nach Lightning-Checkpointing. | Bei vorhandenen versionierten Dateien kann ein älterer Checkpoint geladen werden. |
| Der alte Interpretationspfad verwendet nicht durchgängig die gespeicherten Testindizes und aggregiert teilweise falsch. | `df.head(50000)`, Sample-Normalisierung und wiederholte Gruppensummen. | Alte Important-Word-Ausgaben sind nicht automatisch reine Testerklärungen. |

### 8.3 Präzise Grenzen der Schlussfolgerung

| Frage | Was aus dem Repository folgt |
| --- | --- |
| Gibt es 20 verschiedene Klassen? | Nein. Es gibt 20 Quartals-Outcomes pro Unternehmen, aber nur zwei oder vier Klassenwerte. |
| Ist der letzte LSTM-Hidden-State immer falsch? | Nein. Problematisch ist hier die Kombination aus Endzustand und unmaskiertem Padding. |
| Beweist ein kollabierter Lauf, dass kein LSTM lernen kann? | Nein. Er belegt nur das Verhalten der geprüften Konfiguration. |
| Lädt jedes Trainingsskript zwingend einen alten Checkpoint? | Nein. Gefährdet sind die manuellen Reload-Pfade; der Trainer verwendet `ckpt_path='best'`. |
| Beweist der alte Split, dass Text nutzlos ist? | Nein. Er verhindert nur, den beobachteten Wert sauber als Zukunftsgeneralisierung zu interpretieren. |
| Muss das Target zwingend Q+1 sein? | Nein. Q+1 ist für einen Next-Quarter-Forecast nötig; ein Current-Quarter-Nowcast ist eine andere legitime Aufgabe. |
| Werden Topicergebnisse durch eine schwache Forecast-Evaluation automatisch richtig oder falsch? | Nein. Topicfit, Attribution und Testzuordnung müssen separat geprüft werden. |

### 8.4 Was ohne historische Artefakte offen bleibt

Aus dem sichtbaren Code allein lässt sich nicht sicher bestimmen:

- welcher exakte Checkpoint die publizierten 0,87/0,77 erzeugt hat,
- ob eine berichtete Quartalserkennungsrate von 91,8 % mit genau diesem Datenstand reproduzierbar ist,
- wie sich jede nicht eingecheckte LSTM-Konfiguration verhalten hat,
- welcher Anteil der alten Korrelation aus Finanzinformation, Saison, Quelle oder Periodenerkennung stammt.

Deshalb lautet der direkte Befund nicht „Text funktioniert nicht“, sondern:

> Das alte Evaluationsdesign kann echte Textinformation nicht sauber von Quartals-, Saison- und Quellenwiedererkennung trennen. Genau diese Komponenten müssen im neuen System getrennt modelliert und kontrolliert werden.

### 8.5 Prüfschritt 1: Dokumentversion und publiziertes Protokoll

Das direkt geprüfte Dissertations-PDF umfasst 236 PDF-Seiten. Titelblatt und PDF-Metadaten datieren es auf 2026 beziehungsweise den 10. März 2026. Seitenangaben in dieser Analyse beziehen sich auf die gedruckten Seitenzahlen.

Seite 114 beschreibt zehnfache Kreuzvalidierung, bei der jedes Eingabesample über die Folds sowohl Trainings- als auch Testsample wird, mit Batchgröße 100. Unter den committeten Skripten passt `trainNumbersPredictionModelOnlySubsequentTweetsOrder.py` am besten dazu (`KFold(n_splits=10, shuffle=True, random_state=1337)`, Batchgröße 100). Dieses Protokoll trennt weder Zeit noch Quartale und kann daher keinen echten Zukunftsforecast belegen.

Die Beleggrenze ist wichtig: Ohne Run-Manifest, exakten damaligen Commit, Konfigurationsdump und Checkpoint ist nicht beweisbar, dass genau dieses Skript jede publizierte Zahl erzeugt hat. Es ist die plausibelste Zuordnung im vorhandenen Repository.

### 8.6 Prüfschritt 2: Tabellen und Aussagen der Dissertation

- **Tabelle 9** (binäre Klassenhäufigkeiten, S. 112): Die Amazon-Zeile nennt 16 Rückgänge und 18 Anstiege, also 34 Quartale, obwohl nur 24 existieren. Die archivierte CSV ergibt 6/18; Tabelle 10 nennt auf der Folgeseite selbst sechs Rückgänge. Der Fließtext zählt bei Amazon und Apple offenbar zusätzlich die Basiszeile mit undefinierter Veränderung.
- **Tesla in Tabelle 9:** 8/16 entsteht exakt, wenn die Ratio-Schwelle `>1.0` auf Prozentwerte angewandt wird. Die zwei Anstiege unter einem Prozent (2015Q3: +0,62 %, 2018Q1: +0,37 %) werden dann fälschlich als Rückgänge gezählt. Die archivierten binären Tweetlabels sind davon nicht betroffen.
- **Tabelle 10** (Mehrklassenhäufigkeiten, S. 113): Die Werte stimmen für alle drei Unternehmen mit den archivierten CSVs und den dokumentierten Klassengrenzen überein.
- **Tabelle 16** (S. 123–124): FNN-binär Tesla@20 und LSTM-binär Apple@20 besitzen dasselbe Fünftupel `(0,61, 0,67, 0,60, 0,67, 0,27)`. Das ist verdächtig und mit einem Kopierartefakt vereinbar, ohne Laufprotokolle aber nicht beweisbar. Stichprobengrößen, Konfidenzintervalle und Signifikanzangaben fehlen.
- **Tabelle 17** (S. 125): Der Text kündigt Apple und Tesla an, die Tabelle enthält jedoch auch Amazon und erklärt anschließend Amazon zum besten Ergebnis.
- **Tabellen 11–13 gegen den Code:** Dokumentierte und implementierte FNN-Eingabedimension, LSTM-Dropout, Transformer-Dimension und Batchgröße passen nicht vollständig zusammen. Keine committete Konfiguration reproduziert alle dokumentierten Angaben exakt.
- **Topic-Kohärenz:** `TopicEvaluation` erzeugte Gensims `CoherenceModel` ohne `coherence`-Argument. Gensims Standard ist `c_v`, nicht `c_uci`. Die Thesis bezeichnet die Werte dennoch als UCI-Kohärenz. Ohne einen anderen, nicht committeten Auswertungspfad müssen die historischen Codewerte als `c_v` behandelt werden. Den qualitativen Aussagen „decent quality“ und „underperforming“ fehlt außerdem ein dokumentierter Baseline-Vergleich.

### 8.7 Prüfschritt 3: Daten-, Gruppen- und Labelbefunde

- **Duplikate:** Die Dissertation fordert ihre Entfernung vor dem Training und erklärt im Fazit, exakte Duplikate seien gefiltert worden. Die alte Erzeugung der gelabelten Dataframes enthielt diesen Schritt jedoch nicht. In der binären Apple-Datei wiederholen 194.278 von 1.425.013 Zeilen einen früheren `body` (13,63 %), in der binären Amazon-Datei 113.934 von 718.715 (15,85 %). Die Amazon-Mehrklassendatei besitzt bei gleicher Zeilenzahl nur 66.549 Wiederholungen (9,26 %), was zusätzliche Artefakt- beziehungsweise Versionsdrift zeigt.
- **Labelabhängige Gruppenbildung:** `getSplitIds` filtert zuerst nach Klasse und bildet erst danach Tweetgruppen. Dieselben Eingaben können auf ungelabelten Zukunftsdaten nicht konstruiert werden. Bei Gruppengröße 10 überqueren 27 von 142.503 Apple-Mehrklassengruppen und 28 von 71.873 Amazon-Mehrklassengruppen mindestens eine Kalenderquartalsgrenze.
- **Apple-Klasse als Periodenproxy:** Nach `EqualClassSampler` stammen 99,922 % der Apple-Klasse 1 aus 2015Q3; 0,078 % liegen wegen der archivierten Zeit-/Intervallgrenze in 2015Q2. Die Klasse ist damit fast, aber nicht vollständig, ein einzelnes Quartal.
- **Binär-/Mehrklassen-Inversion:** Der binäre Apple-MCC erreicht höchstens 0,31, der vierklassige MCC auf derselben Kennzahl 0,80. Das ist ein Warnsignal und mit Periodenerkennung vereinbar, aber kein Beweis gegen ein echtes Textsignal. Separate Modelle können sich durch Balancierung, Optimierung, Kalibrierung und Labelrauschen unterschiedlich verhalten.
- **Labeldefinitionsdrift:** Die archivierten `change_4`-Spalten entsprechen ungefähr Grenzen von 0/10/25, während Code und Dissertation 0/15/30 verwenden. Dadurch widersprechen sich archivierte und neu berechnete Klassen in drei Amazon- und vier Tesla-Zeilen.

### 8.8 Prüfschritt 4: Balancierung und Modellmechanik

`EqualClassSampler` behält die ersten `n` Zeilen jeder Klasse. Für Apple mehrklassig ist `n = 93.686`: Klasse 0 stammt zu 100 % aus 2015Q1; Klasse 1 zu 99,922 % aus 2015Q3; Klasse 2 zu 99,878 % aus 2016Q3; Klasse 3 zu 72,268 % aus 2015Q4 und zu 27,589 % aus 2016Q4, ergänzt um kleine Randanteile. Falls der publizierte Lauf diesen Sampler wie auf S. 114 beschrieben verwendete, sah das Modell praktisch keine Tweets nach 2016Q4. Das schafft einen starken Periodenproxy und steht in erklärungsbedürftiger Spannung zu Interpretationen anhand von Ereignissen aus 2017–2019. Das „falls“ ist wesentlich, weil ein Run-Manifest fehlt.

Auch die alten Architektururteile müssen abgestuft werden:

- Das FNN vermeidet den Last-Hidden-State-nach-Padding-Defekt des alten LSTM, mittelt aber unmaskiert über PAD-Positionen.
- Position 0 des Transformers ist kein eingefügtes CLS-Token, sondern der erste reale Token. Zudem fehlt `src_key_padding_mask`, sodass PAD-Tokens an der Attention teilnehmen.
- Beide Architekturen können lernen. Ihre Werte aus dem gemischten Protokoll bleiben aber Mischungen aus möglichem Sach-, Perioden-, Quellen- und Duplikatsignal.
- Das identische Metrikfünftupel in Tabelle 16 beweist weder einen Kopierfehler noch die Korrektheit der übrigen 53 Zeilen. Dafür wären Confusion-Matrizen oder Einzelvorhersagen nötig.

### 8.9 Prüfschritt 5: Erhaltenswerte wissenschaftliche Stärken

Die direkte Prüfung bestätigt mehrere positive Punkte:

1. Das Kreuzvalidierungsprotokoll ist ausreichend konkret beschrieben, um seine Grenzen heute aus dem Dokument selbst zu prüfen.
2. Tabelle 16 zeigt alle 54 Kombinationen ihres Rasters und enthält auch zahlreiche niedrige MCC-Werte; schwache Ergebnisse wurden nicht vollständig verborgen.
3. Tabelle 10 stimmt, und das Explorationskapitel hat das Duplikatproblem selbst quantitativ untersucht.
4. Der gemeinsame Prediction-/Topic-/Important-Word-Pfad ist ein origineller Ansatz, weil er Vorhersage und interpretierbare Hypothesen über Sprachveränderungen verbindet.
5. Das Repository enthält eine breite Testsuite und reproduzierbare Zwischenartefakte. Diese Stärke erleichtert die heutige Nachprüfung, auch wenn das historische Evaluationsprotokoll selbst nicht ausreichend getestet war.

### 8.10 Prüfschritt 6: Umgesetzte Korrekturen und Verifikation

Die folgenden Änderungen korrigieren direkt reproduzierbare Codeprobleme, ohne historische Ergebniszahlen umzudeuten:

| Korrektur | Datei(en) | Verifikation |
| --- | --- | --- |
| Exakte und optional nahe Duplikatentfernung in der Datensatzpipeline | `pipeline/FeatureDataframePipeline.py`, `createTweetsWithNumbers.py` | Der API-Standard bleibt kompatibel; die Datensatzerzeugung aktiviert exakte Entfernung ausdrücklich; ein End-to-End-Test prüft die Entfernung vor dem Finanz-Join. |
| Halboffene Klassenintervalle und geschlossene Lücke unter null | `tweetnumbersconnector/FinancialFiguresClassifier.py`, `tweetpreprocess/FiguresMultiClassCalculator.py` | Grenzwert- und Regressionsprüfungen decken die dokumentierten Klassen ab. |
| Explizite Ratio-/Prozentsemantik | `tweetpreprocess/FiguresIncreaseDecreaseClassCalculator.py` | Der Prozentmodus klassifiziert +0,62 % und +0,37 % korrekt als Anstieg. |
| Warnung bei veralteter `change_4`-Spalte | `tweetpreprocess/FiguresMultiClassCalculator.py` | Die Abweichungen in Amazon und Tesla werden sichtbar gemeldet. |
| Sichereres Checkpointladen mit CPU-Fallback, `map_location` und Schlüsselwarnungen | `classifier/ModelEvaluationHelper.py` | Fest verdrahtete CUDA-Abhängigkeit und stilles State-Dict-Mismatch wurden entfernt. |
| Checkpointpfad direkt aus `Trainer.train` und korrektes Epoch-Sekunden-Parsing | drei temporale Trainingsskripte | Ein versionierter Best-Checkpoint und korrekte Kalenderzeit werden verwendet. |
| Labelfreie Zeitgruppierung mit optionaler Periodengrenze | `nlpvectors/DataframeSplitter.py` | Tests prüfen Labelunabhängigkeit und Neustart an Quartalsgrenzen; historische Resultate werden dadurch nicht rückwirkend repariert. |
| Konfigurierbares LSTM-Dropout | `classifier/LSTMNN.py`, `classifier/CreateClassifierModel.py` | `0.0` erhält bestehendes Verhalten; die dokumentierte Dropoutvariante ist konstruierbar. |
| Explizite Topic-Kohärenz | `topicmodelling/TopicEvaluation.py` | `c_v` bleibt der historische Code-Standard; `c_uci` kann ausdrücklich gewählt werden. |
| Regressionstests für Klassen, Gruppierung, Pipeline und Topic-Metrik | `tests/` | Die vollständige Suite besteht mit 119 von 119 Tests. |

---

## 9. Herleitung und Aufbau des aktuellen Quartalsmodells

### 9.1 Wie das neue System aus dem alten abgeleitet wurde

Das neue System wurde nicht unabhängig von der alten Arbeit erfunden. Es übernimmt deren wissenschaftlich starke Ideen, entfernt aber genau die Abkürzungen, die eine Zukunftsaussage unklar machten.

| Beobachtung oder Baustein im alten System | Entscheidung im neuen System | Warum diese Änderung notwendig ist |
| --- | --- | --- |
| Das fachliche Ziel ist die Veränderung einer Unternehmenskennzahl. | Das Vierklassen-Quartalstarget bleibt erhalten. | Die Forschungsfrage soll nicht durch Aktienkurse oder andere leichter verfügbare Targets ersetzt werden. |
| Viele schwache Texte müssen gemeinsam betrachtet werden. | Texte werden weiterhin aggregiert, jetzt aber einmal pro Company-Quarter. | Die Aggregationsidee bleibt erhalten, ohne ein Finanzereignis durch viele Tweetgruppen künstlich zu vervielfachen. |
| Gruppensamples konnten dasselbe Quartal zwischen Train und Test teilen. | Die primäre Auswertungseinheit ist das vollständige Company-Quarter. | Alle Texte einer Zielrealisierung bleiben auf derselben Seite des Splits. |
| Das klassenweise KFold konnte spätere Blöcke zum Training eines früheren Tests verwenden. | Rolling-Origin-Splits verwenden nur Jahre vor dem Testjahr. | Das Protokoll entspricht einer realistischen zeitlichen Anwendung. |
| Top2Vec wurde auf dem Vollkorpus trainiert. | Das primäre Vorhersagemodell verwendet keine auf Testtexten gefitteten Embeddings. | Transduktive Zukunftsinformation wird aus der Hauptauswertung entfernt. |
| Das LSTM war für nur wenige unabhängige Quartale sehr groß und durch Padding schwer interpretierbar. | Der Textzweig verwendet robuste Quartalsaggregate und regularisierte logistische Regression. | Das Modell passt zur kleinen effektiven Stichprobe und liefert exakt zerlegbare Featurebeiträge. |
| Die alten hohen Werte deuteten auf einen starken Temporal-Fingerprint hin. | Texte werden in `all`, `late_third`, `reported`, `forward_estimate`, `early_reported` und `late_forward_estimate` getrennt. | Zeit- und Ereignissignale werden explizit gemessen, statt unkontrolliert im Embedding zu stecken. |
| Unternehmens- und Kennzahlenbegriffe waren semantisch zentral. | Firmen- und metrikspezifische Marker selektieren zielnahe Texte. | Allgemeines Marktrauschen wird reduziert, während die ursprüngliche Text-Hypothese erhalten bleibt. |
| Saisonalität konnte im alten Split implizit aus der Periode gelernt werden. | Ein eigener Saisonprior wird als sichtbarer Modellzweig und Baseline berechnet. | Saisonleistung wird nicht mehr fälschlich dem Text zugeschrieben. |
| Tesla-Texte enthalten häufig absolute Produktions-, Liefer- und Schätzlevel. | Ein separater Forward-Level-Zweig berechnet erwartete Veränderungen aus frühen und späten Textleveln. | Die wirtschaftliche Struktur des Tesla-Targets unterscheidet sich von Amazon-Umsatz und Apple-EPS. |
| Der alte Interpretationspfad war vom tatsächlichen Testpfad entkoppelt. | Lineare Attribution, Cue-Wörter, past-only Important Words und past-only NMF-Topics werden getrennt ausgegeben. | Jede Erklärung erhält einen klaren Evidenzstatus und verwendet keine Testlabels zum Fit. |
| Vollständige Texte waren für Demo und Interpretation leicht im Code sichtbar. | Neue Ergebnisartefakte speichern nur Aggregate, Begriffe und Topicgewichte. | Reproduzierbare Erklärungen sollen ohne Weitergabe vollständiger Posts möglich sein. |

Die Architektur ist damit keine völlige Abkehr vom alten System. Sie ist eine kontrollierte Zerlegung seiner vermischten Signale:

```text
altes gemeinsames Sprachsignal
        |
        +-- saisonale Wiederholung ----------> expliziter Saisonprior
        +-- zielnahe Zahlen und Erwartungen -> numerischer Textzweig
        +-- Tesla-Levelveränderung ----------> Forward-Level-Zweig
        +-- Themen und Begriffe --------------> past-only Erklärungspfad
```

### 9.2 Aufbau in zehn Schritten

1. **Target einfrieren:** Für jedes Unternehmen bleibt genau die dokumentierte Quartalskennzahl und deren Vierklassenveränderung das Ziel.
2. **Unabhängige Einheit festlegen:** Alle Texte eines Unternehmensquartals werden zu einem einzigen Fall zusammengefasst.
3. **Zeitachse vor dem Featurefit teilen:** Training, Validation und Test werden nach ganzen Jahren als Rolling Origin definiert.
4. **Zielnahe Texte auswählen:** Ein Text muss einen Unternehmens- und einen Kennzahlmarker enthalten.
5. **Informationszeitpunkt trennen:** Frühe, späte, berichtete und erwartungsbezogene Texte werden in getrennten Ansichten aggregiert.
6. **Textsignal separat lernen:** Nur die Trainingsquartale bestimmen Skalierung, Regularisierung und Koeffizienten des numerischen Textklassifikators.
7. **Saison separat berechnen:** Historische Klassen desselben Kalenderquartals bilden einen geglätteten Prior, der ohne Text auskommt.
8. **Zweige kontrolliert fusionieren:** Saison- und Textwahrscheinlichkeiten werden mit festen oder auf Validation bestimmten Regeln kombiniert.
9. **Firmenspezifische Struktur begrenzen:** Nur Tesla erhält den Forward-Level-Zweig; der post hoc entwickelte Konflikt-Gate wird ausdrücklich als explorativ markiert.
10. **Erklärung an die Vorhersage koppeln:** Jede finale Entscheidung speichert die Wahrscheinlichkeiten aller Zweige und trennt exakte Attribution von deskriptiven Topics.

Diese Reihenfolge ist entscheidend. Würde beispielsweise das Topicmodell vor dem Jahressplit oder die Skalierung auf allen Quartalen gefittet, wäre die alte Leakage-Problematik in neuer Form wieder vorhanden.

### 9.3 Wo die Entscheidungen im aktuellen Code umgesetzt sind

| Wissenschaftliche Entscheidung | Aktuelle Implementierung | Prüfbarer Effekt |
| --- | --- | --- |
| Ein Fall entspricht einem Company-Quarter. | `build_company_data` in `trainNumericTextSignalQuarterModel.py` | Texte, Target und Quartalsfeatures werden über einen eindeutigen Quartalsschlüssel verbunden. |
| Nur zielnahe Texte erzeugen Zahlenfeatures. | `contains_company_and_metric` und `numeric_quarter_features` in `NumericQuarterTextFeatures.py` | Jeder berücksichtigte Text erfüllt Firmen- und Kennzahlenmuster; das Ergebnis ist ein fester Featurevektor. |
| Train, Validation und Test folgen der Zeit. | `rolling_fold` | Der Fold erzeugt Jahreslisten, fittet Kandidaten nur auf der Vergangenheit und bewertet das folgende Testjahr. |
| Der reine Textzweig bleibt separat messbar. | `fit_numeric_model`, `candidate_predictions` und `select_numeric_candidate` | Textwahrscheinlichkeiten und Textmetriken werden vor jeder Fusion gespeichert. |
| Saison wird nicht als Text ausgegeben. | `seasonal_probabilities` | Der Prior verwendet ausschließlich frühere Labels desselben Kalenderquartals. |
| Fusion und Tesla-Sonderlogik sind explizit. | `fuse_probabilities`, `tesla_forward_fusion` und `tesla_conflict_gate` | Wahrscheinlichkeiten vor und nach jedem Zweig können getrennt reproduziert werden. |
| Unsicherheit und Textkontrolle werden mitgespeichert. | `wilson_interval` und `paired_accuracy_audit` | Intervall, abweichende richtige Entscheidungen und exakter gepaarter p-Wert werden aus den 36 Fällen berechnet. |
| Erklärungen folgen dem tatsächlichen Modell. | `linear_class_feature_contributions`, `model_linked_cue_words`, `fit_past_only_important_words` und `PastOnlyNmfTopics` | Exakte lineare Beiträge bleiben von deskriptiven Wort- und Topicassoziationen getrennt. |
| Ergebnis und Datenschutz werden beim Export geprüft. | `replay_models` und `_forbidden_output_key_audit` in `extractNumericQuarterTopicsAndImportantWords.py` | Gespeicherte Klassen müssen reproduziert werden; Rohtext- und Identifikatorfelder werden blockiert. |

Diese Zuordnung macht die Herleitung falsifizierbar: Jede methodische Aussage besitzt einen konkreten Codepfad und ein beobachtbares Ergebnisartefakt.

### 9.4 Ziel und Daten

Das Ziel bleibt die vierstufige Veränderung der Quartalskennzahl. Verwendet werden Amazon, Apple und Tesla.

Nicht als aktuelle Eingabefeatures verwendet werden:

- Finanz-CSV,
- der aktuelle Quartalswert,
- die aktuelle prozentuale Zielveränderung,
- Word Embeddings,
- externe Daten.

Vergangene Zielklassen werden für Modelltraining und Saisonprior verwendet.

### 9.5 Zielnahe Textselektion

Ein Text geht in die numerische Aggregation ein, wenn er sowohl einen Unternehmensmarker als auch einen Kennzahlmarker enthält.

Beispiele:

| Unternehmen | Unternehmensmarker | Kennzahlmarker |
| --- | --- | --- |
| Amazon | Amazon, AMZN | revenue, net sales, AWS sales |
| Apple | Apple, AAPL | EPS, earnings per share |
| Tesla | Tesla, TSLA | deliveries, delivery, production |

Das Ergebnis speichert keine vollständigen Texte. Es speichert nur aggregierte Features.

### 9.6 Sechs Textansichten

| Ansicht | Inhalt | Hypothese |
| --- | --- | --- |
| `all` | alle relevanten Texte des Quartals | Grundpegel |
| `late_third` | letztes Drittel | spätere Information ist näher am Periodenende |
| `reported` | reported, actual, announced | bereits berichteter oder rückblickender Stand |
| `forward_estimate` | estimate, consensus, guidance, future | Erwartungssprache |
| `early_reported` | Reportingsprache im ersten Drittel | Proxy für alten Referenzstand |
| `late_forward_estimate` | Schätzungen im letzten Drittel | Proxy für erwarteten neuen Stand |

### 9.7 Featurefamilien

Aus jeder Ansicht werden unter anderem berechnet:

- Gesamtzahl und Zahl relevanter Texte,
- Anteil relevanter Texte,
- Zahl der Prozentangaben,
- positive und negative Prozentwerte,
- Median und Quartile,
- direkte Verteilung der Prozentwerte auf vier Klassen,
- Anteil von reported-, estimate-, guidance-, beat-, miss- und future-Sprache,
- robuste absolute Kennzahllevel,
- Differenzen zwischen frühen, späten, berichteten und erwarteten Levels.

### 9.8 Synthetisches Beispiel

Angenommen, frühe Texte nennen einen berichteten Lieferstand von 70.000 Einheiten. Späte Schätzungen nennen 81.000.

```text
estimated_change = (81.000 / 70.000 - 1) x 100 = 15,7 %
```

15,7 % fällt in Klasse 2. Das ist Feature Engineering aus Text, nicht das Auslesen des echten Testtargets.

### 9.9 Numerischer Textklassifikator

Die Quartalsfeatures werden:

1. nur auf Trainingsquartalen standardisiert,
2. auf den Bereich `[-8, 8]` begrenzt,
3. mit regularisierter logistischer Regression klassifiziert.

Validation wählt:

- Regularisierung,
- aktuelle Features allein oder mit zeitlichen Differenzen,
- optionale Firmenidentität.

### 9.10 Saisonprior

Für ein neues Q2 betrachtet der Saisonprior nur frühere Q2-Labels desselben Unternehmens und erzeugt daraus eine geglättete Klassenverteilung.

Er verwendet keine Finanzwerte als Input, aber historische Targets. Darum ist er kein Textfeature.

### 9.11 Fusion

Der allgemeine Hybrid mischt Saison- und Textwahrscheinlichkeiten mit festem Gewicht.

Bei Tesla kommt ein Forward-Level-Signal hinzu. Es leitet aus späten Schätzleveln und frühen berichteten Levels eine erwartete Veränderung ab.

### 9.12 Tesla-Konflikt-Gate

Der Gate erkennt zwei spezielle Konfliktmuster zwischen Basismodell und numerischem Textmodell. In diesen Fällen ersetzt er die Vorhersage durch die numerische Textverteilung.

Der Gate verbessert 75,00 % auf 80,56 %. Seine Schwellen wurden jedoch nach Betrachtung der Fehler von 2017 bis 2019 entworfen. Deshalb ist er explorativ und nicht bestätigend.

### 9.13 Warum kein CUDA benötigt wird

Das aktuelle beste Modell ist kein LSTM. Regex-Aggregation, Skalierung und logistische Regression laufen auf CPU. CUDA war für die früheren neuronalen Modelle relevant, nicht für den aktuellen 80,56-%-Lauf.

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

Die Wahrscheinlichkeiten der drei Läufe werden pro Company-Quarter gemittelt.

### 10.2 Was im aktuellen Protokoll nicht leakt

- Kein Testlabel geht in Fit, Skalierung oder Auswahl ein.
- Trainingsjahre liegen global vor dem Validierungsjahr.
- Das Validierungsjahr liegt global vor dem Testjahr.
- Jede Company-Quarter-Kombination wird genau einmal gezählt.
- Topic- und Wortmodelle werden nur auf früheren Quartalen gefittet.

### 10.3 Verbleibende Einschränkungen

- Es wird Text aus dem gesamten zu bewertenden Quartal aggregiert.
- Ein Echtzeit-Cutoff bei 25 %, 50 % oder 75 % des Quartals ist noch kein primärer Test.
- Es gibt nur 36 unabhängige Testfälle.
- Der Tesla-Gate ist post hoc.
- Amazon und Apple werden stark von Saisonmustern getragen.
- Der lokale 2020-Bestand ist nicht dicht genug für einen gleichartigen vollständigen neuen Holdout.

---

## 11. Ergebnisse und statistische Einordnung

### 11.1 Vierklassenmetriken

| Modell | Accuracy | MCC | Log Loss | Einordnung |
| --- | ---: | ---: | ---: | --- |
| Numerischer Text allein | 0,5000 | 0,3224 | 1,3078 | reines numerisches Textsignal |
| Saisonprior ohne Finanzfeatures | 0,6111 | 0,4743 | 0,9519 | nur frühere gleiche Kalenderquartale |
| Saison + numerischer Text, fest 50/50 | 0,6944 | 0,5854 | 1,0298 | allgemeiner Hybrid |
| Saison + Tesla Forward | 0,7500 | 0,6633 | 0,9460 | transparente Variante ohne Konflikt-Gate |
| Saison + Tesla Konflikt-Gate | **0,8056** | **0,7387** | 0,9173 | primäres exploratives Ergebnis |
| Primärer Bundle-Shuffle | 0,6944 | 0,5850 | 1,1094 | Textbundle innerhalb der Firma verschoben |

### 11.2 Ergebnis je Unternehmen

| Unternehmen | Richtig | Accuracy | MCC |
| --- | ---: | ---: | ---: |
| Amazon | 10 / 12 | 0,8333 | 0,7828 |
| Apple | 11 / 12 | 0,9167 | 0,8765 |
| Tesla | 8 / 12 | 0,6667 | 0,5664 |
| Gesamt | 29 / 36 | 0,8056 | 0,7387 |

### 11.3 Richtungsdiagnostik

Werden dieselben Wahrscheinlichkeiten nur zu Rückgang gegen Anstieg zusammengefasst, entstehen:

- Accuracy 0,9167,
- MCC 0,8003.

Das ist ein einfacheres Ziel und darf die primäre Vierklassenauswertung nicht ersetzen.

### 11.4 Unsicherheit

29 richtige Entscheidungen bei 36 Fällen ergeben für Accuracy ein Wilson-95-%-Intervall von ungefähr:

```text
0,650 bis 0,902
```

Das Intervall ist breit. Die wahre Leistung kann deutlich unter oder über dem Punktwert liegen.

### 11.5 Vergleich mit der Shuffle-Kontrolle

Das primäre Modell ist in vier Quartalen korrekt, in denen die Bundle-Shuffle-Kontrolle falsch liegt. Der umgekehrte Fall tritt nicht auf.

Der gepaarte zweiseitige exakte Test ergibt:

```text
p = 0,125
```

Das ist eine positive Richtung, aber kein statistisch signifikanter Unterschied auf dem 5-%-Niveau.

### 11.6 Was behauptet werden darf

- Ein leakage-bewusster no-finance Hybrid erreicht explorativ 80,56 % und MCC 0,7387.
- Numerische Text- und Erwartungsfeatures helfen bei bestimmten Tesla-Entscheidungen.
- Der isolierte numerische Textzweig enthält ein positives, aber begrenztes Signal.
- Amazon und Apple besitzen starke saisonale Targetmuster.

### 11.7 Was nicht behauptet werden darf

- Nicht: Ein reines Textmodell erreicht 80 bis 90 %.
- Nicht: Der zusätzliche Textbeitrag ist statistisch bestätigt.
- Nicht: 2017 bis 2019 seien nach Entwicklung des Gates weiterhin ein unberührter finaler Holdout.
- Nicht: Das Modell prognostiziere bereits das nächste Quartal Q+1.
- Nicht: Topics verursachten eine Finanzveränderung.

### 11.8 Nächster bestätigender Test

Regex, Features, Hyperparameter, Fusionsgewichte und Gate-Schwellen müssen eingefroren werden. Danach ist ein vollständig neuer Holdout erforderlich.

Erst ein solcher Test trennt echte Generalisierung von nachträglicher Anpassung.

---

## 12. Topics und Important Words

### 12.1 Warum Integrated Gradients hier nicht die richtige Hauptmethode ist

Der aktuelle numerische Textzweig besteht aus aggregierten Regexfeatures und logistischer Regression. Er hat keine Token-Embedding-Schicht, auf die Integrated Gradients sinnvoll angewandt werden könnte.

Für diesen Zweig ist die exakte lineare Attribution:

```text
Beitrag eines Features = standardisierter Featurewert x Klassenkoeffizient
```

Die Summe dieser Beiträge plus Intercept rekonstruiert den Entscheidungsscore des numerischen Textmodells.

### 12.2 Vier Erklärungsebenen

| Ebene | Berechnung | Aussagekraft |
| --- | --- | --- |
| Exakte Textfeature-Attribution | standardisierter Wert mal OVR-Koeffizient | exakt für den numerischen Textzweig |
| Modellnahe Cue-Wörter | Featurebeiträge werden Sprachfamilien und vorkommenden Cues zugeordnet | Featurefamilie exakt, Verteilung auf Einzelwörter deskriptiv |
| Past-only Important Words | quartalsstabile Klassen-Log-Odds nur aus früheren Quartalen | zeitlich saubere Klassenassoziation, nicht kausal |
| Past-only Topics | TF-IDF plus NMF nur auf früheren Quartalen; Testtexte werden transformiert | Kontextbeschreibung, keine additive Modellattribution |

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

Bei Medianen, Quantilen und Quartalsaggregation ist die Verteilung auf einzelne Wörter nicht mehr mathematisch exakt. Deshalb wird sie ausdrücklich als deskriptive Brücke bezeichnet.

### 12.4 Past-only Important Words

Das Wortlexikon lernt, welche Begriffe in früheren Quartalen stabil mit einer Klasse verbunden waren.

Wichtig ist die Reihenfolge:

1. Nur Training plus Validation werden verwendet.
2. Wörter müssen über mehrere Quartale wiederkehren.
3. Erst danach wird geprüft, welche dieser Wörter im Zukunftsquartal vorkommen.
4. Das Testlabel wird nicht zum Wortfit verwendet.

Damit wird ein einmaliger Begriff aus einem einzigen Quartal weniger leicht zu einem angeblich wichtigen Wort.

### 12.5 Past-only NMF-Topics

Für jedes Unternehmen und Testjahr wird ein kleines Topicmodell auf den bis dahin vergangenen Quartalen trainiert.

Verwendet werden:

- TF-IDF zur Textrepräsentation,
- NMF zur Topiczerlegung,
- maximal 250 relevante Dokumente pro Quartal,
- deterministische, quartalsbalancierte Auswahl.

Das Zukunftsquartal wird nur in das bereits gelernte Topicmodell projiziert.

### 12.6 Vollständiger Entscheidungsweg

Jede Erklärung enthält getrennt:

- Saisonwahrscheinlichkeiten,
- numerische Textwahrscheinlichkeiten,
- Forward-Level-Wahrscheinlichkeiten,
- Wahrscheinlichkeiten vor dem Konflikt-Gate,
- finale Wahrscheinlichkeiten,
- Gate-Aktivierungsanteil,
- exakte Textfeaturebeiträge,
- Cue-Wörter,
- past-only Important Words,
- past-only Topics.

So wird sichtbar, ob ein Topic nur Kontext beschreibt oder ob der Textzweig die finale Klasse tatsächlich beeinflusst hat.

### 12.7 Reproduktionsschutz

Das Erklärungsskript spielt die gespeicherten Foldentscheidungen erneut ab. Es bricht ab, wenn Accuracy, MCC oder finale Klassen nicht mit dem primären Ergebnis übereinstimmen.

Der erfolgreiche Replay reproduzierte:

- Hybrid Accuracy 0,8056,
- Hybrid MCC 0,7387,
- numerische Text-Accuracy 0,5000,
- numerischen Text-MCC 0,3224.

---

## 13. Konkrete Erklärungsbeispiele

Alle Beispiele enthalten nur aggregierte Begriffe und Wahrscheinlichkeiten, keine vollständigen Originaltexte.

### 13.1 Amazon 2017Q1

| Größe | Ergebnis |
| --- | --- |
| Wahre Klasse | 3 |
| Numerische Textklasse | 3 |
| Finale Klasse | 3 |
| Saisonwahrscheinlichkeit für Klasse 3 | 0,750 |
| Numerische Textwahrscheinlichkeit für Klasse 3 | 0,379 |
| Finale Wahrscheinlichkeit für Klasse 3 | 0,564 |

Der starke Saisonpfad wird durch den Text bestätigt, nicht ersetzt.

Größte positive Textfeaturebeiträge für Klasse 3:

- `all__miss_tweet_fraction`: +0,206,
- `forward_estimate__miss_tweet_fraction`: +0,192,
- `early_reported__log_percent_mentions`: +0,137.

Größte negative Beiträge:

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

Past-only Important Words für die finale Klasse beginnen mit:

- business,
- misses,
- miss,
- earnings,
- sales.

Dominante Topicbereiche:

- revenue, Amazon, AWS, cloud,
- growth, revenue growth, year-over-year.

### 13.2 Tesla 2018Q1

Hier verändert der Konflikt-Gate die Entscheidung.

| Zweig | Wahrscheinlichkeiten `[0, 1, 2, 3]` | Argmax |
| --- | --- | ---: |
| Saisonprior | `[0,313; 0,563; 0,063; 0,063]` | 1 |
| Numerischer Text | `[0,143; 0,456; 0,208; 0,193]` | 1 |
| Forward-Level | `[0,042; 0,042; 0,042; 0,875]` | 3 |
| Vor Konflikt-Gate | `[0,135; 0,275; 0,088; 0,501]` | 3 |
| Final nach Gate | `[0,143; 0,456; 0,208; 0,193]` | 1, korrekt |

Größte exakte Beiträge zur numerischen Textklasse 1:

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

Dieses Beispiel zeigt Nutzen und Risiko zugleich: Der Textzweig korrigiert die Forward-Fusion korrekt. Die Entscheidung erfolgt aber über einen nachträglich entwickelten Tesla-Gate. Topicwörter erklären den Kontext; sie validieren nicht die Gate-Schwellen.

---

## 14. Claim Ladder

Die Claim Ladder trennt technische Machbarkeit, empirische Beobachtung, Hypothese und unzulässige Schlussfolgerung.

| Aussage | Status | Wissenschaftlich korrekte Formulierung |
| --- | --- | --- |
| Die Gesamtpipeline ist technisch realisierbar. | belegt | Lokale Texte können mit Berichtsperioden verbunden, gruppiert, klassifiziert und bis zu Wörtern und Topics zurückverfolgt werden. |
| Textgruppen enthalten starke Klassenkorrelationen. | im alten Split belegt | Unter den damaligen Gruppierungs- und Splitbedingungen ist die Zielklasse stark separierbar. |
| Sprache kodiert Zeit, Ereignisse und Marktregime. | starke Hypothese | Alte hohe Werte sowie Saison- und Shuffle-Kontrollen motivieren einen Temporal-Fingerprint-Test. |
| Bestimmte Wörter und Topics sind interpretativ relevant. | Kandidaten | Stabilität muss auf past-only Fits und Zukunftstests erneut gemessen werden. |
| Text sagt unbekannte spätere Quartale mit 87 % voraus. | nicht belegt | Die alte 0,87 darf nicht als strikte Zukunftsgeneralisierung interpretiert werden. |
| Das aktuelle reine Textmodell erreicht 80,56 %. | falsch | 80,56 % gehören zum Hybrid; numerischer Text allein erreicht 50,00 %. |
| Ein Topic verursacht eine Quartalsänderung. | nicht belegt | Attribution und Topiczuordnung zeigen Modellassoziation, keine ökonomische Kausalität. |

### Der erhaltenswerte Dissertationsbeitrag

Der wissenschaftliche Kern liegt in vier Punkten:

1. Entwurf eines modularen Informationssystems, das Social-Media-Text mit Unternehmenskennzahlen verbindet.
2. Empirischer Hinweis auf starke zeitliche und ereignisbezogene Sprachsignaturen.
3. Multi-Scale-Aggregation als Antwort auf schwache Einzeltweets.
4. Mehrstufige Interpretation von Tokenattribution bis Topic- und LLM-Vergleich.

Diese Beiträge bleiben bestehen, auch wenn die alte Forecast-Accuracy neu eingeordnet werden muss.

---

## 15. Datenschutz- und Tweet-Inhaltsaudit

### 15.1 Ergebnis

Der Audit des sichtbaren Branchbaums lautet **FAIL** für eine strikte Null-Tweet-Content-Anforderung.

Es gibt keinen großen eingecheckten Rohdatensatz. Einige Dateien enthalten jedoch vollständige oder tweetartige Inhalte.

### 15.2 Hohe Priorität

| Datei | Befund | Empfohlene Aktion |
| --- | --- | --- |
| `tweetsCompanyNumbersPrediction/src/tests/companyTweetsDummy.csv` | mehrere vollständige Textzeilen mit Handles oder Links | durch deterministische synthetische Texte ersetzen |
| `tweetsCompanyNumbersPrediction/src/predictSingleTweetGroup.py` | fest codierte längere Tweetgruppe und kommentierte Beispiele | synthetisieren oder per CLI laden |
| `tweetsCompanyNumbersPrediction/src/tests/TestNearDuplicateDetector.py` | realistisch wirkende Marktposts | neutral synthetisieren |
| `tweetsCompanyNumbersPrediction/src/tests/TweetSentimentAnalysisTest.py` | Apple-/AAPL-artige Posttexte | neutral synthetisieren |

### 15.3 Niedrigere Priorität

Offensichtlich synthetische Textfixtures finden sich unter anderem in:

- `PipelineTest.py`,
- `TweetTextFilterTransformerTest.py`,
- `nlpvectorstest.py`,
- `HyperlinkRemoverTest.py`.

Sie sind keine Bulk-Rohdaten. Bei einer absolut strikten Null-Content-Policy sollten aber auch diese Beispiele ohne Handles, reale Links oder marktnahe Formulierungen auskommen.

### 15.4 Was bereits sauber ist

- Keine großen CompanyTweets-Gesamtdaten im Repositorybaum.
- Keine eingecheckten trainierten Checkpoints oder exportierten Tweetgruppen.
- `tokenizer.json` enthält Vokabular, keine zusammenhängenden Posts.
- Die neuen Ergebnis-JSONs enthalten Metriken, Features, Begriffe und Topicaggregate, aber keine vollständigen Texte.
- `numeric_text_topics_important_words.json` enthält keine Autoren, Handles, URLs oder Tweet-IDs.

### 15.5 Bereinigungsplan

1. `companyTweetsDummy.csv` durch eindeutig synthetische Kunstsätze ersetzen.
2. Fest codierte Demo-Texte aus `predictSingleTweetGroup.py` entfernen oder über Eingabe laden.
3. Near-Duplicate- und Sentimentfixtures neutral neu formulieren.
4. Einen Pre-Commit-Scanner für `body`-CSV-Schemata, `t.co`, Handles, Cashtags und lange Textliterale hinzufügen.
5. Alle Tests erneut ausführen.
6. Falls für eine Publikation erforderlich, die Git-Historie separat auf historische Blobs prüfen. Eine Bereinigung des aktuellen Baums entfernt keine alten Commits.

### 15.6 Minimaler Freigabestandard

- Kein Hochrisikotreffer im Branch-Tree-Scan.
- Keine CSV- oder JSON-Datei mit mehreren vollständigen Postsätzen.
- Keine Autoren, Handles, Tweet-URLs oder Plattform-IDs in Fixtures.
- Forschungsartefakte speichern nur Aggregationen und Metriken.

---

## 16. Empfohlene Forschungsroadmap

### Priorität P0

#### Gate einfrieren und neu testen

Der Tesla-Konflikt-Gate darf nicht weiter anhand von 2017 bis 2019 verändert werden. Architektur und Schwellen müssen auf neuen Quartalen eingefroren geprüft werden.

#### Baselines immer gemeinsam berichten

Jeder Lauf sollte mindestens enthalten:

- globale Mehrheitsbaseline,
- Saisonbaseline,
- Persistenzbaseline,
- reinen Textzweig,
- Hybrid,
- Text-Shuffle-Kontrolle.

#### Company-Quarter als primäre Einheit

Keine primäre Metrik darf einzelne Tweets oder Tweetgruppen wie unabhängige Finanzereignisse zählen.

### Priorität P1

#### Frühe Nowcast-Cutoffs

Features sollten nach festen Anteilen des Quartals ausgewertet werden:

- 25 %,
- 50 %,
- 75 %,
- 100 %.

So wird sichtbar, ob das Signal früh verfügbar ist oder erst durch nachträgliche Berichts- und Zusammenfassungstexte entsteht.

#### Hierarchisches Firmenmodell

Ein gemeinsames Modell kann allgemeine Sprachmuster lernen und zugleich firmenspezifische Koeffizienten oder Adapter besitzen.

#### Interpretationsstabilität

Important Words und Topics sollten über Seeds, Folds und Unternehmen verglichen werden. Ein Begriff, der nur in einem Lauf erscheint, ist ein schwacher Befund.

#### Target-Provenienz

Pro Unternehmen sollten dokumentiert werden:

- Datenquelle,
- Kennzahlname,
- Einheit,
- Berichtsdatum,
- Reportingperioden,
- Definition der prozentualen Veränderung,
- Klassengrenzen.

### Priorität P2

#### Fairer LSTM-Vergleich

Ein neuer LSTM- oder Attention-Zweig sollte:

- Vokabular und trainierbare Repräsentation nur aus vergangenen Perioden beziehen,
- korrekt maskieren oder packen,
- `padding_idx` setzen,
- Token -> Tweet -> Quartal hierarchisch modellieren,
- Textbeitrag separat ausgeben,
- gegen identische Saison- und Shuffle-Kontrollen antreten.

#### LLM-Vergleich

LSTM, Topicmodell und LLM sollten dieselben held-out Company-Quarters erhalten. Prompts, Temperatureinstellungen und Bewertung müssen vorab feststehen.

### Ziel bleibt Quartalszahl

Mehr unabhängige Daten müssen nicht durch einen täglichen Aktienkurs erzeugt werden. Geeignete Wege sind:

- mehr abgeschlossene Quartale,
- weitere sauber definierte Unternehmen,
- ein externer Holdout,
- mehrere Kennzahlen mit klarer Provenienz.

---

## 17. Reproduktion und Codeevidenz

### 17.1 Aktuelles numerisches Quartalsmodell ausführen

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

Für Topic- und Wortmodelle gilt die quartalsbalancierte Obergrenze von 250 Dokumenten. Die Vorhersagefeatures selbst bleiben unverändert.

### 17.3 Tests

```bash
python -m unittest tests.alltestsuite
```

Beim letzten vollständigen Lauf bestanden:

- 111 registrierte Tests,
- 19 zusätzliche Tests.

### 17.4 Wichtige aktuelle Dateien

| Thema | Datei |
| --- | --- |
| numerische Textmerkmale | [`NumericQuarterTextFeatures.py`](tweetsCompanyNumbersPrediction/src/classifier/NumericQuarterTextFeatures.py) |
| Training, Saison und Tesla-Gate | [`trainNumericTextSignalQuarterModel.py`](tweetsCompanyNumbersPrediction/src/trainNumericTextSignalQuarterModel.py) |
| exakte Attribution und past-only Topics | [`NumericQuarterTextExplanations.py`](tweetsCompanyNumbersPrediction/src/featureinterpretation/NumericQuarterTextExplanations.py) |
| Replay und Erklärungsausgabe | [`extractNumericQuarterTopicsAndImportantWords.py`](tweetsCompanyNumbersPrediction/src/extractNumericQuarterTopicsAndImportantWords.py) |
| Tests der Erklärung | [`NumericQuarterTextExplanationsTest.py`](tweetsCompanyNumbersPrediction/src/tests/NumericQuarterTextExplanationsTest.py) |

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
| Topicqualität | `topicmodelling/TopicEvaluation.py:24-48` |
| Wort, POS und Topic Mapping | `featureinterpretation/InterpretationDataframeUtil.py:8-31` |
| LLM-Topicvergleich | `topicmodelling/llmcomparison/LLMTopicsCompare.py:43-99` |
| Near-Duplicate-Erkennung | `tweetpreprocess/nearduplicates/NearDuplicateDetector.py:34-66` |

### 17.6 Neue Ergebnisartefakte und Datenschutz

Die neue Erklärungsdatei speichert:

- Targets und Vorhersagen auf Company-Quarter-Ebene,
- Wahrscheinlichkeiten je Modellzweig,
- Featurewerte und Koeffizientenbeiträge,
- einzelne aggregierte Cue-Terme,
- past-only Important Words,
- Topicwörter und Topicgewichte.

Sie speichert nicht:

- vollständige Texte,
- Autoren,
- Handles,
- URLs,
- Tweet-IDs,
- Dokument-IDs.

Ein rekursiver Key-Audit blockiert typische Rohtext- und Identifikatorfelder.

---

## 18. Abschließendes Urteil

### Zur alten Implementierung

Die alte `main`-Codebasis ist wissenschaftlich interessant, weil sie:

- ein komplettes Informationssystem realisiert,
- mehrere Unternehmen und Kennzahlen abbildet,
- schwache Einzeltexte aggregiert,
- Vorhersage und Interpretation verbindet,
- Topicqualität quantitativ betrachtet,
- manuelle, neuronale und LLM-Perspektiven zusammenführt,
- einen starken zeitlichen Fingerabdruck der Sprache sichtbar macht.

Sie ist jedoch kein überzeugender Beleg für 87 % echte Zukunftsprognose. Hauptgründe sind Quartalspseudoreplikation, Training auf späteren Blöcken, fehlende Saisonbaseline, transduktive Embeddings und konkrete Codeprobleme.

### Zur direkten Code- und Ergebnisprüfung

Die direkte Prüfung belegt die zentralen Schwachstellen über konkrete Daten- und Aufrufpfade: Das Target wird auf Gruppenebene wiederholt, der Hauptsplit kann Perioden teilen und spätere Blöcke verwenden, Top2Vec sieht den Vollkorpus und mehrere Interpretationsschritte sind technisch fehlerhaft. Daraus folgt eine notwendige Neueinordnung der alten Accuracy, aber keine universelle Aussage, dass Text oder jede LSTM-Architektur unbrauchbar sei.

### Zum aktuellen Modell

Das aktuelle System verbessert die zeitliche Evaluation deutlich und erreicht explorativ 80,56 % Accuracy sowie MCC 0,7387. Dieses Ergebnis gehört zu einem Hybrid. Der reine numerische Textzweig erreicht 50,00 % Accuracy und MCC 0,3224.

Der Tesla-Konflikt-Gate ist post hoc. Deshalb bleibt 75,00 % Accuracy und MCC 0,6633 die transparentere ungated Referenz, bis neue Quartale den Gate bestätigen.

### Zu Topics und Important Words

Die Topic- und Wortanalyse ist nun enger mit dem tatsächlichen Entscheidungsweg verbunden. Exakte Textfeaturebeiträge, modellnahe Cues, past-only Important Words und past-only Topics werden getrennt ausgewiesen.

Die wichtigste Grenze bleibt:

> Topics beschreiben den Kontext eines Modells. Sie beweisen weder eine kausale Wirkung auf die Finanzkennzahl noch einen additiven Beitrag zur Saisonprior oder zum Tesla-Gate.

### Endfazit

Das Projekt hat zwei gleichzeitig wahre Ergebnisse:

1. Die alte hohe Accuracy darf nicht als sauberer Zukunftsforecast interpretiert werden.
2. Die alte Forschungsplattform, die zeitlichen Sprachsignaturen und die Verbindung von Vorhersage mit interpretierbaren Topics sind wissenschaftlich wertvoll und verdienen eine leakage-saubere Fortsetzung.
