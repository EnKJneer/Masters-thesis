
# Machine Learning-basierte Modellierung von flexiblen Prozessen zur Vorhersage von Motorströmen an Werkzeugmaschinen

**Masterarbeit von Jonas Kyrion**
Karlsruher Institut für Technologie (KIT), November 2025

---
## Projektbeschreibung

Die zunehmende Individualisierung von Produkten stellt die Fertigungstechnik vor erhebliche Herausforderungen. Etablierte Methoden zur Prozessüberwachung basieren häufig auf statistischen Analysen historischer Daten, was sie für moderne Produktionsszenarien mit hoher Variantenvielfalt und kleinen Losgrößen ungeeignet macht. Zudem erfordern moderne Überwachungsverfahren oft zusätzliche Sensorik, deren Nachrüstung in Bestandsanlagen mit erheblichem Aufwand und Kosten verbunden ist.

Am wbk Institut für Produktionstechnik wurde daher ein Ansatz entwickelt, der die Prozessüberwachung durch Vergleich des Motorstroms mit einem Referenzsignal ermöglicht. Dieses Referenzsignal wird durch Methoden des maschinellen Lernens (ML) vorhergesagt. Um die Flexibilität dieser Modelle für neue Bauteile zu gewährleisten, werden in dieser Arbeit Methoden analysiert und entwickelt, die die Generalisierungsfähigkeit der Modelle verbessern. Methodisch wird zunächst die Eignung reiner ML-Modelle evaluiert. Anschließend wird die Datenqualität systematisch erhöht, bevor hybride Ansätze entwickelt und analysiert werden, die physikalisches Prozesswissen mit ML-Methoden kombinieren. Dieser **physics-informed ML-Ansatz** führt zur Entwicklung eines Experten-Modells, das unterschiedliche Betriebsbereiche separat modelliert.

Die Ergebnisse zeigen eine Verbesserung der Schätzgenauigkeit. Im Vergleich zum Ausgangszustand konnte der prozentuale Fehler um bis zu **54 % reduziert** werden. Dies demonstriert das Potenzial von **physics-informed ML-Modellen** für die flexible Prozessüberwachung in variantenreichen Produktionsumgebungen mit geringen Losgrößen.

---
## Ordnerstruktur


| Ordner/Datei                     | Beschreibung                                                                                     |
|----------------------------------|-------------------------------------------------------------------------------------------------|
| `DataSets_DMC60H_Plate_Notch_*`   | Experimentelle Datensätze (Referenz, Anomalien, etc.) aus [KIT RADAR](https://radar.kit.edu/radar/en/dataset/ctvuj6dgzepzmk0g) |
| `Models`                         | Grundlegende Implementierung der hybriden ML-Modelle (PiRNN, Random Forests)                     |
| `Experiments_Thesis`             | Skripte der Experimente, die in der Masterarbeit dokumentiert wurden                             |
| `Experiments_other`              | Skripte von weiteren Experimenten, die nicht erfolgreich oder relevant waren                   |
| `Simulation`                     | Materialabtrags- und Prozesskraftsimulationen zur Datenerzeugung                              |
| `Helper`                         | Hilfsfunktionen und Utility-Skripte, z. B. für die Ausführung und Dokumentation von Experimenten |
| `requirements.txt`               | Abhängigkeiten und Installationsanforderungen                                                  |

---
## Installation

1. Repository klonen:
   ```bash
   git clone [Repository-URL]
   ```
2. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```

---
## Nutzung

- Die Experimente sind in den Ordnern `Experiments_Thesis` und `Experiments_other` organisiert.

---
## Ergebnisse

- **Verbesserung der Schätzgenauigkeit:** Reduktion des prozentualen Fehlers um bis zu 54 %.
- **Hybride Modelle:** Kombination von physikalischem Prozesswissen mit ML-Methoden.
- **Visualisierungen:** SHAP-Analysen, Zeitreihenprognosen und Performance-Metriken.

---
## Lizenz

Dieses Projekt steht unter der **[Creative Commons Attribution-Share Alike 4.0 DE License (CC BY-SA 4.0 DE)](https://creativecommons.org/licenses/by-sa/4.0/deed.de)**.

---
## Hinweis

Dieses Repository entstand im Rahmen einer Masterarbeit am Karlsruher Institut für Technologie (KIT). Der enthaltene Code kann Fehler enthalten und wird nicht mehr gewartet.
