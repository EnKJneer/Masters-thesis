import pandas as pd
import os

# Liste der ursprünglichen Dateinamen
dateinamen = [
    "Stahl Plate Normal 1", "Stahl Plate Normal 2", "Stahl Plate SF 1",
    "Stahl Plate SF 2", "Stahl Plate SF 3", "Stahl Plate Depth 1",
    "Stahl Plate Depth 2", "Stahl Plate Depth 3", "Stahl Notch Normal 1",
    "Stahl Notch Normal 2", "Stahl Notch Normal 3", "Stahl Notch Depth 1",
    "Stahl Notch Depth 2", "Stahl Notch Depth 3", "Stahl Plate Normal 3",
    "Stahl Gear Normal 3", "Aluminium Plate Normal 3", "Aluminium Gear Normal 3"
]

# Funktion zur Umbenennung der Dateinamen
def umbenennen(name):
    name = name.replace("Stahl", "S235JR")
    name = name.replace("Aluminium", "AL2007T4")
    name = name.replace(" ", "_")
    return f"DMC60H_{name}"

# Verzeichnis, in dem die Dateien liegen (hier: aktuelles Verzeichnis)
verzeichnis = "Data"

dateinamen = os.listdir(verzeichnis)

# Durch jede Datei iterieren
for alter_name in dateinamen:
    neuer_name = umbenennen(alter_name)
    pfad = os.path.join(verzeichnis, neuer_name + ".csv")
    pfad = os.path.join(verzeichnis,alter_name)
    # Prüfen, ob die Datei existiert
    if os.path.exists(pfad):
        # Datei laden
        df = pd.read_csv(pfad)
        anzahl_zeilen = len(df)

        # Ausgabe
        print(f"Dateiname: {alter_name}, Zeitschritte: {anzahl_zeilen}")
    else:
        print(f"Datei nicht gefunden: {pfad}")
