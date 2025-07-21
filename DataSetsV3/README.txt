RawData:
    Enthält alle Rohdaten.
    Die csv-Dateien enthalten die Sensordaten welche mit DT9836 aufgenommen wurden.
    Die jason-Dateien enthalten die Messungen der Sinumerik.

DataSinumerik_50Hz:
    Ergebnis von Sinumerik_json_to_csv50Hz
    Downsampeld von 500 Hz auf 50 Hz.
    Gefiltert mit FIR-Filter (Cutoff-freq: 24 Hz) und Mittelwertfilter (Fenster: 5).
    Mit Spline interpolation auf 50Hz gebracht.
    Konvertiert die Daten von dem json file in ein csv File.
    Die daten enthalten die positionen, Geschwindigkeiten, beschleunigungen, ströme und reveler inputs.

DT98363_50Hz:
    Ergebnis aus DT98363_to_50Hz.
    Kraftmessungen mit den zugeordneten spalten enthalten.
    Gefiltert mit FIR-Filter (Cutoff-freq: 24 Hz) und Mittelwertfilter (Fenster: Down-Samplingfaktor / 2).
    Mit Spline interpolation auf 50Hz gebracht.

DataMerged:
    Ergebnis aus Merge_Sinumerik_DT9836.
    Zusammengeführte daten von