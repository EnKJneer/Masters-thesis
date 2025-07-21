RawData:
    Enthält alle Rohdaten.
    Die csv-Dateien enthalten die Sensordaten welche mit DT9836 aufgenommen wurden.
    Die jason-Dateien enthalten die Messungen der Sinumerik.

RawData_csv:
    Sinumerik:
    Enthält die mit read_data_from_json umgewandelten Sinumerik daten.
    Die daten enthalten 'DES_POS', 'ENC_POS', 'ENC1_POS', 'ENC2_POS', 'CURRENT', 'CTRL_DIFF', 'CTRL_DIFF2', 'CONT_DEV', 'CMD_SPEED'
    für alle axen.

    DT98363:
    Kraftmessungen mit den zugeordneten spalten enthalten.
    Gefiltert mit FIR-Filter (Cutoff-freq: 499 Hz).
    Mit Spline interpolation auf 1kHz gebracht.

