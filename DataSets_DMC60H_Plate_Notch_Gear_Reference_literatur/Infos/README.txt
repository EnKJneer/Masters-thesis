***************************************************************************
*** Part series dataset of milling processes for time series prediction ***
***************************************************************************

Zusammenfassung: Ziel des Datensatzes ist das Training und die Validierung von Modellen zur Vorhersage von Zeitreihen für Fräsprozesse. Dazu wurden an einer DMC 60H Prozesse mit einer Abtastrate von 500 Hz durch eine Siemens Industrial Edge aufgezeichnet und in einer JSON-Datei gespeichert. Die Maschine wurde steuerungstechnisch aufgerüstet und mit Kraftsensoren (Kistler 9255C) zur Erfassung der Prozesskräfte ausgestattet. Die Kräfte wurden mit einer Abtastrate von 10 kHz aufgezeichnet und in einer Matplab-Datei gespeichert. Insgesamt wurden drei verschiedene Geometrien abgebildet, ein Zahnrad, eine Nut und sowie die Form einer Adapterplatte. Diese wurden sowohl für die Bearbeitung von Stahl als auch von Aluminium verwendet. Zusätzlich wurden für jede der drei Grundgeometrien Variationen im Prozess durchgeführt, indem die Eingriffstiefe, die Geschwindigkeit sowie die Spindeldrehzahl (siehe NC-Codes) verändert wurden.

 
Abstract: The aim of the data set is the training and validation of models for the prediction of time series for milling processes. For this purpose, processes were recorded on a DMC 60H with a sampling rate of 500 Hz by a Siemens Industrial Edge and saved in a JSON file. The machine was upgraded in terms of control technology and equipped with force sensors (Kistler 9255C) to record the process forces. The forces were recorded at a sampling rate of 10 kHz and saved in a Matlab file. A total of three different geometries were milled, a gear, a notch and a plate. These were used for the machining of both steel and aluminium. In addition, variations were made in the process for each of the three basic geometries by changing the depth of engagement, the speed and the spindle speed (see NC codes).

---------------------------------------------------------------------------------------

Documents:
-Design of Experiments: Information on the toolpaths and technological Parameters of the experiments
-Recording information: Information about the recordings with comments
-Data: All recorded datasets. The first level contains the folders of each workpiece. In the next level, the individual executions are located. The individual recordings are stored in the form of a JSON (internal machine data) and a Matlab (external force sensors) file. This consists of a header with all relevant information such as the Signalsources followed by the entries of the recorded time series.
-NC-Code: NC programs executed on the machine
-Tools: Product informations of tools used
-Workpieces: Pictures of the worpieces

Experimental data:
-Machine: Retrofitted DMC 60H
-Material: S235JR, AL 2007 T4
-Tools:
   -VHM-Fräser HPC, TiSi, ⌀ f8 DC: 10mm
   -Schaftfräser HSS-Co8, TiAlN, ⌀ k10 DC: 10mm
-Workpiece blank dimensions: 150x75x50mm

License: This work is licensed under a Creative Commons Attribution 4.0 International License. Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0).