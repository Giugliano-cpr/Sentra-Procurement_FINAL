"""
Exportiert den vollständig bewerteten Datensatz als CSV für Power BI.

Power BI visualisiert nur – die eigentliche Erkennung passiert hier in Python.
Dieses Skript wendet die kombinierte Analyse an und schreibt jede Transaktion
inklusive aller Regel-Flags, des ML-Scores und der kombinierten Kategorie in
eine Datei, die sich direkt in Power BI (Get Data -> Text/CSV) importieren lässt.

Aufruf:
    python export_for_powerbi.py                      # nutzt den Beispieldatensatz
    python export_for_powerbi.py pfad/zu/daten.csv    # eigener (bereits passender) Datensatz

Ergebnis: data/sentra_scored.csv
"""

import sys
import pandas as pd

import detection

EINGABE = sys.argv[1] if len(sys.argv) > 1 else "data/beschaffungsdaten.csv"
AUSGABE = "data/sentra_scored.csv"


def main():
    df = pd.read_csv(EINGABE)
    ergebnis = detection.kombinierte_analyse(df)   # Standardparameter

    # für Power BI lesbarere Spaltennamen der Flags
    umbenennen = {
        "regel_doppelte_rechnung": "Flag_DoppelteRechnung",
        "regel_preisabweichung": "Flag_Preisabweichung",
        "regel_knapp_unter_grenze": "Flag_KnappUnterGrenze",
        "regel_auftragssplitting": "Flag_Auftragssplitting",
        "regel_treffer": "Flag_Regel",
        "ml_anomalie": "Flag_ML",
        "ml_score": "ML_Score",
        "kombinierte_kategorie": "Erkennungsquelle",
    }
    ergebnis = ergebnis.rename(columns=umbenennen)

    # utf-8-sig sorgt dafür, dass Umlaute in Power BI korrekt erscheinen
    ergebnis.to_csv(AUSGABE, index=False, encoding="utf-8-sig")
    print(f"Export geschrieben: {AUSGABE}  ({len(ergebnis)} Zeilen, "
          f"{len(ergebnis.columns)} Spalten)")
    print("Spalten:", ", ".join(ergebnis.columns))


if __name__ == "__main__":
    main()
