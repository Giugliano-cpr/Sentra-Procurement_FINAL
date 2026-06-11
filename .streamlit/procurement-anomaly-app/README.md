# Hybride Anomalieerkennung in Beschaffungsdaten

Prototyp einer datenbasierten Procurement-App im Rahmen des Moduls
*Digital Procurement & Data Science* (THI). Die App kombiniert zwei
Erkennungsschichten:

1. **Regelbasierte Erkennung** – klar definierte, erklärbare Auffälligkeiten
   - doppelte Rechnungen (gleicher Lieferant + Betrag im engen Zeitfenster)
   - starke Preisabweichungen vom Median der Materialkategorie
   - Umgehung von Genehmigungsgrenzen:
     - Beträge knapp unter der Freigabegrenze
     - Auftragssplitting (mehrere Bestellungen je unter, in Summe über der Grenze)
2. **Machine-Learning-Erkennung** – Isolation Forest für unbekannte,
   multivariate Muster in den Beschaffungsdaten

Beide Schichten werden zusammengeführt und jede Buchung wird kategorisiert:
*unauffällig / nur Regel / nur ML / Regel + ML*.

## Projektstruktur

```
procurement-anomaly-app/
├── app.py             # Streamlit-Oberfläche (UI)
├── detection.py       # Erkennungslogik (Regeln + Isolation Forest)
├── generate_data.py   # Generator für den synthetischen Beispieldatensatz
├── data/
│   └── beschaffungsdaten.csv
├── requirements.txt
└── README.md
```

Die Trennung von Oberfläche (`app.py`) und Logik (`detection.py`) hält den
Code testbar und einzelne Funktionen im Bericht zitierbar.

## Installation & Start

```bash
conda create -n procurement-app python=3.11
conda activate procurement-app
pip install -r requirements.txt
streamlit run app.py
```

Beispieldatensatz neu erzeugen (optional, reproduzierbar über festen Seed):

```bash
python generate_data.py
```

Logik ohne Oberfläche testen:

```bash
python detection.py
```

## Bedienung

In der Seitenleiste:
- Datenquelle wählen (Beispieldatensatz oder eigene CSV)
- Bei eigener CSV: **Spalten-Zuordnung** – jede Spalte deiner Datei wird per
  Dropdown einem internen Feld zugeordnet (mit automatischen Vorschlägen).
  Fehlende Felder werden, wo möglich, abgeleitet (z. B. Gesamtbetrag aus
  Menge × Stückpreis). Regeln, für die Daten fehlen, werden übersprungen.
- Regel-Parameter (Zeitfenster, Preisschwelle, Genehmigungsgrenze, Marge, Splitting-Fenster)
- Isolation Forest (Feature-Auswahl, erwarteter Anomalieanteil)

Die vier Tabs zeigen Überblick, regelbasierte Treffer, ML-Anomalien und die
kombinierte Auswertung inklusive CSV-Export.

## Datenformat (für eigene CSV)

Erwartete Spalten: `rechnung_id`, `bestelldatum`, `lieferant`, `kategorie`,
`menge`, `stueckpreis`, `gesamtbetrag`, `lieferzeit_tage`. Numerische Spalten
lassen sich frei als Modell-Features auswählen.

> Hinweis: Der mitgelieferte Datensatz ist **synthetisch** und enthält eine
> Spalte `anomalie_typ` (Ground Truth), die ausschließlich der Evaluation
> dient. Bei echten Daten entfällt diese Spalte.
