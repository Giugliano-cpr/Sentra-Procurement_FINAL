"""
Generator für einen realistischen synthetischen Beschaffungsdatensatz.

Erzeugt Bestell-/Rechnungsdaten mit absichtlich eingebauten Auffälligkeiten,
damit sich sowohl die regelbasierte Erkennung als auch der Isolation Forest
testen lassen. Fester Seed => reproduzierbar (wichtig für den Bericht).

Eingebaute Anomalietypen:
  1. Doppelte Rechnungen      -> Ziel der regelbasierten Schicht
  2. Starke Preisabweichungen -> Ziel der regelbasierten Schicht
  3. Multivariate Ausreißer   -> Ziel des Isolation Forest
"""

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Stammdaten
# ---------------------------------------------------------------------------
LIEFERANTEN = [
    "Alpha Components GmbH", "Beta Metalltechnik", "Gamma Logistik AG",
    "Delta Elektronik", "Epsilon Rohstoffe", "Zeta Verpackung",
    "Eta Industriebedarf", "Theta Werkzeuge",
]

# Materialkategorien mit typischem Stückpreis (Mittelwert, Streuung)
KATEGORIEN = {
    "Elektronik":   (120.0, 25.0),
    "Rohmaterial":  (8.0,   1.5),
    "Verpackung":   (2.5,   0.4),
    "Werkzeuge":    (45.0,  10.0),
    "Logistik":     (300.0, 60.0),
    "Bueromaterial": (15.0, 3.0),
}

N_NORMAL = 600          # Anzahl unauffälliger Datensätze
START_DATUM = pd.Timestamp("2024-01-01")
GENEHMIGUNGSGRENZE = 10000.0   # Referenzwert für die injizierten Umgehungen


def _zufallsdatum():
    tage = int(rng.integers(0, 365))
    return START_DATUM + pd.Timedelta(days=tage)


def erzeuge_basisdaten(n: int) -> pd.DataFrame:
    """Erzeugt n unauffällige Bestellpositionen."""
    zeilen = []
    for i in range(n):
        kategorie = rng.choice(list(KATEGORIEN.keys()))
        mittel, streuung = KATEGORIEN[kategorie]
        stueckpreis = max(0.5, rng.normal(mittel, streuung))
        menge = int(rng.integers(1, 200))
        zeilen.append({
            "rechnung_id": f"INV-{100000 + i}",
            "bestell_id": f"PO-{500000 + i}",
            "bestelldatum": _zufallsdatum(),
            "lieferant": rng.choice(LIEFERANTEN),
            "kategorie": kategorie,
            "material_id": f"MAT-{rng.integers(1000, 1100)}",
            "menge": menge,
            "stueckpreis": round(stueckpreis, 2),
            "lieferzeit_tage": int(np.clip(rng.normal(14, 5), 1, 60)),
        })
    df = pd.DataFrame(zeilen)
    df["gesamtbetrag"] = (df["menge"] * df["stueckpreis"]).round(2)
    df["anomalie_typ"] = "normal"   # Ground truth (nur zur Evaluation!)
    return df


def injiziere_doppelte_rechnungen(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Kopiert zufällige Zeilen mit minimal verändertem Datum (Doppelzahlung)."""
    indizes = rng.choice(df.index, size=n, replace=False)
    kopien = df.loc[indizes].copy()
    # gleicher Lieferant, gleicher Betrag, leicht verschobenes Datum,
    # neue (aber andere) Rechnungs-ID -> klassischer Doppelzahlungsfall
    kopien["bestelldatum"] = kopien["bestelldatum"] + pd.to_timedelta(
        rng.integers(0, 4, size=n), unit="D"
    )
    kopien["rechnung_id"] = [f"INV-DUP-{i}" for i in range(n)]
    kopien["anomalie_typ"] = "doppelte_rechnung"
    return pd.concat([df, kopien], ignore_index=True)


def injiziere_preisanomalien(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Setzt bei zufälligen Zeilen einen stark überhöhten/zu niedrigen Stückpreis."""
    indizes = rng.choice(df.index, size=n, replace=False)
    for idx in indizes:
        faktor = rng.choice([rng.uniform(3.0, 6.0), rng.uniform(0.1, 0.3)])
        df.loc[idx, "stueckpreis"] = round(df.loc[idx, "stueckpreis"] * faktor, 2)
        df.loc[idx, "gesamtbetrag"] = round(
            df.loc[idx, "menge"] * df.loc[idx, "stueckpreis"], 2
        )
        df.loc[idx, "anomalie_typ"] = "preisanomalie"
    return df


def injiziere_multivariate_ausreisser(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Ungewöhnliche Kombinationen (hohe Menge + lange Lieferzeit + hoher Preis).
    Jeder Einzelwert wirkt plausibel, erst die Kombination ist auffällig
    -> typischer Fall für den Isolation Forest."""
    neue = []
    naechste_id = 900000
    for k in range(n):
        kategorie = rng.choice(list(KATEGORIEN.keys()))
        mittel, _ = KATEGORIEN[kategorie]
        menge = int(rng.integers(400, 800))            # ungewöhnlich hohe Menge
        stueckpreis = round(mittel * rng.uniform(1.8, 2.5), 2)
        neue.append({
            "rechnung_id": f"INV-MV-{k}",
            "bestell_id": f"PO-{naechste_id + k}",
            "bestelldatum": _zufallsdatum(),
            "lieferant": rng.choice(LIEFERANTEN),
            "kategorie": kategorie,
            "material_id": f"MAT-{rng.integers(1000, 1100)}",
            "menge": menge,
            "stueckpreis": stueckpreis,
            "lieferzeit_tage": int(rng.integers(45, 60)),  # lange Lieferzeit
            "gesamtbetrag": round(menge * stueckpreis, 2),
            "anomalie_typ": "multivariater_ausreisser",
        })
    return pd.concat([df, pd.DataFrame(neue)], ignore_index=True)


def injiziere_knapp_unter_grenze(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Setzt bei zufälligen Zeilen den Betrag gezielt knapp unter die Grenze
    (klassische Umgehung, um eine Freigabe zu vermeiden)."""
    indizes = rng.choice(df.index, size=n, replace=False)
    for idx in indizes:
        betrag = round(rng.uniform(GENEHMIGUNGSGRENZE * 0.955, GENEHMIGUNGSGRENZE * 0.995), 2)
        menge = int(df.loc[idx, "menge"]) or 1
        df.loc[idx, "gesamtbetrag"] = betrag
        df.loc[idx, "stueckpreis"] = round(betrag / menge, 2)
        df.loc[idx, "anomalie_typ"] = "knapp_unter_grenze"
    return df


def injiziere_auftragssplitting(df: pd.DataFrame, n_gruppen: int = 6) -> pd.DataFrame:
    """Erzeugt Gruppen aufgeteilter Bestellungen: gleicher Lieferant + Kategorie,
    kurzer Zeitraum, jede Einzelbestellung unter der Grenze, Summe darüber."""
    neue = []
    laufnr = 700000
    for g in range(n_gruppen):
        lieferant = rng.choice(LIEFERANTEN)
        kategorie = rng.choice(list(KATEGORIEN.keys()))
        material = f"MAT-{rng.integers(1000, 1100)}"
        basisdatum = _zufallsdatum()
        teile = int(rng.integers(3, 5))   # 3-4 Teilbestellungen
        for t in range(teile):
            betrag = round(rng.uniform(0.55, 0.85) * GENEHMIGUNGSGRENZE, 2)  # je < Grenze
            menge = int(rng.integers(20, 120))
            neue.append({
                "rechnung_id": f"INV-SPLIT-{g}-{t}",
                "bestell_id": f"PO-{laufnr}",
                "bestelldatum": basisdatum + pd.Timedelta(days=int(rng.integers(0, 6))),
                "lieferant": lieferant,
                "kategorie": kategorie,
                "material_id": material,
                "menge": menge,
                "stueckpreis": round(betrag / menge, 2),
                "lieferzeit_tage": int(np.clip(rng.normal(14, 5), 1, 60)),
                "gesamtbetrag": betrag,
                "anomalie_typ": "auftragssplitting",
            })
            laufnr += 1
    return pd.concat([df, pd.DataFrame(neue)], ignore_index=True)


def main():
    df = erzeuge_basisdaten(N_NORMAL)
    df = injiziere_preisanomalien(df, n=20)
    df = injiziere_doppelte_rechnungen(df, n=15)
    df = injiziere_multivariate_ausreisser(df, n=15)
    df = injiziere_knapp_unter_grenze(df, n=12)
    df = injiziere_auftragssplitting(df, n_gruppen=6)

    # Reihenfolge mischen, damit Anomalien nicht am Ende klumpen
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df["bestelldatum"] = pd.to_datetime(df["bestelldatum"]).dt.date

    spalten = [
        "rechnung_id", "bestell_id", "bestelldatum", "lieferant", "kategorie",
        "material_id", "menge", "stueckpreis", "gesamtbetrag",
        "lieferzeit_tage", "anomalie_typ",
    ]
    df = df[spalten]
    df.to_csv("data/beschaffungsdaten.csv", index=False)
    print(f"Datensatz erzeugt: {len(df)} Zeilen -> data/beschaffungsdaten.csv")
    print(df["anomalie_typ"].value_counts())


if __name__ == "__main__":
    main()
