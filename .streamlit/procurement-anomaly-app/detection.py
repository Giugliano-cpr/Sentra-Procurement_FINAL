"""
Erkennungslogik für den hybriden Ansatz.

Zwei Schichten, bewusst getrennt:

  1) regelbasierte_erkennung(): klar definierte, erklärbare Regeln
       - doppelte Rechnungen (gleicher Lieferant + Betrag, naher Zeitraum)
       - starke Preisabweichung vom Kategorie-Median

  2) ml_erkennung(): Isolation Forest für multivariate, unbekannte Muster

Die Funktion kombinierte_analyse() führt beide zusammen und kategorisiert
jede Zeile: unauffällig / nur Regel / nur ML / Regel + ML.

Bewusst UI-frei gehalten -> in app.py importierbar und unabhängig testbar.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 0) Spalten-Zuordnung / Standardisierung beliebiger CSV-Dateien
# ---------------------------------------------------------------------------
# Interne Zielfelder, mit denen die Erkennungslogik arbeitet:
ZIEL_FELDER = [
    "rechnung_id", "bestelldatum", "lieferant", "kategorie",
    "menge", "stueckpreis", "gesamtbetrag", "lieferzeit_tage",
]
NUMERISCHE_FELDER = ["menge", "stueckpreis", "gesamtbetrag", "lieferzeit_tage"]


def _hat(df: pd.DataFrame, *cols: str) -> bool:
    """True, wenn alle angegebenen Spalten vorhanden sind."""
    return all(c in df.columns for c in cols)


def standardisiere(df_roh: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Übersetzt eine beliebige CSV in das interne Schema.

    mapping: {zielfeld -> spaltenname_in_df_roh oder None}
    Fehlende Felder werden, wo möglich, abgeleitet (Gesamtbetrag aus
    Menge x Stückpreis bzw. Stückpreis aus Gesamtbetrag / Menge) oder mit
    sinnvollen Standardwerten gefüllt. Nicht herleitbare Felder bleiben weg;
    die zugehörigen Regeln werden später automatisch übersprungen.
    """
    out = pd.DataFrame(index=df_roh.index)
    for ziel, quelle in mapping.items():
        if quelle and quelle in df_roh.columns:
            out[ziel] = df_roh[quelle]

    for f in NUMERISCHE_FELDER:
        if f in out.columns:
            out[f] = pd.to_numeric(out[f], errors="coerce")

    # Ableitungen
    if "gesamtbetrag" not in out.columns and _hat(out, "menge", "stueckpreis"):
        out["gesamtbetrag"] = (out["menge"] * out["stueckpreis"]).round(2)
    if "stueckpreis" not in out.columns and _hat(out, "menge", "gesamtbetrag"):
        out["stueckpreis"] = (out["gesamtbetrag"] / out["menge"].replace(0, np.nan)).round(2)

    # Standardwerte für nicht zwingende Felder
    if "kategorie" not in out.columns:
        out["kategorie"] = "Alle"
    if "lieferant" not in out.columns:
        out["lieferant"] = "Unbekannt"
    if "rechnung_id" not in out.columns:
        out["rechnung_id"] = [f"ROW-{i}" for i in range(len(out))]

    return out


# ---------------------------------------------------------------------------
# 1) Regelbasierte Schicht
# ---------------------------------------------------------------------------
def regel_doppelte_rechnungen(df: pd.DataFrame, tage_fenster: int = 5) -> pd.Series:
    """Markiert Zeilen, die eine wahrscheinliche Doppelzahlung darstellen:
    gleicher Lieferant UND gleicher Gesamtbetrag, deren Bestelldaten höchstens
    `tage_fenster` Tage auseinanderliegen.

    Rückgabe: bool-Series (True = verdächtig). Fehlen benötigte Spalten,
    wird die Regel übersprungen (alles False).
    """
    if not _hat(df, "lieferant", "gesamtbetrag", "bestelldatum"):
        return pd.Series(False, index=df.index)
    datum = pd.to_datetime(df["bestelldatum"], errors="coerce")
    treffer = pd.Series(False, index=df.index)

    # nur Gruppen mit potenziellen Dubletten betrachten (gleicher Lieferant+Betrag)
    gruppen = df.groupby(["lieferant", "gesamtbetrag"]).groups
    for schluessel, idx in gruppen.items():
        if len(idx) < 2:
            continue
        teil = datum.loc[idx].sort_values()
        # liegen zwei Buchungen innerhalb des Zeitfensters?
        for i in range(len(teil)):
            nah = (teil - teil.iloc[i]).abs().dt.days <= tage_fenster
            if nah.sum() >= 2:
                treffer.loc[teil.index[nah]] = True
    return treffer


def regel_preisabweichung(df: pd.DataFrame, schwelle_prozent: float = 70.0) -> pd.Series:
    """Markiert Zeilen, deren Stückpreis stark vom Median ihrer Kategorie abweicht.

    schwelle_prozent = 100 bedeutet: Preis weicht um mehr als 100 % (also > 2x
    oder < 0.5x) vom Kategorie-Median ab.

    Rückgabe: bool-Series (True = verdächtig).
    """
    if not _hat(df, "stueckpreis", "kategorie"):
        return pd.Series(False, index=df.index)
    median_je_kat = df.groupby("kategorie")["stueckpreis"].transform("median")
    abweichung = (df["stueckpreis"] - median_je_kat).abs() / median_je_kat * 100
    return abweichung > schwelle_prozent

def regel_knapp_unter_grenze(
    df: pd.DataFrame, grenze: float = 10000.0, marge_prozent: float = 5.0
) -> pd.Series:
    """Markiert Bestellungen, deren Gesamtbetrag knapp UNTER einer
    Genehmigungsgrenze liegt (mögliche Umgehung einer Freigabe).

    Treffer, wenn:  grenze * (1 - marge/100)  <=  betrag  <  grenze
    """
    if not _hat(df, "gesamtbetrag"):
        return pd.Series(False, index=df.index)
    untergrenze = grenze * (1 - marge_prozent / 100)
    betrag = df["gesamtbetrag"]
    return (betrag >= untergrenze) & (betrag < grenze)


def regel_auftragssplitting(
    df: pd.DataFrame, grenze: float = 10000.0, tage_fenster: int = 7
) -> pd.Series:
    """Markiert mögliches Auftragssplitting: mehrere Einzelbestellungen, die je
    unter der Grenze liegen, aber bei gleichem Lieferant + gleicher Kategorie
    innerhalb eines kurzen Zeitfensters in Summe die Grenze überschreiten.
    """
    if not _hat(df, "lieferant", "kategorie", "gesamtbetrag", "bestelldatum"):
        return pd.Series(False, index=df.index)
    datum = pd.to_datetime(df["bestelldatum"], errors="coerce")
    treffer = pd.Series(False, index=df.index)

    for _, idx in df.groupby(["lieferant", "kategorie"]).groups.items():
        idx = list(idx)
        # nur Einzelbestellungen unter der Grenze mit gültigem Datum
        unter = [i for i in idx
                 if df.loc[i, "gesamtbetrag"] < grenze and pd.notna(datum.loc[i])]
        if len(unter) < 2:
            continue
        unter = sorted(unter, key=lambda i: datum.loc[i])
        for i in unter:
            start = datum.loc[i]
            fenster = [j for j in unter
                       if 0 <= (datum.loc[j] - start).days <= tage_fenster]
            if len(fenster) >= 2 and df.loc[fenster, "gesamtbetrag"].sum() > grenze:
                treffer.loc[fenster] = True
    return treffer


def regelbasierte_erkennung(
    df: pd.DataFrame,
    tage_fenster: int = 5,
    preis_schwelle_prozent: float = 70.0,
    grenze: float = 10000.0,
    grenze_marge_prozent: float = 5.0,
    split_fenster: int = 7,
) -> pd.DataFrame:
    """Wendet alle Regeln an und ergänzt das DataFrame um Regel-Spalten."""
    out = df.copy()
    out["regel_doppelte_rechnung"] = regel_doppelte_rechnungen(out, tage_fenster)
    out["regel_preisabweichung"] = regel_preisabweichung(out, preis_schwelle_prozent)
    out["regel_knapp_unter_grenze"] = regel_knapp_unter_grenze(
        out, grenze, grenze_marge_prozent)
    out["regel_auftragssplitting"] = regel_auftragssplitting(
        out, grenze, split_fenster)
    out["regel_treffer"] = (
        out["regel_doppelte_rechnung"]
        | out["regel_preisabweichung"]
        | out["regel_knapp_unter_grenze"]
        | out["regel_auftragssplitting"]
    )
    return out


# ---------------------------------------------------------------------------
# 2) Machine-Learning-Schicht (Isolation Forest)
# ---------------------------------------------------------------------------
STANDARD_FEATURES = ["stueckpreis", "menge", "gesamtbetrag", "lieferzeit_tage"]


def ml_erkennung(
    df: pd.DataFrame,
    features: list[str] | None = None,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Isolation Forest auf numerischen Features.

    contamination = erwarteter Anteil an Anomalien (Stellschraube für den Nutzer).
    Ergänzt:
      - ml_score      : Anomalie-Score (je niedriger, desto auffälliger)
      - ml_anomalie   : bool (True = vom Modell als Anomalie markiert)
    """
    out = df.copy()
    if features is None:
        features = STANDARD_FEATURES
    # nur tatsächlich vorhandene Features verwenden
    features = [f for f in features if f in out.columns]
    if not features:
        # ohne Features kann der Isolation Forest nicht arbeiten
        out["ml_score"] = 0.0
        out["ml_anomalie"] = False
        return out

    X = out[features].apply(pd.to_numeric, errors="coerce")
    # fehlende Werte je Spalte mit dem Median auffüllen (robust gegen NaN)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    X_scaled = StandardScaler().fit_transform(X.values)

    modell = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    label = modell.fit_predict(X_scaled)          # -1 = Anomalie, 1 = normal
    out["ml_score"] = modell.decision_function(X_scaled)
    out["ml_anomalie"] = label == -1
    return out


# ---------------------------------------------------------------------------
# 3) Kombination beider Schichten
# ---------------------------------------------------------------------------
def kombinierte_analyse(
    df: pd.DataFrame,
    tage_fenster: int = 5,
    preis_schwelle_prozent: float = 70.0,
    grenze: float = 10000.0,
    grenze_marge_prozent: float = 5.0,
    split_fenster: int = 7,
    features: list[str] | None = None,
    contamination: float = 0.05,
) -> pd.DataFrame:
    """Führt regelbasierte und ML-Erkennung aus und kategorisiert jede Zeile."""
    out = regelbasierte_erkennung(
        df, tage_fenster, preis_schwelle_prozent,
        grenze, grenze_marge_prozent, split_fenster,
    )
    out = ml_erkennung(out, features, contamination)

    def kategorisieren(row):
        regel, ml = row["regel_treffer"], row["ml_anomalie"]
        if regel and ml:
            return "Regel + ML"
        if regel:
            return "nur Regel"
        if ml:
            return "nur ML"
        return "unauffaellig"

    out["kombinierte_kategorie"] = out.apply(kategorisieren, axis=1)
    return out


# ---------------------------------------------------------------------------
# Kennzahlen-Helfer für die UI / Evaluation
# ---------------------------------------------------------------------------
def kennzahlen(df: pd.DataFrame) -> dict:
    return {
        "gesamt": len(df),
        "regel_treffer": int(df["regel_treffer"].sum()),
        "ml_anomalien": int(df["ml_anomalie"].sum()),
        "beide": int((df["regel_treffer"] & df["ml_anomalie"]).sum()),
        "mindestens_eine": int((df["regel_treffer"] | df["ml_anomalie"]).sum()),
    }


if __name__ == "__main__":
    # kleiner Selbsttest gegen den synthetischen Datensatz
    daten = pd.read_csv("data/beschaffungsdaten.csv")
    ergebnis = kombinierte_analyse(daten)
    print(kennzahlen(ergebnis))
    print(ergebnis["kombinierte_kategorie"].value_counts())
    # Plausibilität: wie gut deckt sich die Erkennung mit der Ground Truth?
    print(pd.crosstab(ergebnis["anomalie_typ"], ergebnis["kombinierte_kategorie"]))
