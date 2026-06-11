"""
SENTRA - Procurement Anomaly Intelligence
Streamlit-Oberfläche für die hybride Anomalieerkennung.

Start:  streamlit run app.py

Designprinzip: konsequente Reduktion (Enterprise-Minimalismus).
Ein zentrales Farb- und Typografiesystem, das Oberfläche und Diagramme teilen.
"""

import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

import detection

# Verzeichnis dieser Datei – macht Dateipfade unabhängig vom Arbeitsverzeichnis
BASIS_VERZEICHNIS = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Markenidentität (hier zentral änderbar)
# ---------------------------------------------------------------------------
PRODUKT_NAME = "SENTRA"
PRODUKT_TAGLINE = "Procurement Anomaly Intelligence"

# Logo: Datenpunktfeld mit einer hervorgehobenen Anomalie (passt zur Bildsprache)
LOGO_SVG = """
<svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="8"  cy="8"  r="2.3" fill="#CBD5E1"/>
  <circle cx="18" cy="8"  r="2.3" fill="#94A3B8"/>
  <circle cx="8"  cy="18" r="2.3" fill="#94A3B8"/>
  <circle cx="18" cy="18" r="2.3" fill="#CBD5E1"/>
  <circle cx="8"  cy="28" r="2.3" fill="#CBD5E1"/>
  <circle cx="18" cy="28" r="2.3" fill="#94A3B8"/>
  <circle cx="28" cy="28" r="2.3" fill="#CBD5E1"/>
  <circle cx="28" cy="18" r="5.0" fill="none" stroke="#B42318" stroke-width="1.5"/>
  <circle cx="28" cy="18" r="2.6" fill="#B42318"/>
</svg>
"""

# ---------------------------------------------------------------------------
# Zentrales Farbsystem (einzige Quelle der Wahrheit, auch für Diagramme)
# ---------------------------------------------------------------------------
INK = "#0F172A"          # Primärtext / dominant
MUTED = "#64748B"        # Sekundärtext
BORDER = "#E6E8EB"       # Hairlines
GRID = "#EEF0F2"         # Diagrammraster
ACCENT = "#B42318"       # einziger Signalakzent (sparsam!)

# Sequenzielle Schweregrad-Skala (hell -> dunkel, Akzent nur fuer das Kritischste)
SEVERITY = {
    "Unauffällig": "#CBD5E1",
    "Nur Regel": "#475569",
    "Nur ML": "#94A3B8",
    "Regel + ML": ACCENT,
}
ANZEIGE = {
    "unauffaellig": "Unauffällig",
    "nur Regel": "Nur Regel",
    "nur ML": "Nur ML",
    "Regel + ML": "Regel + ML",
}
ANZEIGE_REVERSE = {v: k for k, v in ANZEIGE.items()}
BOOL_MAP = {True: ACCENT, False: "#CBD5E1"}

st.set_page_config(page_title=f"{PRODUKT_NAME} – {PRODUKT_TAGLINE}", layout="wide")


# ---------------------------------------------------------------------------
# Designsystem (CSS)
# ---------------------------------------------------------------------------
def stil_anwenden():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        html, body, [class*="css"], .stMarkdown, .stDataFrame, button, input, label {
            font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Standard-Streamlit-Chrome dezent entfernen, aber die Sidebar-Steuerung
           (Pfeil zum Auf-/Zuklappen) erhalten -> Header NICHT komplett ausblenden */
        #MainMenu, footer, [data-testid="stToolbar"],
        [data-testid="stAppDeployButton"] { display: none; }
        header[data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"] { visibility: visible !important; }

        .block-container {
            max-width: 1180px;
            padding-top: 1.2rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3 {
            color: #0F172A;
            font-weight: 600;
            letter-spacing: -0.01em;
        }
        h2 { font-size: 1.15rem !important; margin-top: 0.4rem; }
        h3 { font-size: 1.0rem !important; }

        /* Kopfzeile / Wortmarke */
        .sentra-header { display: flex; align-items: center; gap: 0.85rem; padding: 0.2rem 0 1.0rem 0; }
        .sentra-logo { line-height: 0; }
        .sentra-mark {
            font-size: 1.55rem;
            font-weight: 600;
            letter-spacing: 0.20em;
            color: #0F172A;
        }
        .sentra-tag {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.70rem;
            text-transform: uppercase;
            letter-spacing: 0.26em;
            color: #94A3B8;
            margin-top: 0.25rem;
        }
        .sentra-rule {
            border: none;
            border-top: 1px solid #E6E8EB;
            margin: 0.4rem 0 1.4rem 0;
        }
        .sentra-sub { color: #64748B; font-size: 0.92rem; }

        /* Kennzahlen-Karten */
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E6E8EB;
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
        }
        [data-testid="stMetricValue"] {
            font-family: 'IBM Plex Mono', monospace;
            font-weight: 500;
            color: #0F172A;
        }
        [data-testid="stMetricLabel"] p {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem !important;
            color: #64748B;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #F6F7F9;
            border-right: 1px solid #E6E8EB;
        }
        [data-testid="stSidebar"] h2 { font-size: 0.78rem !important;
            text-transform: uppercase; letter-spacing: 0.10em; color: #64748B; }

        /* Tabs */
        [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid #E6E8EB; }
        [data-baseweb="tab"] {
            font-size: 0.9rem; color: #64748B;
            padding-left: 0; padding-right: 0;
        }
        [aria-selected="true"][data-baseweb="tab"] { color: #0F172A; font-weight: 500; }
        [data-baseweb="tab-highlight"] { background-color: #0F172A; }

        /* Buttons */
        [data-testid="stDownloadButton"] button,
        .stButton button {
            background: #0F172A; color: #FFFFFF;
            border: none; border-radius: 8px;
            font-weight: 500; letter-spacing: 0.02em;
        }
        [data-testid="stDownloadButton"] button:hover { background: #1E293B; color:#FFFFFF; }

        [data-testid="stDataFrame"] { border: 1px solid #E6E8EB; border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kopfzeile():
    st.markdown(
        f"""
        <div class="sentra-header">
            <div class="sentra-logo">{LOGO_SVG}</div>
            <div>
                <div class="sentra-mark">{PRODUKT_NAME}</div>
                <div class="sentra-tag">{PRODUKT_TAGLINE}</div>
            </div>
        </div>
        <hr class="sentra-rule"/>
        """,
        unsafe_allow_html=True,
    )


def diagramm_stil(fig):
    """Einheitliches, reduziertes Diagramm-Layout fuer alle Charts."""
    fig.update_layout(
        font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0,
            title="", font=dict(size=12, color=MUTED),
        ),
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, ticks="outside",
                     tickcolor=BORDER, title_font=dict(color=MUTED, size=12))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     linecolor=BORDER, title_font=dict(color=MUTED, size=12))
    return fig


# ---------------------------------------------------------------------------
# Daten
# ---------------------------------------------------------------------------
@st.cache_data
def lade_beispieldaten() -> pd.DataFrame:
    return pd.read_csv(BASIS_VERZEICHNIS / "data" / "beschaffungsdaten.csv")


# Beschriftungen der internen Zielfelder für die Spalten-Zuordnung
FELD_LABELS = {
    "lieferant": "Lieferant",
    "bestelldatum": "Bestelldatum",
    "kategorie": "Warengruppe / Kategorie",
    "menge": "Menge",
    "stueckpreis": "Stückpreis",
    "gesamtbetrag": "Gesamtbetrag",
    "lieferzeit_tage": "Lieferzeit (Tage)",
    "rechnung_id": "Rechnungs- / Beleg-ID",
}

# Stichwörter zur automatischen Vorab-Zuordnung (Substring, klein geschrieben)
SYNONYME = {
    "lieferant": ["lieferant", "supplier", "vendor", "seller", "verkäufer", "verkaeufer"],
    "bestelldatum": ["bestelldatum", "order_date", "orderdate", "datum", "date", "invoice_date"],
    "kategorie": ["kategorie", "category", "warengruppe", "produktkategorie", "material_group", "item_category", "segment"],
    "stueckpreis": ["stueckpreis", "stückpreis", "unit_price", "unitprice", "einzelpreis", "preis", "price"],
    "menge": ["menge", "quantity", "qty", "order_quantity", "anzahl"],
    "gesamtbetrag": ["gesamtbetrag", "total_amount", "total", "line_total", "betrag", "amount", "value", "spend", "summe"],
    "lieferzeit_tage": ["lieferzeit", "lead_time", "leadtime", "delivery_days", "liefertage"],
    "rechnung_id": ["rechnung", "invoice", "beleg", "po_id", "order_id", "bestell"],
}


def rate_spalte(ziel: str, spalten: list[str]) -> str | None:
    """Schlägt anhand der Spaltennamen eine passende Quellspalte vor."""
    klein = {s.lower(): s for s in spalten}
    for stichwort in SYNONYME.get(ziel, []):
        for low, original in klein.items():
            if stichwort in low:
                return original
    return None


def vorhandene(df: pd.DataFrame, spalten: list[str]) -> list[str]:
    """Filtert eine Spaltenliste auf die tatsächlich vorhandenen Spalten."""
    return [s for s in spalten if s in df.columns]


# ---------------------------------------------------------------------------
# Aufbau
# ---------------------------------------------------------------------------
stil_anwenden()
kopfzeile()

with st.sidebar:
    st.markdown(f"### {PRODUKT_NAME}")
    st.header("Datenquelle")
    quelle = st.radio(
        "Quelle", ["Beispieldatensatz", "Eigene CSV hochladen", "Keine Daten"],
        label_visibility="collapsed",
    )

    df_basis = None
    if quelle == "Beispieldatensatz":
        df_basis = lade_beispieldaten()
    elif quelle == "Eigene CSV hochladen":
        datei = st.file_uploader("CSV-Datei", type="csv")
        if datei is not None:
            df_roh = pd.read_csv(datei)
            st.header("Spalten-Zuordnung")
            st.caption("Ordne die Spalten deiner Datei den Feldern zu. "
                       "Vorschläge sind bereits vorausgewählt.")
            KEINE = "(nicht vorhanden)"
            mapping = {}
            for ziel, label in FELD_LABELS.items():
                optionen = [KEINE] + list(df_roh.columns)
                vorschlag = rate_spalte(ziel, list(df_roh.columns))
                idx = optionen.index(vorschlag) if vorschlag in optionen else 0
                wahl = st.selectbox(label, optionen, index=idx, key=f"map_{ziel}")
                mapping[ziel] = None if wahl == KEINE else wahl
            df_basis = detection.standardisiere(df_roh, mapping)
        else:
            st.caption("Noch keine Datei hochgeladen.")
    # quelle == "Keine Daten": df_basis bleibt None

    if df_basis is not None:
        st.header("Regelbasierte Erkennung")
        tage_fenster = st.slider(
            "Zeitfenster Doppelzahlung (Tage)", 0, 30, 5,
            help="Gleicher Lieferant + gleicher Betrag innerhalb dieser Tage gilt als Dublette.",
        )
        preis_schwelle = st.slider(
            "Preisabweichung vom Kategorie-Median (%)", 20, 300, 70, step=10,
            help="Ab dieser prozentualen Abweichung gilt der Stückpreis als auffällig.",
        )
        grenze = st.number_input(
            "Genehmigungsgrenze (€)", min_value=1000, max_value=100000,
            value=10000, step=1000,
            help="Bestellwert, ab dem eine Freigabe erforderlich ist.",
        )
        grenze_marge = st.slider(
            "Marge knapp unter Grenze (%)", 1, 20, 5,
            help="Wie nah unter der Grenze ein Betrag als Umgehung gilt.",
        )
        split_fenster = st.slider(
            "Splitting-Zeitfenster (Tage)", 1, 30, 7,
            help="Zeitraum, in dem aufgeteilte Bestellungen zusammengezählt werden.",
        )

        st.header("Isolation Forest")
        num_spalten = df_basis.select_dtypes("number").columns.tolist()
        standard = [s for s in detection.STANDARD_FEATURES if s in num_spalten]
        features = st.multiselect(
            "Features für das Modell", num_spalten, default=standard or num_spalten,
        )
        contamination = st.slider(
            "Erwarteter Anomalieanteil", 0.01, 0.20, 0.05, step=0.01,
            help="Geschätzter Anteil auffälliger Datensätze. Steuert die Sensitivität.",
        )

# Leerer Zustand, wenn keine Daten gewählt sind
if df_basis is None:
    st.info("Keine Datenquelle ausgewählt. Wähle in der Seitenleiste links den "
            "Beispieldatensatz oder lade eine eigene CSV, um das Dashboard zu füllen.")
    st.stop()

if not features:
    st.warning("Bitte mindestens ein numerisches Feature für den Isolation Forest "
               "auswählen (in der Seitenleiste). Bei eigenen Daten zuerst die "
               "Spalten zuordnen.")
    st.stop()

df = detection.kombinierte_analyse(
    df_basis,
    tage_fenster=tage_fenster,
    preis_schwelle_prozent=preis_schwelle,
    grenze=float(grenze),
    grenze_marge_prozent=grenze_marge,
    split_fenster=split_fenster,
    features=features,
    contamination=contamination,
)
kpi = detection.kennzahlen(df)

# Hinweis, welche Regeln mit den vorhandenen Spalten aktiv sind
_regel_bedarf = {
    "Doppelte Rechnungen": ["lieferant", "gesamtbetrag", "bestelldatum"],
    "Preisabweichung": ["stueckpreis", "kategorie"],
    "Knapp unter Grenze": ["gesamtbetrag"],
    "Auftragssplitting": ["lieferant", "kategorie", "gesamtbetrag", "bestelldatum"],
}
_inaktiv = [name for name, cols in _regel_bedarf.items()
            if not all(c in df_basis.columns for c in cols)]
if _inaktiv:
    st.info("Folgende Regeln sind mit den zugeordneten Spalten nicht aktiv: "
            + ", ".join(_inaktiv) + ". Ordne die fehlenden Felder zu, um sie zu nutzen.")

tab_dash, tab_ueber, tab_regel, tab_ml, tab_komb = st.tabs(
    ["Dashboard", "Überblick", "Regelbasiert", "Isolation Forest", "Kombiniert"]
)

# ----- Dashboard (Einkäufer-Sicht) -----------------------------------------
with tab_dash:
    st.markdown(
        '<p class="sentra-sub">Entscheidungsrelevante Übersicht für den Einkauf '
        </p>',
        unsafe_allow_html=True,
    )
    auffaellig = df[df["kombinierte_kategorie"] != "unauffaellig"].copy()
    hohe_prio = int((df["kombinierte_kategorie"] == "Regel + ML").sum())

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Zu prüfen", len(auffaellig),
              help="Anzahl der Vorgänge, die von mindestens einer Schicht als "
                   "auffällig markiert wurden und manuell geprüft werden sollten.")
    if "gesamtbetrag" in auffaellig.columns:
        wert = auffaellig["gesamtbetrag"].sum()
        d2.metric("Wert auffällig (€)", f"{wert:,.0f}".replace(",", "."),
                  help="Summe der Gesamtbeträge aller auffälligen Vorgänge – das "
                       "finanzielle Volumen, das zur Prüfung ansteht.")
    else:
        d2.metric("Wert auffällig (€)", "—",
                  help="Summe der Gesamtbeträge aller auffälligen Vorgänge. "
                       "Nicht verfügbar, da keine Betragsspalte zugeordnet ist.")
    anteil = len(auffaellig) / len(df) * 100 if len(df) else 0
    d3.metric("Anteil auffällig", f"{anteil:.1f} %",
              help="Anteil der auffälligen Vorgänge an allen Transaktionen.")
    d4.metric("Hohe Priorität", hohe_prio, help="Von Regel UND Modell erkannt.")

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Top-Lieferanten nach Auffälligkeiten")
        if len(auffaellig):
            top_lief = (auffaellig["lieferant"].value_counts().head(8)
                        .rename_axis("Lieferant").reset_index(name="Auffällige Vorgänge"))
            fig_l = px.bar(top_lief, x="Auffällige Vorgänge", y="Lieferant",
                           orientation="h")
            fig_l.update_traces(marker_color="#475569")
            fig_l.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(diagramm_stil(fig_l), use_container_width=True)
        else:
            st.caption("Keine auffälligen Vorgänge.")
    with g2:
        st.subheader("Auffälligkeiten nach Kategorie")
        if len(auffaellig):
            kat = (auffaellig["kategorie"].value_counts()
                   .rename_axis("Kategorie").reset_index(name="Anzahl"))
            fig_k = px.bar(kat, x="Kategorie", y="Anzahl")
            fig_k.update_traces(marker_color="#475569")
            st.plotly_chart(diagramm_stil(fig_k), use_container_width=True)
        else:
            st.caption("Keine auffälligen Vorgänge.")

    st.subheader("Priorisierte Prüfliste")
    if len(auffaellig):
        if "gesamtbetrag" in auffaellig.columns:
            prio = auffaellig.sort_values("gesamtbetrag", ascending=False)
        else:
            prio = auffaellig.sort_values("ml_score")
        prio = prio.copy()
        prio["Erkennungsquelle"] = prio["kombinierte_kategorie"].map(ANZEIGE)
        _spalten = vorhandene(prio, ["rechnung_id", "lieferant", "kategorie",
                                     "gesamtbetrag"]) + ["Erkennungsquelle"]
        st.dataframe(prio[_spalten].head(15), use_container_width=True)
    else:
        st.caption("Aktuell sind keine Vorgänge zu prüfen.")

# ----- Überblick -----------------------------------------------------------
with tab_ueber:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Datensätze", kpi["gesamt"])
    c2.metric("Regel-Treffer", kpi["regel_treffer"])
    c3.metric("ML-Anomalien", kpi["ml_anomalien"])
    c4.metric("Von beiden erkannt", kpi["beide"])

    st.subheader("Verteilung nach Erkennungsquelle")
    st.caption("Tipp: Auf einen Balken klicken, um die zugehörigen Transaktionen "
               "unten anzuzeigen.")
    verteilung = (
        df["kombinierte_kategorie"].map(ANZEIGE).value_counts()
        .rename_axis("Kategorie").reset_index(name="Anzahl")
    )
    fig = px.bar(verteilung, x="Kategorie", y="Anzahl", color="Kategorie",
                 color_discrete_map=SEVERITY)
    fig.update_layout(showlegend=False)
    auswahl_event = st.plotly_chart(
        diagramm_stil(fig), use_container_width=True,
        on_select="rerun", key="verteilung_chart",
    )

    # Drill-down: angeklickte Kategorie ermitteln und Transaktionen anzeigen
    gewaehlt = None
    try:
        punkte = auswahl_event["selection"]["points"]
        if punkte:
            gewaehlt = punkte[0].get("x")
    except (TypeError, KeyError, IndexError):
        gewaehlt = None

    if gewaehlt:
        intern = ANZEIGE_REVERSE.get(gewaehlt, gewaehlt)
        treffer = df[df["kombinierte_kategorie"] == intern]
        st.markdown(f"**{len(treffer)} Transaktion(en) in der Kategorie "
                    f"{gewaehlt}**")
        _sp = vorhandene(treffer, ["rechnung_id", "lieferant", "kategorie", "menge",
                                   "stueckpreis", "gesamtbetrag"])
        st.dataframe(treffer[_sp], use_container_width=True)
    else:
        st.subheader("Datenvorschau")
        st.dataframe(df_basis.head(20), use_container_width=True)

# ----- Regelbasiert --------------------------------------------------------
with tab_regel:
    st.markdown(
        '<p class="sentra-sub">Erklärbare Regeln mit klarer Begründung je Treffer – '
        'nachvollziehbar für Compliance und Revision.</p>',
        unsafe_allow_html=True,
    )
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Doppelte Rechnungen", int(df["regel_doppelte_rechnung"].sum()))
    r2.metric("Preisabweichungen", int(df["regel_preisabweichung"].sum()))
    r3.metric("Knapp unter Grenze", int(df["regel_knapp_unter_grenze"].sum()))
    r4.metric("Auftragssplitting", int(df["regel_auftragssplitting"].sum()))

    treffer = df[df["regel_treffer"]].copy()
    treffer["grund"] = treffer.apply(
        lambda r: ", ".join(
            g for g, b in [
                ("doppelte Rechnung", r["regel_doppelte_rechnung"]),
                ("Preisabweichung", r["regel_preisabweichung"]),
                ("knapp unter Grenze", r["regel_knapp_unter_grenze"]),
                ("Auftragssplitting", r["regel_auftragssplitting"]),
            ] if b
        ),
        axis=1,
    )
    st.subheader("Regel-Treffer")
    _spalten = vorhandene(treffer, ["rechnung_id", "lieferant", "kategorie",
                                    "menge", "stueckpreis", "gesamtbetrag"]) + ["grund"]
    st.dataframe(treffer[_spalten], use_container_width=True)

# ----- Isolation Forest ----------------------------------------------------
with tab_ml:
    st.markdown(
        '<p class="sentra-sub">Der Isolation Forest bewertet jede Buchung anhand der '
        'gewählten Merkmale. Ein niedriger Score bedeutet ungewöhnlich.</p>',
        unsafe_allow_html=True,
    )
    st.subheader("Verteilung der Anomalie-Scores")
    fig_score = px.histogram(
        df, x="ml_score", color="ml_anomalie", nbins=40,
        color_discrete_map=BOOL_MAP,
        labels={"ml_score": "Anomalie-Score", "ml_anomalie": "Anomalie"},
    )
    st.plotly_chart(diagramm_stil(fig_score), use_container_width=True)

    if {"stueckpreis", "menge"}.issubset(df.columns):
        st.subheader("Menge gegen Stückpreis")
        fig_scatter = px.scatter(
            df, x="menge", y="stueckpreis", color="ml_anomalie",
            color_discrete_map=BOOL_MAP,
            hover_data=vorhandene(df, ["rechnung_id", "lieferant", "kategorie", "gesamtbetrag"]),
            labels={"menge": "Menge", "stueckpreis": "Stückpreis", "ml_anomalie": "Anomalie"},
        )
        fig_scatter.update_traces(marker=dict(size=7, opacity=0.75))
        st.plotly_chart(diagramm_stil(fig_scatter), use_container_width=True)

    st.subheader("Auffälligste Buchungen")
    _spalten = vorhandene(df, ["rechnung_id", "lieferant", "kategorie", "menge",
                               "stueckpreis", "gesamtbetrag", "lieferzeit_tage"]) + ["ml_score"]
    st.dataframe(df.nsmallest(15, "ml_score")[_spalten], use_container_width=True)

# ----- Kombiniert ----------------------------------------------------------
with tab_komb:
    st.markdown(
        '<p class="sentra-sub">Gegenüberstellung beider Schichten. Die Überschneidung '
        '(Regel + ML) ist besonders belastbar; Nur ML liefert Hinweise ohne feste Regel.</p>',
        unsafe_allow_html=True,
    )
    auffaellig = df[df["kombinierte_kategorie"] != "unauffaellig"].copy()
    anzeige = auffaellig.copy()
    anzeige["kombinierte_kategorie"] = anzeige["kombinierte_kategorie"].map(ANZEIGE)
    st.subheader("Auffällige Buchungen")
    _spalten = vorhandene(anzeige, ["rechnung_id", "lieferant", "kategorie", "menge",
                                    "stueckpreis", "gesamtbetrag"]) + \
        ["regel_treffer", "ml_anomalie", "kombinierte_kategorie"]
    st.dataframe(
        anzeige[_spalten].sort_values("kombinierte_kategorie"),
        use_container_width=True,
    )

    csv = auffaellig.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Auffällige Buchungen als CSV exportieren",
        data=csv, file_name="anomalien_export.csv", mime="text/csv",
    )

    if "anomalie_typ" in df.columns:
        with st.expander("Plausibilitätscheck gegen Ground Truth (nur Beispieldaten)"):
            st.dataframe(
                pd.crosstab(df["anomalie_typ"], df["kombinierte_kategorie"].map(ANZEIGE)),
                use_container_width=True,
            )
