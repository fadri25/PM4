"""
Erstellt am: 27. April 2025
Autor: manuw
streamlit run dashboard.py

"""
import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import json
import pydeck as pdk
import numpy as np
import time
from datetime import datetime, timedelta


# --- Demo-Login ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login erforderlich")
    username = st.text_input("Benutzername")
    password = st.text_input("Passwort", type="password")

    if st.button("Einloggen"):
        if username == "0550":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("❌ Ungültiger Benutzername. Zugriff verweigert.")

    st.stop()



# Session State initialisieren
if "page" not in st.session_state:
    st.session_state.page = "Aktueller Stand"

# Seitenkonfiguration
st.set_page_config(
    page_title="Kreditkartenbetrug Überwachung",
    page_icon="💳",
    layout="wide"
)

# Navigation Sidebar
page_choice = st.sidebar.radio("Gehe zu:", [
    "Aktueller Stand",
    "Übersicht Transaktionen",
    "Detaillierte Transaktion",
    "Second Level Support",
    "System",
    "Feedback Loop",
    "Retraining",
    "Support"
], index=[
    "Aktueller Stand",
    "Übersicht Transaktionen",
    "Detaillierte Transaktion",
    "Second Level Support",
    "System",
    "Feedback Loop",
    "Retraining",
    "Support"
].index(st.session_state.page))

if page_choice != st.session_state.page:
    st.session_state.page = page_choice

page = st.session_state.page

# CSV-Datei laden
@st.cache_data(ttl=60)
def load_data():
    df = pd.read_csv("C:/PM4/processed-data/transactions_first_50_kürzer.csv", sep=";", low_memory=False)
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
    return df

df = load_data()

# === Seiteninhalt ===

if page == "Aktueller Stand":
    st_autorefresh(interval=11 * 1000, key="auto_refresh")
    st.title("💳 Aktueller Stand - Live Dashboard")

    # === Daten vorbereiten ===
    df['TX_DATE'] = pd.to_datetime(df['TX_DATETIME'], errors='coerce').dt.date
    verfügbare_tage = sorted(df["TX_DATE"].dropna().unique(), reverse=True)
    default_tag = df["TX_DATE"].dropna().max()

    ausgewählter_tag = st.selectbox(
        "📆 Tag auswählen zur Analyse:",
        options=verfügbare_tage,
        index=verfügbare_tage.index(default_tag)
    )

    df_heute = df[df["TX_DATE"] == ausgewählter_tag]
    df_gestern = df[df["TX_DATE"] == (ausgewählter_tag - timedelta(days=1))]

    # === KPIs berechnen ===
    anzahl_transaktionen_heute = len(df_heute)
    anzahl_betrug_heute = df_heute["TX_FRAUD"].sum()
    betrag_durchschnitt_heute = df_heute["TX_AMOUNT"].mean()
    kundenanzahl_heute = df_heute["CUSTOMER_ID"].nunique()
    tx_pro_kunde = anzahl_transaktionen_heute / kundenanzahl_heute if kundenanzahl_heute else 0

    # === Vergleich zu gestern ===
    betrug_ges = df_gestern["TX_FRAUD"].sum()
    delta_betrug = anzahl_betrug_heute - betrug_ges
    delta_betrug_anzeige = f"{delta_betrug:+}"
    farbe = "normal"
    if delta_betrug > 0:
        farbe = "inverse"
    elif delta_betrug < 0:
        farbe = "off"

    # === Anzeige KPIs ===
    st.markdown("## 📊 Überblick – Auswahl: " + str(ausgewählter_tag))
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📅 Transaktionen", anzahl_transaktionen_heute)
    col2.metric("🚨 Betrugsfälle", anzahl_betrug_heute, delta_betrug_anzeige, delta_color=farbe)
    col3.metric("💵 Ø Betrag ($)", f"{betrag_durchschnitt_heute:.2f}")
    col4.metric("👤 Kunden", kundenanzahl_heute)
    col5.metric("🔁 Ø Tx pro Kunde", f"{tx_pro_kunde:.2f}")

    st.markdown("---")

    # === Letzte 5 verdächtige Transaktionen mit Statusanzeige ===
    st.subheader("🕵️‍♂️ Letzte 5 zu überprüfende Betrugsfälle")

    letzte_betrugsfaelle = df_heute[df_heute["TX_FRAUD"] == 1].sort_values(by="TX_DATETIME", ascending=False).head(5).copy()
    status_liste = [
        "🟥 Offen",
        "🟥 Offen",
        "🟧 In Bearbeitung (0452)",
        "🟧 In Bearbeitung (0572)",
        "✅ Abgeschlossen"
    ]
    letzte_betrugsfaelle["Status"] = status_liste[:len(letzte_betrugsfaelle)]
    letzte_betrugsfaelle["Zeit"] = letzte_betrugsfaelle["TX_DATETIME"].dt.strftime("%H:%M")
    anzeige_df = letzte_betrugsfaelle[[
        "TRANSACTION_ID", "Zeit", "TX_AMOUNT", "CUSTOMER_ID", "TERMINAL_ID", "Status"
    ]].rename(columns={
        "TRANSACTION_ID": "Transaktion",
        "TX_AMOUNT": "Betrag ($)",
        "CUSTOMER_ID": "Kunde",
        "TERMINAL_ID": "Terminal"
    })
    st.dataframe(anzeige_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    # === Transaktionen nach Stunden ===
    st.subheader("⏰ Verteilung nach Stunde")
    df_heute["Hour"] = pd.to_datetime(df_heute["TX_DATETIME"]).dt.hour
    fig = px.histogram(df_heute, x='Hour', title="Transaktionen nach Stunde", nbins=24)
    st.plotly_chart(fig, use_container_width=True)

    # === Top 5 Risikoterminals ===
    st.subheader("🏧 Top 5 Risikoterminals (nach Betrugsfällen)")
    top_terminals = df_heute[df_heute['TX_FRAUD'] == 1]['TERMINAL_ID'].value_counts().head(5)
    st.bar_chart(top_terminals)

    # === Karte mit Live-Koordinaten (optional) ===
    try:
        with open('live_coords.txt', 'r') as f:
            coords = json.load(f)
        df_coords = pd.DataFrame(coords)

        layer = pdk.Layer(
            'ScatterplotLayer',
            data=df_coords,
            get_position='[longitude, latitude]',
            get_color='[255, 0, 0, 160]',
            get_radius=50000,
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=df_coords['latitude'].mean(),
            longitude=df_coords['longitude'].mean(),
            zoom=4,
            pitch=0,
        )

        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style='mapbox://styles/mapbox/light-v9',
            tooltip={"text": "Lat: {latitude}\nLon: {longitude}"}
        )

        st.subheader("🌍 Karte mit verdächtigen Aktivitäten")
        st.pydeck_chart(r)

    except Exception as e:
        st.info("ℹ️ Keine aktuellen Koordinaten verfügbar.")

elif page == "Übersicht Transaktionen":
    st.title("📄 Übersicht Transaktionen")

    amount_threshold = st.sidebar.slider(
        "Mindestbetrag ($)",
        min_value=int(df['TX_AMOUNT'].min()),
        max_value=int(df['TX_AMOUNT'].max()),
        value=100
    )

    columns_to_show = ['TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_FRAUD']
    filtered_df = df[df['TX_AMOUNT'] >= amount_threshold]
    st.dataframe(filtered_df[columns_to_show])

elif page == "Detaillierte Transaktion":
    st.title("🔍 Detaillierte Transaktion")

    transaction_id_input = st.text_input("Transaktions-ID eingeben:")

    if transaction_id_input:
        try:
            transaction_id = int(transaction_id_input)
            result = df[df['TRANSACTION_ID'] == transaction_id]

            if not result.empty:
                st.success(f"Transaktion {transaction_id} gefunden!")

                customer_id = result['CUSTOMER_ID'].values[0]
                terminal_id = result['TERMINAL_ID'].values[0]
                amount = result['TX_AMOUNT'].values[0]
                is_fraud = result['TX_FRAUD'].values[0]

                st.subheader("🔎 Basisinformationen")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Transaktions-ID", f"{transaction_id}")
                col2.metric("Kunde", f"{customer_id}")
                col3.metric("Betrag ($)", f"{amount:.2f}")
                col4.metric("Betrugsfall?", "JA" if is_fraud == 1 else "NEIN")

                st.markdown("---")

                # Kunden- und Terminalhistorie
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 📈 Kundenhistorie")
                    st.dataframe(pd.DataFrame({
                        "Zeitfenster": ["1 Tag", "7 Tage", "30 Tage"],
                        "Anzahl Transaktionen": [
                            result["CUSTOMER_ID_NB_TX_1DAY_WINDOW"].values[0],
                            result["CUSTOMER_ID_NB_TX_7DAY_WINDOW"].values[0],
                            result["CUSTOMER_ID_NB_TX_30DAY_WINDOW"].values[0],
                        ],
                        "Ø Betrag ($)": [
                            result["CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW"].values[0],
                            result["CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW"].values[0],
                            result["CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW"].values[0],
                        ]
                    }), use_container_width=True)

                with col2:
                    st.markdown("### 🏧 Terminalhistorie")
                    st.dataframe(pd.DataFrame({
                        "Zeitfenster": ["1 Tag", "7 Tage", "30 Tage"],
                        "Anzahl Transaktionen": [
                            result["TERMINAL_ID_NB_TX_1DAY_WINDOW"].values[0],
                            result["TERMINAL_ID_NB_TX_7DAY_WINDOW"].values[0],
                            result["TERMINAL_ID_NB_TX_30DAY_WINDOW"].values[0],
                        ],
                        "Risiko-Level": [
                            result["TERMINAL_ID_RISK_1DAY_WINDOW"].values[0],
                            result["TERMINAL_ID_RISK_7DAY_WINDOW"].values[0],
                            result["TERMINAL_ID_RISK_30DAY_WINDOW"].values[0],
                        ]
                    }), use_container_width=True)

                st.markdown("---")

                # === Betrugs-Szenario anzeigen ===
                scenario_id = int(result['TX_FRAUD_SCENARIO'].values[0])
                szenario_text = {
                    0: "Unregelmäßigkeit im Transaktionsmuster",
                    1: "Auffällige Häufigkeit an einem Terminal",
                    2: "Unerwartet hoher Betrag",
                    3: "Geografisch verdächtige Aktivität",
                    4: "Verdacht auf Mehrfachversuche"
                }.get(scenario_id, "Unbekanntes Szenario")

                st.markdown("### 📌 Erkanntes Betrugsszenario")
                st.info(f"**Szenario {scenario_id}** – {szenario_text}")
                with st.expander("🧭 Legende aller Betrugsszenarien anzeigen"):
                    st.markdown("""
                    | Szenario-ID | Beschreibung                                      |
                    |-------------|---------------------------------------------------|
                    | 0           | Unregelmäßigkeit im Transaktionsmuster           |
                    | 1           | Auffällige Häufigkeit an einem Terminal          |
                    | 2           | Unerwartet hoher Betrag                          |
                    | 3           | Geografisch verdächtige Aktivität                |
                    | 4           | Verdacht auf Mehrfachversuche                   |
                    """)
                
                
                # Vergleich mit 30-Tage-Kundendurchschnitt
                ø_30 = result['CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW'].values[0]
                abweichung = ((amount - ø_30) / ø_30) * 100 if ø_30 else 0

                st.markdown("### 📊 Vergleich mit Ø-Kundenbetrag (30 Tage)")
                colA, colB = st.columns(2)
                colA.metric("Aktueller Betrag", f"${amount:.2f}")
                colB.metric("Ø Betrag (30 Tage)", f"${ø_30:.2f}", f"{abweichung:+.1f} %")

                # Verlauf Betrag (Dummy-Daten)
                st.subheader("📉 Verlauf der letzten 10 Kunden-Transaktionen")
                kunden_betraege = pd.DataFrame({
                    "Datum": pd.date_range(end=pd.Timestamp.today(), periods=10),
                    "Betrag ($)": (amount * (1 + 0.1 * (np.random.randn(10)))).round(2)
                })
                fig = px.line(kunden_betraege, x="Datum", y="Betrag ($)", title="Kundenbetrag Verlauf")
                st.plotly_chart(fig, use_container_width=True)

                # Heatmap Kundenverhalten
                st.markdown("### 🕒 Kundenverhalten: Wochentag vs. Uhrzeit")
                kundendaten = df[df["CUSTOMER_ID"] == customer_id].copy()
                kundendaten["Wochentag"] = kundendaten["TX_DATETIME"].dt.day_name()
                kundendaten["Stunde"] = kundendaten["TX_DATETIME"].dt.hour
                heatmap_data = kundendaten.groupby(["Wochentag", "Stunde"]).size().unstack(fill_value=0)
                heatmap_data = heatmap_data.reindex([
                    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
                ])
                fig_heatmap = px.imshow(
                    heatmap_data,
                    labels=dict(x="Stunde", y="Wochentag", color="Anzahl Transaktionen"),
                    aspect="auto",
                    title="Transaktionshäufigkeit des Kunden"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

                # Karte
                st.subheader("🌍 Bewegungsmuster der Transaktion")
                reise_df = pd.DataFrame({
                    "latitude": [48.8566, 50.1109, 25.276987],
                    "longitude": [2.3522, 8.6821, 55.296249],
                    "color": [[0, 128, 255], [0, 128, 255], [255, 0, 0]]
                })
                line_layer = pdk.Layer(
                    "LineLayer",
                    data=pd.DataFrame({
                        "source_lat": [48.8566, 50.1109],
                        "source_lon": [2.3522, 8.6821],
                        "target_lat": [50.1109, 25.276987],
                        "target_lon": [8.6821, 55.296249],
                    }),
                    get_source_position='[source_lon, source_lat]',
                    get_target_position='[target_lon, target_lat]',
                    get_width=3,
                    get_color=[0, 0, 255],
                    pickable=True
                )
                scatter_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=reise_df,
                    get_position='[longitude, latitude]',
                    get_fill_color='color',
                    get_radius=80000,
                )
                view_state = pdk.ViewState(latitude=40, longitude=20, zoom=1.5, pitch=0)
                deck = pdk.Deck(layers=[line_layer, scatter_layer], initial_view_state=view_state,
                                tooltip={"text": "Location: {latitude}, {longitude}"})
                st.pydeck_chart(deck)

                # Aktionen
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🔧 Aktionen")
                    weiter_support = st.checkbox("Weiter an Second Support")
                    kunden_anrufen = st.checkbox("Kunden Anrufen")
                    kunden_hat_fragen = st.checkbox("Kunde hat Fragen")
                with col2:
                    st.markdown("### 📝 Kommentar zum Fall")
                    kommentar = st.text_area("Kommentar eingeben:", placeholder="Hier Bemerkungen einfügen...")

                with st.container():
                    st.markdown(
                        """
                        <div style="background-color: #f0f0f0; padding: 20px; border: 3px solid red; border-radius: 10px;">
                        <h3 style="color:black;">🚨 Weitere Maßnahmen</h3>                        
                        """,
                        unsafe_allow_html=True,
                    )
                    kunde_sperren = st.checkbox("Kunde sperren")
                    nur_online = st.checkbox("Nur Online-Zahlungen zulassen")
                    nur_ausland = st.checkbox("Nur ausländische Zahlungen erlauben")
                    betrag_ab = st.number_input("Beträge ab (CHF)", min_value=0.00, step=10.00, format="%.2f")
                    whitelist_only = st.checkbox("Nur vertrauenswürdige Händler (Whitelisted)")
                    if st.button("💾 Maßnahmen übernehmen"):
                        st.success("✅ Maßnahmen erfolgreich gespeichert!")
                    st.markdown("</div>", unsafe_allow_html=True)

            else:
                st.warning("Keine Transaktion mit dieser ID gefunden.")

        except ValueError:
            st.error("Bitte eine gültige numerische Transaktions-ID eingeben.")

elif page == "Second Level Support":
    st.title("🔎 Second Level Support – Detailanalyse")

    # --- Gefilterte Fälle anzeigen ---
    st.markdown("### 📋 Weitergeleitete Betrugsfälle")
    # Falls die Spalte nicht existiert → Demo-Simulation: nimm z. B. die letzten 10 Fraud-Fälle
    if "WEITERGELEITET_AN_SUPPORT" in df.columns:
        weitergeleitet_df = df[df["WEITERGELEITET_AN_SUPPORT"] == 1].copy()
    else:
        st.info("")
        weitergeleitet_df = df[df["TX_FRAUD"] == 1].sort_values("TX_DATETIME", ascending=False).head(10).copy()

    if weitergeleitet_df.empty:
        st.info("Keine weitergeleiteten Fälle vorhanden.")
        st.stop()

    weitergeleitet_df = weitergeleitet_df.sort_values("TX_DATETIME", ascending=False).head(10)
    selected_id = st.selectbox("Transaktions-ID auswählen", weitergeleitet_df["TRANSACTION_ID"])

    fall = weitergeleitet_df[weitergeleitet_df["TRANSACTION_ID"] == selected_id].iloc[0]

    st.markdown("### 🧾 Fallinformationen")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transaktion", selected_id)
    col2.metric("Kunde", fall["CUSTOMER_ID"])
    col3.metric("Betrag ($)", f"{fall['TX_AMOUNT']:.2f}")
    col4.metric("Szenario", int(fall["TX_FRAUD_SCENARIO"]))

    # Vergleich mit Kunden-Ø
    ø_30 = fall["CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW"]
    abw = ((fall["TX_AMOUNT"] - ø_30) / ø_30) * 100 if ø_30 else 0
    st.markdown("#### 📊 Vergleich zum 30-Tage-Ø-Betrag des Kunden")
    colA, colB = st.columns(2)
    colA.metric("Aktueller Betrag", f"${fall['TX_AMOUNT']:.2f}")
    colB.metric("Ø Betrag (30 Tage)", f"${ø_30:.2f}", f"{abw:+.1f} %")

    # Modellscore (optional)
    if "MODEL_SCORE" in fall:
        st.markdown("#### 🤖 Modellbewertung")
        st.progress(fall["MODEL_SCORE"])

    st.markdown("### 📝 Fallbewertung")

    # Statuswahl
    status = st.selectbox("Bearbeitungsstatus wählen", [
        "In Bearbeitung", "Bestätigter Betrug", "Kein Betrug", "Weiter an Analyst"
    ])

    kommentar = st.text_area("Kommentar zum Fall", placeholder="Was wurde festgestellt?")

    if st.button("💾 Fall aktualisieren"):
        # Hier würdest du den Status + Kommentar speichern (z. B. in DB oder CSV)
        st.success("✅ Fallstatus gespeichert!")

    st.markdown("---")

    # Feedback an ML-Team
    st.markdown("### 📡 Rückmeldung ans Modell-Team")

    if st.button("🚨 Diesen Fall als Modellfehler melden"):
        # Feedbacksystem/Logging
        st.warning("Danke! Fall wurde zur Modellüberprüfung markiert.")


elif page == "System":
    st.title("🖥️ Systemübersicht")

    # === 1. Modellinformationen ===
    st.markdown("### 🧠 Modellinformationen")
    st.write("""
    - Modelltyp: **XGBoost Classifier**
    - Modellversion: **v2.1**
    - Trainiert am: **2025-04-25**
    - Speicherort: `/models/fraud_model_v2.1.pkl`
    - Features genutzt: 22
    """)

    st.divider()

    # === 2. Datenquelle ===
    st.markdown("### 🗂 Datenquellen & Zeitraum")
    st.write("""
    - Quelle: **Live-Streaming + historische CSV-Dateien**
    - Aktiver Datenzeitraum: **2023-01-01** bis **2025-04-30**
    - Zeitzone: **Europe/Zurich (UTC+1)**
    """)

    st.divider()

    # === 3. Rollen & Benutzer ===
    st.markdown("### 👥 Rollen & aktive Benutzer")
    rollen_df = pd.DataFrame({
        "Rolle": ["Admin", "Analyst", "Support", "Read Only"],
        "Anzahl aktive Nutzer": [3, 5, 4, 2]
    })

    st.dataframe(rollen_df, hide_index=True, use_container_width=True)

    st.markdown("""
    - **Admins**: Vollzugriff (Modelle, Training, Nutzerverwaltung)  
    - **Analysten**: Zugriff auf Feedback & Performance  
    - **Support**: Zugriff auf Transaktionsdetails & Maßnahmen  
    - **Read Only**: Einsicht ohne Schreibrechte
    """)

    st.divider()

    # === 4. Logs und Wartung ===
    st.markdown("### 🛠 System-Logs & Wartung")
    st.write("""
    - Letzter Neustart: **2025-04-28, 03:12 Uhr**
    - Logs: `./logs/app.log`
    - Geplante Wartung: **2025-05-10, 00:00–04:00**
    - Monitoring: [interner Statuslink](http://localhost:8501/status) *(nur intern verfügbar)*
    """)

    st.divider()

    # === 5. (Platzhalter) Systemmetriken live (optional erweiterbar) ===
    st.markdown("### 📈 Systemmetriken (Demo)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Vorhersagedauer", "18 ms")
    col2.metric("Modellgröße", "5.3 MB")
    col3.metric("Live-API-Status", "✅ Online")


elif page == "Feedback Loop":
    st.title("🔁 Feedback Loop Monitoring")

    st_autorefresh(interval=10_000, key="refresh_feedback")

    # === Metriken laden ===
    precision_df = pd.read_csv("precision_history.csv")
    recall_df = pd.read_csv("recall_history.csv")
    roc_auc_df = pd.read_csv("roc_auc_history.csv")
    f1_score_df = pd.read_csv("f1_score_history.csv")

    latest_precision = precision_df["Wert"].iloc[-1]
    latest_recall = recall_df["Wert"].iloc[-1]
    latest_roc_auc = roc_auc_df["Wert"].iloc[-1]
    latest_f1_score = f1_score_df["Wert"].iloc[-1]

    # === 1. Modell-Kontext anzeigen ===
    st.markdown("### 🧠 Modellinformationen")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📅 Letztes Training", "2025-04-25")  # Beispielwert
    col2.metric("🧠 Modell", "XGBoost v2.1")
    col3.metric("📁 Datensätze", "50'000")
    col4.metric("📊 Features", "22")

    st.markdown("---")

    # === 2. KPI Übersicht mit Alarm ===
    st.markdown("### 📋 Aktuelle Modellmetriken")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Precision", f"{latest_precision:.3f}")
    col2.metric("Recall", f"{latest_recall:.3f}")
    col3.metric("ROC AUC", f"{latest_roc_auc:.3f}")
    col4.metric("F1-Score", f"{latest_f1_score:.3f}")

    # Alarmanzeige bei schlechter Performance
    if latest_f1_score < 0.75:
        st.error("⚠️ F1-Score unter 0.75 – Modellperformance kritisch!")
    elif latest_f1_score < 0.85:
        st.warning("⚠️ F1-Score mäßig – Performance beobachten.")

    st.markdown("---")

    # === 3. Metrikverläufe ===
    st.markdown("### 📈 Verlauf der Modellmetriken")
    col5, col6 = st.columns(2)

    with col5:
        st.subheader(f"Precision Verlauf")
        st.line_chart(precision_df.set_index("Zeit"))
        st.subheader(f"ROC AUC Verlauf")
        st.line_chart(roc_auc_df.set_index("Zeit"))

    with col6:
        st.subheader(f"Recall Verlauf")
        st.line_chart(recall_df.set_index("Zeit"))
        st.subheader(f"F1-Score Verlauf")
        st.line_chart(f1_score_df.set_index("Zeit"))

    st.markdown("---")

    # === 4. Konfusionsmatrix (Demo) ===
    st.markdown("### 🧮 Konfusionsmatrix")
    conf_matrix = pd.DataFrame({
        "Pred: Fraud": [15, 3],
        "Pred: Normal": [5, 100]
    }, index=["True: Fraud", "True: Normal"])

    fig_matrix = px.imshow(conf_matrix, text_auto=True, color_continuous_scale='Blues',
                           title="Konfusionsmatrix", aspect="auto")
    st.plotly_chart(fig_matrix, use_container_width=True)

    # === 5. Button für Retraining-Seite ===
    st.markdown("---")
    if st.button("🚀 Zum Retraining gehen"):
        st.session_state.page = "Retraining"
        st.rerun()

elif page == "Retraining":
    st.title("♻️ Manuelles Retraining")

    if "retraining_started" not in st.session_state:
        st.session_state.retraining_started = False
    if "retraining_start_time" not in st.session_state:
        st.session_state.retraining_start_time = None
    if "retraining_total_seconds" not in st.session_state:
        st.session_state.retraining_total_seconds = 10  # Wirkliche Dauer (10s)
    if "retraining_simulated_hours" not in st.session_state:
        st.session_state.retraining_simulated_hours = 3  # Simulierte Dauer (3h)

    # Button zum Starten
    if not st.session_state.retraining_started:
        if st.button("🚀 Manuelles Retraining starten"):
            st.session_state.retraining_started = True
            st.session_state.retraining_start_time = datetime.now()

    # Wenn Retraining läuft
    if st.session_state.retraining_started:
        elapsed = (datetime.now() - st.session_state.retraining_start_time).total_seconds()
        progress = min(elapsed / st.session_state.retraining_total_seconds, 1.0)

        st.info("Retraining läuft... Bitte warten.")
        progress_bar = st.progress(int(progress * 100))
        status_text = st.empty()

        simulated_total_seconds = st.session_state.retraining_simulated_hours * 3600
        remaining_seconds = int(simulated_total_seconds * (1 - progress))

        hours = remaining_seconds // 3600
        minutes = (remaining_seconds % 3600) // 60
        seconds = (remaining_seconds % 60)

        status_text.markdown(f"⏳ Fortschritt: {int(progress*100)}% - Verbleibende Zeit: {hours}h {minutes}m {seconds}s")

        if progress >= 1.0:
            st.success("✅ Retraining abgeschlossen!")
            st.balloons()
            st.session_state.retraining_started = False
    else:
        st.info("Drücke den Start-Button, um das Retraining zu beginnen.")

    # Automatisch aktualisieren während Retraining läuft
    if st.session_state.retraining_started:
        st_autorefresh(interval=500, key="retraining_progress_refresh")


elif page == "Support":
    st.title("🛠️ Support & Hilfezentrum")

    # === 1. Support-Infos ===
    st.markdown("### 📞 Kontaktmöglichkeiten")
    st.markdown("""
    Bei Problemen oder Fragen stehen wir dir gerne zur Verfügung:

    - 📧 E-Mail: [support@fkm.com](mailto:support@fkm.com)  
    - 📞 Telefon: +41 58 934 45 50  
    - 💬 Live-Chat: *Demnächst verfügbar*  
    - 🕒 Supportzeiten: 24/7 
    """)

    st.divider()

    # === 2. FAQ ===
    st.markdown("### 📘 Häufig gestellte Fragen (FAQ)")

    with st.expander("💡 Was mache ich, wenn eine Transaktion fälschlich als Betrug markiert wurde?"):
        st.write("""
        Gehe auf die Detailseite der Transaktion und markiere sie manuell als „Kein Betrug“.
        Optional kannst du dies im Kommentarbereich begründen. 
        """)

    with st.expander("🤖 Wie funktioniert das Betrugserkennungsmodell?"):
        st.write("""
        Das System basiert auf einem Machine-Learning-Modell (aktuell XGBoost v2.1), das auf mehreren hunderttausend historischen Transaktionen trainiert wurde. Es bewertet Muster in Betrag, Zeit, Ort und Kundenverhalten.
        """)

    with st.expander("🔁 Wann wird das Modell neu trainiert?"):
        st.write("""
        In der Regel alle 1–2 Wochen, abhängig vom Feedback-Loop. Du kannst auch manuell ein Retraining starten.
        """)

    with st.expander("🔐 Wer hat Zugriff auf welche Funktionen?"):
        st.write("""
        - **Support**: Zugriff auf alle Transaktionen & Detailanalyse  
        - **Analysten**: Zugriff auf Feedbackdaten & Performanceübersicht  
        - **Admins**: Zugriff auf System, Training und Rollenvergabe
        """)

    st.divider()

    # === 3. Feedback-Formular ===
    st.markdown("### 📝 Feedback oder Fehler melden")

    with st.form("feedback_form"):
        trans_id = st.text_input("⚙️ Betroffene Transaktions-ID (optional)")
        kategorie = st.selectbox("Kategorie", ["Fehlalarm", "Nicht erkannt", "Allgemein", "Lob", "Vorschlag"])
        nachricht = st.text_area("Nachricht", placeholder="Beschreibe dein Anliegen so genau wie möglich...")
        datei = st.file_uploader("Optionaler Anhang (Screenshot, Logdatei etc.)", type=["png", "jpg", "pdf", "txt"])
        abgeschickt = st.form_submit_button("📤 Absenden")

        if abgeschickt:
            st.success("✅ Vielen Dank! Dein Feedback wurde übermittelt.")

    st.divider()
    # === 4. Videotutorial === 
    st.divider()
    st.markdown("### 🎓 Video-Tutorial")
    
    # Variante 1: YouTube-Link (ersetze mit deinem Link)
    st.video("https://www.youtube.com/watch?v=mqOxQ3DUJPE")
    

    # === 5. Systemstatus ===
    st.markdown("### 📡 Systemstatus")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("API-Verbindung", "✅ Aktiv")
        st.metric("Modellstatus", "🟢 Einsatzbereit")
    with col2:
        st.metric("Letztes Training", "2025-04-25")
        st.metric("Live-Daten", "✅ Eingehend")
