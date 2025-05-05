# -*- coding: utf-8 -*-
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

    total_transactions = len(df)
    total_frauds = df['TX_FRAUD'].sum()
    fraud_rate = (total_frauds / total_transactions) * 100
    avg_amount = df['TX_AMOUNT'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Transaktionen", f"{total_transactions}")
    col2.metric("Betrugsrate (%)", f"{fraud_rate:.2f}")
    col3.metric("Ø Betrag ($)", f"{avg_amount:.2f}")

    df['Hour'] = df['TX_DATETIME'].dt.hour
    fig = px.histogram(df, x='Hour', title="Transaktionen nach Stunde", nbins=24)
    st.plotly_chart(fig, use_container_width=True)

    top_terminals = df[df['TX_FRAUD'] == 1]['TERMINAL_ID'].value_counts().head(5)
    st.subheader("Top 5 Risikoterminals")
    st.bar_chart(top_terminals)

    # Map
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

    st.pydeck_chart(r)

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

    # Eingabefeld für Transaktions-ID
    transaction_id_input = st.text_input("Transaktions-ID eingeben:")

    if transaction_id_input:
        try:
            transaction_id = int(transaction_id_input)
            result = df[df['TRANSACTION_ID'] == transaction_id]

            if not result.empty:
                st.success(f"Transaktion {transaction_id} gefunden!")

                # Basisdaten extrahieren
                customer_id = result['CUSTOMER_ID'].values[0]
                terminal_id = result['TERMINAL_ID'].values[0]
                amount = result['TX_AMOUNT'].values[0]
                is_fraud = result['TX_FRAUD'].values[0]

                # Oberer Bereich: KPIs
                st.subheader("🔎 Basisinformationen")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Transaktions-ID", f"{transaction_id}")
                col2.metric("Kunde", f"{customer_id}")
                col3.metric("Betrag ($)", f"{amount:.2f}")
                col4.metric("Betrugsfall?", "JA" if is_fraud == 1 else "NEIN")

                st.markdown("---")

               # --- Zwei Spalten erzeugen für Historien ---
                col1, col2 = st.columns(2)
                
                # Linke Spalte: Kundenhistorie
                with col1:
                    st.markdown("<h3>📈 Kundenhistorie</h3>", unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame({
                        "Zeitfenster": ["1 Tag", "7 Tage", "30 Tage"],
                        "Anzahl Transaktionen": [1, 1, 1],
                        "Ø Betrag ($)": [37.94, 37.94, 37.94]
                    }))
                
                # Rechte Spalte: Terminalhistorie
                with col2:
                    st.markdown("<h3>🏧 Terminalhistorie</h3>", unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame({
                        "Zeitfenster": ["1 Tag", "7 Tage", "30 Tage"],
                        "Anzahl Transaktionen": [0, 0, 0],
                        "Risiko-Level": [0, 0, 0]
                    }))


                st.markdown("---")

                # Unterer Bereich: Diagramm zu Kundenverlauf
                st.subheader("📊 Verlauf der Beträge des Kunden")
                # (Optional: nur wenn du historische Kundendaten hast, sonst Dummy-Daten generieren)
                # Hier zum Beispiel Dummy-Daten:
                kunden_betraege = pd.DataFrame({
                    "Datum": pd.date_range(end=pd.Timestamp.today(), periods=10),
                    "Betrag ($)": (amount * (1 + 0.1 * (np.random.randn(10)))).round(2)
                })

                fig = px.line(kunden_betraege, x="Datum", y="Betrag ($)", title="Verlauf der letzten 10 Transaktionen")
                st.plotly_chart(fig, use_container_width=True)
                
                #Karte
                st.subheader("🌍 Bewegungsmuster der Transaktion")

                # Drei fiktive Koordinaten (kannst du natürlich später dynamisch aus echten Daten erzeugen)
                reise_df = pd.DataFrame({
                    "latitude": [48.8566, 50.1109, 25.276987],     # Paris → Frankfurt → Dubai
                    "longitude": [2.3522, 8.6821, 55.296249],
                    "color": [[0, 128, 255], [0, 128, 255], [255, 0, 0]]  # Letzter Punkt rot (verdächtig)
                })
                
                # Linienlayer (verbindet die Punkte)
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
                
                # Punkte Layer
                scatter_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=reise_df,
                    get_position='[longitude, latitude]',
                    get_fill_color='color',
                    get_radius=80000,
                )
                
                # View auf alle Punkte zoomen
                view_state = pdk.ViewState(
                    latitude=40, 
                    longitude=20, 
                    zoom=1.5,
                    pitch=0
                )
                
                # Deck zusammenbauen
                deck = pdk.Deck(
                    layers=[line_layer, scatter_layer],
                    initial_view_state=view_state,
                    tooltip={"text": "Location: {latitude}, {longitude}"}
                )
                
                # Karte anzeigen
                st.pydeck_chart(deck)
                
                left_column, right_column = st.columns([1, 2])
                col1, col2 = st.columns(2)
                
                # Linke Spalte: Aktionen
                with col1:
                    st.markdown("<h3>🔧 Aktionen</h3>", unsafe_allow_html=True)
                    weiter_support = st.checkbox("Weiter an Second Support")
                    kunden_anrufen = st.checkbox("Kunden Anrufen")
                    kunden_hat_fragen = st.checkbox("Kunde hat Fragen")
                
                with col2:
                    st.markdown("<h3>📝 Kommentar zum Fall</h3>", unsafe_allow_html=True)
                    kommentar = st.text_area("Kommentar eingeben:", placeholder="Hier Bemerkungen einfügen...")
               
                # Hintergrundbox mit style
                with st.container():
                    st.markdown(
                        """
                        <div style="background-color: #f0f0f0; padding: 20px; border: 3px solid red; border-radius: 10px;">
                        <h3 style="color:black;">🚨 Weitere Maßnahmen</h3>                        
                        """,
                        unsafe_allow_html=True,
                        
                    )
                    
                    # Innerhalb des Containers → die echten Streamlit-Elemente:
                    kunde_sperren = st.checkbox("Kunde sperren")
                    nur_online = st.checkbox("Nur Online-Zahlungen zulassen")
                    nur_ausland = st.checkbox("Nur ausländische Zahlungen erlauben")
                    betrag_ab = st.number_input("Beträge ab (CHF)", min_value=0.00, step=10.00, format="%.2f")
                    whitelist_only = st.checkbox("Nur vertrauenswürdige Händler (Whitelisted)")
                    
                    if st.button("💾 Maßnahmen übernehmen"):
                        st.success("✅ Maßnahmen erfolgreich gespeichert!")
                    # Abschluss des Containers
                    st.markdown("</div>", unsafe_allow_html=True)
             
                with left_column:
                    pass
                with right_column:
                    pass
 
            else:
                st.warning("Keine Transaktion mit dieser ID gefunden.")

        except ValueError:
            st.error("Bitte eine gültige numerische Transaktions-ID eingeben.")

elif page == "Second Level Support":
    st.title("🔎 Second Level Support – Detailanalyse")
    # Dein Second Level Code (wie du schon geschrieben hast)

elif page == "System":
    st.title("🖥️ System")
    st.write("""
    - **Datenquelle**: Live Transaktionsdaten
    - **Datenaktualisierung**: Alle 10 Sekunden
    - **Betrugserkennung**: Basierend auf Regeln und Machine Learning Modellen
    - **Serverstatus**: ✅ Online
    """)

elif page == "Feedback Loop":
    st.title("🔁 Feedback Loop Monitoring")

    st_autorefresh(interval=10_000, key="refresh_feedback")

    precision_df = pd.read_csv("precision_history.csv")
    recall_df = pd.read_csv("recall_history.csv")
    roc_auc_df = pd.read_csv("roc_auc_history.csv")
    f1_score_df = pd.read_csv("f1_score_history.csv")

    latest_precision = precision_df["Wert"].iloc[-1]
    latest_recall = recall_df["Wert"].iloc[-1]
    latest_roc_auc = roc_auc_df["Wert"].iloc[-1]
    latest_f1_score = f1_score_df["Wert"].iloc[-1]

    st.subheader("📋 Aktuelle Modellmetriken")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Precision", f"{latest_precision:.3f}")
    col2.metric("Recall", f"{latest_recall:.3f}")
    col3.metric("ROC AUC", f"{latest_roc_auc:.3f}")
    col4.metric("F1-Score", f"{latest_f1_score:.3f}")

    st.markdown("---")

    st.subheader("📈 Verlauf der Modellmetriken")
    col5, col6 = st.columns(2)

    with col5:
        st.subheader(f"Precision Verlauf ({latest_precision:.3f})")
        st.line_chart(precision_df.set_index("Zeit"))
        st.subheader(f"ROC AUC Verlauf ({latest_roc_auc:.3f})")
        st.line_chart(roc_auc_df.set_index("Zeit"))

    with col6:
        st.subheader(f"Recall Verlauf ({latest_recall:.3f})")
        st.line_chart(recall_df.set_index("Zeit"))
        st.subheader(f"F1-Score Verlauf ({latest_f1_score:.3f})")
        st.line_chart(f1_score_df.set_index("Zeit"))

    # Button zum Wechsel auf Retraining
    st.markdown("---")
    if st.button("🚀 Zum Retraining gehen"):
        st.session_state.page = "Retraining"
        st.stop()

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
    st.title("🛠️ Support")
    st.write("""
    Bei Problemen oder Fragen:
    - 📧 E-Mail: support@FKM.com
    - 📞 Telefon: +41 58 934 45 50
    - 🕒 Supportzeiten: 24/7
    """)

