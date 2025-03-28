from shiny import App, render, ui
from shinywidgets import output_widget, render_widget
import pandas as pd
import plotly.express as px
import numpy as np

# Daten direkt aus Maschine.xlsx laden
df = pd.read_excel("Daten.xlsx")
df["TX_FRAUD"] = df["TX_FRAUD"].fillna(0).astype(int)
df["TX_AMOUNT"] = df["TX_AMOUNT"].fillna(0).astype(float)
df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
df["Hour"] = df["TX_DATETIME"].dt.hour

# Dummy-Koordinaten für Karte
df["LAT"] = 50 + np.random.randn(len(df)) * 0.1
df["LON"] = 8 + np.random.randn(len(df)) * 0.1

app_ui = ui.page_fluid(
    ui.h2("\U0001F4B3 Credit Card Fraud Detection Dashboard"),
    ui.layout_columns(
        ui.card(
            ui.input_slider("amount", "\U0001F50D Mindestbetrag (€)", min=0, max=5000, value=0, step=50),
            style="width: 300px"
        ),
        ui.div(
            ui.output_text("summary"),
            output_widget("fraud_pie"),
            output_widget("fraud_heatmap"),
            output_widget("fraud_trend"),
            output_widget("fraud_map"),
            output_widget("top_customers"),
            output_widget("terminal_risk_plot"),
            ui.output_table("filtered_table")
        ),
        col_widths=(3, 9)
    )
)

def server(input, output, session):

    @render.text
    def summary():
        total = len(df)
        frauds = df[df["TX_FRAUD"] == 1].shape[0]
        ratio = frauds / total * 100 if total > 0 else 0
        return f"Insgesamt {total} Transaktionen, davon {frauds} Betrugsfälle ({ratio:.2f}%)."

    @render_widget
    def fraud_pie():
        fraud_counts = df["TX_FRAUD"].value_counts().reset_index()
        fraud_counts.columns = ["is_fraud", "count"]
        fraud_counts["label"] = fraud_counts["is_fraud"].map({0: "Normal", 1: "Betrug"})
        if len(fraud_counts) < 2:
            fraud_counts = pd.concat([
                fraud_counts,
                pd.DataFrame([{"is_fraud": 1, "count": 0, "label": "Betrug"}])
            ], ignore_index=True)
        return px.pie(fraud_counts, values="count", names="label", title="Verteilung: Betrug vs. Normal")

    @render_widget
    def fraud_heatmap():
        fraud_df = df[df["TX_FRAUD"] == 1]
        if fraud_df.empty:
            dummy = pd.DataFrame([list(range(1, 25))], columns=list(range(24)))
            return px.imshow(dummy, labels=dict(x="Stunde", color="Betrugsfälle"), x=dummy.columns,
                             y=["Demo"], title="Demo Heatmap")
        hourly = fraud_df.groupby("Hour").size().reindex(range(24), fill_value=0)
        heatmap_df = pd.DataFrame([hourly.values.astype(float)], columns=hourly.index)
        return px.imshow(heatmap_df, labels=dict(x="Stunde", color="Betrugsfälle"), x=hourly.index,
                         y=["Betrugsaktivität"], title="Betrugsaktivität nach Stunde")

    @render_widget
    def fraud_trend():
        trend_df = df[df["TX_FRAUD"] == 1].copy()
        trend_df["Datum"] = trend_df["TX_DATETIME"].dt.strftime("%Y-%m-%d")
        trend = trend_df.groupby("Datum").size().reset_index(name="Betrugsfälle")
        if trend.empty:
            trend = pd.DataFrame({"Datum": [pd.Timestamp.today().strftime("%Y-%m-%d")], "Betrugsfälle": [0]})
        return px.line(trend, x="Datum", y="Betrugsfälle", title="Betrugsfälle pro Tag")

    @render_widget
    def fraud_map():
        map_df = df[df["TX_FRAUD"] == 1]
        if map_df.empty:
            return px.scatter_mapbox(lat=[50], lon=[8], zoom=5, height=300,
                                     mapbox_style="open-street-map", title="Keine Betrugsdaten")
        return px.scatter_mapbox(map_df, lat="LAT", lon="LON", color="TX_AMOUNT",
                                  size="TX_AMOUNT", hover_name="CUSTOMER_ID",
                                  zoom=5, mapbox_style="open-street-map", title="Betrugsstandorte")

    @render_widget
    def top_customers():
        top = df.groupby("CUSTOMER_ID")["TX_AMOUNT"].sum().sort_values(ascending=False).head(10).reset_index()
        return px.bar(top, x="CUSTOMER_ID", y="TX_AMOUNT", title="Top 10 Kunden nach Umsatz")

    @render_widget
    def terminal_risk_plot():
        risk = df.groupby("TERMINAL_ID")["TERMINAL_ID_RISK_30DAY_WINDOW"].mean().sort_values(ascending=False).head(10).reset_index()
        return px.bar(risk, x="TERMINAL_ID", y="TERMINAL_ID_RISK_30DAY_WINDOW", title="Höchste Terminal-Risiken")

    @render.table
    def filtered_table():
        return df[df["TX_AMOUNT"] >= input.amount()]

app = App(app_ui, server)
