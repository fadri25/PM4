from shiny import App, render, ui
from shinywidgets import output_widget, render_widget
import pandas as pd
import plotly.express as px
import numpy as np

# CSV-Daten laden (Dateipfad ggf. anpassen)
df = pd.read_csv("transactions_first_100000.csv")
df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"], errors="coerce")
df["Hour"] = df["TX_DATETIME"].dt.hour
df["Weekday"] = df["TX_DATETIME"].dt.day_name()

print("Data Preview:", df.head())
print("Fraud Count:", df["TX_FRAUD"].value_counts())

# UI mit frei kombinierbaren Plotgrößen und Layout
app_ui = ui.page_fluid(
    ui.h2("Kreditkartenbetrug - Fraud Detection Dashboard"),
    ui.navset_tab(
        ui.nav_panel("Allgemein",
            ui.row(
                ui.column(6, output_widget("plot1")),
                ui.column(6, output_widget("plot5"))
            )
        ),
        ui.nav_panel("Zeitverläufe",
            ui.row(
                ui.column(6, output_widget("plot2")),
                ui.column(6, output_widget("plot3"))
            ),
            ui.row(
                ui.column(12, output_widget("plot4"))
            )
        ),
        ui.nav_panel("Kundenverhalten",
            ui.row(
                ui.column(12, output_widget("plot6"))
            )
        ),
        ui.nav_panel("Terminals",
            ui.row(
                ui.column(6, output_widget("plot7")),
                ui.column(6, output_widget("plot8"))
            )
        ),
        ui.nav_panel("Zusammenhänge",
            ui.row(
                ui.column(6, output_widget("plot9")),
                ui.column(6, output_widget("plot10"))
            )
        ),
        ui.nav_panel("Test",
            ui.row(
                ui.column(12, output_widget("plot_test"))
            )
        )
    )
)

# Server-Logik
def server(input, output, session):

    @output
    @render_widget
    def plot_test():
        return px.scatter(x=[1, 2, 3], y=[4, 5, 6], title="Testplot", width=500, height=350)

    @output
    @render_widget
    def plot1():
        if df.empty:
            return px.scatter(title="Keine Daten verfügbar")
        return px.histogram(df, x="TX_AMOUNT", nbins=50, title="Verteilung der Transaktionsbeträge", width=500, height=350)

    @output
    @render_widget
    def plot2():
        data = df[df["TX_FRAUD"] == 1]
        if data.empty:
            return px.scatter(title="Keine Betrugsfälle verfügbar")
        fraud_per_day = data.groupby("TX_TIME_DAYS").size().reset_index(name="Fraud_Count")
        return px.line(fraud_per_day, x="TX_TIME_DAYS", y="Fraud_Count", title="Betrugsfälle pro Tag", width=500, height=350)

    @output
    @render_widget
    def plot3():
        if df.empty:
            return px.scatter(title="Keine Daten verfügbar")
        fraud_rate_by_hour = df.groupby("Hour")["TX_FRAUD"].mean().reset_index(name="Fraud_Rate")
        return px.bar(fraud_rate_by_hour, x="Hour", y="Fraud_Rate", title="Betrugsrate nach Uhrzeit", width=500, height=350)

    @output
    @render_widget
    def plot4():
        if df.empty:
            return px.scatter(title="Keine Daten verfügbar")
        weekday_fraud = df.groupby("Weekday")["TX_FRAUD"].mean().reset_index()
        return px.bar(weekday_fraud, x="Weekday", y="TX_FRAUD", title="Betrugsrate nach Wochentag", width=1000, height=350)

    @output
    @render_widget
    def plot5():
        if df.empty:
            return px.scatter(title="Keine Daten verfügbar")
        return px.box(df, x="TX_FRAUD", y="TX_AMOUNT", title="Transaktionsbetrag: Betrug vs. Nicht-Betrug", width=500, height=350)

    @output
    @render_widget
    def plot6():
        if df.empty:
            return px.scatter(title="Keine Daten verfügbar")
        return px.scatter(df, x="CUSTOMER_ID_NB_TX_1DAY_WINDOW", y="CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW", color="TX_FRAUD", title="Kundenverhalten vs. Betrug", width=1000, height=400)

    @output
    @render_widget
    def plot7():
        if df.empty:
            return px.scatter(title="Keine Daten verfügbar")
        terminal_risk = df.groupby("TERMINAL_ID")["TERMINAL_ID_RISK_30DAY_WINDOW"].mean().nlargest(10).reset_index()
        return px.bar(terminal_risk, x="TERMINAL_ID", y="TERMINAL_ID_RISK_30DAY_WINDOW", title="Top 10 riskanteste Terminals", width=500, height=350)

    @output
    @render_widget
    def plot8():
        if df.empty:
            return px.scatter(title="Keine Daten verfügbar")
        return px.density_heatmap(df, x="TERMINAL_ID_NB_TX_30DAY_WINDOW", y="TERMINAL_ID_RISK_30DAY_WINDOW", title="Heatmap: Terminalrisiko vs. Transaktionsanzahl", width=500, height=350)

    @output
    @render_widget
    def plot9():
        if df.empty:
            return px.scatter(title="Keine Daten verfügbar")
        df["amount_bin"] = pd.cut(df["TX_AMOUNT"], bins=10).astype(str)
        fraud_by_amount = df.groupby("amount_bin")["TX_FRAUD"].mean().reset_index(name="Fraud_Rate")
        return px.line(fraud_by_amount, x="amount_bin", y="Fraud_Rate", title="Betrugsrate nach Transaktionsbetrag", width=500, height=350)

    @output
    @render_widget
    def plot10():
        if df.empty:
            return px.scatter(title="Keine Daten verfügbar")
        numeric_cols = df.select_dtypes(include='number')
        corr = numeric_cols.corr()[["TX_FRAUD"]].drop("TX_FRAUD")
        corr.reset_index(inplace=True)
        return px.bar(corr, x="index", y="TX_FRAUD", title="Korrelation mit Betrug", width=500, height=350)

# App erstellen
app = App(app_ui, server)