# PM2.5-Vorhersage für Beijing – Kaggle Challenge

Vorhersage des täglichen Feinstaub-Mittelwerts (PM2.5) für 12 Luftqualitätsstationen in Beijing.
Das Projekt entstand im Rahmen einer hochschulinternen Kaggle-Challenge im Modul **Umweltmonitoring** (Data Science, Hochschule Karlsruhe, SS 2026).

## Aufgabenstellung

Auf Basis von Luftqualitäts- und Wetterdaten soll der PM2.5-Tagesmittelwert (µg/m³) für einen zukünftigen Zeitraum vorhergesagt werden. Bewertet wird anhand des **Mean Squared Error (MSE)** auf dem Testdatensatz.

Im Vordergrund steht neben einem guten Score ein **sauberer, reproduzierbarer Modellierungsprozess** mit nachvollziehbaren Entscheidungen.

## Daten

Grundlage ist der *Beijing Multi-Site Air Quality* Datensatz, von Stunden- auf Tageswerte aggregiert.

| Eigenschaft | Wert |
|---|---|
| Trainingsdaten | 14.319 Zeilen (März 2013 – Juni 2016) |
| Testdaten | 2.845 Zeilen (Juli 2016 – Dezember 2016) |
| Stationen | 12 |
| Zielvariable | `pm25` – Feinstaub-Tagesmittelwert in µg/m³ |

**Features im Rohdatensatz:**
- Luftqualität: `pm10`, `so2`, `no2`, `co`, `o3`
- Wetter: Temperatur, Luftdruck, Taupunkt, Niederschlag, Windgeschwindigkeit
- Kategorisch: Station, Windrichtung
- Zeitlich: Datum

> Die Daten sind nicht in diesem Repository enthalten. Sie können über die Kaggle-Challenge bezogen werden.

## Vorgehen

### 1. Explorative Datenanalyse
- PM2.5 ist stark rechtsschief verteilt (Ausreißer bis ca. 570 µg/m³)
- Stärkste Korrelationen mit PM2.5: `pm10` (0,90), `co` (0,81), `no2` (0,69); Windgeschwindigkeit wirkt negativ (−0,38)
- Deutliche Saisonalität mit höheren Werten im Winter (Heizperiode)
- Fehlende Werte moderat (max. 4,4 % bei `co`), ohne systematische Häufung an einzelnen Stationen

### 2. Feature Engineering
- **Zeit:** zyklische Kodierung von Monat (sin/cos), Tag des Jahres, Quartal
- **Wetter-Interaktionen:** Temperatur-Taupunkt-Differenz (Proxy für Luftfeuchtigkeit), Wind × Luftdruck, Temperatur × Wind
- **Schadstoff-Interaktionen:** Gesamtbelastung, Verhältnis `pm10`/`co`
- **Lag-Features:** PM2.5 und PM10 vom Vortag (Autokorrelation Lag 1: 0,55)
- **Stationsübergreifend:** täglicher PM10-Durchschnitt aller Stationen

### 3. Vorverarbeitung
scikit-learn-Pipeline mit `ColumnTransformer`:
- Numerisch: Median-Imputation + `StandardScaler`
- Kategorisch: Most-Frequent-Imputation + `OneHotEncoder`

Durch die Pipeline wird der Preprocessor nur auf Trainingsdaten gefittet. So entsteht kein Data Leakage.

### 4. Validierungsstrategie
Zeitlicher Split statt zufälliger Aufteilung:

| Split | Zeitraum |
|---|---|
| Training | März 2013 – Dezember 2015 |
| Validierung | Januar 2016 – Juni 2016 |
| Test (Kaggle) | Juli 2016 – Dezember 2016 |

Für das Hyperparameter-Tuning wurde `TimeSeriesSplit` verwendet.

### 5. Modellierung
Getestet wurden Lineare Regression (Baseline), Decision Tree, Random Forest, ExtraTrees, Gradient Boosting, LightGBM, XGBoost, HistGradientBoosting und CatBoost.
LightGBM, HistGradientBoosting und CatBoost wurden per `RandomizedSearchCV` getunt.

Das finale Modell ist ein **Stacking-Ensemble** aus acht Basismodellen mit Ridge-Regression als Meta-Modell (5-fache Cross-Validation).

## Ergebnisse

Validierungs-RMSE (Januar – Juni 2016):

| Modell | RMSE |
|---|---|
| **Stacking** | **13,35** |
| CatBoost | 13,43 |
| LightGBM | 14,11 |
| HistGradientBoosting | 14,14 |
| XGBoost | 14,27 |
| ExtraTrees | 15,85 |
| Random Forest | 16,27 |
| Lineare Regression (Baseline) | 19,24 |
| Decision Tree | 19,67 |

Das Stacking-Ensemble verbessert die Baseline um rund 30 %. Die Overfitting-Analyse zeigt, dass CatBoost und HistGradientBoosting die geringste Lücke zwischen Trainings- und Validierungsfehler aufweisen. Das Ridge-Meta-Modell dämpft die Overfitting-Tendenz einzelner Basismodelle.

## Technologien

- Python 3.11
- pandas, NumPy
- scikit-learn
- LightGBM, XGBoost, CatBoost
- Matplotlib, Seaborn
- `holidays` (chinesische Feiertage)
- Jupyter Notebook

## Ausführung

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn holidays
```

1. Trainings- und Testdaten von Kaggle herunterladen
2. Dateipfade (`train_path`, `test_path`, `output_path`) im Notebook anpassen
3. Notebook von oben nach unten ausführen

## KI-Nutzung

Claude (Anthropic) wurde unterstützend eingesetzt, vergleichbar mit Dokumentation oder Stack Overflow: zur Erklärung von Konzepten (z. B. Stacking, TimeSeriesSplit), zur Fehlersuche (z. B. Data Leakage) und als Syntaxhilfe. Feature-Auswahl, Modellwahl, Validierungsstrategie und Interpretation der Ergebnisse wurden eigenständig erarbeitet.

## Autor

Nicolas Breitner – Data Science (B.Sc.), Hochschule Karlsruhe
