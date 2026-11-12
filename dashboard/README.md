# AquaTwin Grafana dashboard (simulated data)

`aquatwin_dashboard.json` is a Grafana dashboard that reads `aquatwin_dashboard_data.csv` straight from this GitHub repository, using the free **Infinity** data source plugin. No database or server is needed, so it works on Grafana Cloud as is.

## Set it up in Grafana Cloud (about 5 minutes)

1. **Install Infinity.** In Grafana: *Connections → Add new connection*, search **Infinity**, click **Install**.
2. **Add the data source.** On the Infinity page click **Add new data source**, keep the defaults, and click **Save & test**.
   If a later panel says the URL is not allowed, open this data source again and add `https://raw.githubusercontent.com` under *Security → Allowed hosts*, then save.
3. **Import the dashboard.** *Dashboards → New → Import*, upload `aquatwin_dashboard.json` (or paste its contents), pick the Infinity data source in the drop-down, and click **Import**.

The dashboard opens on the simulated window, 17 Nov 2026 14:00 to 18 Nov 02:00 UTC (8:00 a.m. to 8:00 p.m. Central). It is imported as a **new** dashboard (`AquaTwin Digital Twin (simulated data)`), so the existing AquaTwin Dashboard is not changed.

## What it shows

| Section | Panels |
|---|---|
| Status | Detector status, suspected signal, injected fault (ground truth), peak anomaly score, time in alarm |
| Anomaly detection | Alarm vs injected fault timeline; anomaly score of the residual z-score chart and Isolation Forest, as a multiple of each alarm threshold (log scale, threshold line at 1) |
| Hydraulics | Flow, filter inlet pressure, filter head loss, tank level: measured (blue) vs twin expected (dashed gray) |
| Water quality | Turbidity in and out, dissolved oxygen, pH, conductivity: measured vs twin; temperature |

The status tiles summarize the file as a whole: "latest" means the end of the 12-hour window, not the current time.

## Regenerate the data

```
python simulate.py                 # optional: new simulated dataset
python export_dashboard_data.py    # writes dashboard/aquatwin_dashboard_data.csv
python dashboard/build_dashboard.py
```
Commit and push the CSV; the dashboard picks it up on the next refresh.

## Notes

- The repository must stay **public** for Grafana to read the CSV. If you make it private, host the CSV somewhere public or switch to the SQL database planned for the live testbed.
- This replays recorded results. For the live testbed, the same panels should query the SQL database instead. Keep the column names in `aquatwin_dashboard_data.csv` as the table schema so the panels carry over.
- Grafana Cloud reaches GitHub over the internet. It cannot reach a database on the lab laptop without exposing it or installing Grafana's Private Data Source Connect agent.
