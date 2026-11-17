# Customer Segmentation for an Online Store (RFM + K-Means)

Groups customers by **Recency** (days since last order), **Frequency** (number of orders) and
**Monetary** value (total spend), and gives a marketing action for each segment. The Streamlit app
shows the segments, checks a single customer, and segments a whole uploaded transactions file.

## Results
4,334 customers from 391k clean transaction lines (UCI Online Retail, Dec 2010 - Dec 2011).

| Algorithm (K = 4) | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|
| **K-Means (selected)** | **0.338** | **1.013** | **3,330** |
| Gaussian Mixture | 0.175 | 1.708 | 2,168 |
| Agglomerative (reference only) | 0.243 | 1.226 | 2,780 |

| Segment | Customers | Revenue share | Median days since order | Median orders | Action |
|---|---|---|---|---|---|
| Champions | 16% | 64% | 8 | 10 | VIP perks, early access, referrals |
| At Risk | 27% | 24% | 52 | 4 | Win-back campaign now |
| New / Promising | 19% | 5% | 17 | 2 | Onboarding, second-purchase offer |
| Lost / Inactive | 38% | 7% | 177 | 1 | Low-cost reactivation only |

Full details: [reports/evaluation_report.md](reports/evaluation_report.md).

## Project structure
```
customer_segmentation.ipynb  cleaning, RFM, choosing K, algorithm comparison, profiling, saving
segmentation.py              cleaning, RFM and segment naming shared by the notebook and the app
app.py                       Streamlit app
model/                       best_model.joblib (pipeline + segment names) + metrics.json
data/                        customer_segments.csv, sample_transactions.csv (try the upload with it)
reports/                     evaluation report and charts
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Uploads need the columns `InvoiceNo, InvoiceDate, Quantity, UnitPrice, CustomerID`; `StockCode` is
optional. To retrain, put `data.csv` (Kaggle `carrie1/ecommerce-data`) in `data/` and run the notebook.

## Deploy (Streamlit Community Cloud)
Push this folder to GitHub, open share.streamlit.io, choose **New app**, select the repo and set the
main file to `app.py`.

## Limitations
- Clustering has no single correct answer. Segment names come from rules on the group medians and
  should be reviewed with the marketing team.
- One year of data from one UK gift shop (many customers are wholesalers). Refit on the client's own
  customers before real use.
- Recency depends on the date the data ends, so re-score customers regularly.
