"""Cleaning, RFM and segment naming shared by the training notebook and the Streamlit app."""
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice", "CustomerID"]
RFM = ["Recency", "Frequency", "Monetary"]
NON_PRODUCT_CODES = {"POST", "D", "M", "BANK CHARGES", "AMAZONFEE", "DOT", "CRUK", "PADS", "C2"}

ACTIONS = {
    "Champions": "Reward them: VIP perks, early access to new products, ask for reviews and referrals.",
    "Loyal Customers": "Upsell and cross-sell, loyalty points, personalised recommendations.",
    "New / Promising": "Onboarding emails, a second-purchase discount, show best-sellers.",
    "At Risk": "Win-back campaign now: personal email, time-limited offer, ask what went wrong.",
    "Lost / Inactive": "Low-cost reactivation (newsletter, big seasonal sale); do not overspend.",
}


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    info = {"rows_raw": len(df)}
    df = df.copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    info["missing_customer_id"] = int(df["CustomerID"].isna().sum())
    df = df.dropna(subset=["CustomerID"])
    cancel = df["InvoiceNo"].str.startswith("C")
    info["cancellations"] = int(cancel.sum())
    df = df[~cancel]
    bad = (df["Quantity"] <= 0) | (df["UnitPrice"] <= 0)
    info["non_positive_qty_or_price"] = int(bad.sum())
    df = df[~bad]
    if "StockCode" in df:
        non_prod = df["StockCode"].astype(str).str.upper().isin(NON_PRODUCT_CODES)
        info["non_product_lines"] = int(non_prod.sum())
        df = df[~non_prod]
    info["duplicates"] = int(df.duplicated().sum())
    df = df.drop_duplicates()
    df["CustomerID"] = df["CustomerID"].astype(float).astype(int).astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    info["rows_clean"] = len(df)
    return df.reset_index(drop=True), info


def compute_rfm(tx: pd.DataFrame, snapshot=None) -> pd.DataFrame:
    """One row per customer. Recency = days since last purchase, measured from `snapshot`
    (default: the day after the last transaction in the data)."""
    snapshot = pd.Timestamp(snapshot) if snapshot is not None else tx["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = tx.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda d: (snapshot - d.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
    )
    return rfm


def name_segments(profile: pd.DataFrame) -> dict:
    """Give each cluster a business name from its median Recency/Frequency/Monetary.
    profile: index = cluster id, columns = RFM medians; overall medians in profile.attrs."""
    ranks = (profile["Recency"].rank(ascending=False) + profile["Frequency"].rank()
             + profile["Monetary"].rank())
    order = ranks.sort_values(ascending=False).index.tolist()
    med = profile.attrs["overall_median"]
    names = {order[0]: "Champions", order[-1]: "Lost / Inactive"}
    for c in order[1:-1]:
        recent = profile.loc[c, "Recency"] <= med["Recency"]
        frequent = profile.loc[c, "Frequency"] > med["Frequency"]
        if recent and frequent:
            name = "Loyal Customers"
        elif recent:
            name = "New / Promising"
        else:
            name = "At Risk"
        while name in names.values():  # keep names unique if two clusters look alike
            name += " (2)"
        names[c] = name
    return names


def action_for(name: str) -> str:
    return ACTIONS.get(name.replace(" (2)", ""), "")
