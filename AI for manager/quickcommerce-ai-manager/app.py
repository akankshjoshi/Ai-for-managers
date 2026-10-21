import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Application setup
# --------------------------------------------------

st.set_page_config(page_title="QuickCommerce AI Manager", page_icon="📦", layout="wide")

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

st.title("📦 QuickCommerce AI Manager")

REQUIRED_FILES = ["cart_data.csv", "inventory_data.csv", "cart_abandonment_model.pkl",
                  "inventory_demand_model.pkl", "inventory_features.pkl", "kpis.json"]
missing = [f for f in REQUIRED_FILES if not (BASE_DIR / f).exists()]
if missing:
    st.error("These project files are missing: " + ", ".join(missing)
             + ". Run the notebook QuickCommerce_Cart_Abandonment_Replenishment.ipynb "
               "from top to bottom first; it creates them in this folder.")
    st.stop()

# --------------------------------------------------
# 2. Load saved models and data
# --------------------------------------------------

@st.cache_resource
def load_models():
    return (joblib.load(BASE_DIR / "cart_abandonment_model.pkl"),
            joblib.load(BASE_DIR / "inventory_demand_model.pkl"),
            joblib.load(BASE_DIR / "inventory_features.pkl"))


@st.cache_data
def load_data():
    return (pd.read_csv(BASE_DIR / "cart_data.csv"),
            pd.read_csv(BASE_DIR / "inventory_data.csv"),
            json.load(open(BASE_DIR / "kpis.json")))


cart_model, inventory_model, inventory_features = load_models()
cart_data, inventory_data, kpis = load_data()

CART_FEATURES = ["cart_value", "delivery_fee", "items_in_cart", "out_of_stock_items",
                 "eta_minutes", "previous_orders", "discount_percent"]
REVIEW_DAYS = 2


def forecast_next_day(history):
    """Forecast tomorrow's demand for one product from its sales history."""
    recent = history["daily_sales"].tail(7)
    row = pd.DataFrame([{
        "sales_lag_1": recent.iloc[-1], "sales_lag_2": recent.iloc[-2],
        "sales_lag_3": recent.iloc[-3], "sales_7day_avg": recent.mean(),
        "weekday": int(history["day"].iloc[-1]) % 7,
    }])[inventory_features]
    return max(0.0, float(inventory_model.predict(row)[0]))


@st.cache_data
def build_trigger_board(inventory):
    rows = []
    for product, history in inventory.sort_values("day").groupby("product"):
        latest = history.iloc[-1]
        demand = forecast_next_day(history)
        lead, safety, stock = latest["supplier_lead_time"], latest["safety_stock"], latest["current_stock"]
        reorder_point = demand * lead + safety
        trigger = stock <= reorder_point
        rows.append({
            "Product": product,
            "Current Stock": int(stock),
            "Forecast Demand / day": round(demand, 1),
            "Lead Time (days)": int(lead),
            "Safety Stock": int(safety),
            "Reorder Point": round(reorder_point, 1),
            "Days Until Stock-out": round(stock / demand, 1) if demand > 0 else None,
            "Trigger": "REORDER NOW" if trigger else "OK",
            "Suggested Order": int(np.ceil(demand * (lead + REVIEW_DAYS) + safety - stock)) if trigger else 0,
        })
    return pd.DataFrame(rows).sort_values(["Trigger", "Days Until Stock-out"], ascending=[False, True])


trigger_board = build_trigger_board(inventory_data)

# --------------------------------------------------
# 3. Navigation
# --------------------------------------------------

overview_tab, cart_tab, stock_tab, about_tab = st.tabs([
    "📊 Overview", "🛒 Cart Predictor", "📦 Replenishment Triggers", "ℹ️ About Us"])

# --------------------------------------------------
# 4. Overview
# --------------------------------------------------

with overview_tab:
    st.subheader("Business Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cart Sessions Analysed", f"{kpis['sessions_analysed']:,}")
    c2.metric("Cart Abandonment Rate", f"{kpis['abandonment_rate_pct']:.1f}%")
    c3.metric("Revenue Lost to Stock-out Abandonment",
              f"₹{kpis['revenue_lost_to_stockout_abandonment'] / 1e5:.1f} lakh")
    c4.metric("Products to Reorder Now", int((trigger_board["Trigger"] == "REORDER NOW").sum()),
              help="Products at or below their reorder point today")

    st.subheader("Impact of the AI solution (back-test, last 20 days)")
    d1, d2, d3 = st.columns(3)
    d1.metric("Fill Rate", f"{kpis['fill_rate_ai_pct']:.2f}%",
              f"{kpis['fill_rate_ai_pct'] - kpis['fill_rate_current_pct']:+.1f} pts vs current")
    d2.metric("Stock-out Product-days", kpis["stockout_days_ai"],
              f"{kpis['stockout_days_ai'] - kpis['stockout_days_current']} vs current", delta_color="inverse")
    saved = kpis["lost_revenue_current_20d"] - kpis["lost_revenue_ai_20d"]
    d3.metric("Revenue Lost to Stock-outs", f"₹{kpis['lost_revenue_ai_20d']:,.0f}",
              f"-₹{saved:,.0f} vs current", delta_color="inverse")

    left, right = st.columns(2)
    with left:
        st.markdown("**Abandonment rate by number of unavailable items**")
        st.bar_chart((cart_data.groupby("out_of_stock_items")["abandoned"].mean() * 100)
                     .rename("Abandonment %"))
    with right:
        st.markdown("**Abandonment rate by delivery fee (₹)**")
        st.bar_chart((cart_data.groupby("delivery_fee")["abandoned"].mean() * 100)
                     .rename("Abandonment %"))

    st.caption(f"Carts with an unavailable item are abandoned {kpis['abandonment_rate_with_oos_pct']:.0f}% "
               f"of the time vs {kpis['abandonment_rate_no_oos_pct']:.0f}% otherwise – "
               "better replenishment directly reduces cart abandonment.")

# --------------------------------------------------
# 5. Cart abandonment prediction
# --------------------------------------------------

with cart_tab:
    st.subheader("Cart Abandonment Predictor")
    st.write(f"Model in use: **{kpis['selected_cart_model']}** "
             f"(ROC-AUC {kpis['cart_model_roc_auc']:.2f}, catches "
             f"{kpis['cart_model_recall']:.0%} of abandoning carts).")

    with st.form("cart_form"):
        c1, c2 = st.columns(2)
        with c1:
            cart_value = st.number_input("Cart value (₹)", min_value=1.0, value=450.0, step=50.0)
            delivery_fee = st.selectbox("Delivery fee (₹)", [0, 10, 20, 30, 40, 50], index=4)
            items_in_cart = st.number_input("Items in cart", min_value=1, max_value=50, value=3)
            out_of_stock_items = st.number_input("Unavailable items", min_value=0, max_value=10, value=1)
        with c2:
            eta_minutes = st.slider("Estimated delivery time (minutes)", 5, 45, 22)
            previous_orders = st.number_input("Previous orders", min_value=0, value=1)
            discount_percent = st.selectbox("Discount (%)", [0, 5, 10, 15, 20])
        submitted = st.form_submit_button("Predict Abandonment Risk")

    if submitted:
        cart_input = pd.DataFrame([{
            "cart_value": cart_value, "delivery_fee": delivery_fee,
            "items_in_cart": items_in_cart, "out_of_stock_items": out_of_stock_items,
            "eta_minutes": eta_minutes, "previous_orders": previous_orders,
            "discount_percent": discount_percent}])[CART_FEATURES]
        probability = float(cart_model.predict_proba(cart_input)[0, 1])

        st.metric("Estimated Abandonment Probability", f"{probability:.1%}")
        st.progress(probability)

        # Pick the most relevant recovery action for this cart
        if out_of_stock_items > 0:
            action = "Suggest an in-stock substitute for the unavailable item(s)."
        elif delivery_fee > 0:
            action = f"Waive the ₹{delivery_fee} delivery fee or show how close the cart is to free delivery."
        elif eta_minutes > 20:
            action = "Show a clear, honest ETA and offer a slot when the rider is nearer."
        else:
            action = "Send a gentle reminder."

        threshold = kpis["best_nudge_threshold"]
        if probability >= threshold:
            st.warning(f"High risk (≥ {threshold:.0%} – the most profitable nudge threshold). "
                       f"Recommended action: {action}")
        elif probability >= 0.40:
            st.info(f"Medium risk. Optional: {action}")
        else:
            st.success("Low risk. Continue the normal customer journey – no incentive needed.")

# --------------------------------------------------
# 6. Replenishment triggers
# --------------------------------------------------

with stock_tab:
    st.subheader("Dark-Store Replenishment Triggers")
    st.write("Reorder point = forecast daily demand × lead time + safety stock "
             "(safety stock = 1.65 × σ × √lead time). A trigger fires when stock is at or below it.")

    def colour_trigger(value):
        return "background-color: #f8d7da" if value == "REORDER NOW" else ""

    st.dataframe(trigger_board.style.map(colour_trigger, subset=["Trigger"])
                 .format({"Forecast Demand / day": "{:.1f}", "Reorder Point": "{:.1f}",
                          "Days Until Stock-out": "{:.1f}"}),
                 width="stretch", hide_index=True)

    st.divider()
    st.subheader("Check one product")
    selected_product = st.selectbox("Select product", sorted(inventory_data["product"].unique()))
    history = inventory_data[inventory_data["product"] == selected_product].sort_values("day")
    latest = history.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    current_stock = c1.number_input("Current stock (units)", min_value=0, value=int(latest["current_stock"]))
    incoming_stock = c2.number_input("Incoming stock (units)", min_value=0, value=0)
    lead_time = c3.number_input("Lead time (days)", min_value=1, max_value=30,
                                value=int(latest["supplier_lead_time"]))
    safety_stock = c4.number_input("Safety stock (units)", min_value=0, value=int(latest["safety_stock"]))

    predicted_demand = forecast_next_day(history)
    reorder_point = predicted_demand * lead_time + safety_stock
    inventory_position = current_stock + incoming_stock
    order_quantity = int(np.ceil(max(0, predicted_demand * (lead_time + REVIEW_DAYS)
                                     + safety_stock - inventory_position))) \
        if inventory_position <= reorder_point else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Forecast Daily Demand", f"{predicted_demand:.1f} units")
    m2.metric("Reorder Point", f"{reorder_point:.1f} units")
    m3.metric("Suggested Order", f"{order_quantity} units")

    if inventory_position <= reorder_point:
        st.warning(f"Replenishment trigger: {selected_product} is at or below its reorder point.")
    else:
        st.success(f"{selected_product} is above its reorder point – no order needed.")

    st.markdown("**Daily sales history**")
    st.line_chart(history.set_index("day")["daily_sales"])

# --------------------------------------------------
# 7. About us
# --------------------------------------------------

with about_tab:
    st.subheader("About QuickCommerce AI Manager")
    st.write("QuickCommerce AI Manager helps a quick-commerce dark store with two daily questions: "
             "**which carts are about to be abandoned**, and **which products need restocking before "
             "they run out**. It turns both answers into clear actions for the store team.")

    st.markdown("#### Why it matters")
    p1, p2, p3 = st.columns(3)
    p1.metric("Carts abandoned", f"{kpis['abandonment_rate_pct']:.1f}%")
    p2.metric("Abandonment with an out-of-stock item", f"{kpis['abandonment_rate_with_oos_pct']:.0f}%",
              f"+{kpis['abandonment_rate_with_oos_pct'] - kpis['abandonment_rate_no_oos_pct']:.0f} pts vs in stock",
              delta_color="inverse")
    p3.metric("Revenue lost to stock-out abandonment", f"₹{kpis['revenue_lost_to_stockout_abandonment'] / 1e5:.1f} lakh")
    st.write("When an item runs out, customers leave the whole cart. Cart abandonment and stock-outs are "
             "one connected problem, so the app tackles both together.")

    st.markdown("#### What the app does")
    a, b, c = st.columns(3)
    with a.container(border=True):
        st.markdown("**🛒 Predicts risky carts**")
        st.write("Scores each cart's chance of being abandoned from its value, delivery fee, "
                 "delivery time, unavailable items and the customer's order history.")
    with b.container(border=True):
        st.markdown("**🎯 Suggests the right action**")
        st.write(f"Only carts with risk of {kpis['best_nudge_threshold']:.0%} or more get a nudge, matched to the "
                 "reason: a substitute, a fee waiver or a clearer delivery time.")
    with c.container(border=True):
        st.markdown("**📦 Triggers restocking**")
        st.write("Forecasts tomorrow's demand for every product and flags what to reorder, "
                 "how much, and how many days are left before it runs out.")

    st.markdown("#### How it works")
    s1, s2, s3, s4 = st.columns(4)
    with s1.container(border=True):
        st.markdown("**1. Data**")
        st.caption("Cart sessions and daily sales are cleaned: duplicates removed, missing values filled.")
    with s2.container(border=True):
        st.markdown("**2. Cart model**")
        st.caption("Class-balanced Logistic Regression, chosen because it catches the most abandoners and is easy to explain.")
    with s3.container(border=True):
        st.markdown("**3. Demand model**")
        st.caption("Random Forest forecast from the last 3 days, the 7-day average and the day of the week.")
    with s4.container(border=True):
        st.markdown("**4. Trigger rule**")
        st.caption("Reorder point = forecast × lead time + safety stock. At or below it, the app says reorder.")

    st.markdown("#### Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Abandoners caught", f"{kpis['cart_model_recall']:.0%}")
    m2.metric("Net profit of targeted nudges", f"₹{kpis['nudge_net_profit_test']:,}",
              f"+₹{kpis['nudge_net_profit_test'] - kpis['nudge_everyone_net_profit_test']:,} vs nudging everyone")
    m3.metric("Fill rate", f"{kpis['fill_rate_ai_pct']:.2f}%",
              f"+{kpis['fill_rate_ai_pct'] - kpis['fill_rate_current_pct']:.1f} pts vs weekly ordering")
    m4.metric("Stock-out days (20 days)", kpis["stockout_days_ai"],
              f"{kpis['stockout_days_ai'] - kpis['stockout_days_current']} vs weekly ordering", delta_color="inverse")

    with st.expander("How to use this app"):
        st.markdown("""
1. **Overview** – the business problem and the impact of the AI in a 20-day back-test.
2. **Cart Predictor** – enter a cart's details and press *Predict* to see its risk and the recommended action.
3. **Replenishment Triggers** – today's reorder list, most urgent first. Pick a product to check it and try changing incoming stock.
""")
    with st.expander("Data, privacy and limitations"):
        st.markdown("""
- All data is **synthetic**, generated in Python; it is not Blinkit or Zepto data.
- Money figures rest on stated assumptions (₹20 per nudge, 25% of carts saved, 20% margin).
- Designed for India's **DPDP Act 2023**: consent, purpose limitation, pseudonymised customer IDs, no protected attributes.
- Nudges must be honest – no fake urgency. The AI recommends; the store manager decides.
- Next step: a pilot on real data in a few dark stores with an A/B test.
""")
    st.caption("Results come from a back-test on synthetic data. Built with Python, pandas, scikit-learn and Streamlit.")
