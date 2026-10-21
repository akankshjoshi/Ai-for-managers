import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

# --------------------------------------------------
# 1. Application setup & Path Resolver
# --------------------------------------------------

st.set_page_config(
    page_title="QuickCommerce AI Manager",
    page_icon="📦",
    layout="wide"
)

# Fix for NameError: '__file__' in interactive notebooks vs CLI
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

st.title("📦 QuickCommerce AI Manager")
st.caption(
    "AI-powered cart abandonment prediction "
    "and dark-store inventory replenishment"
)

st.info(
    "Academic prototype using synthetic data. "
    "Predictions do not represent actual Blinkit or Zepto operations."
)

# --------------------------------------------------
# 2. Automated Model & Data Generator (Self-Healing)
# --------------------------------------------------

def ensure_project_files_exist():
    """Generates synthetic datasets and models if they are missing."""
    cart_csv = BASE_DIR / "cart_data.csv"
    inv_csv = BASE_DIR / "inventory_data.csv"
    cart_model_pkl = BASE_DIR / "cart_abandonment_model.pkl"
    inv_model_pkl = BASE_DIR / "inventory_demand_model.pkl"
    inv_feats_pkl = BASE_DIR / "inventory_features.pkl"

    # 1. Create cart_data.csv & cart model if missing
    if not (cart_csv.exists() and cart_model_pkl.exists()):
        np.random.seed(42)
        n_carts = 2000
        c_val = np.round(np.random.normal(520, 150, n_carts).clip(120, 1500), 2)
        d_fee = np.random.choice([0, 15, 25, 40], size=n_carts, p=[0.3, 0.4, 0.2, 0.1])
        items = np.random.randint(1, 15, n_carts)
        oos = np.random.binomial(n=3, p=0.15, size=n_carts)
        eta = np.random.randint(8, 35, n_carts)
        prev_ord = np.random.poisson(lam=5, size=n_carts)
        disc = np.random.choice([0, 5, 10, 15, 20], size=n_carts)

        # Churn/Abandonment logic
        logits = (
            0.08 * (eta - 10)
            + 0.95 * oos
            + 0.04 * d_fee
            - 0.001 * c_val
            - 0.15 * prev_ord
            - 0.03 * disc
            - 0.5
        )
        prob = 1 / (1 + np.exp(-logits))
        abandoned = (prob > 0.45).astype(int)

        cart_df = pd.DataFrame({
            "cart_value": c_val,
            "delivery_fee": d_fee,
            "items_in_cart": items,
            "out_of_stock_items": oos,
            "eta_minutes": eta,
            "previous_orders": prev_ord,
            "discount_percent": disc,
            "abandoned": abandoned
        })
        cart_df.to_csv(cart_csv, index=False)

        cart_X = cart_df.drop(columns=["abandoned"])
        cart_y = cart_df["abandoned"]
        c_model = LogisticRegression(max_iter=1000)
        c_model.fit(cart_X, cart_y)
        joblib.dump(c_model, cart_model_pkl)

    # 2. Create inventory_data.csv & inventory model if missing
    if not (inv_csv.exists() and inv_model_pkl.exists() and inv_feats_pkl.exists()):
        products = ["Amul Gold Milk 500ml", "Brown Bread 400g", "Farm Eggs (Tray 6)", 
                    "Fresh Coriander 100g", "Paneer 200g", "Curd 400g"]
        inv_records = []
        np.random.seed(42)

        for prod in products:
            base_demand = np.random.randint(40, 90)
            lead_time = np.random.choice([1, 2])
            safety = int(base_demand * 0.4)
            current = int(base_demand * np.random.uniform(0.7, 1.8))

            for day in range(1, 31):
                sales = max(5, int(np.random.normal(base_demand, 12)))
                inv_records.append({
                    "day": day,
                    "product": prod,
                    "daily_sales": sales,
                    "current_stock": current if day == 30 else current + np.random.randint(-10, 15),
                    "supplier_lead_time": lead_time,
                    "safety_stock": safety
                })

        inv_df = pd.DataFrame(inv_records)
        inv_df.to_csv(inv_csv, index=False)

        # Train inventory lag regression model
        training_rows = []
        for prod in products:
            p_hist = inv_df[inv_df["product"] == prod].sort_values("day")
            sales_arr = p_hist["daily_sales"].values
            days_arr = p_hist["day"].values
            for i in range(3, len(sales_arr)):
                training_rows.append({
                    "sales_lag_1": sales_arr[i - 1],
                    "sales_lag_2": sales_arr[i - 2],
                    "sales_lag_3": sales_arr[i - 3],
                    "sales_7day_avg": np.mean(sales_arr[max(0, i - 7):i]),
                    "weekday": int(days_arr[i]) % 7,
                    "target_sales": sales_arr[i]
                })

        train_data = pd.DataFrame(training_rows)
        inv_feats = ["sales_lag_1", "sales_lag_2", "sales_lag_3", "sales_7day_avg", "weekday"]
        inv_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        inv_model.fit(train_data[inv_feats], train_data["target_sales"])

        joblib.dump(inv_model, inv_model_pkl)
        joblib.dump(inv_feats, inv_feats_pkl)

ensure_project_files_exist()

# --------------------------------------------------
# 3. Load saved models and data
# --------------------------------------------------

@st.cache_resource
def load_models():
    cart_model = joblib.load(BASE_DIR / "cart_abandonment_model.pkl")
    inventory_model = joblib.load(BASE_DIR / "inventory_demand_model.pkl")
    inventory_features = joblib.load(BASE_DIR / "inventory_features.pkl")
    return cart_model, inventory_model, inventory_features


@st.cache_data
def load_data():
    cart = pd.read_csv(BASE_DIR / "cart_data.csv")
    inventory = pd.read_csv(BASE_DIR / "inventory_data.csv")
    return cart, inventory


try:
    cart_model, inventory_model, inventory_features = load_models()
    cart_data, inventory_data = load_data()
except Exception as error:
    st.error(
        "Could not load project files. Check that app.py, "
        "both model files, inventory_features.pkl and both "
        "CSV files are in the same folder."
    )
    st.exception(error)
    st.stop()

# --------------------------------------------------
# 4. Main navigation
# --------------------------------------------------

dashboard, cart_tab, stock_tab, about_tab = st.tabs([
    "📊 Overview",
    "🛒 Cart Predictor",
    "📦 Inventory Replenishment",
    "ℹ️ About Project"
])

# --------------------------------------------------
# 5. Overview dashboard
# --------------------------------------------------

with dashboard:
    st.subheader("Business Overview")

    total_sessions = len(cart_data)
    abandonment_rate = cart_data["abandoned"].mean() * 100
    product_count = inventory_data["product"].nunique()

    latest_stock = (
        inventory_data.sort_values("day")
        .groupby("product")
        .tail(1)
        .copy()
    )

    stock_alert_count = 0

    for _, row in latest_stock.iterrows():
        product_history = (
            inventory_data[
                inventory_data["product"] == row["product"]
            ].sort_values("day")
        )

        recent_sales = product_history["daily_sales"].tail(7)
        forecast_input = pd.DataFrame([{
            "sales_lag_1": recent_sales.iloc[-1],
            "sales_lag_2": recent_sales.iloc[-2],
            "sales_lag_3": recent_sales.iloc[-3],
            "sales_7day_avg": recent_sales.mean(),
            "weekday": int(row["day"]) % 7
        }])[inventory_features]

        predicted_demand = max(
            0,
            float(inventory_model.predict(forecast_input)[0])
        )

        reorder_point = (
            predicted_demand * row["supplier_lead_time"]
            + row["safety_stock"]
        )

        if row["current_stock"] <= reorder_point:
            stock_alert_count += 1

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Simulated Cart Sessions", f"{total_sessions:,}")
    col2.metric("Cart Abandonment Rate", f"{abandonment_rate:.1f}%")
    col3.metric("Products Tracked", product_count)
    col4.metric("Products Below Reorder Point", stock_alert_count)

    st.subheader("Cart Outcomes")

    outcome_data = (
        cart_data["abandoned"]
        .map({0: "Completed", 1: "Abandoned"})
        .value_counts()
    )

    st.bar_chart(outcome_data)

    st.caption(
        "These are descriptive statistics from the synthetic dataset, "
        "not measured results from a live business."
    )

# --------------------------------------------------
# 6. Cart abandonment prediction
# --------------------------------------------------

with cart_tab:
    st.subheader("Cart Abandonment Predictor")

    st.write(
        "Enter the shopping session details to estimate "
        "the probability of cart abandonment."
    )

    with st.form("cart_form"):
        c1, c2 = st.columns(2)

        with c1:
            cart_value = st.number_input(
                "Cart value (₹)",
                min_value=0.0,
                value=450.0,
                step=50.0
            )

            delivery_fee = st.number_input(
                "Delivery fee (₹)",
                min_value=0.0,
                value=40.0,
                step=5.0
            )

            items_in_cart = st.number_input(
                "Items in cart",
                min_value=1,
                max_value=100,
                value=3
            )

            out_of_stock_items = st.number_input(
                "Unavailable items",
                min_value=0,
                max_value=100,
                value=1
            )

        with c2:
            eta_minutes = st.number_input(
                "Estimated delivery time (minutes)",
                min_value=1,
                max_value=180,
                value=22
            )

            previous_orders = st.number_input(
                "Previous orders",
                min_value=0,
                value=1
            )

            discount_percent = st.number_input(
                "Discount (%)",
                min_value=0,
                max_value=100,
                value=0
            )

        submitted = st.form_submit_button("Predict Abandonment Risk")

    if submitted:
        cart_input = pd.DataFrame([{
            "cart_value": cart_value,
            "delivery_fee": delivery_fee,
            "items_in_cart": items_in_cart,
            "out_of_stock_items": out_of_stock_items,
            "eta_minutes": eta_minutes,
            "previous_orders": previous_orders,
            "discount_percent": discount_percent
        }])

        probability = cart_model.predict_proba(cart_input)[0, 1]

        st.subheader("Prediction Result")
        st.metric(
            "Estimated Abandonment Probability",
            f"{probability:.1%}"
        )

        st.progress(float(probability))

        if probability >= 0.70:
            st.warning(
                "High estimated risk. Review availability, "
                "delivery information and total order cost. "
                "Consider testing a suitable retention action."
            )
        elif probability >= 0.40:
            st.info(
                "Medium estimated risk. Consider clearer "
                "delivery information or an optional reminder."
            )
        else:
            st.success(
                "Low estimated risk. Continue the normal "
                "customer journey."
            )

# --------------------------------------------------
# 7. Inventory replenishment
# --------------------------------------------------

with stock_tab:
    st.subheader("Dark-Store Replenishment")

    st.write(
        "Select a product to estimate demand and calculate "
        "a basic replenishment recommendation."
    )

    products = sorted(inventory_data["product"].unique())

    selected_product = st.selectbox(
        "Select product",
        products
    )

    history = (
        inventory_data[
            inventory_data["product"] == selected_product
        ].sort_values("day")
    )

    latest = history.iloc[-1]
    recent_sales = history["daily_sales"].tail(7)

    current_stock = st.number_input(
        "Current stock (units)",
        min_value=0,
        value=int(latest["current_stock"]),
        step=1
    )

    incoming_stock = st.number_input(
        "Incoming stock (units)",
        min_value=0,
        value=0,
        step=1
    )

    lead_time = st.number_input(
        "Supplier lead time (days)",
        min_value=1,
        max_value=30,
        value=int(latest["supplier_lead_time"])
    )

    safety_stock = st.number_input(
        "Safety stock (units)",
        min_value=0,
        value=int(latest["safety_stock"])
    )

    inventory_input = pd.DataFrame([{
        "sales_lag_1": recent_sales.iloc[-1],
        "sales_lag_2": recent_sales.iloc[-2],
        "sales_lag_3": recent_sales.iloc[-3],
        "sales_7day_avg": recent_sales.mean(),
        "weekday": int(latest["day"]) % 7
    }])[inventory_features]

    predicted_demand = max(
        0,
        float(inventory_model.predict(inventory_input)[0])
    )

    reorder_point = (
        predicted_demand * lead_time + safety_stock
    )

    inventory_position = current_stock + incoming_stock
    order_quantity = int(
        np.ceil(max(0, reorder_point - inventory_position))
    )

    st.divider()

    m1, m2, m3 = st.columns(3)
    m1.metric("Forecast Daily Demand", f"{predicted_demand:.1f} units")
    m2.metric("Reorder Point", f"{reorder_point:.1f} units")
    m3.metric("Suggested Order", f"{order_quantity} units")

    if inventory_position <= reorder_point:
        st.warning(
            f"Replenishment alert: {selected_product} is at or "
            "below the calculated reorder point."
        )
    else:
        st.success(
            f"{selected_product} is currently above the "
            "calculated reorder point."
        )

    st.subheader("Inventory History")
    st.line_chart(
        history.set_index("day")["daily_sales"]
    )

    st.subheader("All Product Inventory")

    all_products = []

    for product in products:
        product_history = (
            inventory_data[
                inventory_data["product"] == product
            ].sort_values("day")
        )

        latest_row = product_history.iloc[-1]
        recent = product_history["daily_sales"].tail(7)

        forecast_input = pd.DataFrame([{
            "sales_lag_1": recent.iloc[-1],
            "sales_lag_2": recent.iloc[-2],
            "sales_lag_3": recent.iloc[-3],
            "sales_7day_avg": recent.mean(),
            "weekday": int(latest_row["day"]) % 7
        }])[inventory_features]

        demand = max(
            0,
            float(inventory_model.predict(forecast_input)[0])
        )

        point = (
            demand * latest_row["supplier_lead_time"]
            + latest_row["safety_stock"]
        )

        all_products.append({
            "Product": product,
            "Current Stock": int(latest_row["current_stock"]),
            "Forecast Demand": round(demand, 1),
            "Reorder Point": round(point, 1),
            "Status": (
                "Reorder Required"
                if latest_row["current_stock"] <= point
                else "Stock Sufficient"
            )
        })

    st.dataframe(
        pd.DataFrame(all_products),
        width="stretch",
        hide_index=True
    )

# --------------------------------------------------
# 8. About the project
# --------------------------------------------------

with about_tab:
    st.subheader("About QuickCommerce AI Manager")

    st.markdown("""
    **Business problem**

    Quick-commerce businesses need to improve customer conversion
    while maintaining product availability at dark stores.

    **AI module 1: Cart abandonment**

    - Logistic Regression and Random Forest classification
    - Estimated probability of cart abandonment
    - Illustrative retention recommendations

    **AI module 2: Inventory replenishment**

    - Random Forest demand forecasting
    - Seven-day moving-average baseline
    - Reorder-point calculation and stock alerts

    **Data and limitations**

    This academic prototype uses synthetic data. It is not
    connected to Blinkit or Zepto systems, and its outputs are
    not validated against real customer or inventory outcomes.

    **Responsible use**

    Predictions should support, not replace, management decisions.
    Any production deployment would require real authorized data,
    monitoring, privacy controls and operational validation.
    """)