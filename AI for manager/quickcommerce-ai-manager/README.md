# QuickCommerce AI Manager

Blinkit / Zepto – Cart Abandonment & Dark-Store Replenishment Triggers.
Course project for AI Foundations for Managers (Semester 1, 2026). All data is synthetic.

- **Cart Predictor** – class-balanced Logistic Regression scores a cart's abandonment risk and suggests an action.
- **Replenishment Triggers** – Random Forest demand forecast + reorder point with safety stock.

Run locally:

```
pip install -r requirements.txt
streamlit run app.py
```

The data files and models are created by the project notebook `QuickCommerce_Cart_Abandonment_Replenishment.ipynb`.
