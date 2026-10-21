# Quick automatic check of app.py without opening a browser.
# Run from a Jupyter cell with:  !python test_app.py
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py", default_timeout=90).run()
assert not at.exception, at.exception
assert not at.error, [e.value for e in at.error]
print("App loaded OK. Tabs:", [t.label for t in at.tabs])
print("Overview:", [(m.label, m.value) for m in at.metric[:4]])

at.button[0].click().run()                      # press "Predict Abandonment Risk"
assert not at.exception, at.exception
print("Cart predictor works:", [m.value for m in at.metric if "Probability" in m.label])

product_box = [s for s in at.selectbox if s.label == "Select product"][0]
for product in product_box.options[:5]:         # try a few products
    product_box.select(product).run()
    assert not at.exception, at.exception
    print(f"  {product}: {[m.value for m in at.metric[-3:]]}")
    product_box = [s for s in at.selectbox if s.label == "Select product"][0]
print("ALL CHECKS PASSED")
