# Quick automatic check of app.py without opening a browser.
# Run from a Jupyter cell with:  !python test_app.py
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py", default_timeout=60).run()
assert not at.exception, at.exception
print("App loaded OK. Tabs:", [t.label for t in at.tabs])
print("Overview:", [(m.label, m.value) for m in at.metric[:4]])

at.button[0].click().run()                      # press "Predict Abandonment Risk"
assert not at.exception, at.exception
print("Cart predictor works.")

for product in at.selectbox[0].options[:5]:     # try a few products
    at.selectbox[0].select(product).run()
    assert not at.exception, at.exception
    print(f"  {product}: {[m.value for m in at.metric[-3:]]}")
print("ALL CHECKS PASSED")
