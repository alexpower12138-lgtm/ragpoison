"""Streamlit frontend (WP8).

Planned modules:
    data_layer.py     pure Python: load CSVs/JSONL into plain structures (unit tested)
    charts.py         pure Python: figure construction, also exportable as PNG
    streamlit_app.py  layout only: five tabs, no business logic
    ui_check.py       headless verification through streamlit.testing.v1.AppTest

The data layer and the chart layer deliberately have no Streamlit import, so that
they can be tested on a machine without a browser.
"""

from __future__ import annotations
