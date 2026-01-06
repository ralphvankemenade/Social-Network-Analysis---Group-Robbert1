"""Expose Streamlit page modules with valid names.

The page files are named with numeric prefixes (e.g. `1_upload_overview.py`) to
control their ordering in the repository.  Python identifiers cannot start
with digits, so they cannot be imported directly using standard syntax.  To
work around this, we import each module using `importlib` and assign it to
a variable whose name begins with an underscore.  The top‑level `app.py`
then imports these underscore variables to access the `page()` functions.
"""

from importlib import import_module

_1_upload_overview = import_module("dss.pages.1_upload_overview")
_2_centrality = import_module("dss.pages.2_centrality")
_3_roles = import_module("dss.pages.3_roles")
_4_communities_robustness = import_module("dss.pages.4_communities_robustness")
_5_kemeny_interactive = import_module("dss.pages.5_kemeny_interactive")
_6_arrest_optimization = import_module("dss.pages.6_arrest_optimization")
_7_user_manual = import_module("dss.pages.7_user_manual")