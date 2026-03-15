# Vulture whitelist — false positives that should not be flagged.
#
# These are variables/parameters required by framework callback signatures
# (SQLAlchemy event listeners, structlog processors) or code that vulture
# cannot statically determine is reachable.

# SQLAlchemy event.listens_for("connect") requires this parameter
connection_record  # noqa

# structlog processor/renderer callbacks require method_name parameter
method_name  # noqa

# vulture cannot see that `while True: ... break` makes trailing code reachable
_ = "fix.py:258 — code after while-True with break is reachable"
