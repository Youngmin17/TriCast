"""TriCast Studio: compare native and emulated model outputs (see app/README.md)."""

import sys

# requires-python is >= 3.10, but a venv made with macOS /usr/bin/python3 (3.9) still installs the page's deps
# and then fails inside pydantic on `str | None`; say so in one line instead.
if sys.version_info < (3, 10):  # noqa: UP036
    raise SystemExit(f"TriCast needs Python 3.10 or newer; this is Python {sys.version.split()[0]} "
                     f"({sys.executable}). On macOS /usr/bin/python3 is 3.9: install a newer Python "
                     "(for example `brew install python@3.12`) and create the virtual environment with it.")
