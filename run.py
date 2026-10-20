#!/usr/bin/env python
"""Command-line entry point.

Usage:
    python run.py --help
    python run.py secret set
    python run.py selftest
"""

from __future__ import annotations

import sys

from ragpoison.cli import main

if __name__ == "__main__":
    sys.exit(main())
