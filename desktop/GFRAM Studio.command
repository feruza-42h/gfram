#!/bin/zsh
# GFRAM Studio from the source repository (developer machine)
cd "$(dirname "$0")/.."
exec venv/bin/python -m gfram.studio
