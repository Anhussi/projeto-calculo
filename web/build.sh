#!/bin/sh
set -e
rm -rf dist
mkdir -p dist/.streamlit dist/apresentacao/shots
cp web/index.html dist/
cp app.py engine.py surto.py relatorio.py dist/
cp .streamlit/config.toml dist/config.toml
cp apresentacao/index.html dist/apresentacao/
cp apresentacao/shots/*.png dist/apresentacao/shots/