#!/bin/bash
#
# Pre-install commands for read-the-docs
#
set -euxo pipefail

if [[ "${READTHEDOCS:-no}" == "no" ]]; then
  echo "This script is only intended to be run in the read-the-docs environment"
  exit 1
fi

# Try and fetch the shallow submodule
git submodule update --init --depth=1 --single-branch

# Ensure we get a version that lets us use groups
python3 -m pip install --upgrade pip
python3 -m pip install --group docs

# So since kaleido does not install chrome, we have to do it
# This is only necessary when it works, otherwise, skip it!
kaleido_get_chrome || true

# Cache MathJax for kaleido, docs/conf.py looks for it in <venv>/share/kaleido
# Kaleido otherwise downloads it for every figure (which may hang).
mathjax="$(python3 -c 'import sys; print(sys.prefix)')/share/kaleido/tex-svg.min.js"
mkdir -p "$(dirname "$mathjax")"
for url in \
    https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.min.js \
    https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js ; do
  curl -fsSL --connect-timeout 10 --max-time 60 -o "$mathjax" "$url" && break
  rm -f "$mathjax"
done
