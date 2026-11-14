#!/bin/bash
# Runs smoke.mjs with Playwright from DEVMAP_PLAYWRIGHT (default
# ~/.local/share/devmap-browser: `npm install playwright && npx playwright
# install chromium` there) and, when that directory has a libs/ of
# unpacked system libraries, with them on the library path.
here=$(cd "$(dirname "$0")" && pwd)
export DEVMAP_PLAYWRIGHT=${DEVMAP_PLAYWRIGHT:-$HOME/.local/share/devmap-browser}
libs=$DEVMAP_PLAYWRIGHT/libs/usr/lib/x86_64-linux-gnu
[ -d "$libs" ] && export LD_LIBRARY_PATH=$libs:$LD_LIBRARY_PATH
exec node "$here/smoke.mjs" "$@"
