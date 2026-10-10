''' scdata works without the plotting libraries (the plot extra). Run apart: it blocks their import '''
import os
import subprocess
import sys

SCRIPT = '''
import sys
for module in BLOCKED:
    sys.modules[module] = None

from scdata import Device, Test

loaded = [name for name in ["matplotlib", "seaborn", "bokeh", "panel"] if sys.modules.get(name)]
assert Test.ts_plot.__name__ == "ts_plot"
try:
    # Unbound: no device needed (creating one requests the Smart Citizen API)
    Device.ts_plot(None)
except ImportError as error:
    print("ERROR", error)
print("LOADED", loaded)
'''


import pytest


@pytest.mark.parametrize('blocked', [
    ["matplotlib", "seaborn", "bokeh", "panel", "folium", "branca", "hvplot", "plotly"],
    # bokeh and panel installed, matplotlib not: importing scdata must not load scdata.plot
    ["matplotlib", "seaborn"],
])
def test_scdata_without_plotting_libraries(tmp_path, blocked):
    # Python on Windows needs the system variables (SYSTEMROOT...): extend the environment
    env = dict(os.environ, HOME=str(tmp_path), XDG_CONFIG_HOME=str(tmp_path / 'config'),
               XDG_CACHE_HOME=str(tmp_path / 'cache'), APPDATA=str(tmp_path / 'config'))
    script = SCRIPT.replace('BLOCKED', repr(blocked))
    result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True, timeout=300)

    assert result.returncode == 0, result.stderr
    assert 'ERROR ts_plot needs the plotting libraries: pip install "scdata[plot]"' in result.stdout
    assert 'LOADED []' in result.stdout
