''' scdata works without the plotting libraries (the plot extra). Run apart: it blocks their import '''
import subprocess
import sys

SCRIPT = '''
import sys
for module in ["matplotlib", "seaborn", "bokeh", "panel", "folium", "branca", "hvplot", "plotly"]:
    sys.modules[module] = None

from scdata import Device, Test

loaded = [name for name in ["matplotlib", "seaborn"] if sys.modules.get(name)]
assert Test.ts_plot.__name__ == "ts_plot"
try:
    # Unbound: no device needed (creating one requests the Smart Citizen API)
    Device.ts_plot(None)
except ImportError as error:
    print("ERROR", error)
print("LOADED", loaded)
'''


def test_scdata_without_plotting_libraries(tmp_path):
    env = {'HOME': str(tmp_path), 'XDG_CONFIG_HOME': str(tmp_path / 'config'), 'XDG_CACHE_HOME': str(tmp_path / 'cache'),
           'APPDATA': str(tmp_path / 'config'), 'PATH': ''}
    result = subprocess.run([sys.executable, '-c', SCRIPT], env=env, capture_output=True, text=True, timeout=300)

    assert result.returncode == 0, result.stderr
    assert 'ERROR ts_plot needs the plotting libraries: pip install "scdata[plot]"' in result.stdout
    assert 'LOADED []' in result.stdout
