''' Metadata urls from BASE_POSTPROCESSING_URL. scdata reads its configuration on import: run it apart '''
import functools
import http.server
import json
import os
import subprocess
import sys
import threading
from os.path import join

import pytest
import yaml

from conftest import ROOT

SCRIPT = '''
import json
from scdata._config import config
print(json.dumps({
    "calibrations_urls": config.calibrations_urls,
    "blueprints_urls": config.blueprints_urls,
    "blueprints": sorted(config.blueprints),
    "calibrations": sorted(config.calibrations),
}))
'''


@pytest.fixture
def metadata_server(tmp_path):
    ''' Serves blueprints and calibrations like flows, recording the requested paths '''
    root = tmp_path / 'server'
    api = root / 'api' / 'v1'
    (api / 'blueprints').mkdir(parents=True)
    (api / 'calibrations').mkdir()
    (api / 'blueprints' / 'sc_air.json').write_text(open(join(ROOT, 'blueprints', 'sc_air.json')).read())
    (api / 'calibrations' / 'calibrations.json').write_text(json.dumps({'10-002911': {'t20': 20, 'v20': '0.3'}}))

    requested = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            requested.append(self.path)

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{server.server_port}/api/v1', requested
    server.shutdown()


def run_scdata(tmp_path, base_url):
    # Configuration folder: XDG_CONFIG_HOME, or APPDATA on Windows
    env = dict(os.environ, BASE_POSTPROCESSING_URL=base_url, APPDATA=str(tmp_path / 'config'),
               XDG_CONFIG_HOME=str(tmp_path / 'config'), XDG_CACHE_HOME=str(tmp_path / 'cache'))
    result = subprocess.run([sys.executable, '-c', SCRIPT], env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_metadata_from_base_url(tmp_path, metadata_server):
    base_url, requested = metadata_server
    # Urls saved in config.yaml by older versions are ignored
    (tmp_path / 'config' / 'scdata').mkdir(parents=True)
    (tmp_path / 'config' / 'scdata' / 'config.yaml').write_text(yaml.dump({
        'calibrations_urls': ['https://raw.githubusercontent.com/fablabbcn/smartcitizen-data/old/calibrations.json'],
        'blueprints_urls': ['https://raw.githubusercontent.com/fablabbcn/smartcitizen-data/old/sc_air.json'],
    }))

    result = run_scdata(tmp_path, base_url)

    assert result['calibrations_urls'] == [f'{base_url}/calibrations/calibrations.json']
    assert result['blueprints_urls'] == [f'{base_url}/blueprints/sc_air.json']
    assert result['blueprints'] == ['sc_air']
    assert result['calibrations'] == ['10-002911']
    assert requested == ['/api/v1/blueprints/sc_air.json', '/api/v1/calibrations/calibrations.json']

    saved = yaml.safe_load((tmp_path / 'config' / 'scdata' / 'config.yaml').read_text())
    assert not {'calibrations_urls', 'blueprints_urls', 'names_urls'} & set(saved)


def test_saved_config_keeps_only_overrides(tmp_path, metadata_server):
    base_url, _ = metadata_server
    custom = tmp_path / 'my-raw'
    custom.mkdir()
    (tmp_path / 'config' / 'scdata').mkdir(parents=True)
    # As saved by older versions: every setting, with paths of another machine
    (tmp_path / 'config' / 'scdata' / 'config.yaml').write_text(yaml.dump({
        'data': {'cached_data_margin': '5Min'},
        'framework': 'script',
        'paths': {'config': '/nowhere/config', 'data': '/nowhere/data', 'raw': str(custom)},
        'zenodo_real_base_url': 'https://zenodo.org',
    }))
    script = SCRIPT.replace('print(json.dumps({', 'print(json.dumps({"data": config.data, "paths": config.paths,')

    env = dict(os.environ, BASE_POSTPROCESSING_URL=base_url, APPDATA=str(tmp_path / 'config'),
               XDG_CONFIG_HOME=str(tmp_path / 'config'), XDG_CACHE_HOME=str(tmp_path / 'cache'))
    result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    result = json.loads(result.stdout.strip().splitlines()[-1])

    # Keys added by later versions keep their default
    assert result['data'] == {'cached_data_margin': '5Min', 'reload_metadata': True}
    # Paths that do not exist are ignored, the config folder is where the file is
    # Windows keeps data in APPDATA too
    cache = tmp_path / ('config' if sys.platform == 'win32' else 'cache')
    assert result['paths']['data'] == str(cache / 'scdata')
    assert result['paths']['config'] == str(tmp_path / 'config' / 'scdata')
    assert result['paths']['raw'] == str(custom)
    saved = yaml.safe_load((tmp_path / 'config' / 'scdata' / 'config.yaml').read_text())
    assert saved == {'data': {'cached_data_margin': '5Min'}, 'paths': {'raw': str(custom)}}


def test_fresh_config_saves_nothing(tmp_path, metadata_server):
    base_url, _ = metadata_server

    run_scdata(tmp_path, base_url)

    assert yaml.safe_load((tmp_path / 'config' / 'scdata' / 'config.yaml').read_text()) == {}
