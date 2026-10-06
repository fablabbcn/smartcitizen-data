import pytest
from types import SimpleNamespace
from scdata.test.export.to_file import to_html

def test_to_html_disabled():
    with pytest.raises(NotImplementedError, match='old test descriptor'):
        to_html(SimpleNamespace(name='t', path='/tmp'))
