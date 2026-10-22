"""First-run configuration must work without pre-created persistence folders."""
from pathlib import Path

import pytest

from odyssey.config import get_config_dir, get_data_dir, load_settings
from odyssey.storage import db


def test_new_nested_data_directory_supports_real_sqlite(monkeypatch, tmp_path):
    target = tmp_path / 'new' / 'data'
    monkeypatch.setenv('ODYSSEY_DATA_DIR', str(target))
    monkeypatch.setattr(db, '_conn', None)
    try:
        task_id = db.add_task('first launch')
        assert db.get_task(task_id)['title'] == 'first launch'
        assert (target / 'odyssey.db').is_file()
    finally:
        if db._conn is not None:
            db._conn.close()
            db._conn = None


def test_config_directory_is_ready_for_history_and_config(monkeypatch, tmp_path):
    target = tmp_path / 'new' / 'config'
    monkeypatch.setenv('ODYSSEY_CONFIG_DIR', str(target))
    monkeypatch.setenv('ODYSSEY_DATA_DIR', str(tmp_path / 'data'))
    path = get_config_dir()
    (path / 'config.toml').write_text('fast_model = "custom-model"\n')
    assert load_settings().fast_model == 'custom-model'
    assert get_config_dir() == path  # repeated calls preserve existing content


@pytest.mark.parametrize('variable,getter', [
    ('ODYSSEY_DATA_DIR', get_data_dir), ('ODYSSEY_CONFIG_DIR', get_config_dir),
])
def test_override_expands_home(monkeypatch, tmp_path, variable, getter):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv(variable, '~/odyssey/nested')
    assert getter() == tmp_path / 'odyssey' / 'nested'
    assert getter().is_dir()


@pytest.mark.parametrize('variable,getter', [
    ('ODYSSEY_DATA_DIR', get_data_dir), ('ODYSSEY_CONFIG_DIR', get_config_dir),
])
def test_existing_file_is_not_overwritten(monkeypatch, tmp_path, variable, getter):
    path = tmp_path / 'not-a-directory'
    path.write_text('preserve me')
    monkeypatch.setenv(variable, str(path))
    with pytest.raises(OSError):
        getter()
    assert path.read_text() == 'preserve me'
