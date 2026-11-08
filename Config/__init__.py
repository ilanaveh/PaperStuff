"""Machine-local path configuration."""

from importlib import import_module

try:
    path_config = import_module('.path_config', __name__)
except ModuleNotFoundError as exc:
    if exc.name != __name__ + '.path_config':
        raise
    raise ModuleNotFoundError(
        'Create Config/path_config.py by copying Config/path_config.example.py, '
        'then edit its paths for this computer.'
    ) from exc
