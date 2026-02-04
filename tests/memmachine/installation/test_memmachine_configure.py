from pathlib import Path
from unittest.mock import patch

import pytest

from memmachine.installation.memmachine_configure import (
    ConfigurationWizard,
    LinuxInstaller,
    MacosInstaller,
    WindowsInstaller,
)


def mock_wizard_run(self) -> None:
    Path(self.args.destination).mkdir(parents=True, exist_ok=True)
    Path(self.args.destination, "cfg.yml").touch()


def mock_wizard_init(self, args: ConfigurationWizard.Params) -> None:
    self.args = args


@pytest.fixture
def mock_wizard():
    with (
        patch.object(ConfigurationWizard, "__init__", mock_wizard_init),
        patch.object(ConfigurationWizard, "run_wizard", mock_wizard_run),
    ):
        yield


@pytest.mark.parametrize(
    "installer_cls", [LinuxInstaller, MacosInstaller, WindowsInstaller]
)
def test_install_creates_config(installer_cls, mock_wizard, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    installer = installer_cls()
    installer.install()
    config_path = Path("~/.config/memmachine/cfg.yml").expanduser()
    assert config_path.exists()
