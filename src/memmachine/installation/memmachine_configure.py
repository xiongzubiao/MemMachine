import logging
import logging
import sys
from pathlib import Path

from memmachine import setup_nltk
from memmachine.installation.configuration_wizard import ConfigurationWizard

logger = logging.getLogger("MemMachineInstaller")


def get_memmachine_config_dir() -> str:
    """Get the MemMachine configuration directory path."""
    return str(Path("~/.config/memmachine").expanduser())


class Installer:
    """Configuration-only installer for MemMachine."""

    def install(self, prompt: bool = True) -> None:
        """Generate the MemMachine configuration file."""
        wizard_args = ConfigurationWizard.Params(
            destination=get_memmachine_config_dir(),
            prompt=prompt,
        )
        wizard = ConfigurationWizard(wizard_args)
        wizard.run_wizard()
        logger.info("MemMachine configuration completed.")


class LinuxInstaller(Installer):
    """Linux installer entry point."""


class MacosInstaller(Installer):
    """macOS installer entry point."""


class WindowsInstaller(Installer):
    """Windows installer entry point."""


def install_memmachine() -> None:
    """Install and configure MemMachine."""
    Installer().install()


def main() -> None:
    """Execute the MemMachine configuration script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    try:
        install_memmachine()
        setup_nltk()
        logger.info("MemMachine setup complete.")
        logger.info("Use 'memmachine-server' to start the API server.")
    except KeyboardInterrupt:
        logger.warning("Configuration cancelled by user.")
        sys.exit(130)
    except RuntimeError:
        logger.exception("Configuration failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
