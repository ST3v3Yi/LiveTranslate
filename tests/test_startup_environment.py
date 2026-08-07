from pathlib import Path


READY_MARKER = ".venv\\.livetranslate-ready"


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").lower()


def test_source_launcher_rejects_an_incomplete_environment():
    launcher = _read("start.bat")

    assert READY_MARKER in launcher
    assert "setup is incomplete" in launcher


def test_install_and_update_only_mark_verified_environments_ready():
    installer = _read("install.ps1")
    updater = _read("update.bat")

    assert READY_MARKER in installer
    assert installer.rindex("set-content -literalpath $readymarker") > installer.rindex(
        "pip check"
    )
    assert updater.rindex(f'> "{READY_MARKER}" echo') > updater.rindex("pip check")


def test_portable_launcher_repairs_interrupted_bootstraps():
    builder = _read("build_release.ps1")

    assert READY_MARKER in builder
    assert "--allow-existing" in builder
    assert "pip check --python $py" in builder
    assert builder.rindex("set-content -literalpath $ready") > builder.rindex(
        "pip check --python $py"
    )
