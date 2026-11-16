"""Shared fixtures. pytest.ini puts the repo root on sys.path, so `bot`, `vsai`,
`ingest` import as packages. The bot only opens the vector DB in main()."""
import pytest


@pytest.fixture(scope="session")
def watchdog():
    from bot import watchdog as wd

    return wd
