from pathlib import Path



def pytest_sessionstart(session):
    root = Path("tests/fixtures/xbd/train")
    (root/"images").mkdir(parents=True, exist_ok=True); (root/"labels").mkdir(parents=True, exist_ok=True)
