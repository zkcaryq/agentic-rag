from app.utils.path_util import PROJECT_ROOT, get_project_root


def test_project_root_uses_tracked_marker() -> None:
    root = get_project_root()

    assert root == PROJECT_ROOT
    assert (root / "pyproject.toml").is_file()
