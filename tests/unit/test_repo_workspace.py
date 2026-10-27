"""Tests for the session-scoped, jail-sandboxed repository workspace.

Every fixture builds a *local* git repository inside ``tmp_path`` and clones it
with ``allow_local=True``; no test touches the network and no test writes into
an existing checkout.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

from tensortruth.services import repo_workspace as rw

GIT_ENV = {
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "LC_ALL": "C",
}

README_TEXT = "# Demo repo\n\nHello TensorTruth.\nSecond line.\n"
MAIN_TEXT = (
    "def main():\n"
    "    # TODO: wire this up\n"
    "    value = compute(2)\n"
    "    return value\n"
    "\n"
    "\n"
    "def compute(x):\n"
    "    return x * 2\n"
)


def _git(cwd: Path, *args: str) -> None:
    env = dict(os.environ)
    env.update(GIT_ENV)
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _make_repo(base: Path, name: str = "source-repo", commits: int = 2) -> Path:
    """Create a small committed git repository at ``base/name``."""
    repo = base / name
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "commit.gpgsign", "false")

    (repo / "README.md").write_text(README_TEXT, encoding="utf-8")
    src = repo / "src"
    src.mkdir()
    (src / "main.py").write_text(MAIN_TEXT, encoding="utf-8")
    (src / "util.py").write_text("VALUE = 1\n", encoding="utf-8")
    docs = repo / "docs"
    docs.mkdir()
    (docs / "guide.rst").write_text("Guide\n=====\n", encoding="utf-8")
    data = repo / "data"
    data.mkdir()
    (data / "blob.bin").write_bytes(b"\x00\x01\x02BINARY\x00")
    # A symlink committed on purpose: it must never be followed out of the jail.
    os.symlink("/etc/passwd", repo / "escape")

    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "initial commit")
    for index in range(1, commits):
        (repo / f"changelog{index}.md").write_text(
            f"# change {index}\n", encoding="utf-8"
        )
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", f"change {index}")
    return repo


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    """A local git repository that tests may clone from."""
    return _make_repo(tmp_path, "source-repo")


@pytest.fixture
def scope(tmp_path: Path) -> Path:
    """An empty session scope directory."""
    path = tmp_path / "scope"
    path.mkdir()
    return path


@pytest.fixture
def cloned(scope: Path, source_repo: Path):
    """The default single repository, cloned into the scope."""
    record = rw.clone_repo(scope, str(source_repo), allow_local=True)
    _, root = rw.find_repo(scope, record.name)
    return record, root


# ---------------------------------------------------------------------------
# clone / list / find
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClone:
    def test_clone_writes_metadata_and_derives_sha(self, scope, source_repo):
        record = rw.clone_repo(scope, str(source_repo), allow_local=True)

        assert record.name == "local/source-repo"
        assert record.slug == "local__source-repo"
        assert record.shallow is True
        assert record.branch == "main"
        assert len(record.sha) == 40
        assert record.short_sha == record.sha[:8]

        meta = rw.repos_dir(scope) / record.slug / rw.META_FILENAME
        assert meta.is_file()
        payload = rw.RepoRecord.from_dict(json.loads(meta.read_text(encoding="utf-8")))
        assert payload.sha == record.sha
        # The metadata file lives beside the checkout, never inside it.
        checkout_meta = (
            rw.repos_dir(scope) / record.slug / rw.CLONE_DIRNAME / rw.META_FILENAME
        )
        assert not checkout_meta.exists()

    def test_reclone_replaces_previous_checkout(self, scope, source_repo):
        first = rw.clone_repo(scope, str(source_repo), allow_local=True)
        second = rw.clone_repo(scope, str(source_repo), allow_local=True)
        assert first.slug == second.slug
        assert len(rw.list_repos(scope)) == 1

    def test_rejects_unknown_host_without_allow_local(self, scope):
        with pytest.raises(rw.RepoError) as excinfo:
            rw.clone_repo(scope, "https://gitlab.com/owner/repo.git")
        assert "gitlab.com" in str(excinfo.value)
        assert "not allowed" in str(excinfo.value)

    def test_rejects_malformed_url(self, scope):
        with pytest.raises(rw.RepoError):
            rw.clone_repo(scope, "https://github.com/onlyowner")


@pytest.mark.unit
class TestRegistry:
    def test_list_repos_is_empty_for_missing_dir(self, scope):
        assert rw.list_repos(scope) == []
        assert not rw.repos_dir(scope).exists()

    def test_list_repos_skips_corrupt_metadata(self, scope, source_repo):
        rw.clone_repo(scope, str(source_repo), allow_local=True)
        broken = rw.repos_dir(scope) / "broken__repo"
        broken.mkdir()
        (broken / rw.META_FILENAME).write_text("{not json", encoding="utf-8")

        assert [record.name for record in rw.list_repos(scope)] == ["local/source-repo"]

    def test_list_repos_sorted_by_clone_time(self, tmp_path, scope):
        first_src = _make_repo(tmp_path, "alpha")
        second_src = _make_repo(tmp_path, "beta")
        rw.clone_repo(scope, str(first_src), allow_local=True)
        rw.clone_repo(scope, str(second_src), allow_local=True)

        names = [record.name for record in rw.list_repos(scope)]
        assert names == ["local/alpha", "local/beta"]

    def test_find_repo_by_slug_and_by_name(self, scope, cloned):
        record, root = cloned
        by_slug, slug_root = rw.find_repo(scope, record.slug)
        by_name, name_root = rw.find_repo(scope, record.name)
        assert by_slug.slug == by_name.slug == record.slug
        assert slug_root == name_root == root

    def test_find_repo_none_resolves_single(self, scope, cloned):
        record, root = cloned
        resolved, resolved_root = rw.find_repo(scope, None)
        assert resolved.slug == record.slug
        assert resolved_root == root

    def test_find_repo_none_is_ambiguous_with_two(self, tmp_path, scope, source_repo):
        rw.clone_repo(scope, str(source_repo), allow_local=True)
        other = _make_repo(tmp_path, "other-repo")
        rw.clone_repo(scope, str(other), allow_local=True)

        with pytest.raises(rw.RepoError) as excinfo:
            rw.find_repo(scope, None)
        message = str(excinfo.value)
        assert "local/source-repo" in message
        assert "local/other-repo" in message

    def test_find_repo_unknown_name_lists_available(self, scope, cloned):
        record, _ = cloned
        with pytest.raises(rw.RepoError) as excinfo:
            rw.find_repo(scope, "nope/nothing")
        assert "nope/nothing" in str(excinfo.value)
        assert record.name in str(excinfo.value)

    def test_drop_repo_removes_directory(self, scope, cloned):
        record, _root = cloned
        dropped = rw.drop_repo(scope, record.name)
        assert dropped.slug == record.slug
        assert not (rw.repos_dir(scope) / record.slug).exists()
        assert rw.list_repos(scope) == []
        with pytest.raises(rw.RepoError):
            rw.drop_repo(scope, record.name)


# ---------------------------------------------------------------------------
# prompt section
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPromptSection:
    def test_empty_when_no_repos(self, scope):
        assert rw.render_prompt_section(scope) == ""

    def test_lists_repo_with_sha_and_hint(self, scope, cloned):
        record, _root = cloned
        section = rw.render_prompt_section(scope)
        assert section.startswith("CLONED REPOSITORIES")
        assert record.short_sha in section
        assert "branch main" in section
        assert 'inspect(action="read", repo="local/source-repo"' in section
        assert "files in" in section and "dirs" in section

    def test_reports_remaining_repositories(self, tmp_path, scope, source_repo):
        rw.clone_repo(scope, str(source_repo), allow_local=True)
        rw.clone_repo(scope, str(_make_repo(tmp_path, "second")), allow_local=True)
        section = rw.render_prompt_section(scope, max_repos=1)
        assert "and 1 more" in section


# ---------------------------------------------------------------------------
# tree
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTree:
    def test_header_dir_suffix_and_sizes(self, cloned):
        record, root = cloned
        output = rw.tree(record, root)
        first_line = output.splitlines()[0]
        assert first_line.startswith(f"Repository {record.name} @ {record.short_sha}")
        assert " entries" in first_line
        assert "src/" in output
        size = len(README_TEXT.encode("utf-8"))
        assert f"README.md  {size} B" in output

    def test_depth_limits_recursion(self, cloned):
        record, root = cloned
        shallow = rw.tree(record, root, depth=1)
        deep = rw.tree(record, root, depth=3)
        assert "src/" in shallow
        assert "main.py" not in shallow
        assert "main.py" in deep

    def test_subdirectory_path(self, cloned):
        record, root = cloned
        output = rw.tree(record, root, path="src")
        assert "main.py" in output
        assert "README.md" not in output

    def test_entry_limit_marker(self, cloned):
        record, root = cloned
        output = rw.tree(record, root, max_entries=2)
        assert "[truncated: entry limit]" in output
        assert "2 of " in output.splitlines()[0]

    def test_missing_path_raises(self, cloned):
        record, root = cloned
        with pytest.raises(rw.RepoError):
            rw.tree(record, root, path="does/not/exist")


# ---------------------------------------------------------------------------
# read_text
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReadText:
    def test_numbering_and_header(self, cloned):
        record, root = cloned
        output = rw.read_text(record, root, "src/main.py")
        lines = output.splitlines()
        assert lines[0].startswith("File: src/main.py @ ")
        size = len(MAIN_TEXT.encode("utf-8"))
        assert f"(8 lines, {size} bytes)" in lines[0]
        assert lines[1] == "    1| def main():"
        assert lines[2] == "    2|     # TODO: wire this up"

    def test_start_line_and_truncation_note(self, cloned):
        record, root = cloned
        output = rw.read_text(record, root, "src/main.py", start_line=3, max_lines=2)
        lines = output.splitlines()
        assert lines[1].startswith("    3| ")
        assert lines[2].startswith("    4| ")
        assert "[truncated: showing lines 3-4 of 8" in output
        assert "start_line=5" in output

    def test_perfect_slice_has_no_truncation(self, cloned):
        record, root = cloned
        output = rw.read_text(record, root, "README.md")
        assert "[truncated" not in output

    def test_binary_file_rejected(self, cloned):
        record, root = cloned
        with pytest.raises(rw.RepoError) as excinfo:
            rw.read_text(record, root, "data/blob.bin")
        assert "binary" in str(excinfo.value)

    def test_directory_rejected(self, cloned):
        record, root = cloned
        with pytest.raises(rw.RepoError) as excinfo:
            rw.read_text(record, root, "src")
        assert "directory" in str(excinfo.value)

    def test_missing_file_rejected(self, cloned):
        record, root = cloned
        with pytest.raises(rw.RepoError):
            rw.read_text(record, root, "src/nope.py")

    def test_oversized_file_rejected(self, cloned, monkeypatch):
        record, root = cloned
        monkeypatch.setattr(rw, "MAX_READ_BYTES", 10)
        with pytest.raises(rw.RepoError) as excinfo:
            rw.read_text(record, root, "README.md")
        assert "read limit" in str(excinfo.value)


# ---------------------------------------------------------------------------
# jail
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestJail:
    @pytest.mark.parametrize(
        "bad_path",
        [
            "../etc/passwd",
            "../../../../etc/passwd",
            "/etc/passwd",
            "escape",
            ".git/config",
        ],
    )
    def test_escaping_paths_are_rejected(self, cloned, bad_path):
        record, root = cloned
        with pytest.raises(rw.RepoError):
            rw.read_text(record, root, bad_path)

    def test_nul_byte_rejected(self, cloned):
        record, root = cloned
        with pytest.raises(rw.RepoError) as excinfo:
            rw.read_text(record, root, "src/main.py\x00.txt")
        assert "NUL" in str(excinfo.value)

    def test_symlink_escape_rejected_for_tree(self, cloned):
        record, root = cloned
        with pytest.raises(rw.RepoError):
            rw.tree(record, root, path="escape")

    def test_grep_does_not_follow_escaping_symlink(self, cloned):
        record, root = cloned
        # 'root:' appears in /etc/passwd; the symlink must never be read.
        with pytest.raises(rw.RepoError):
            rw.grep(record, root, "root:", path="escape")
        assert "escape:" not in rw.grep(record, root, "root")

    def test_git_path_escape_rejected(self, cloned):
        record, root = cloned
        with pytest.raises(rw.RepoError):
            rw.git(record, root, "blame", path="../outside.txt")


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGrep:
    def test_finds_matches_with_relpath_and_lineno(self, cloned):
        record, root = cloned
        output = rw.grep(record, root, "TODO")
        assert "src/main.py:2: # TODO: wire this up" in output
        assert "— 1 matches in 1 files" in output

    def test_case_sensitivity(self, cloned):
        record, root = cloned
        assert "No matches for todo" not in rw.grep(
            record, root, "todo", case_sensitive=False
        )
        assert rw.grep(record, root, "todo", case_sensitive=True).startswith(
            "No matches"
        )

    def test_no_matches_message(self, cloned):
        record, root = cloned
        assert rw.grep(record, root, "zzz-not-here") == "No matches for zzz-not-here"

    def test_match_limit_marker(self, cloned):
        record, root = cloned
        output = rw.grep(record, root, "e", max_matches=2)
        assert "[truncated: match limit]" in output
        assert "— 2 matches" in output

    def test_invalid_regex_raises_with_reason(self, cloned):
        record, root = cloned
        with pytest.raises(rw.RepoError) as excinfo:
            rw.grep(record, root, "([unclosed")
        assert "Invalid regular expression" in str(excinfo.value)

    def test_skips_binary_files(self, cloned):
        record, root = cloned
        assert "blob.bin" not in rw.grep(record, root, "BINARY")


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGit:
    def test_log_lists_commits(self, cloned):
        record, root = cloned
        output = rw.git(record, root, "log")
        assert "initial commit" in output
        assert "change 1" in output
        assert record.sha[:7] in output

    def test_show_head(self, cloned):
        record, root = cloned
        output = rw.git(record, root, "show", rev="HEAD", path="changelog1.md")
        assert "change 1" in output
        assert record.sha in output

    def test_blame_requires_path(self, cloned):
        record, root = cloned
        with pytest.raises(rw.RepoError) as excinfo:
            rw.git(record, root, "blame")
        assert "path" in str(excinfo.value)

    def test_blame_works_with_path(self, cloned):
        record, root = cloned
        assert "def main" in rw.git(record, root, "blame", path="src/main.py")

    def test_unsupported_action_rejected(self, cloned):
        record, root = cloned
        with pytest.raises(rw.RepoError) as excinfo:
            rw.git(record, root, "run")
        assert "Unsupported git action" in str(excinfo.value)

    @pytest.mark.parametrize("bad_rev", ["--upload-pack=evil", "-n1", "a b", "x" * 200])
    def test_invalid_revision_rejected(self, cloned, bad_rev):
        record, root = cloned
        with pytest.raises(rw.RepoError):
            rw.git(record, root, "log", rev=bad_rev)

    def test_output_is_capped(self, cloned):
        record, root = cloned
        output = rw.git(record, root, "log", max_lines=1)
        assert "[truncated: showing 1 of" in output


@pytest.mark.unit
class TestUrlHandling:
    def test_github_url_is_normalised(self):
        clone_url, owner, repo, safe_url, had_credentials = rw._parse_remote_url(
            "https://github.com/owner/repo", allow_local=False
        )
        assert (owner, repo) == ("owner", "repo")
        assert clone_url == safe_url == "https://github.com/owner/repo.git"
        assert had_credentials is False

    def test_scp_style_url_is_accepted(self):
        clone_url, owner, repo, safe_url, _ = rw._parse_remote_url(
            "git@github.com:owner/repo.git", allow_local=False
        )
        assert (owner, repo) == ("owner", "repo")
        assert clone_url.startswith("ssh://git@github.com/")
        assert "@" not in safe_url.replace("git@", "")

    def test_token_url_is_split_from_safe_url(self):
        clone_url, _owner, _repo, safe_url, had_credentials = rw._parse_remote_url(
            "https://x-access-token:supersecret@github.com/owner/repo.git",
            allow_local=False,
        )
        assert "supersecret" in clone_url
        assert "supersecret" not in safe_url
        assert had_credentials is True
        assert safe_url == "https://github.com/owner/repo.git"

    def test_strip_credentials_is_idempotent(self):
        assert (
            rw._strip_credentials("https://user:tok@github.com/o/r.git")
            == "https://github.com/o/r.git"
        )
        assert rw._strip_credentials("https://github.com/o/r.git") == (
            "https://github.com/o/r.git"
        )


@pytest.mark.unit
class TestRecord:
    def test_to_dict_round_trip(self):
        record = rw.RepoRecord(
            name="owner/repo",
            slug="owner__repo",
            owner="owner",
            repo="repo",
            url="https://github.com/owner/repo.git",
            sha="a" * 40,
            branch="main",
            cloned_at=1234.5,
            shallow=True,
        )
        assert rw.RepoRecord.from_dict(record.to_dict()) == record
        assert record.short_sha == "a" * 8
