"""Session-scoped, jail-sandboxed git repository workspace.

This module is the state + sandbox layer behind the read-only ``inspect`` tool.
A chat session clones a repository once (``clone_repo``) and the agent then
inspects it through a fixed set of actions (``tree``, ``read_text``, ``grep``,
``git``).  Every function is *synchronous*: the async caller is expected to
wrap the calls in ``asyncio.to_thread``.

Security model
--------------
* NO SHELL ANYWHERE.  There is no ``shell=True``, no ``os.system``, no
  ``os.popen``, no ``eval``/``exec`` and no command-string building in this
  module.  The only subprocess use is ``git`` (clone / rev-parse / remote
  set-url and the ``git`` inspection action) and it always receives an argv
  *list* with ``shell=False``.
* Every user-supplied relative path is funnelled through exactly one helper,
  ``_resolve_in_root``, which rejects NUL bytes, absolute paths, ``..`` escapes,
  symlinks that resolve outside the root and anything under ``.git/``.
* No function writes *inside the clone*.  The only files written are the
  per-repository metadata file and the clone itself, both inside
  ``<scope>/repos/<slug>/``; the checkout lives in ``<scope>/repos/<slug>/repo``
  so that metadata never pollutes the work tree.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPOS_DIRNAME = "repos"
META_FILENAME = ".repo_meta.json"

MAX_REPO_BYTES = 800 * 1024 * 1024
MAX_READ_BYTES = 512 * 1024
MAX_READ_LINES = 2000
MAX_GREP_MATCHES = 200
MAX_TREE_ENTRIES = 2000

ALLOWED_GIT_HOSTS = {"github.com"}

#: Name of the checkout directory inside ``<scope>/repos/<slug>/``.
CLONE_DIRNAME = "repo"

_GIT_ENV = {
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "LC_ALL": "C",
    "GIT_PAGER": "cat",
}

_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_REV_RE = re.compile(r"^[A-Za-z0-9_./@~^:-]{1,160}$")
_SCHEME_URL_RE = re.compile(
    r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*)://"
    r"(?:(?P<userinfo>[^@/]*)@)?"
    r"(?P<host>[^/:]+)"
    r"(?::(?P<port>\d+))?"
    r"(?P<path>/.*)$"
)
_SCP_URL_RE = re.compile(
    r"^(?:(?P<userinfo>[^@/]*)@)?(?P<host>[^:/]+):(?!//)(?P<path>.+)$"
)

_GIT_ACTIONS = ("log", "show", "blame")

_SKIP_DIRS = {".git"}
_GREP_MAX_FILE_BYTES = 1024 * 1024
_BINARY_SNIFF_BYTES = 8192
_COUNT_CAP = 20000
_TREE_HARD_CAP = 200000


class RepoError(Exception):
    """User-facing error.

    ``str(exc)`` is safe to hand straight to an LLM: it never contains tokens,
    credentials or stack traces.
    """


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepoRecord:
    """Immutable description of one cloned repository."""

    name: str
    slug: str
    owner: str
    repo: str
    url: str
    sha: str
    branch: str
    cloned_at: float
    shallow: bool

    @property
    def short_sha(self) -> str:
        """First 8 characters of the commit sha."""
        return self.sha[:8]

    def to_dict(self) -> Dict[str, object]:
        """Serialise to a JSON-friendly mapping."""
        return {
            "name": self.name,
            "slug": self.slug,
            "owner": self.owner,
            "repo": self.repo,
            "url": self.url,
            "sha": self.sha,
            "branch": self.branch,
            "cloned_at": self.cloned_at,
            "shallow": self.shallow,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "RepoRecord":
        """Deserialise from a metadata mapping (unknown keys are ignored)."""
        if not isinstance(data, dict):
            raise TypeError("repo metadata is not an object")
        return cls(
            name=str(data["name"]),
            slug=str(data["slug"]),
            owner=str(data["owner"]),
            repo=str(data["repo"]),
            url=str(data["url"]),
            sha=str(data["sha"]),
            branch=str(data["branch"]),
            cloned_at=float(data["cloned_at"]),  # type: ignore[arg-type]
            shallow=bool(data["shallow"]),
        )


# ---------------------------------------------------------------------------
# Paths / metadata helpers
# ---------------------------------------------------------------------------


def repos_dir(scope_dir: Path) -> Path:
    """Return ``<scope_dir>/repos`` without creating it."""
    return Path(scope_dir) / REPOS_DIRNAME


def _slug_dir(scope_dir: Path, slug: str) -> Path:
    return repos_dir(scope_dir) / slug


def _clone_path(scope_dir: Path, slug: str) -> Path:
    return _slug_dir(scope_dir, slug) / CLONE_DIRNAME


def _meta_path(scope_dir: Path, slug: str) -> Path:
    return _slug_dir(scope_dir, slug) / META_FILENAME


def _read_meta_file(path: Path) -> Optional[Dict[str, object]]:
    """Read a metadata file, returning ``None`` for anything unusable."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _write_meta(path: Path, payload: Dict[str, object]) -> None:
    """Write metadata atomically-ish inside the workspace (never in the clone)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _count_entries(root: Path, cap: int = _COUNT_CAP) -> Tuple[int, int, bool]:
    """Cheaply count ``(files, dirs, complete)`` under ``root``, skipping .git."""
    files = 0
    dirs = 0
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for _ in dirnames:
            dirs += 1
            seen += 1
        for _ in filenames:
            files += 1
            seen += 1
        if seen >= cap:
            return files, dirs, False
    return files, dirs, True


def _repo_size(root: Path) -> int:
    """Total on-disk size of the clone, bounded by MAX_REPO_BYTES + 1."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
            if total > MAX_REPO_BYTES:
                return total
    return total


# ---------------------------------------------------------------------------
# URL handling
# ---------------------------------------------------------------------------


def _sanitize_name(value: str) -> str:
    """Reduce an arbitrary name to ``[A-Za-z0-9._-]`` and forbid dot-only names."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not cleaned:
        cleaned = "repo"
    return cleaned


def _parse_remote_url(
    url: str, *, allow_local: bool
) -> Tuple[str, str, str, str, bool]:
    """Normalise a repository URL.

    Returns ``(clone_url, owner, repo, safe_url, had_credentials)`` where
    ``clone_url`` is what git is actually asked to fetch and ``safe_url`` is a
    credential-free URL that may be shown to a user or persisted.
    """
    raw = (url or "").strip()
    if not raw:
        raise RepoError("No repository URL was provided.")
    if "\x00" in raw or "\n" in raw or "\r" in raw:
        raise RepoError("Repository URL contains invalid control characters.")

    scheme_match = _SCHEME_URL_RE.match(raw)
    scp_match = None if scheme_match else _SCP_URL_RE.match(raw)

    if scheme_match:
        scheme = scheme_match.group("scheme").lower()
        if scheme not in ("http", "https", "ssh", "git", "file"):
            raise RepoError(
                f"Unsupported URL scheme '{scheme}'. "
                "Use an https, ssh or git URL for a public repository."
            )
        if scheme == "file":
            if not allow_local:
                raise RepoError(
                    "file:// URLs are only accepted when local repositories are allowed."
                )
            return _parse_local_url(scheme_match.group("path"), raw)
        host = (scheme_match.group("host") or "").lower()
        userinfo = scheme_match.group("userinfo") or ""
        path = scheme_match.group("path") or ""
    elif scp_match:
        host = (scp_match.group("host") or "").lower()
        userinfo = scp_match.group("userinfo") or ""
        path = scp_match.group("path") or ""
    else:
        return _parse_local_url(raw, raw)

    if host not in ALLOWED_GIT_HOSTS and not allow_local:
        allowed = ", ".join(sorted(ALLOWED_GIT_HOSTS))
        raise RepoError(
            f"Host '{host}' is not allowed. Allowed hosts: {allowed}. "
            "Only public repositories on an allowed host can be cloned."
        )

    owner, repo = _split_owner_repo(path, host)
    clone_url = f"https://{host}/{owner}/{repo}.git"
    if scp_match or (scheme_match and scheme_match.group("scheme").lower() == "ssh"):
        clone_url = f"ssh://git@{host}/{owner}/{repo}.git"
    safe_url = clone_url
    had_credentials = bool(userinfo and ":" in userinfo)
    if had_credentials:
        # Never hand the token to anything that persists it in .git/config.
        clone_url = _inject_userinfo(raw, userinfo, host, owner, repo)
        safe_url = f"https://{host}/{owner}/{repo}.git"
    return clone_url, owner, repo, safe_url, had_credentials


def _inject_userinfo(raw: str, userinfo: str, host: str, owner: str, repo: str) -> str:
    """Rebuild the original URL (credentials included) for the clone call."""
    path = f"/{owner}/{repo}.git"
    if "://" in raw:
        scheme = raw.split("://", 1)[0].lower()
        if scheme == "http":
            return f"http://{userinfo}@{host}{path}"
        return f"https://{userinfo}@{host}{path}"
    return f"https://{userinfo}@{host}{path}"


def _split_owner_repo(path: str, host: str) -> Tuple[str, str]:
    """Split a URL path into a validated ``(owner, repo)`` pair."""
    cleaned = path.strip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[: -len(".git")]
    parts = [part for part in cleaned.split("/") if part]
    if len(parts) < 2:
        raise RepoError(
            f"Could not determine an owner/repository from '{host}/{cleaned}'. "
            "Expected a URL like https://github.com/owner/repo."
        )
    owner, repo = parts[-2], parts[-1]
    for label, value in (("owner", owner), ("repository", repo)):
        if not _NAME_RE.match(value) or value in (".", ".."):
            raise RepoError(f"Invalid {label} name '{value}' in repository URL.")
    return owner, repo


def _parse_local_url(value: str, raw: str) -> Tuple[str, str, str, str, bool]:
    """Validate a local path used as a clone source (dev/test only)."""
    candidate = value
    if candidate.startswith("file://"):
        candidate = urlsplit(candidate).path
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        raise RepoError(f"Local repository path '{value}' could not be resolved.")
    if not resolved.is_dir():
        raise RepoError(f"Local repository path '{value}' is not a directory.")
    repo = _sanitize_name(resolved.name)
    owner = "local"
    return str(resolved), owner, repo, f"file://{resolved}", False


def _strip_credentials(url: str) -> str:
    """Drop a ``user:token@`` userinfo component from a URL, if present."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if parts.username is None and parts.password is None:
        return url
    host = parts.hostname or ""
    netloc = f"{host}:{parts.port}" if parts.port else host
    return parts._replace(netloc=netloc).geturl()


# ---------------------------------------------------------------------------
# Git invocation (argv lists only, never a shell)
# ---------------------------------------------------------------------------


def _git_env() -> Dict[str, str]:
    env = dict(os.environ)
    env.update(_GIT_ENV)
    return env


def _run_git(
    argv: List[str],
    *,
    timeout: int = 180,
    secrets: Tuple[str, ...] = (),
) -> subprocess.CompletedProcess:
    """Run git with an argv list.  Raises RepoError on a non-zero exit."""
    try:
        result = subprocess.run(
            argv,
            shell=False,
            timeout=timeout,
            capture_output=True,
            text=True,
            env=_git_env(),
        )
    except subprocess.TimeoutExpired:
        raise RepoError(f"Git command timed out after {timeout}s.")
    except FileNotFoundError:
        raise RepoError("The 'git' executable is not available on this system.")
    except OSError as exc:
        raise RepoError(f"Git could not be executed: {exc.strerror or exc}")
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        for secret in secrets:
            if secret:
                message = message.replace(secret, "<redacted>")
        message = message.splitlines()[-1] if message else "unknown error"
        if len(message) > 400:
            message = message[:400] + "…"
        raise RepoError(f"Git failed: {message}")
    return result


# ---------------------------------------------------------------------------
# Clone lifecycle
# ---------------------------------------------------------------------------


def list_repos(scope_dir: Path) -> List[RepoRecord]:
    """Return every readable repository record, oldest clone first."""
    base = repos_dir(scope_dir)
    if not base.is_dir():
        return []
    records: List[RepoRecord] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        data = _read_meta_file(entry / META_FILENAME)
        if data is None:
            continue
        try:
            records.append(RepoRecord.from_dict(data))
        except (KeyError, TypeError, ValueError):
            continue
    records.sort(key=lambda record: (record.cloned_at, record.name))
    return records


def _available_names(scope_dir: Path) -> str:
    names = [record.name for record in list_repos(scope_dir)]
    return ", ".join(names) if names else "none"


def _match(record: RepoRecord, selector: str) -> bool:
    return selector in (record.name, record.slug)


def find_repo(scope_dir: Path, selector: Optional[str]) -> Tuple[RepoRecord, Path]:
    """Resolve ``selector`` to a record and the absolute checkout path."""
    records = list_repos(scope_dir)
    chosen: Optional[RepoRecord] = None

    if selector is None or not str(selector).strip():
        if len(records) == 1:
            chosen = records[0]
        elif not records:
            raise RepoError(
                "No repositories have been cloned in this session. "
                'Use inspect(action="clone", url="https://github.com/owner/repo").'
            )
        else:
            raise RepoError(
                "Several repositories are cloned; specify repo=. Available: "
                f"{_available_names(scope_dir)}"
            )
    else:
        key = str(selector).strip()
        matches = [record for record in records if _match(record, key)]
        if len(matches) == 1:
            chosen = matches[0]
        elif not matches:
            raise RepoError(
                f"No cloned repository named '{key}'. Available: {_available_names(scope_dir)}"
            )
        else:
            raise RepoError(
                f"Selector '{key}' is ambiguous. Available: {_available_names(scope_dir)}"
            )

    root = _clone_path(scope_dir, chosen.slug)
    if not root.is_dir():
        raise RepoError(
            f"The checkout for '{chosen.name}' is missing; "
            'clone it again with inspect(action="clone").'
        )
    return chosen, root


def clone_repo(
    scope_dir: Path,
    url: str,
    *,
    full_history: bool = False,
    allow_local: bool = False,
    timeout: int = 180,
) -> RepoRecord:
    """Clone ``url`` into the session workspace and record its metadata."""
    clone_url, owner, repo, safe_url, had_credentials = _parse_remote_url(
        url, allow_local=allow_local
    )
    name = f"{owner}/{repo}"
    slug = f"{owner}__{repo}"
    if not _NAME_RE.match(slug) or slug in (".", ".."):
        raise RepoError(
            f"Repository '{name}' cannot be used as a workspace directory name."
        )

    dest_parent = _slug_dir(scope_dir, slug)
    dest = dest_parent / CLONE_DIRNAME
    if dest_parent.exists():
        if dest_parent.is_symlink() or not dest_parent.is_dir():
            raise RepoError(f"Workspace path for '{name}' is not a directory.")
        shutil.rmtree(dest_parent)
    dest_parent.mkdir(parents=True, exist_ok=True)

    argv = ["git", "clone"]
    if not full_history:
        argv += [
            "--depth",
            "1",
            "--single-branch",
            "--no-tags",
            "--no-recurse-submodules",
        ]
    argv += [clone_url, str(dest)]

    try:
        _run_git(argv, timeout=timeout, secrets=(clone_url,))
    except RepoError:
        shutil.rmtree(dest_parent, ignore_errors=True)
        raise

    if not (dest / ".git").exists():
        shutil.rmtree(dest_parent, ignore_errors=True)
        raise RepoError(f"Cloning '{name}' did not produce a git repository.")

    if had_credentials:
        _run_git(
            ["git", "-C", str(dest), "remote", "set-url", "origin", safe_url],
            timeout=timeout,
            secrets=(clone_url,),
        )

    sha = _run_git(["git", "-C", str(dest), "rev-parse", "HEAD"]).stdout.strip()
    branch = _run_git(
        ["git", "-C", str(dest), "rev-parse", "--abbrev-ref", "HEAD"]
    ).stdout.strip()
    if not sha:
        shutil.rmtree(dest_parent, ignore_errors=True)
        raise RepoError(f"Could not read the commit sha of '{name}'.")

    size = _repo_size(dest)
    if size > MAX_REPO_BYTES:
        shutil.rmtree(dest_parent, ignore_errors=True)
        raise RepoError(
            f"Repository '{name}' is {_human_size(size)}, which exceeds the "
            f"{_human_size(MAX_REPO_BYTES)} limit."
        )

    record = RepoRecord(
        name=name,
        slug=slug,
        owner=owner,
        repo=repo,
        url=safe_url,
        sha=sha,
        branch=branch or "HEAD",
        cloned_at=time.time(),
        shallow=not full_history,
    )
    files, dirs, _complete = _count_entries(dest)
    payload: Dict[str, object] = record.to_dict()
    payload["n_files"] = files
    payload["n_dirs"] = dirs
    _write_meta(_meta_path(scope_dir, slug), payload)
    return record


def drop_repo(scope_dir: Path, selector: str) -> RepoRecord:
    """Delete a cloned repository and return its record."""
    record, _root = find_repo(scope_dir, selector)
    shutil.rmtree(_slug_dir(scope_dir, record.slug), ignore_errors=False)
    return record


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def render_prompt_section(scope_dir: Path, max_repos: int = 10) -> str:
    """Render the 'CLONED REPOSITORIES' section of the system prompt."""
    records = list_repos(scope_dir)
    if not records:
        return ""
    limit = max(1, int(max_repos))
    shown = records[:limit]
    lines = ["CLONED REPOSITORIES", ""]
    for record in shown:
        files, dirs = _counts_for(scope_dir, record)
        lines.append(
            f"- {record.name} @ {record.short_sha} — branch {record.branch}, "
            f"{files} files in {dirs} dirs — inspect with "
            'inspect(action="tree"), '
            f'inspect(action="read", repo="{record.name}", path=...), '
            'inspect(action="grep", ...), inspect(action="git", ...)'
        )
    remaining = len(records) - len(shown)
    if remaining > 0:
        lines.append(f"- … and {remaining} more")
    return "\n".join(lines)


def _counts_for(scope_dir: Path, record: RepoRecord) -> Tuple[object, object]:
    """File/dir counts from metadata, falling back to a bounded walk."""
    data = _read_meta_file(_meta_path(scope_dir, record.slug))
    if data is not None:
        files = data.get("n_files")
        dirs = data.get("n_dirs")
        if isinstance(files, int) and isinstance(dirs, int):
            return files, dirs
    root = _clone_path(scope_dir, record.slug)
    files, dirs, complete = _count_entries(root)
    return (files, dirs) if complete else (f"{files}+", f"{dirs}+")


# ---------------------------------------------------------------------------
# Jail
# ---------------------------------------------------------------------------


def _resolve_in_root(root: Path, path: str, *, allow_git: bool = False) -> Path:
    """Resolve a user-supplied relative path, guaranteeing it stays in ``root``.

    This is the *only* place user paths are turned into filesystem paths.
    """
    raw = "" if path is None else str(path)
    if "\x00" in raw:
        raise RepoError("Path contains a NUL byte.")
    if raw.startswith("~"):
        raise RepoError(
            f"Path '{raw}' is not allowed; use a path relative to the repository."
        )
    candidate = Path(raw)
    if candidate.is_absolute():
        raise RepoError(
            f"Path '{raw}' is absolute; use a path relative to the repository."
        )

    root_resolved = root.resolve()
    try:
        target = (root_resolved / candidate).resolve(strict=False)
    except (OSError, RuntimeError):
        raise RepoError(f"Path '{raw}' could not be resolved.")
    try:
        relative = target.relative_to(root_resolved)
    except ValueError:
        raise RepoError(
            f"Path '{raw}' escapes the repository sandbox and was rejected."
        )
    if relative.parts and relative.parts[0] == ".git" and not allow_git:
        raise RepoError("The .git directory is not readable through this tool.")
    return target


def _rel(root: Path, target: Path) -> str:
    try:
        return target.relative_to(root.resolve()).as_posix()
    except ValueError:
        return "."


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _link_target(path: Path) -> str:
    try:
        return os.readlink(path)
    except OSError:
        return "?"


def _is_binary_bytes(chunk: bytes) -> bool:
    return b"\x00" in chunk


def _is_binary_file(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            return _is_binary_bytes(handle.read(_BINARY_SNIFF_BYTES))
    except OSError:
        return True


# ---------------------------------------------------------------------------
# Inspection actions
# ---------------------------------------------------------------------------


def tree(
    record: RepoRecord,
    root: Path,
    path: str = "",
    depth: int = 3,
    max_entries: int = MAX_TREE_ENTRIES,
) -> str:
    """Render a depth-limited text listing of the repository."""
    start = _resolve_in_root(root, path)
    if not start.exists():
        raise RepoError(f"Path '{path or '.'}' does not exist in {record.name}.")
    if not start.is_dir():
        raise RepoError(f"Path '{path}' is a file, not a directory; use action='read'.")

    max_depth = max(0, int(depth))
    limit = max(1, int(max_entries))
    entries: List[Tuple[int, str, bool, int]] = []
    total = 0
    exhausted = False

    def walk(directory: Path, level: int) -> None:
        nonlocal total, exhausted
        if exhausted or level > max_depth:
            return
        try:
            children = sorted(
                directory.iterdir(),
                key=lambda item: (not item.is_dir(), item.name.lower()),
            )
        except OSError:
            return
        for child in children:
            if child.name in _SKIP_DIRS:
                continue
            total += 1
            if total > _TREE_HARD_CAP:
                exhausted = True
                return
            try:
                is_link = child.is_symlink()
                is_dir = child.is_dir() and not is_link
                size = 0 if is_dir else child.lstat().st_size
            except OSError:
                is_link, is_dir, size = False, False, 0
            name = f"{child.name} -> {_link_target(child)}" if is_link else child.name
            entries.append((level, name, is_dir, size))
            if is_dir:
                walk(child, level + 1)

    walk(start, 1)

    shown = entries[:limit]
    header = (
        f"Repository {record.name} @ {record.short_sha} — "
        f"{len(shown)} of {total} entries"
    )
    if path:
        header += f" — under {_rel(root, start)}/"
    lines = [header]
    for level, name, is_dir, size in shown:
        indent = "  " * level
        if is_dir:
            lines.append(f"{indent}{name}/")
        else:
            lines.append(f"{indent}{name}  {_human_size(size)}")
    if len(shown) < total or exhausted:
        lines.append("[truncated: entry limit]")
    return "\n".join(lines)


def read_text(
    record: RepoRecord,
    root: Path,
    path: str,
    start_line: int = 1,
    max_lines: int = MAX_READ_LINES,
) -> str:
    """Return a slice of a text file with 1-based numbered lines."""
    if not str(path or "").strip():
        raise RepoError("A path is required to read a file.")
    target = _resolve_in_root(root, path)
    if not target.exists():
        raise RepoError(f"File '{path}' does not exist in {record.name}.")
    if target.is_dir():
        raise RepoError(f"Path '{path}' is a directory; use action='tree'.")
    try:
        size = target.stat().st_size
    except OSError:
        raise RepoError(f"File '{path}' could not be inspected.")
    if size > MAX_READ_BYTES:
        raise RepoError(
            f"File '{path}' is {_human_size(size)}, above the "
            f"{_human_size(MAX_READ_BYTES)} read limit."
        )
    try:
        with open(target, "rb") as handle:
            raw = handle.read(MAX_READ_BYTES + 1)
    except OSError:
        raise RepoError(f"File '{path}' could not be read.")
    if _is_binary_bytes(raw[:_BINARY_SNIFF_BYTES]):
        raise RepoError(f"File '{path}' looks binary and cannot be read as text.")

    text = raw.decode("utf-8", errors="replace")
    all_lines = text.splitlines()
    total_lines = len(all_lines)

    first = max(1, int(start_line))
    count = max(0, int(max_lines))
    selected = all_lines[first - 1 : first - 1 + count] if count else []

    rel = _rel(root, target)
    lines = [f"File: {rel} @ {record.short_sha} ({total_lines} lines, {size} bytes)"]
    for offset, content in enumerate(selected):
        lines.append(f"{first + offset:>5}| {content}")
    if not selected:
        lines.append("[no lines in the requested range]")
    end = first + len(selected) - 1
    if selected and end < total_lines:
        lines.append(
            f"[truncated: showing lines {first}-{end} of {total_lines} — "
            f"call again with start_line={end + 1}]"
        )
    return "\n".join(lines)


def grep(
    record: RepoRecord,
    root: Path,
    pattern: str,
    path: str = "",
    *,
    case_sensitive: bool = False,
    max_matches: int = MAX_GREP_MATCHES,
) -> str:
    """Search the checkout with a pure-python regex walk (no shell, no rg)."""
    if not pattern:
        raise RepoError("A search pattern is required.")
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        raise RepoError(f"Invalid regular expression '{pattern}': {exc}")

    start = _resolve_in_root(root, path)
    if not start.exists():
        raise RepoError(f"Path '{path or '.'}' does not exist in {record.name}.")

    limit = max(1, int(max_matches))
    matches: List[str] = []
    matched_files = set()
    truncated = False

    if start.is_file():
        candidates: List[Path] = [start]
    else:
        candidates = []
        for dirpath, dirnames, filenames in os.walk(start):
            dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
            for filename in sorted(filenames):
                candidates.append(Path(dirpath) / filename)

    for candidate in candidates:
        if len(matches) >= limit:
            truncated = True
            break
        try:
            if candidate.is_symlink():
                continue
            if (
                not candidate.is_file()
                or candidate.stat().st_size > _GREP_MAX_FILE_BYTES
            ):
                continue
            with open(candidate, "rb") as handle:
                raw = handle.read()
        except OSError:
            continue
        if _is_binary_bytes(raw[:_BINARY_SNIFF_BYTES]):
            continue
        rel = _rel(root, candidate)
        for lineno, line in enumerate(
            raw.decode("utf-8", errors="replace").splitlines(), 1
        ):
            if regex.search(line):
                matches.append(f"{rel}:{lineno}: {line.strip()[:300]}")
                matched_files.add(rel)
                if len(matches) >= limit:
                    truncated = True
                    break

    if not matches:
        return f"No matches for {pattern}"

    lines = list(matches)
    lines.append("")
    lines.append(f"— {len(matches)} matches in {len(matched_files)} files")
    if truncated:
        lines.append("[truncated: match limit]")
    return "\n".join(lines)


def git(
    record: RepoRecord,
    root: Path,
    action: str,
    *,
    path: str = "",
    rev: str = "HEAD",
    max_lines: int = 200,
) -> str:
    """Run one of a fixed set of read-only git commands on the checkout."""
    if action not in _GIT_ACTIONS:
        raise RepoError(
            f"Unsupported git action '{action}'. Allowed: {', '.join(_GIT_ACTIONS)}."
        )
    revision = (rev or "HEAD").strip()
    if revision.startswith("-") or not _REV_RE.match(revision):
        raise RepoError(
            f"Invalid revision '{rev}'. Use a commit sha, tag or branch name."
        )

    rel_path = ""
    if path:
        raw = str(path)
        if raw.startswith("-"):
            raise RepoError("Path may not start with '-'.")
        rel_path = _rel(root, _resolve_in_root(root, raw, allow_git=True))

    if action == "blame" and not rel_path:
        raise RepoError("The 'blame' action requires path=.")

    argv = ["git", "-C", str(root), "-c", "color.ui=false", "--no-pager"]
    if action == "log":
        argv += ["log", "--format=%h %ad %an: %s", "--date=short"]
        if rel_path:
            argv += [revision, "--", rel_path]
        else:
            argv += [revision]
    elif action == "show":
        argv += ["show", "--stat", "--format=%H%n%an <%ae>%n%ad%n%n%s", revision]
        if rel_path:
            argv += ["--", rel_path]
    else:  # blame
        argv += ["blame", revision, "--", rel_path]

    result = _run_git(argv)
    output = result.stdout.rstrip("\n")
    if not output.strip():
        return "(no output)"
    lines = output.splitlines()
    limit = max(1, int(max_lines))
    if len(lines) > limit:
        lines = lines[:limit]
        lines.append(
            f"[truncated: showing {limit} of {len(output.splitlines())} lines]"
        )
    return "\n".join(lines)
