# Tensor-Truth: AI Agent Context Guide

**Purpose**: Guide AI coding agents working on this local RAG pipeline project.

---

## CRITICAL INSTRUCTIONS FOR AI AGENTS

### Before Implementing Anything (MANDATORY)

**ALWAYS check for existing implementations first:**

1. **Understand project structure**: Use Glob tool to explore module organization
2. **Search for similar functions**: Use Grep tool to search for patterns across the codebase
3. **Read existing utilities**: Check `utils/`, `core/`, and `app_utils/` for reusable functions
4. **Avoid duplication**: If similar functionality exists, reuse or extend it rather than reimplementing

**Why this matters**: This codebase has many utilities already implemented. Creating duplicate functions wastes effort and creates maintenance burden.

### Code Quality Workflow (MANDATORY)
**After editing any Python code, you MUST:**

Scripts within scripts/ directory and methods contained within should be used for:
- lint: scripts/lint.sh
- format: scripts/format.sh
- test: scripts/test.sh

Always run format script after major file edits, and then run linting. Instead of running wwhole script, run corresponding mechanism for the file in question (depending if python or typescript file). Run individual tests are you're implementing particular modules, and ALAWYS finish a considerable edits by running full test.sh script.

**Never skip these steps.** This applies to all `.py` files in `src/`, `tests/`, and `scripts/`.

### Testing Requirements
**When creating new code, write tests:**

- **Unit tests** in `tests/unit/` for individual functions/classes
- **Integration tests** in `tests/integration/` for multi-component workflows (use `@pytest.mark.integration`)
- Mock external dependencies (file I/O, APIs, databases)
- Test both success and error paths
- IMPORTANT: write only sensible tests, limit those with trivial logic or heavy mocking

**Then run tests as part of the Code Quality Workflow above.**

### Code Style Guidelines

**Emoji usage:**
- Use ONLY for functional UI purposes (status indicators, interactive buttons)
- NEVER in logs, comments, docstrings, CLI output, or error messages
- Rationale: Prevents encoding issues, improves clarity
- For frontend use proper icons, always avoid emojis.

**Documentation:**
- Do NOT create/modify README.md or other docs unless explicitly requested
- Keep inline comments minimal and focused on "why", not "what"

---

## Project Overview

- Consult README to get acquainted generally with the project.
- If prompted to work on docker, read docs/DOCKER.md
- To get acquainted more with general mechanisms around vector indexes read docs/INDEXES.md
- To get acquainted with extensions (user defined tools, agents and commands) read docs/EXTENSIONS.md

**For implementation details, always read the code first:**
- Component behavior → Read the module
- Configuration schema → Check `pyproject.toml` and `config/sources.json`
- Test patterns → Examine existing tests in `tests/`
- User features → See `README.md`

