# CLAUDE.md

## Ask before creating or editing anything

Do not create, edit, delete, move, or overwrite any file, directory, or piece of information until the user has explicitly told you to, or you have asked and they said yes.

This covers:
- Source code, notebooks, scripts, and config files (`pyproject.toml`, `.env*`, `.gitignore`, etc.)
- Data, results, and outputs (`questions/`, `runs/`, `sweep/`, experiment folders)
- Documentation (`README.md`, this file, new docs)
- Dependencies and environments (`uv add`, `pip install`, changes to `.venv` or `uv.lock`)
- Git operations that change state (commits, branches, resets)
- Anything outside the repo: Claude memory, external services, and connectors (Docs, Drive, Slack, Jira, etc.)

What you may do without asking:
- Read files, search the codebase, and run read-only commands to understand the project.
- Write throwaway files to the session scratchpad directory, never inside the project.

How to ask:
- Say exactly what you plan to change, where, and why. Then wait for an answer.
- If you're asked a question or asked to analyze something, answer it. Do not treat the question as permission to make changes.
- Approval covers only the change that was approved. Do not stretch it to related or follow-up edits; ask again for each new change.
- If unsure whether something counts as an edit, assume it does and ask.
