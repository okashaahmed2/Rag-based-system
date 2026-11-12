# CLAUDE.md

@AGENTS.md

- Run Python only as `.venv\Scripts\python.exe` (also for pip and pytest); never activate the venv or change the execution policy.
- The primary shell is Windows PowerShell 5.1: no `&&`, `||` or `?:`; use `;` and `if ($?) { ... }`.
- After each task, report as AGENTS.md requires, then stop and wait for the user.
