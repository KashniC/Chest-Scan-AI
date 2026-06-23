---
name: Artifact working directory
description: Replit artifact services execute from the artifact's own directory, not the workspace root.
---

When a service is launched via artifact.toml `[services.development] run = "..."`, the working directory is the artifact folder itself (e.g. `/home/runner/workspace/artifacts/chestscan-ai/`), NOT the workspace root.

**Rule:** Use bare filenames or paths relative to the artifact dir:
- Correct: `python app.py`
- Wrong: `python artifacts/chestscan-ai/app.py` (doubles the path)

**For workflows configured via configureWorkflow():** use `cd /absolute/path && command` to be explicit, since workflows may or may not inherit the artifact's working directory depending on how they are triggered.

**Why:** Replit's artifact runner `cd`s into the artifact directory before executing the service command. Using a workspace-relative path from there produces a doubled invalid path like `/home/runner/workspace/artifacts/chestscan-ai/artifacts/chestscan-ai/app.py`.
