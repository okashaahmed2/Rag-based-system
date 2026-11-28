# n8n workflows

n8n only orchestrates: it triggers the FastAPI backend and watches the job.
All processing (extraction, chunking, embedding, storage) happens in Python.

## `workflows/rag_ingest_scheduled.json` — "RAG ingest (scheduled)"

```
Schedule every 5 min --+
                       +--> Start scan (POST /ingest/scan)
Manual run ------------+          |
                                  v
                         Already running? --true--> Skip: scan already running
                                  | false
                                  v
                +-----------> Wait 3s
                |                 |
                |                 v
                |          Check status (GET /status/{job_id})
                |                 |
                |                 v
                +--true--- Still running?   (status queued/running AND $runIndex < 40)
                                  | false
                                  v
                          Job succeeded? --true--> Done
                                  | false
                                  v
                          Scan not succeeded (Stop and Error)
```

- Both HTTP nodes: timeout 10 s, retry on fail (3 tries, 5 s apart). If the
  API stays unreachable the execution fails (error shown in n8n).
- The poll loop stops after 40 rounds (about 2 minutes); a job still running
  then ends in "Scan not succeeded" with its current status. The job itself
  keeps running in the API and the next scheduled run reports it as already
  running.
- `active` is `false` in the file: nothing runs on a schedule until the
  workflow is published in the editor.

## Import

1. Start the API (staging while building: `scripts\run_api_staging.py serve`).
2. Start n8n and open `http://127.0.0.1:5678`.
3. Overview -> Create workflow -> "..." menu (top right) -> **Import from File...**
   -> choose `n8n\workflows\rag_ingest_scheduled.json` -> Save.
4. After import, open workflow **Settings** and set **Timezone** to
   `Asia/Karachi`: on 2026-11-28 the imported `settings.timezone` was not
   applied (schedule output showed America/New_York until set in the UI).
5. Test with **Execute workflow** (uses "Manual run"); publish only when it works.

## Daily run commands (PowerShell, one terminal each)

```powershell
# Live API (project root)
.venv\Scripts\python.exe -m uvicorn rag_pipeline.api.app:create_app --factory --host 127.0.0.1 --port 8000

# OR staging API on an isolated inbox copy (project root)
.venv\Scripts\python.exe scripts\run_api_staging.py prepare
.venv\Scripts\python.exe scripts\run_api_staging.py serve

# n8n, started from the user profile folder
cd $env:USERPROFILE; n8n
```

## Rules

- Call the API at **`http://127.0.0.1:8000`**, never `localhost`: Uvicorn binds
  IPv4 only, Node resolves `localhost` to `::1` first and n8n disables the IPv4
  fallback, so the connection is refused (verified 2026-11-27).
- Start n8n from `%USERPROFILE%` (`cd $env:USERPROFILE; n8n`), not from the
  project folder: n8n loads a `.env` from its working directory.
- Edited in the UI? Re-export (Download) the workflow, overwrite the JSON in
  `n8n/workflows/`, then run `.venv\Scripts\python.exe -m pytest tests\test_n8n_workflow.py`.
  Check the export has no credentials and no pinned data before committing.
- Only **one** published workflow may call the API (two schedules would race
  for the same scan job).
- `n8n/reference/` holds raw UI exports used as the source of truth for node
  versions and parameter shapes; they are reviewed by the test, not enforced.
