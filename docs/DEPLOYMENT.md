# Deployment: FastAPI on Render, Streamlit on Streamlit Community Cloud

Nothing in this guide has been executed against Render or Streamlit Cloud by the project tooling. What *was* verified locally is listed in [What was verified](#what-was-verified-and-what-was-not). **Do not treat the deployment as working until `/health` on your Render URL answers (step 6).**

## Why the dashboard said "Backend Offline"

The frontend takes its API address from one place, `frontend/config.py`: the `API_BASE_URL` environment variable, or Streamlit secrets, or a local `.env`, falling back to `http://localhost:8000`. Your `.env` is git-ignored (correctly), so it was not in the deployed copy, and no `API_BASE_URL` was set in Streamlit Cloud. The app therefore called `http://localhost:8000` **on the Streamlit Cloud server itself**, where nothing is listening, and showed "Backend offline". Nothing else in the runtime code hardcodes localhost (the remaining mentions are local launch scripts and documentation).

Fix: deploy the backend somewhere with a public https URL (below) and set `API_BASE_URL` in Streamlit Cloud Secrets. The dashboard now also says this explicitly instead of just "offline".

## Architecture

```
Browser -> Streamlit Cloud (frontend/app.py) --HTTPS, X-API-Key--> Render web service (FastAPI)
                                                                     |-> ChromaDB on a persistent disk (/var/data/chroma)
                                                                     |-> AWS Bedrock (ap-south-1): Titan V2 + Nova Lite (apac profile)
```
The two halves are separate deployments. The frontend never talks to Bedrock or ChromaDB; its requests come from Streamlit's server, so no CORS configuration is needed.

## 1. Files involved

| File | Purpose |
|---|---|
| `render.yaml` | Render Blueprint for the backend (paid plan, persistent disk, env vars, health check) |
| `requirements-backend.txt` | Backend-only dependencies for Render (no Streamlit) |
| `frontend/requirements.txt` | Frontend-only dependencies (`streamlit`, `httpx`) for Streamlit Cloud |
| `requirements.txt`, `requirements-dev.txt` | Unchanged: everything, for local Windows development |
| `.streamlit/secrets.toml.example` | Template for the Streamlit Cloud Secrets box (the real `secrets.toml` is git-ignored) |
| `frontend/config.py`, `frontend/api_client.py` | API URL / key lookup and clear connection errors |
| `app/api/main.py`, `app/core/config.py` | Optional `API_KEY` protection |

Commit and push these (and `reports/eval_results.json`, `reports/evaluation_report.md` if you want the Evaluation page to show results on the hosted backend). Never commit `.env` or `.streamlit/secrets.toml`.

## 2. Backend on Render

Render is **not** AWS, so there are no IAM roles to attach. The backend authenticates to Bedrock with an IAM user's access key supplied as environment variables (section 4). `aws login` profiles (like your local `hcl-rag`) only exist on your machine and must not be used on Render: leave `AWS_PROFILE` **unset**.

**Option A, Blueprint (recommended):** Render Dashboard -> *New* -> *Blueprint* -> connect the GitHub repository -> Render reads `render.yaml` -> review -> *Apply*. The blueprint uses the paid `starter` plan because only paid services can have a persistent disk. Then enter the two secrets it asks for (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`); `API_KEY` is generated for you.

**Option B, manual web service:** *New* -> *Web Service* -> the repository, then:

| Setting | Value |
|---|---|
| Runtime | Python 3 |
| Region | Singapore (closest to ap-south-1) |
| Instance type | Starter or larger (a disk requires a paid type) |
| Build command | `pip install --upgrade pip && pip install -r requirements-backend.txt` |
| Start command | `uvicorn app.api.main:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/health` |
| Auto-deploy | Off (recommended) |
| Disk | name `chroma-data`, mount path `/var/data`, size 1 GB |

`$PORT` is supplied by Render; `0.0.0.0` makes the server reachable from outside the container. The entry point is `app.api.main:app` (verified).

## 3. Environment variables (Render -> Environment)

| Variable | Value | Secret? |
|---|---|---|
| `PYTHON_VERSION` | `3.12.7` (any 3.12.x Render offers) | no |
| `AWS_REGION` | `ap-south-1` | no |
| `BEDROCK_EMBED_MODEL_ID` | `amazon.titan-embed-text-v2:0` | no |
| `BEDROCK_EMBED_DIM` | `1024` | no |
| `BEDROCK_LLM_MODEL_ID` | `apac.amazon.nova-lite-v1:0` | no |
| `CHROMA_PATH` | `/var/data/chroma` (must be inside the disk mount path) | no |
| `CHROMA_COLLECTION` | `enterprise_docs` | no |
| `REPORTS_DIR` | `reports` | no |
| `AWS_ACCESS_KEY_ID` | the IAM user's access key id | **yes** |
| `AWS_SECRET_ACCESS_KEY` | the IAM user's secret key | **yes** |
| `API_KEY` | long random string (Render can generate it) | **yes** |
| `AWS_PROFILE` | **do not set** | n/a |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | optional; see the chunk-size note in the README (set before uploading documents) | no |

Do not put `AWS_*` secrets in `render.yaml`, in the repository, or in chat. If you ever paste a key somewhere public, deactivate it in IAM immediately.

## 4. AWS credentials, safely

Create a dedicated IAM user (for example `hcl-docqa-render`) with **no console access**, no admin rights, and only this inline policy (replace `ACCOUNT_ID`):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "InvokeTitanEmbeddings",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": "arn:aws:bedrock:ap-south-1::foundation-model/amazon.titan-embed-text-v2:0"
    },
    {
      "Sid": "InvokeNovaLiteThroughApacProfile",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": [
        "arn:aws:bedrock:ap-south-1:ACCOUNT_ID:inference-profile/apac.amazon.nova-lite-v1:0",
        "arn:aws:bedrock:*::foundation-model/amazon.nova-lite-v1:0"
      ]
    }
  ]
}
```
(The Converse API is authorised by `bedrock:InvokeModel`. The `apac.` profile routes across Asia-Pacific regions, hence the wildcard region on the Nova foundation-model ARN.) Create one access key for that user, paste it into Render's environment, and rotate it periodically. Model access for Titan V2 and Nova Lite must already be enabled in the Bedrock console for your account.

**If you want an IAM role instead of keys:** run the backend on AWS (for example App Runner, ECS/Fargate with a task role, or EC2 with an instance profile) and attach a role with the same policy. boto3 then finds the role automatically and you set **no** `AWS_*` keys. Render cannot do this.

## 5. ChromaDB persistence: a safe strategy

ChromaDB stores everything in a local folder (`CHROMA_PATH`), so on a host with an ephemeral filesystem **every restart, deploy or spin-down erases all uploaded documents**.

* **Use a persistent disk** (the blueprint does): mounted at `/var/data`, with `CHROMA_PATH=/var/data/chroma`. Data then survives restarts and redeploys. Constraints: a disk belongs to one instance (do not scale to several instances), it requires a paid plan, and a service with a disk has a brief downtime on each deploy.
* **Do not use the free plan for real data.** It has no disk; documents would vanish whenever the service restarts or sleeps.
* **Never change the embedding model or `BEDROCK_EMBED_DIM` on an existing disk.** The collection records both, and the backend refuses to open it on a mismatch (the service still starts, but every request that needs the store fails with a clear error) because it would mix vector spaces. To change them, use a new `CHROMA_COLLECTION` and re-upload. The same applies to `CHUNK_SIZE`: it only takes effect when a document is ingested, so after changing it delete and re-upload.
* **Keep your source files.** The index can always be rebuilt by re-uploading, which costs only a few Titan calls. Treat the disk as a cache you can regenerate, and enable Render disk snapshots/backups as an extra safeguard (check Render's current options).
* **Your existing local store** (`data/chroma`, 56 chunks at the time of writing) is not uploaded by this process. The hosted backend starts empty: upload your documents through the dashboard after deploying. Do not copy the local `data/chroma` folder into the repository (it is git-ignored on purpose).
* **Verify persistence after the first deploy:** upload a document, note the chunk count in `/health`, then *Manual Deploy -> Restart* and check that `chroma_count` is unchanged.

## 6. Test the hosted backend

Replace `YOUR-SERVICE` with the name in your Render URL.

```powershell
# PowerShell: health (no key needed; this is also what Render's health check calls)
curl.exe -i https://YOUR-SERVICE.onrender.com/health
# expected: HTTP/1.1 200 OK  and  {"status":"ok","chroma_count":0,"embed_model":"amazon.titan-embed-text-v2:0","llm_model":"apac.amazon.nova-lite-v1:0"}

# a protected route must be refused without the key (expect 401) and accepted with it (expect 200 and [])
curl.exe -i https://YOUR-SERVICE.onrender.com/documents
curl.exe -i -H "X-API-Key: YOUR_API_KEY" https://YOUR-SERVICE.onrender.com/documents

# real end-to-end check: this calls Titan and therefore proves AWS access from Render (a 502 shows the AWS error)
curl.exe -H "X-API-Key: YOUR_API_KEY" -F "files=@C:\path\to\small.txt" https://YOUR-SERVICE.onrender.com/ingest
curl.exe -X POST -H "X-API-Key: YOUR_API_KEY" -H "Content-Type: application/json" -d "{\"question\": \"What does the document say?\"}" https://YOUR-SERVICE.onrender.com/query
```
`/health` does not call AWS, so it passing proves the service is up, not that Bedrock works; the `/ingest` and `/query` calls prove that. A 401 means the key is missing or wrong; a 502 `bedrock_error` includes the AWS reason (missing credentials, permissions, model access).

## 7. Frontend on Streamlit Community Cloud

1. *New app* -> repository and branch -> **Main file path: `frontend/app.py`** (Streamlit Cloud looks for a requirements file next to the main file first, which should make it install `frontend/requirements.txt` instead of the heavy root file; confirm in the app's build log, and if it installed the root file the app still works, just more slowly).
2. *Advanced settings* -> Python version **3.12**.
3. *Advanced settings -> Secrets* (or *App settings -> Secrets* later), paste:

```toml
API_BASE_URL = "https://YOUR-SERVICE.onrender.com"
API_KEY = "the-same-value-as-the-render-API_KEY"
# API_TIMEOUT_SECONDS = 60    # optional, useful if the backend can be asleep
```
   `API_BASE_URL` is the bare public address: no `/health`, no `/docs`, no trailing slash needed (a missing `https://` is added for you). Do **not** put AWS keys in Streamlit; the frontend never needs them.
4. Save and *Reboot app* (secrets are read at startup).
5. Check the sidebar: **Backend online** with your URL under it. If it still says offline, the sidebar now states why (see below).

| Symptom in the dashboard | Cause / fix |
|---|---|
| "API_BASE_URL is not configured: it still points at localhost" | Secrets missing or saved without a reboot |
| "waking from sleep" / timeout | Free-tier backend spinning up (up to ~1 min); wait, or raise `API_TIMEOUT_SECONDS`, or use a paid always-on instance |
| "Backend error 401 unauthorized" | `API_KEY` in Secrets differs from Render's `API_KEY` |
| "Backend error 502 bedrock_error" | AWS credentials/permissions/model access on Render (see the message) |
| "hosting platform returned an error page" | Render service crashed or is deploying; check Render logs |
| Backend online but `chroma_count` is 0 after a restart | No persistent disk, or `CHROMA_PATH` is not under the disk mount path |

## 8. Security notes

* The backend has **no user accounts**; the optional `API_KEY` is a shared secret. Without it, anyone who finds the URL can upload, delete and run billable Bedrock queries. Always set it for a public deployment.
* Streamlit Cloud apps are public by default unless you restrict viewers in the app's sharing settings; the API key lives only in Streamlit Secrets and Render environment variables, not in the browser.
* Use https URLs only. Rotate the IAM key and `API_KEY` if either is ever exposed.

## What was verified, and what was not

Verified locally (not on Render or Streamlit Cloud):
* The Render start command was run for real in a clean virtualenv built **only** from `requirements-backend.txt`, from a source copy without `.env`, with `PORT`, `--host 0.0.0.0`, `API_KEY` and a disk-style `CHROMA_PATH`: `/health` returned 200 without a key, other routes returned 401 without or with a wrong key and 200 with the right one, the store was created on the disk path, and an upload without AWS credentials returned the clear 502.
* The frontend resolves its URL from the environment, from real Streamlit secrets (in a subprocess), and from `.env`, in that precedence; it sends `X-API-Key`; and it produces specific messages for localhost-on-cloud, sleeping hosts, 401s and hosting error pages. 461 automated tests pass, and local Windows development (`scripts\start_api.ps1`, `start_ui.ps1`, `start_all.ps1`) was re-checked.

**Not verified:** anything on Render or Streamlit Cloud themselves: the Render build, the persistent disk, Bedrock access with the IAM user's keys from Render, and the Streamlit Cloud install and secrets. No service was created. The deployment is not complete until step 6 succeeds against your real URL.
