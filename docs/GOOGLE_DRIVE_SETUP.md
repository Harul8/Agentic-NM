# Storing Nyaymalaw data in Google Drive

You can keep **chat history**, **login credentials**, **vector DB**, **BareActs**, and **CaseLaws** in a single folder—for example a Google Drive folder—so they sync across machines and are backed up.

## How it works

1. **Mount Google Drive** on your PC (e.g. with [Google Drive for Desktop](https://www.google.com/drive/download/)). You’ll get a path like:
   - **Windows:** `G:\My Drive\`
   - **macOS:** `/Users/you/Google Drive/`

2. **Create a folder** for Nyaymalaw data, e.g. `G:\My Drive\Nyaymalaw` (Windows) or `~/Google Drive/Nyaymalaw` (macOS). This project is configured to use **G:\My Drive\Nyaymalaw** (see `.env`).

3. **Set the environment variable** before starting the API server and any scripts:

   **Windows (PowerShell):**
   ```powershell
   $env:NYAYMALAW_DATA_ROOT = "G:\My Drive\Nyaymalaw"
   python api_server.py
   ```

   **Windows (Command Prompt):**
   ```cmd
   set NYAYMALAW_DATA_ROOT=G:\My Drive\Nyaymalaw
   python api_server.py
   ```

   **macOS / Linux:**
   ```bash
   export NYAYMALAW_DATA_ROOT="$HOME/Google Drive/Nyaymalaw"
   python api_server.py
   ```

   Or put the same line in a `.env` file in the project root and load it (e.g. with `python-dotenv` if you add it).

4. **Optional — shareable links for local files:** If you share your Drive folders (BareActs, CaseLaws) and want bare act / case law titles in the app to link to those folders when there is no web URL, set:

   **Windows (PowerShell):**
   ```powershell
   $env:NYAYMALAW_GOOGLE_DRIVE_BARE_ACTS_URL = "https://drive.google.com/drive/folders/YOUR_BARE_ACTS_FOLDER_ID"
   $env:NYAYMALAW_GOOGLE_DRIVE_CASE_LAWS_URL = "https://drive.google.com/drive/folders/YOUR_CASE_LAWS_FOLDER_ID"
   ```

   **macOS / Linux:**
   ```bash
   export NYAYMALAW_GOOGLE_DRIVE_BARE_ACTS_URL="https://drive.google.com/drive/folders/YOUR_BARE_ACTS_FOLDER_ID"
   export NYAYMALAW_GOOGLE_DRIVE_CASE_LAWS_URL="https://drive.google.com/drive/folders/YOUR_CASE_LAWS_FOLDER_ID"
   ```

   To get the folder ID: open the folder in Google Drive in your browser, copy the URL. It looks like `https://drive.google.com/drive/folders/1abc...xyz` — the part after `/folders/` is the ID.

5. **First run:** The app will create under that folder:
   - `chat_history/app.db` — login credentials + chat history
   - `vector_store/` — `bareacts.index`, `bareacts_chunks.json`, `caselaws.index`, `caselaws_chunks.json`
   - `BareActs/` — your bare act PDFs/text files
   - `CaseLaws/` — case law files

Everything (API server, MCP, Ingestion scripts) will use this data root. **MCP protocol is unchanged**; only the on-disk paths are configurable.

## Notes

- Use a **local path** to the Drive folder (the path after Drive is mounted). The app does not use the Google Drive API.
- Ensure the Drive folder is synced and available before starting the server.
- If `NYAYMALAW_DATA_ROOT` is not set, the app uses `project_root/data` as before.

### FAISS vector store on Windows

The vector store uses FAISS, which can fail on **Windows** when the data path contains spaces or is on a network/Drive path (e.g. `G:\My Drive\Nyaymalaw`). If you see errors like `FileIOWriter` or "couldn't complete your search", either:

1. **Use a path without spaces** for `NYAYMALAW_DATA_ROOT`, e.g. `C:\NyaymalawData`, and sync that folder to Google Drive with a sync tool, or  
2. **Leave `NYAYMALAW_DATA_ROOT` unset** so the app uses `project_root/data` (no spaces). Chat history and DB will still work; you can copy the `data` folder to Drive manually for backup.

When FAISS cannot read or write the index, the app still runs and uses web search for results; it just won’t persist new vector index entries until the path is fixed.
