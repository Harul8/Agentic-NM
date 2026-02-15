# Nyaymalaw on Cloudflare Free – Step-by-step

Use **Cloudflare Pages** for the frontend (nyaymalaw.in) and **Cloudflare Tunnel** to expose your local FastAPI backend at **api.nyaymalaw.in**. All of this works on the **free** plan.

---

## What you need

- Domain **nyaymalaw.in** on Cloudflare (DNS managed by Cloudflare).
- Frontend already deployed on **Cloudflare Pages** at https://nyaymalaw.in.
- Your PC (or a server) where you run the API and the tunnel.

---

## Step 1: Expose the API with a tunnel (api.nyaymalaw.in)

### 1.1 Install cloudflared

- **Windows:** Download from [GitHub Releases](https://github.com/cloudflare/cloudflared/releases) (e.g. `cloudflared-windows-amd64.exe`) and add to PATH, or run:
  ```bash
  winget install Cloudflare.cloudflared
  ```

### 1.2 Log in to Cloudflare (one-time)

```bash
cloudflared tunnel login
```

A browser window opens. Log in and choose the domain **nyaymalaw.in**. This saves a certificate under `C:\Users\<You>\.cloudflared\`.

### 1.3 Create a named tunnel (one-time)

```bash
cloudflared tunnel create nyaymalaw-api
```

Copy the **Tunnel ID** (e.g. `fdaa2d23-8887-4802-b7fa-aee1503af37a`). You will use it in the next steps.

### 1.4 Add DNS record for api.nyaymalaw.in

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com) → **Websites** → **nyaymalaw.in** → **DNS** → **Records**.
2. Click **Add record**:
   - **Type:** `CNAME`
   - **Name:** `api`
   - **Target:** `<TUNNEL-ID>.cfargotunnel.com` (replace with your tunnel ID)
   - **Proxy status:** Proxied (orange cloud)
3. Save.

So **api.nyaymalaw.in** will resolve to your tunnel.

### 1.5 Create tunnel config

Create or edit the config file (e.g. `C:\Users\rahul\.cloudflared\config.yml`):

```yaml
tunnel: <TUNNEL-ID>
credentials-file: C:\Users\rahul\.cloudflared\<TUNNEL-ID>.json

ingress:
  - hostname: api.nyaymalaw.in
    service: http://localhost:8000
  - service: http_status:404
```

Replace both `<TUNNEL-ID>` with your actual tunnel ID. The `credentials-file` path should match where `cloudflared tunnel login` saved the JSON file (usually `C:\Users\rahul\.cloudflared\<TUNNEL-ID>.json`).

### 1.6 Run the API and the tunnel

Every time you want the live site to work:

- **Terminal 1** (project root):
  ```bash
  python api_server.py
  ```
  Leave it running (API on http://127.0.0.1:8000).

- **Terminal 2**:
  ```bash
  cloudflared tunnel run nyaymalaw-api
  ```
  Or, if you use the config path explicitly:
  ```bash
  cloudflared tunnel --config C:\Users\rahul\.cloudflared\config.yml run <TUNNEL-ID>
  ```

When both are running, **https://api.nyaymalaw.in** will forward to your local API.

---

## Step 2: Point the frontend to the API

1. In the project, create **Frontend/.env.production** with:
   ```env
   VITE_API_BASE=https://api.nyaymalaw.in
   ```

2. Build the frontend:
   ```bash
   cd Frontend
   npm run build
   ```

3. Deploy the **Frontend/dist** folder to Cloudflare Pages (drag-and-drop in the dashboard, or push to your connected Git repo and set build output to `dist`).

After deployment, https://nyaymalaw.in will call https://api.nyaymalaw.in for all API requests.

---

## Step 3: Verify

1. Open **https://nyaymalaw.in** in the browser.
2. Use the app (e.g. type a case and submit).
3. Open DevTools (F12) → **Network** tab. You should see requests to **https://api.nyaymalaw.in** (e.g. `/submit_case`, `/conversation/continue`) with status 200 and JSON responses.

If you see "invalid or empty response", the frontend is not reaching the API: ensure both `python api_server.py` and `cloudflared tunnel run ...` are running and that you rebuilt/redeployed the frontend with `VITE_API_BASE=https://api.nyaymalaw.in`.

---

## Quick test without DNS (URL changes each time)

If you only want to test and don’t want to set up api.nyaymalaw.in yet:

```bash
cloudflared tunnel --url http://localhost:8000
```

Cloudflare will print a URL like **https://xxxxx-xxxx.trycloudflare.com**. Use it as the API base:

1. Create **Frontend/.env.production** with:
   ```env
   VITE_API_BASE=https://xxxxx-xxxx.trycloudflare.com
   ```
2. Run `npm run build` in `Frontend` and deploy `dist` to Pages.

The URL changes every time you restart the command, so use the **api.nyaymalaw.in** setup above for a stable production URL.
