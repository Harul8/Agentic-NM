# Deploying Nyaymalaw (nyaymalaw.in)

**Using Cloudflare Free?** See **[CLOUDFLARE_FREE_SETUP.md](CLOUDFLARE_FREE_SETUP.md)** for a full step-by-step (tunnel + api.nyaymalaw.in + frontend).

---

## Why “invalid response” or “API server running?” on login

If the **frontend** at https://nyaymalaw.in loads but **login** shows:

- *“Server at https://nyaymalaw.in returned an invalid response”*, or  
- *“Network error. Is the API server running?”*

then the browser is calling **https://nyaymalaw.in/auth/login**, but that URL is **not** your FastAPI app—it’s returning HTML (e.g. the SPA or a 404 page), not JSON. So the API is either not deployed or not reachable at that host.

---

## What you need in production

1. **Frontend**  
   - Built with `npm run build` in `Frontend/` and served from your CDN/host (e.g. Cloudflare Pages) at https://nyaymalaw.in.

2. **Backend (FastAPI)**  
   - Must be running and reachable at a URL the frontend can call.  
   - The app expects: `/auth/login`, `/auth/register`, `/chat/*`, `/conversation/*`, `/chats`, `/bareacts/*`, etc.

You can do either of the following.

---

### Option A: API on the same domain (https://nyaymalaw.in)

- Run the FastAPI server (e.g. `uvicorn api_server:app` or `python api_server.py`) on a machine that Cloudflare (or your reverse proxy) can reach.
- Configure your proxy so that:
  - `https://nyaymalaw.in/` → static frontend (e.g. Cloudflare Pages).
  - `https://nyaymalaw.in/auth/*`, `/chat/*`, `/conversation/*`, `/chats`, `/bareacts/*`, etc. → **forward to your FastAPI backend** (e.g. `http://your-server:8000`).
- Do **not** set `VITE_API_BASE`; the frontend will use `https://nyaymalaw.in` and the proxy will send those paths to the API.

---

### Option B: API on a subdomain (e.g. https://api.nyaymalaw.in)

1. Deploy the FastAPI app so it is served at e.g. **https://api.nyaymalaw.in** (Cloudflare Workers, a VPS, or any host).
2. In the **Frontend** build, set the API base URL. For Vite, use an env file:
   - Create `Frontend/.env.production` (or set in your CI/CD):
   ```env
   VITE_API_BASE=https://api.nyaymalaw.in
   ```
   - Rebuild: `cd Frontend && npm run build`.
3. Ensure the FastAPI app allows CORS from your frontend origin (e.g. `https://nyaymalaw.in`). The default in `api_server.py` is `allow_origins=["*"]`, which is fine for broad access; tighten in production if you prefer.

Then the frontend will call **https://api.nyaymalaw.in/auth/login** (and other endpoints) and login will work.

---

### Option C: Cloudflare Tunnel (cloudflared)

If you use **Cloudflare Tunnel** to expose your backend:

1. **Run the FastAPI backend** on your machine: `python api_server.py` (runs on `http://127.0.0.1:8000`).
2. **Configure the tunnel** so a public hostname (e.g. `api.nyaymalaw.in`) forwards to `http://localhost:8000`.
3. **Point the frontend to that URL:** create `Frontend/.env.production` with `VITE_API_BASE=https://api.nyaymalaw.in`, then rebuild and redeploy the frontend.
4. In **Cloudflare Dashboard** (Zero Trust / Tunnels), add a public hostname: `api.nyaymalaw.in` → Service `http://localhost:8000`.

If the **entire** site goes through one tunnel, use **path-based routing**: `/` → frontend; `/auth`, `/chat`, `/conversation`, `/chats`, `/submit_case`, `/interview_step`, `/bareacts` → `http://localhost:8000` (in the tunnel ingress rules).

---

## Summary

- **“Invalid response”** = the URL you’re calling (e.g. https://nyaymalaw.in/auth/login) is returning HTML, not the API.  
- **Fix:** Run the FastAPI backend and either **proxy** its routes (Option A), **host on a subdomain** and set **`VITE_API_BASE`** (Option B), or expose via **Cloudflare Tunnel** and point the frontend to it (Option C); then rebuild/redeploy the frontend where needed.
