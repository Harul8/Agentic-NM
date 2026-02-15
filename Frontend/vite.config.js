import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: ["nyaymalaw.in", "www.nyaymalaw.in"],
    hmr: false, // disable WebSocket HMR when using tunnel (nyaymalaw.in → localhost:5173)
    proxy: {
      "/search": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        secure: false
      },
      "/chat": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        secure: false
      },
      "/conversation": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        secure: false
      }
    }
  }
});
