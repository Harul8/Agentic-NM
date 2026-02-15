import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: ["nyaymalaw.in", "www.nyaymalaw.in"],
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
