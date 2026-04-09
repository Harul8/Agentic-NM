import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Help Rollup resolve core-js internals when ?commonjs-external is appended
function stripCommonJsExternal() {
  return {
    name: "strip-commonjs-external",
    resolveId(id, importer) {
      if (id && id.endsWith("?commonjs-external")) {
        const clean = id.replace(/\?commonjs-external$/, "");
        if (importer && clean.startsWith(".")) {
          return path.resolve(path.dirname(importer), clean);
        }
        return clean;
      }
      return null;
    },
  };
}

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    include: ["core-js"],
  },
  build: {
    chunkSizeWarningLimit: 600,
    commonjsOptions: {
      include: [/core-js/, /node_modules/],
      transformMixedEsModules: true,
      defaultIsModuleExports: "auto",
      requireReturnsDefault: "auto",
    },
    rollupOptions: {
      plugins: [stripCommonJsExternal()],
      output: {
        manualChunks(id) {
          if (!id || !id.includes("node_modules")) return null;
          if (id.includes("react-dom") || id.includes("react/")) return "vendor-react";
          if (id.includes("jspdf") || id.includes("html2canvas")) return "vendor-export";
          if (id.includes("xlsx")) return "vendor-xlsx";
          if (id.includes("react-markdown") || id.includes("rehype") || id.includes("parse5")) return "vendor-markdown";
          if (id.includes("core-js")) return "vendor-corejs";
          return null;
        },
      },
    },
  },
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
      },
      "/bareacts": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        secure: false
      },
      "/caselaws": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        secure: false
      },
      "/eval": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        secure: false
      }
    }
  }
});
