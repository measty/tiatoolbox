import { fileURLToPath, URL } from "node:url";

import react from "@vitejs/plugin-react-swc";
import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  plugins: [react()],
  build: {
    emptyOutDir: true,
    outDir: fileURLToPath(
      new URL("../../tiatoolbox/data/visualization/app", import.meta.url),
    ),
    sourcemap: false,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:5000",
      "/tileserver": "http://127.0.0.1:5000",
    },
  },
});
