import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => (
          path === "/api/health" || path === "/api/generate"
            ? path.replace(/^\/api/, "")
            : path
        ),
      },
    },
  },
});
