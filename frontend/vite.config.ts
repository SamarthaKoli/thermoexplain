import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const frontendRoot = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        app: resolve(frontendRoot, "index.html"),
        apiWorkspace: resolve(frontendRoot, "app.html"),
        overview: resolve(frontendRoot, "overview.html"),
        design: resolve(frontendRoot, "design.html"),
        explainable: resolve(frontendRoot, "explainable.html"),
        generative: resolve(frontendRoot, "generative.html"),
        validation: resolve(frontendRoot, "validation.html"),
        thermal: resolve(frontendRoot, "thermal.html"),
      },
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/tests/setup.ts",
  },
});
