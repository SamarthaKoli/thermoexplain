import { defineConfig, type Plugin } from "vitest/config";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const frontendRoot = fileURLToPath(new URL(".", import.meta.url));

function overviewAtRoot(): Plugin {
  const redirectToOverview = (
    request: { url?: string },
    response: {
      writeHead: (status: number, headers: Record<string, string>) => void;
      end: () => void;
    },
    next: () => void,
  ) => {
    if (request.url?.split("?")[0] !== "/") {
      next();
      return;
    }
    response.writeHead(302, { Location: "/overview.html" });
    response.end();
  };

  return {
    name: "thermoexplain-overview-root",
    configureServer(server) {
      server.middlewares.use(redirectToOverview);
    },
    configurePreviewServer(server) {
      server.middlewares.use(redirectToOverview);
    },
  };
}

export default defineConfig({
  plugins: [react(), overviewAtRoot()],
  build: {
    rollupOptions: {
      input: {
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
