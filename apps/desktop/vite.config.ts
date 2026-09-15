/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import path from "path";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],

  root: ".",
  base: "./",

  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },

  build: {
    outDir: "dist",
    emptyOutDir: true,
    target: "esnext",
    minify: "esbuild",
  },

  server: {
    port: 1420,
    strictPort: true,
  },

  clearScreen: false,
  envPrefix: ["VITE_", "TAURI_"],

  test: {
    environment: "jsdom",
    include: ["src/**/*.test.{ts,tsx}"],
    // Globals let @testing-library/react register its afterEach cleanup without a setup file.
    globals: true,
    restoreMocks: true,
  },
});
