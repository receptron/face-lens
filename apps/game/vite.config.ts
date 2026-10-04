import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const root = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  // Relative asset paths, so the build works under any sub-path.
  base: "./",
  // Use the library's source directly, so edits to it show up without a rebuild.
  resolve: { alias: { "@receptron/face-lens": join(root, "../../packages/face-lens/src/index.ts") } },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
