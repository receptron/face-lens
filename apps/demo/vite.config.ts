import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

// Cross-origin isolation lets ONNX Runtime use threaded WASM.
const root = fileURLToPath(new URL(".", import.meta.url));

const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

// Dev-only endpoint: the capture panel posts face crops here, and they land in
// teacher/data/captures/<tag>/ for teacher labeling and training.
function capturePlugin(): Plugin {
  return {
    name: "face-lens-capture",
    configureServer(server) {
      server.middlewares.use("/api/capture", (req, res) => {
        if (req.method !== "POST") {
          res.statusCode = 405;
          res.end();
          return;
        }
        const chunks: Buffer[] = [];
        req.on("data", (c: Buffer) => chunks.push(c));
        req.on("end", () => {
          try {
            const { tag, image } = JSON.parse(Buffer.concat(chunks).toString());
            if (!/^[a-z0-9=_-]+$/.test(tag)) throw new Error(`bad tag ${tag}`);
            const dir = join(root, "../../teacher/data/captures", tag);
            mkdirSync(dir, { recursive: true });
            const file = join(dir, `${Date.now()}-${Math.random().toString(36).slice(2, 6)}.jpg`);
            writeFileSync(file, Buffer.from(image.replace(/^data:image\/jpeg;base64,/, ""), "base64"));
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ ok: true }));
          } catch (e) {
            res.statusCode = 400;
            res.end(JSON.stringify({ ok: false, error: String(e) }));
          }
        });
      });
    },
  };
}

export default defineConfig({
  // Relative asset paths, so the build works under any sub-path (e.g. swarmstrike.com/lens/).
  base: "./",
  plugins: [capturePlugin()],
  // Use the library's source directly, so edits to it show up without a rebuild.
  resolve: { alias: { "@receptron/face-lens": join(root, "../../packages/face-lens/src/index.ts") } },
  server: { headers: isolation },
  preview: { headers: isolation },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
