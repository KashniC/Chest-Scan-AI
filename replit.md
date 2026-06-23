# ChestScan AI

A CNN-powered chest X-ray classifier that detects four conditions — Healthy, Pneumonia, Tuberculosis, and COVID-19 — from uploaded X-ray images.

## Run & Operate

- `PORT=22267 python artifacts/chestscan-ai/app.py` — run the ChestScan AI server
- `pnpm --filter @workspace/api-server run dev` — run the background API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- ML backend: Python 3.11, FastAPI, Uvicorn, TensorFlow/Keras
- Frontend: Vanilla HTML/CSS/JS served directly by FastAPI
- DB: PostgreSQL + Drizzle ORM (not yet used)
- API codegen: Orval (from OpenAPI spec)

## Where things live

- `artifacts/chestscan-ai/app.py` — FastAPI server (ML inference + HTML serving)
- `artifacts/chestscan-ai/templates/index.html` — Frontend UI
- `artifacts/chestscan-ai/cnn_model_lung_detection.keras` — Trained CNN model (128×128 input)
- `artifacts/chestscan-ai/sample_images/` — 80 sample X-ray images (healthy, pneumonia, tuberculosis, covid)
- `artifacts/chestscan-ai/class_names.json` — Class label list
- `artifacts/chestscan-ai/model_setup.py` — Path resolver for model files
- `lib/api-spec/openapi.yaml` — API spec (health check only)

## Architecture decisions

- Python FastAPI serves both the HTML frontend and the ML inference API to avoid Node.js/Python inter-process complexity.
- Routes use `/app/` prefix (not `/api/`) to avoid conflict with the existing Express API server at `/api`.
- The model runs on CPU (no CUDA); TF oneDNN warnings are cosmetic.
- All model files are local (no HuggingFace Hub downloads needed — `hub_repo_id: null`).
- Sample images use CSS `filter: invert(1)` since X-ray PNGs are stored as bright-on-dark.

## Product

- Hero section with tagline and condition chips
- Drag-and-drop image upload with preview
- 80 sample X-ray images filterable by condition (Healthy / Pneumonia / TB / COVID-19)
- Animated confidence bars for all 4 classes after prediction
- Fun facts panel about lung diseases
- Model info card (input size, status, class count)

## Gotchas

- Always run with `PORT=22267` so the Replit proxy routes correctly.
- The `/api/*` path prefix is taken by the Express API server — Python routes use `/app/*`.
- TF CUDA errors in logs are harmless (CPU-only environment).

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
