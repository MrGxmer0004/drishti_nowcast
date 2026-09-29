// Data access. Today: model inference exported by
// scripts/export_frontend_data.py into public/data. Later: the FastAPI
// backend, by setting NEXT_PUBLIC_API_BASE (same JSON shapes).

const API = (process.env.NEXT_PUBLIC_API_BASE || "").replace(/\/$/, "");
const BASE = process.env.NEXT_PUBLIC_BASE_PATH || "";

async function getJSON(path) {
  const url = API ? `${API}${path.replace(/\.json$/, "")}` : `${BASE}/data${path}`;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Could not load ${url} (HTTP ${r.status})`);
  return r.json();
}

export const loadIndex = () => getJSON("/index.json");
export const loadScene = (id) => getJSON(`/${id}.json`);
