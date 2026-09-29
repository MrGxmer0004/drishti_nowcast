// Shared vocabulary: hazards, tiers, colour ramps, terrain shading, drainage.

export const HEADS = ["thunderstorm", "cloudburst", "flash_flood"];
export const HEAD_NAME = { thunderstorm: "Thunderstorm", cloudburst: "Cloudburst", flash_flood: "Flash flood" };
export const HEAD_COLOR = { thunderstorm: "#9FB2CE", cloudburst: "#E8635A", flash_flood: "#7FA8D9" };
export const TH = { watch: 0.35, warning: 0.55, critical: 0.75 };

export const TIER = {
  normal:   { label: "Normal",   fg: "#9FB2CE", bg: "#132038", ring: "#2A3E60", dot: "#6E85AC" },
  watch:    { label: "Watch",    fg: "#D9AE45", bg: "#26200E", ring: "#5C4A1E", dot: "#D9AE45" },
  warning:  { label: "Warning",  fg: "#E28F4E", bg: "#2A1C0F", ring: "#6B4420", dot: "#E28F4E" },
  critical: { label: "Critical", fg: "#E8635A", bg: "#301418", ring: "#6E2B2A", dot: "#DB4A42" },
};
export const tierOf = (p) => (p >= TH.critical ? "critical" : p >= TH.warning ? "warning" : p >= TH.watch ? "watch" : "normal");

// Zone status: severe heads (cloudburst, flash flood) set the tier; a
// thunderstorm on its own never raises a zone above Watch.
export function zoneStatus(zone, leads) {
  let best = { p: 0, head: null, lead: null };
  for (const h of ["cloudburst", "flash_flood"]) {
    zone.p[h].forEach((p, j) => { if (p > best.p) best = { p, head: h, lead: leads[j] }; });
  }
  const tsMax = Math.max(...zone.p.thunderstorm);
  let tier = tierOf(best.p);
  if (tier === "normal" && tsMax >= TH.watch) { tier = "watch"; best = { p: tsMax, head: "thunderstorm", lead: leads[zone.p.thunderstorm.indexOf(tsMax)] }; }
  const onsetIdx = leads.findIndex((_, j) => zone.p.cloudburst[j] >= TH.warning || zone.p.flash_flood[j] >= TH.warning);
  return { tier, ...best, onset: onsetIdx >= 0 ? leads[onsetIdx] : null };
}

// Probability ramp: transparent below 0.2, then the tier colours.
const STOPS = [[0.2, [110, 133, 172, 0]], [0.35, [217, 174, 69, 0.45]], [0.55, [226, 143, 78, 0.62]],
               [0.75, [219, 74, 66, 0.75]], [1, [255, 107, 94, 0.85]]];
export function probRGBA(p) {
  if (p <= STOPS[0][0]) return [0, 0, 0, 0];
  for (let i = 1; i < STOPS.length; i++) {
    if (p <= STOPS[i][0]) {
      const [p0, c0] = STOPS[i - 1], [p1, c1] = STOPS[i], f = (p - p0) / (p1 - p0);
      return c0.map((v, k) => v + (c1[k] - v) * f);
    }
  }
  return STOPS[STOPS.length - 1][1];
}

const lerp = (a, b, f) => a + (b - a) * f;
function ramp(stops, t) {
  t = Math.max(0, Math.min(1, t));
  const n = stops.length - 1, i = Math.min(n - 1, Math.floor(t * n)), f = t * n - i;
  return stops[i].map((v, k) => lerp(v, stops[i + 1][k], f));
}
// Base-layer colour maps, all tuned to sit under the probability overlay.
export const BASE = {
  terrain: { label: "Terrain (DEM)" },
  iwv_rise: { label: "IWV rise, 90 min", unit: "mm", range: [-2, 9],
    color: (v) => ramp([[14, 26, 44], [22, 52, 92], [44, 104, 170], [120, 180, 230]], (v + 2) / 11) },
  ctt: { label: "Cloud-top temperature", unit: "K", range: [205, 285],
    color: (v) => ramp([[240, 240, 250], [180, 90, 170], [80, 40, 110], [20, 30, 52]], (v - 205) / 80) },
  convergence: { label: "850 hPa convergence", unit: "1e-5 s⁻¹", range: [-4, 4],
    color: (v) => ramp([[70, 110, 190], [22, 34, 56], [190, 90, 80]], (v + 4) / 8) },
};

// Bilinear upsample of a square grid.
export function upsample(grid, factor) {
  const n = grid.length, m = n * factor, out = new Float32Array(m * m);
  for (let y = 0; y < m; y++) {
    const gy = Math.min(n - 1.001, Math.max(0, (y + 0.5) / factor - 0.5)), y0 = Math.floor(gy), fy = gy - y0;
    for (let x = 0; x < m; x++) {
      const gx = Math.min(n - 1.001, Math.max(0, (x + 0.5) / factor - 0.5)), x0 = Math.floor(gx), fx = gx - x0;
      const a = grid[y0][x0], b = grid[y0][x0 + 1], c = grid[y0 + 1][x0], d = grid[y0 + 1][x0 + 1];
      out[y * m + x] = lerp(lerp(a, b, fx), lerp(c, d, fx), fy);
    }
  }
  return { data: out, size: m };
}

// Hillshade from an upsampled DEM, lit from the north-west.
export function hillshadeRGB(dem, factor = 4) {
  const { data, size } = upsample(dem, factor);
  let lo = Infinity, hi = -Infinity;
  for (const v of data) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const px = new Uint8ClampedArray(size * size * 4), cell = 4000 / factor;
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const i = y * size + x;
    const zx = (data[y * size + Math.min(size - 1, x + 1)] - data[y * size + Math.max(0, x - 1)]) / (2 * cell);
    const zy = (data[Math.min(size - 1, y + 1) * size + x] - data[Math.max(0, y - 1) * size + x]) / (2 * cell);
    const shade = Math.max(0, Math.min(1, 0.5 + (-zx - zy) * 14));
    const h = (data[i] - lo) / (hi - lo + 1e-9);
    px[i * 4] = 12 + shade * 40 + h * 30;
    px[i * 4 + 1] = 22 + shade * 52 + h * 36;
    px[i * 4 + 2] = 38 + shade * 64 + h * 40;
    px[i * 4 + 3] = 255;
  }
  return { px, size };
}

// D8 receivers for drawing the drainage network (mirrors risk_engine.py).
export function d8Receivers(dem) {
  const n = dem.length, rec = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    let best = -1, drop = 0;
    for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) {
      if (!di && !dj) continue;
      const a = i + di, b = j + dj;
      if (a < 0 || b < 0 || a >= n || b >= n) continue;
      const d = (dem[i][j] - dem[a][b]) / Math.hypot(di, dj);
      if (d > drop) { drop = d; best = a * n + b; }
    }
    rec.push(best);
  }
  return rec;
}

export const leadLabel = (h) => `T+${h} h`;
