"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { BASE, HEAD_NAME, d8Receivers, hillshadeRGB, leadLabel, probRGBA, upsample } from "@/lib/risk";

const UP = 4; // display upsampling of the 4 km grid (bilinear)

function toImage(px, size) {
  const c = document.createElement("canvas");
  c.width = size; c.height = size;
  c.getContext("2d").putImageData(new ImageData(px, size, size), 0, 0);
  return c;
}

function baseImage(scene, base) {
  if (base === "terrain") {
    const { px, size } = hillshadeRGB(scene.layers.dem, UP);
    return toImage(px, size);
  }
  const spec = BASE[base];
  const { data, size } = upsample(scene.layers[base], UP);
  const px = new Uint8ClampedArray(size * size * 4);
  data.forEach((v, i) => {
    const [r, g, b] = spec.color(v);
    px[i * 4] = r; px[i * 4 + 1] = g; px[i * 4 + 2] = b; px[i * 4 + 3] = 255;
  });
  return toImage(px, size);
}

function probImage(grid) {
  const { data, size } = upsample(grid, UP);
  const px = new Uint8ClampedArray(size * size * 4);
  data.forEach((p, i) => {
    const [r, g, b, a] = probRGBA(p);
    px[i * 4] = r; px[i * 4 + 1] = g; px[i * 4 + 2] = b; px[i * 4 + 3] = a * 255;
  });
  return toImage(px, size);
}

// Cell-edge outline of a binary mask, as SVG path data in cell units.
function outline(mask) {
  const n = mask.length;
  const on = (i, j) => i >= 0 && j >= 0 && i < n && j < n && mask[i][j] === 1;
  let d = "";
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    if (!on(i, j)) continue;
    if (!on(i - 1, j)) d += `M${j},${i}h1`;
    if (!on(i + 1, j)) d += `M${j},${i + 1}h1`;
    if (!on(i, j - 1)) d += `M${j},${i}v1`;
    if (!on(i, j + 1)) d += `M${j + 1},${i}v1`;
  }
  return d;
}

export default function RiskMap({ scene, head, leadIdx, base, showDrain, showZones, showObs, zoneId, onZone }) {
  const wrap = useRef(null), cv = useRef(null);
  const [size, setSize] = useState(640);
  const n = scene.grid, km = n * scene.dx_km;
  const lead = scene.leads_h[leadIdx];

  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setSize(Math.round(e.contentRect.width)));
    ro.observe(wrap.current);
    return () => ro.disconnect();
  }, []);

  const baseImg = useMemo(() => baseImage(scene, base), [scene, base]);
  const probImg = useMemo(() => probImage(scene.prob[head][leadIdx]), [scene, head, leadIdx]);
  const drains = useMemo(() => {
    const rec = d8Receivers(scene.layers.dem), segs = [];
    scene.layers.channel.forEach((row, i) => row.forEach((c, j) => {
      const r = rec[i * n + j];
      if (c && r >= 0 && scene.layers.channel[Math.floor(r / n)][r % n]) segs.push([j + 0.5, i + 0.5, (r % n) + 0.5, Math.floor(r / n) + 0.5]);
    }));
    return segs;
  }, [scene, n]);
  const obsPath = useMemo(() => outline(scene.obs[head][leadIdx]), [scene, head, leadIdx]);

  useEffect(() => {
    const c = cv.current, dpr = window.devicePixelRatio || 1;
    c.width = size * dpr; c.height = size * dpr;
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
    g.globalAlpha = base === "terrain" ? 1 : 0.85;
    g.drawImage(baseImg, 0, 0, size, size);
    g.globalAlpha = 1;
    g.drawImage(probImg, 0, 0, size, size);
  }, [baseImg, probImg, size, base]);

  const s = size / n;                          // px per cell
  const click = (e) => {
    const r = wrap.current.getBoundingClientRect();
    const j = Math.floor(((e.clientX - r.left) / r.width) * n), i = Math.floor(((e.clientY - r.top) / r.height) * n);
    const z = scene.zones.find((z) => z.row === Math.floor(i / 8) && z.col === Math.floor(j / 8));
    if (z) onZone(z.id);
  };
  const sel = scene.zones.find((z) => z.id === zoneId);
  const baseSpec = BASE[base];

  return (
    <div className="map" ref={wrap} onClick={click} role="img"
      aria-label={`${HEAD_NAME[head]} probability at ${leadLabel(lead)}, scene ${scene.id}`}>
      <canvas ref={cv} />
      <svg viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
        {showDrain && drains.map(([x1, y1, x2, y2], k) => (
          <line key={k} x1={x1 * s} y1={y1 * s} x2={x2 * s} y2={y2 * s} stroke="#8FB4E0" strokeOpacity=".45" strokeWidth="1.3" strokeLinecap="round" />
        ))}
        {showZones && [1, 2, 3].map((k) => (
          <g key={k} stroke="#EDF2F9" strokeOpacity=".16" strokeDasharray="3 5">
            <line x1={k * 8 * s} y1="0" x2={k * 8 * s} y2={size} />
            <line x1="0" y1={k * 8 * s} x2={size} y2={k * 8 * s} />
          </g>
        ))}
        {showZones && scene.zones.map((z) => (
          <text key={z.id} x={z.col * 8 * s + 7} y={z.row * 8 * s + 16} fill="#C8D4E2" fillOpacity=".75" fontSize="11" fontFamily="var(--mono)"
            paintOrder="stroke" stroke="#0a0f18" strokeWidth="3">{z.id}</text>
        ))}
        {showObs && obsPath && (
          <path d={obsPath} transform={`scale(${s})`} fill="none" stroke="#FFFFFF" strokeWidth={2 / s} strokeDasharray={`${4 / s} ${3 / s}`} />
        )}
        {sel && <rect x={sel.col * 8 * s + 1} y={sel.row * 8 * s + 1} width={8 * s - 2} height={8 * s - 2} fill="none" stroke="#C77D26" strokeWidth="2" rx="3" />}
        {[0, 32, 64, 96, 128].map((k) => (
          <text key={k} x={Math.min(size - 18, Math.max(4, (k / km) * size))} y={size - 6} fill="#8CA0C2" fontSize="9.5" fontFamily="var(--mono)"
            textAnchor={k === 0 ? "start" : k === 128 ? "end" : "middle"} paintOrder="stroke" stroke="#0a0f18" strokeWidth="3">{k} km</text>
        ))}
        <g transform={`translate(${size - 30},${size - 64})`}>
          <path d="M0,-15 L7,6 L0,1 L-7,6 z" fill="#C8D4E2" />
          <text y="19" textAnchor="middle" fill="#8CA0C2" fontSize="10" fontWeight="600">N</text>
        </g>
      </svg>
      <div className="valid">
        <div className="t num">{leadLabel(lead)}</div>
        <div className="s">Valid time · issued at T+0</div>
      </div>
      <div className="legend">
        <span>{HEAD_NAME[head]} probability</span>
        <div className="ramp" />
        <div className="ticks"><span>0.2</span><span>0.35</span><span>0.55</span><span>0.75</span><span>1.0</span></div>
        {showObs && <span><span style={{ display: "inline-block", width: 18, borderTop: "2px dashed #fff", verticalAlign: "middle", marginRight: 6 }} />Observed event</span>}
        {base !== "terrain" && <span>Base: {baseSpec.label} ({baseSpec.range[0]} to {baseSpec.range[1]} {baseSpec.unit})</span>}
      </div>
    </div>
  );
}
