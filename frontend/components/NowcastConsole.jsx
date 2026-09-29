"use client";

import { useEffect, useMemo, useState } from "react";
import { loadIndex, loadScene } from "@/lib/data";
import { BASE, HEADS, HEAD_NAME, TH, TIER, leadLabel, zoneStatus } from "@/lib/risk";
import RiskMap from "./RiskMap";
import ModelPanel from "./ModelPanel";
import { AlertQueue, PrecursorChart, XaiPanel, ZoneDetail, ZoneTable } from "./Panels";

function SceneStrip({ scenes, current, onPick }) {
  const tone = (p) => TIER[p >= TH.critical ? "critical" : p >= TH.warning ? "warning" : p >= TH.watch ? "watch" : "normal"].dot;
  return (
    <div className="scenes" role="group" aria-label="Test scenes">
      {scenes.map((s) => (
        <button key={s.id} type="button" className="scene" aria-pressed={s.id === current} onClick={() => onPick(s.id)}
          title={`Peak forecast: thunderstorm ${s.peak.thunderstorm}, cloudburst ${s.peak.cloudburst}, flash flood ${s.peak.flash_flood}`}>
          <span className="sid">Scene {s.id}</span>
          <span className="dots">
            {HEADS.map((h) => <span key={h} className="d" style={{ background: tone(s.peak[h]), opacity: s.peak[h] < 0.2 ? 0.35 : 1 }} />)}
          </span>
        </button>
      ))}
    </div>
  );
}

export default function NowcastConsole() {
  const [index, setIndex] = useState(null);
  const [scene, setScene] = useState(null);
  const [sceneId, setSceneId] = useState(null);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState("ops");
  const [head, setHead] = useState("cloudburst");
  const [leadIdx, setLeadIdx] = useState(0);
  const [base, setBase] = useState("terrain");
  const [showDrain, setShowDrain] = useState(true);
  const [showZones, setShowZones] = useState(true);
  const [showObs, setShowObs] = useState(false);
  const [zoneId, setZoneId] = useState(null);
  const [xaiHead, setXaiHead] = useState("cloudburst");

  useEffect(() => {
    loadIndex().then((ix) => { setIndex(ix); setSceneId(ix.scenes[0].id); }).catch((e) => setError(e.message));
  }, []);
  useEffect(() => {
    if (!sceneId) return;
    loadScene(sceneId).then((s) => {
      setScene(s);
      const ranked = s.zones.map((z) => ({ z, st: zoneStatus(z, s.leads_h) })).sort((a, b) => b.st.p - a.st.p);
      setZoneId(ranked[0].z.id);
    }).catch((e) => setError(e.message));
  }, [sceneId]);

  const summary = useMemo(() => {
    if (!scene) return null;
    const st = scene.zones.map((z) => ({ z, s: zoneStatus(z, scene.leads_h) }));
    const high = st.filter(({ s }) => s.tier === "warning" || s.tier === "critical");
    const onsets = st.filter(({ s }) => s.onset).sort((a, b) => a.s.onset - b.s.onset);
    // zone-level verification for the selected head and lead at the Warning threshold
    let hit = 0, miss = 0, fa = 0;
    scene.zones.forEach((z) => {
      const f = z.p[head][leadIdx] >= TH.warning, o = z.obs[head][leadIdx] === 1;
      if (f && o) hit++; else if (o) miss++; else if (f) fa++;
    });
    const counts = Object.fromEntries(HEADS.map((h) => [h, scene.zones.filter((z) => z.p[h][leadIdx] >= TH.watch).length]));
    const hot = Object.fromEntries(HEADS.map((h) => [h, scene.zones.some((z) => z.p[h][leadIdx] >= TH.warning)]));
    return { high, first: onsets[0], veto: st.filter(({ s }) => s.tier === "warning").length, hit, miss, fa, counts, hot };
  }, [scene, head, leadIdx]);

  const zone = scene?.zones.find((z) => z.id === zoneId);
  const zoneOf = (cell) => `${"ABCD"[Math.floor(cell[0] / 8)]}${Math.floor(cell[1] / 8) + 1}`;

  return (
    <div className="page">
      <header className="mast">
        <div className="mast-in">
          <div className="brand"><h1>DRISHTI</h1><span className="sub">Severe Weather Nowcast · thunderstorm, cloudburst, flash flood</span></div>
          <div className="feeds">
            <span className="feed"><span className="led" />Model <b>NowcastNet v0</b></span>
            <span className="feed"><span className="led" />Inputs: IWV · CTT · QPE · CAPE/CIN · wind · DEM</span>
            <span className="feed"><span className="led" />4 km grid · 2–6 h horizon</span>
            <span className="badge-syn" title="Held-out synthetic test scenes. Probabilities are real model output.">Synthetic test data</span>
          </div>
        </div>
        <nav className="tabs" role="tablist">
          <button className="tab" role="tab" aria-selected={tab === "ops"} onClick={() => setTab("ops")}>Operations</button>
          <button className="tab" role="tab" aria-selected={tab === "model"} onClick={() => setTab("model")}>Model performance</button>
        </nav>
      </header>

      {error && <div className="err">Could not load model output: {error}. Run <span className="mono">python scripts/export_frontend_data.py</span> first.</div>}
      {!error && !index && <div className="err" style={{ color: "#8CA0C2" }}>Loading model output…</div>}

      {index && tab === "model" && <div style={{ marginTop: 14 }}><ModelPanel index={index} /></div>}

      {index && tab === "ops" && scene && zone && (
        <div className="wrap" style={{ marginTop: 14 }}>
          <SceneStrip scenes={index.scenes} current={sceneId} onPick={setSceneId} />

          <div className="bar" role="toolbar" aria-label="Forecast controls">
            <div className="grp"><span className="lbl">Hazard head</span>
              <div className="seg">
                {HEADS.map((h) => (
                  <button key={h} type="button" aria-pressed={h === head} onClick={() => setHead(h)}>
                    {HEAD_NAME[h]} <span className={`cnt${summary.hot[h] ? " hot" : ""}`}>{summary.counts[h]}</span>
                  </button>
                ))}
              </div>
            </div>
            <div className="grp"><span className="lbl">Lead time</span>
              <div className="seg">
                {scene.leads_h.map((L, j) => <button key={L} type="button" aria-pressed={j === leadIdx} onClick={() => setLeadIdx(j)}>{leadLabel(L)}</button>)}
              </div>
            </div>
            <div className="grp"><span className="lbl">Base</span>
              <select className="sel" id="base-layer" value={base} onChange={(e) => setBase(e.target.value)} aria-label="Base layer">
                {Object.entries(BASE).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
              </select>
              <label className="chk"><input type="checkbox" id="ly-drn" checked={showDrain} onChange={(e) => setShowDrain(e.target.checked)} /> Drainage</label>
              <label className="chk"><input type="checkbox" id="ly-zone" checked={showZones} onChange={(e) => setShowZones(e.target.checked)} /> Zones</label>
              <label className="chk"><input type="checkbox" id="ly-obs" checked={showObs} onChange={(e) => setShowObs(e.target.checked)} /> Show observed events</label>
            </div>
          </div>

          <section className="stats" aria-label="Situation summary">
            <div className="stat"><span className="lbl">Zones at Warning or above</span>
              <span className="v num">{summary.high.length} <small>of 16</small></span>
              <span className="d">{summary.high.map(({ z }) => z.id).join(", ") || "None"}</span></div>
            <div className="stat"><span className="lbl">Earliest expected onset</span>
              <span className="v num">{summary.first ? leadLabel(summary.first.s.onset) : "–"}</span>
              <span className="d">{summary.first ? `Zone ${summary.first.z.id} · ${HEAD_NAME[summary.first.s.head].toLowerCase()}` : "No severe hazard in 6 h"}</span></div>
            <div className="stat"><span className="lbl">Alerts in veto window</span>
              <span className="v num">{summary.veto}</span>
              <span className="d">Auto-broadcast via SACHET unless cancelled</span></div>
            <div className="stat"><span className="lbl">Verification · {HEAD_NAME[head].toLowerCase()} {leadLabel(scene.leads_h[leadIdx])}</span>
              <span className="v num">{summary.hit} <small>hit</small> · {summary.miss} <small>miss</small> · {summary.fa} <small>false</small></span>
              <span className="d">Zone level at p ≥ {TH.warning} against observed events</span></div>
          </section>

          <div className="grid">
            <div className="stack">
              <section className="panel" aria-labelledby="map-h">
                <div className="ph"><h2 id="map-h">Hyper-local probability risk map</h2>
                  <span className="meta">Scene {scene.id} · {HEAD_NAME[head]} head · 128 km domain, 4 km grid, smoothed for display</span></div>
                <RiskMap scene={scene} head={head} leadIdx={leadIdx} base={base} showDrain={showDrain} showZones={showZones}
                  showObs={showObs} zoneId={zoneId} onZone={setZoneId} />
                <p className="note">Click a zone to inspect it. Probabilities are the trained model's output on a held-out synthetic scene.</p>
              </section>
              <PrecursorChart scene={scene} zone={zone} head={head} />
            </div>
            <div className="stack">
              <ZoneDetail scene={scene} zone={zone} head={head} leadIdx={leadIdx} showObs={showObs} />
              <XaiPanel scene={scene} head={xaiHead} setHead={setXaiHead} zoneOf={zoneOf} />
              <AlertQueue scene={scene} zoneId={zoneId} onZone={setZoneId} showObs={showObs} />
            </div>
          </div>
          <ZoneTable scene={scene} zoneId={zoneId} onZone={setZoneId} showObs={showObs} />

          <div className="foot">
            <span>Output of <span className="mono">artifacts/nowcast_v0.pt</span> exported by <span className="mono">scripts/export_frontend_data.py</span></span>
            <span>Synthetic test scenes · not observational data</span>
          </div>
        </div>
      )}
    </div>
  );
}
