"use client";

import { useEffect, useState } from "react";
import { HEADS, HEAD_COLOR, HEAD_NAME, TH, TIER, leadLabel, tierOf, zoneStatus } from "@/lib/risk";

export function Pill({ tier, children }) {
  const t = TIER[tier];
  return (
    <span className="pill" style={{ color: t.fg, background: t.bg, borderColor: t.ring }}>
      <i style={{ background: t.dot }} />{children ?? t.label}
    </span>
  );
}

/* ---------------- zone detail: hazard x lead matrix ---------------- */
export function ZoneDetail({ scene, zone, head, leadIdx, showObs }) {
  const st = zoneStatus(zone, scene.leads_h);
  return (
    <section className="panel" aria-labelledby="zone-h">
      <div className="wd-top">
        <div>
          <div className="wd-name" id="zone-h">Zone {zone.id}</div>
          <div className="wd-sub">32 × 32 km · 64 grid cells · mean elevation {zone.elev_mean.toLocaleString("en-IN")} m · peak CAPE {zone.cape_max.toLocaleString("en-IN")} J/kg</div>
        </div>
        <Pill tier={st.tier} />
      </div>
      <div className="facts">
        <div><div className="k">Expected onset</div><div className="v num">{st.onset ? leadLabel(st.onset) : "None in 6 h"}</div></div>
        <div><div className="k">Driving hazard</div><div className="v">{st.head ? HEAD_NAME[st.head] : "–"}</div></div>
        <div><div className="k">Peak probability</div><div className="v num">{st.p.toFixed(2)}{st.lead ? <span className="dim" style={{ fontSize: 12, fontWeight: 500 }}> at {leadLabel(st.lead)}</span> : null}</div></div>
      </div>
      <span className="lbl">Multi-task output · zone maximum by lead time</span>
      <table className="mx" aria-label="Probability by hazard and lead time">
        <thead><tr><th />{scene.leads_h.map((L) => <th key={L}>{leadLabel(L)}</th>)}</tr></thead>
        <tbody>
          {HEADS.map((h) => (
            <tr key={h}>
              <th className="rh" scope="row">{HEAD_NAME[h]}</th>
              {zone.p[h].map((p, j) => {
                const t = TIER[tierOf(p)];
                return (
                  <td key={j} className={`num${h === head && j === leadIdx ? " cur" : ""}`}
                    style={{ background: t.bg, borderColor: t.ring, color: t.fg }}>
                    {p.toFixed(2)}
                    {showObs && zone.obs[h][j] === 1 && <span className="obs" title="Event observed" />}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {showObs && <p className="note">A white dot marks a lead time at which the event was actually observed in this zone.</p>}
    </section>
  );
}

/* ---------------- XAI ---------------- */
export function XaiPanel({ scene, head, setHead, zoneOf }) {
  const avail = HEADS.filter((h) => scene.xai[h]);
  const h = scene.xai[head] ? head : avail[0];
  const x = h ? scene.xai[h] : null;
  const fams = x ? Object.entries(x.families) : [];
  const top = fams.length ? Math.max(...fams.map(([, v]) => v)) : 1;
  return (
    <section className="panel" aria-labelledby="xai-h">
      <div className="ph">
        <h2 id="xai-h">Why this forecast</h2>
        <span className="meta">Integrated Gradients · share of attribution</span>
      </div>
      {!x ? <div className="empty">No hazard in this scene reaches p ≥ 0.20, so there is no forecast to explain.</div> : (
        <>
          <div className="xhead">
            <div className="seg" role="group" aria-label="Explain hazard">
              {avail.map((k) => (
                <button key={k} type="button" aria-pressed={k === h} onClick={() => setHead(k)}>{HEAD_NAME[k]}</button>
              ))}
            </div>
          </div>
          <p className="note" style={{ margin: "0 0 10px" }}>
            Strongest {HEAD_NAME[h].toLowerCase()} forecast in the scene: <b style={{ color: "#EDF2F9" }}>p {x.p.toFixed(2)}</b> at {leadLabel(x.lead_h)}, cell ({x.cell[0]}, {x.cell[1]}) in zone {zoneOf(x.cell)}.
          </p>
          <div className="xai">
            {fams.map(([name, v]) => (
              <div className="xr" key={name}>
                <div className="n">{name}</div>
                <div className="xbar"><span style={{ width: `${(v / top) * 100}%` }} /></div>
                <div className="p num">{Math.round(v * 100)}%</div>
              </div>
            ))}
          </div>
          <p className="note">Attribution from a neutral baseline to the actual inputs, summed per input channel and grouped by predictor family. Computed by <span className="mono">xai.py</span> on the trained model.</p>
        </>
      )}
    </section>
  );
}

/* ---------------- precursor chart ---------------- */
export function PrecursorChart({ scene, zone, head }) {
  const W = 760, H = 300, L = 74, R = 18, x0 = -3, x1 = 6;
  const X = (t) => L + ((t - x0) / (x1 - x0)) * (W - L - R);
  const ts = scene.hist_times_h, se = zone.series;
  const rows = [{ y: 22, h: 60 }, { y: 104, h: 60 }, { y: 190, h: 84 }];
  const iv = se.iwv_max, cv = se.ctt_min;
  const iLo = Math.floor(Math.min(...se.iwv_mean) - 2), iHi = Math.ceil(Math.max(...iv) + 2);
  const cLo = Math.min(205, Math.floor(Math.min(...cv) - 3)), cHi = Math.max(275, Math.ceil(Math.max(...cv) + 3));
  const Yi = (v) => rows[0].y + rows[0].h - ((v - iLo) / (iHi - iLo)) * rows[0].h;
  const Yc = (v) => rows[1].y + ((cHi - v) / (cHi - cLo)) * rows[1].h;
  const Yp = (p) => rows[2].y + rows[2].h - p * rows[2].h;
  const line = (xs, ys) => xs.map((x, i) => `${i ? "L" : "M"}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(" ");
  const lab = (r, a, b) => (
    <g>
      <text x={L - 8} y={r.y + 11} textAnchor="end" fill="#8CA0C2" fontSize="10.5" fontWeight="600">{a}</text>
      <text x={L - 8} y={r.y + 25} textAnchor="end" fill="#5E7396" fontSize="9.5">{b}</text>
    </g>
  );
  const leads = scene.leads_h;
  let lx = X(0) + 8;
  return (
    <section className="panel" aria-labelledby="tl-h">
      <div className="ph"><h2 id="tl-h">Precursor signals and forecast</h2><span className="meta">Zone {zone.id} · 30-min satellite frames</span></div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }} role="img"
        aria-label={`Observed IWV and cloud-top temperature and forecast probabilities for zone ${zone.id}`}>
        {rows.map((r, k) => <rect key={k} x={L} y={r.y} width={W - L - R} height={r.h} fill="#0A1528" rx="4" />)}
        <rect x={X(0)} y={12} width={X(6) - X(0)} height={H - 34} fill="#C77D26" fillOpacity=".05" />
        {[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6].map((t) => (
          <g key={t}>
            <line x1={X(t)} y1={14} x2={X(t)} y2={H - 22} stroke="#fff" strokeOpacity=".05" />
            <text x={X(t)} y={H - 8} textAnchor="middle" fill="#8CA0C2" fontSize="10" fontFamily="var(--mono)">{t === 0 ? "T+0" : t > 0 ? `+${t}h` : `${t}h`}</text>
          </g>
        ))}
        <line x1={X(0)} y1={10} x2={X(0)} y2={H - 22} stroke="#C77D26" strokeWidth="1.5" />
        <text x={X(0) + 5} y={14} fill="#E3A15A" fontSize="10" fontWeight="600">Issue time</text>
        <text x={X(-1.5)} y={14} textAnchor="middle" fill="#5E7396" fontSize="10" letterSpacing=".8">OBSERVED</text>
        <text x={X(3.2)} y={14} textAnchor="middle" fill="#5E7396" fontSize="10" letterSpacing=".8">NOWCAST</text>

        {lab(rows[0], "IWV", "mm")}
        <path d={`${line(ts.map(X), se.iwv_mean.map(Yi))}`} fill="none" stroke="#7FA8D9" strokeOpacity=".5" strokeWidth="1.4" strokeDasharray="3 3" />
        <path d={line(ts.map(X), iv.map(Yi))} fill="none" stroke="#7FA8D9" strokeWidth="1.8" />
        <circle cx={X(0)} cy={Yi(iv[iv.length - 1])} r="3.5" fill="#7FA8D9" />
        <text x={X(0) + 8} y={Yi(iv[iv.length - 1]) + 4} fill="#C7D4E5" fontSize="11" fontFamily="var(--mono)">{iv[iv.length - 1].toFixed(1)} max</text>
        <text x={X(0) + 8} y={Yi(se.iwv_mean[6]) + 4 + (Math.abs(Yi(se.iwv_mean[6]) - Yi(iv[6])) < 12 ? 12 : 0)} fill="#7F93B0" fontSize="10" fontFamily="var(--mono)">{se.iwv_mean[6].toFixed(1)} mean</text>

        {lab(rows[1], "CTT", "K, min")}
        <line x1={L} x2={X(0)} y1={Yc(235)} y2={Yc(235)} stroke="#E8635A" strokeOpacity=".5" strokeDasharray="3 3" />
        <text x={L + 4} y={Yc(235) - 3} fill="#E8635A" fillOpacity=".85" fontSize="9">235 K deep convection</text>
        <path d={line(ts.map(X), cv.map(Yc))} fill="none" stroke="#C8D4E2" strokeWidth="1.8" />
        <circle cx={X(0)} cy={Yc(cv[cv.length - 1])} r="3.5" fill="#C8D4E2" />
        <text x={X(0) + 8} y={Yc(cv[cv.length - 1]) + 4} fill="#C7D4E5" fontSize="11" fontFamily="var(--mono)">{cv[cv.length - 1].toFixed(0)} K</text>

        {lab(rows[2], "Prob.", "zone max")}
        {Object.entries(TH).map(([k, v]) => (
          <line key={k} x1={X(0)} x2={W - R} y1={Yp(v)} y2={Yp(v)} stroke={TIER[k].dot} strokeOpacity=".45" strokeDasharray="2 4" />
        ))}
        {HEADS.map((h) => {
          const xs = leads.map(X), ys = zone.p[h].map(Yp), on = h === head;
          return (
            <g key={h}>
              <path d={line(xs, ys)} fill="none" stroke={HEAD_COLOR[h]} strokeWidth={on ? 2.6 : 1.5} strokeOpacity={on ? 1 : 0.7} />
              {xs.map((x, j) => zone.obs[h][j] ? (
                <circle key={j} cx={x} cy={ys[j]} r={on ? 5 : 4} fill={HEAD_COLOR[h]} stroke="#EDF2F9" strokeWidth="1.5" />
              ) : (
                <circle key={j} cx={x} cy={ys[j]} r={on ? 3.2 : 2.4} fill="#0A1528" stroke={HEAD_COLOR[h]} strokeWidth="1.5" />
              ))}
            </g>
          );
        })}
        {HEADS.map((h) => {
          const g = (
            <g key={h}>
              <rect x={lx} y={rows[2].y - 11} width="10" height="3" fill={HEAD_COLOR[h]} />
              <text x={lx + 14} y={rows[2].y - 6} fill="#8CA0C2" fontSize="10">{HEAD_NAME[h]}</text>
            </g>
          );
          lx += HEAD_NAME[h].length * 6 + 34;
          return g;
        })}
        <text x={W - R} y={rows[2].y - 6} textAnchor="end" fill="#8CA0C2" fontSize="10">filled = event observed</text>
      </svg>
    </section>
  );
}

/* ---------------- alert queue ---------------- */
const CAP = { critical: "Extreme · Immediate · Likely", warning: "Severe · Expected · Likely", watch: "Moderate · Expected · Possible" };

export function AlertQueue({ scene, zoneId, onZone, showObs }) {
  const order = { critical: 0, warning: 1, watch: 2 };
  const alerts = scene.zones
    .map((z) => ({ z, st: zoneStatus(z, scene.leads_h) }))
    .filter(({ st }) => st.tier !== "normal")
    .sort((a, b) => order[a.st.tier] - order[b.st.tier] || b.st.p - a.st.p);
  const [clock, setClock] = useState({});
  const [acted, setActed] = useState({});
  useEffect(() => { setClock({}); setActed({}); }, [scene.id]);
  useEffect(() => {
    const t = setInterval(() => setClock((c) => {
      const n = { ...c };
      alerts.forEach(({ z, st }, k) => { if (st.tier === "warning") n[z.id] = Math.max(0, (n[z.id] ?? 300 - k * 37) - 1); });
      return n;
    }), 1000);
    return () => clearInterval(t);
  }, [scene.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const fmt = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  return (
    <section className="panel" aria-labelledby="al-h">
      <div className="ph"><h2 id="al-h">Automated categorized alerts</h2><span className="meta">Tiered dispatch · CAP</span></div>
      {!alerts.length && <div className="empty">No zone reaches Watch in the next 6 hours. Nothing to dispatch.</div>}
      <div className="alerts">
        {alerts.map(({ z, st }, k) => {
          const t = TIER[st.tier], observed = st.head && z.obs[st.head].some((v) => v === 1);
          const secs = clock[z.id] ?? 300 - k * 37, done = acted[z.id];
          let right;
          if (done) right = <><Pill tier={st.tier}>{done === "veto" ? "Vetoed" : done === "dismiss" ? "Dismissed" : "Sent"}</Pill><span className="cd done">by operator</span></>;
          else if (st.tier === "critical") right = <><Pill tier="critical">Tier 1 · Auto</Pill><span className="cd sent">Broadcast at issue</span></>;
          else if (st.tier === "warning") right = (
            <><Pill tier="warning">Tier 2 · Veto</Pill>
              <span className="cd num">{secs > 0 ? fmt(secs) : "Sent"}</span>
              <span className="btns">
                <button type="button" className="btn" onClick={(e) => { e.stopPropagation(); setActed({ ...acted, [z.id]: "veto" }); }}>Veto</button>
                <button type="button" className="btn pri" onClick={(e) => { e.stopPropagation(); setActed({ ...acted, [z.id]: "send" }); }}>Send now</button>
              </span></>
          );
          else right = (
            <><Pill tier="watch">Tier 3 · Review</Pill>
              <span className="btns">
                <button type="button" className="btn" onClick={(e) => { e.stopPropagation(); setActed({ ...acted, [z.id]: "dismiss" }); }}>Dismiss</button>
                <button type="button" className="btn pri" onClick={(e) => { e.stopPropagation(); setActed({ ...acted, [z.id]: "send" }); }}>Approve</button>
              </span></>
          );
          return (
            <div key={z.id} className={`al${z.id === zoneId ? " on" : ""}`} onClick={() => onZone(z.id)}>
              <span className="stripe" style={{ background: t.dot }} />
              <div>
                <div className="h">Zone {z.id} · {HEAD_NAME[st.head]}</div>
                <div className="m">p {st.p.toFixed(2)} at {leadLabel(st.lead)}{st.onset ? ` · onset ${leadLabel(st.onset)}` : ""}</div>
                <div className="cap">CAP {CAP[st.tier]} · SACHET cell broadcast</div>
                {showObs && <div style={{ marginTop: 6 }}><span className={`verify ${observed ? "hit" : "fa"}`}>{observed ? "Verified: event occurred" : "False alarm: no event"}</span></div>}
              </div>
              <div className="right">{right}</div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

/* ---------------- zone table ---------------- */
export function ZoneTable({ scene, zoneId, onZone, showObs }) {
  const [all, setAll] = useState(false);
  const ranked = scene.zones.map((z) => ({ z, st: zoneStatus(z, scene.leads_h) }))
    .sort((a, b) => b.st.p - a.st.p);
  const rows = all ? ranked : ranked.slice(0, 8);
  const cell = (z, h) => {
    const p = Math.max(...z.p[h]), j = z.p[h].indexOf(p), obs = z.obs[h].some((v) => v);
    return (
      <span className="cellp num">
        <i style={{ background: TIER[tierOf(p)].dot }} />{p.toFixed(2)} <span className="dim">{leadLabel(scene.leads_h[j])}</span>
        {showObs && obs && <span title="Observed" style={{ color: "#EDF2F9" }}>●</span>}
      </span>
    );
  };
  return (
    <section className="panel" aria-labelledby="zt-h">
      <div className="ph"><h2 id="zt-h">Zone outlook · peak probability over the next 6 h</h2><span className="meta">Sorted by risk · select a zone to inspect it</span></div>
      <div className="tbl-wrap">
        <table className="zt">
          <thead><tr><th>Zone</th><th>Thunderstorm</th><th>Cloudburst</th><th>Flash flood</th><th>Onset</th><th>Status</th></tr></thead>
          <tbody>
            {rows.map(({ z, st }) => (
              <tr key={z.id} className={z.id === zoneId ? "on" : ""} tabIndex={0} onClick={() => onZone(z.id)}
                onKeyDown={(e) => { if (e.key === "Enter") onZone(z.id); }}>
                <td><b className="mono">{z.id}</b></td>
                <td>{cell(z, "thunderstorm")}</td><td>{cell(z, "cloudburst")}</td><td>{cell(z, "flash_flood")}</td>
                <td className="mono">{st.onset ? leadLabel(st.onset) : "–"}</td>
                <td><Pill tier={st.tier} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ marginTop: 10 }}>
        <button type="button" className="btn" onClick={() => setAll(!all)}>{all ? "Show top 8 zones" : `Show all ${ranked.length} zones`}</button>
      </div>
    </section>
  );
}
