"use client";

import { HEADS, HEAD_NAME } from "@/lib/risk";

const f2 = (v) => (v == null || Number.isNaN(v) ? "–" : v.toFixed(2));

function PrAucChart({ metrics, head }) {
  const leads = [2, 3, 4, 6], W = 360, H = 190, L = 34, B = 26, T = 12;
  const series = [
    ["model", "NowcastNet v0", "#E8635A"],
    ["gradient_boosting", "Gradient boosting", "#7FA8D9"],
    ["persistence", "Persistence", "#5E7396"],
  ];
  const gw = (W - L - 8) / leads.length, bw = (gw - 16) / series.length;
  const Y = (v) => T + (1 - v) * (H - T - B);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }} role="img"
      aria-label={`PR-AUC by lead time for ${HEAD_NAME[head]}`}>
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={v}>
          <line x1={L} x2={W - 4} y1={Y(v)} y2={Y(v)} stroke="#fff" strokeOpacity={v === 0 ? 0.25 : 0.06} />
          <text x={L - 6} y={Y(v) + 3} textAnchor="end" fill="#5E7396" fontSize="9" fontFamily="var(--mono)">{v.toFixed(2)}</text>
        </g>
      ))}
      {leads.map((ld, i) => (
        <g key={ld}>
          {series.map(([k, , c], s) => {
            const v = metrics[k][`${head}@+${ld}h`]?.pr_auc ?? 0;
            const x = L + i * gw + 8 + s * bw;
            return (
              <g key={k}>
                <rect x={x} y={Y(v)} width={bw - 3} height={Math.max(0.5, Y(0) - Y(v))} fill={c} rx="2" />
                {k === "model" && <text x={x + (bw - 3) / 2} y={Y(v) - 4} textAnchor="middle" fill="#EDF2F9" fontSize="9.5" fontFamily="var(--mono)">{v.toFixed(2)}</text>}
              </g>
            );
          })}
          <text x={L + i * gw + gw / 2} y={H - 8} textAnchor="middle" fill="#8CA0C2" fontSize="10" fontFamily="var(--mono)">T+{ld} h</text>
        </g>
      ))}
    </svg>
  );
}

export default function ModelPanel({ index }) {
  const { model, metrics } = index;
  const keys = HEADS.flatMap((h) => [2, 3, 4, 6].map((l) => `${h}@+${l}h`));
  return (
    <div className="wrap">
      <div className="callout">
        <b>Synthetic data.</b> The model is trained and tested on physically structured synthetic scenes
        (<span className="mono">drishti_nowcast/synthetic.py</span>), because MOSDAC and IMDAA ingestion is still in build.
        These scores show the pipeline learns the encoded precursors (IWV pooling, cloud-top cooling, CAPE, convergence,
        terrain runoff). They are not estimates of skill on real observations.
      </div>

      <div className="perf-grid">
        {HEADS.map((h) => (
          <section className="panel" key={h}>
            <div className="ph"><h2>{HEAD_NAME[h]}</h2><span className="meta">Test PR-AUC by lead time</span></div>
            <PrAucChart metrics={metrics} head={h} />
          </section>
        ))}
      </div>
      <div className="bar" style={{ gap: 18 }}>
        {[["#E8635A", "NowcastNet v0 (this model)"], ["#7FA8D9", "Gradient boosting, per-pixel features"], ["#5E7396", "Persistence"]].map(([c, t]) => (
          <span key={t} className="chk" style={{ cursor: "default" }}><span style={{ width: 12, height: 12, borderRadius: 3, background: c, display: "inline-block" }} />{t}</span>
        ))}
      </div>

      <div className="grid">
        <section className="panel">
          <div className="ph"><h2>Test-set metrics</h2><span className="meta">400 held-out scenes · threshold chosen on validation (max CSI)</span></div>
          <div className="tbl-wrap">
            <table className="mt">
              <thead><tr><th>Hazard · lead</th><th>Base rate</th><th>PR-AUC</th><th>GBM PR-AUC</th><th>POD</th><th>FAR</th><th>CSI</th><th>GBM CSI</th><th>FSS 20 km</th></tr></thead>
              <tbody>
                {keys.map((k) => {
                  const m = metrics.model[k], g = metrics.gradient_boosting[k];
                  const [h, l] = k.split("@");
                  return (
                    <tr key={k}>
                      <td style={{ fontFamily: "var(--sans)" }}>{HEAD_NAME[h]} <span className="dim">{l.replace("+", "T+").replace("h", " h")}</span></td>
                      <td>{(m.base_rate * 100).toFixed(2)}%</td>
                      <td className={m.pr_auc > g.pr_auc ? "win" : ""}>{f2(m.pr_auc)}</td>
                      <td className={g.pr_auc > m.pr_auc ? "win" : ""}>{f2(g.pr_auc)}</td>
                      <td>{f2(m.pod)}</td><td>{f2(m.far)}</td>
                      <td className={m.csi > g.csi ? "win" : ""}>{f2(m.csi)}</td>
                      <td className={g.csi > m.csi ? "win" : ""}>{f2(g.csi)}</td>
                      <td>{f2(m.fss_20km)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="note">Green marks the better of model and gradient boosting. Persistence scores are in <span className="mono">results/metrics.json</span>.</p>
        </section>
        <div className="stack">
          <section className="panel">
            <div className="ph"><h2>Model card</h2></div>
            <dl className="kv">
              <dt>Model</dt><dd>{model.model}</dd>
              <dt>Parameters</dt><dd>{model.params.toLocaleString("en-IN")}</dd>
              <dt>Inputs</dt><dd>3 h of 30-min frames + issue-time context</dd>
              <dt>Heads</dt><dd>{model.heads.join(", ")}</dd>
              <dt>Lead times</dt><dd>{model.leads_h.map((l) => `+${l} h`).join(", ")}</dd>
              <dt>Loss</dt><dd>focal loss, severe heads weighted 1.5×</dd>
              <dt>Epochs</dt><dd>{model.epochs}</dd>
              <dt>Best val PR-AUC</dt><dd>{model.best_val_pr_auc_severe.toFixed(3)} (cloudburst + flash flood)</dd>
              <dt>Data</dt><dd>{model.data}</dd>
            </dl>
          </section>
          <section className="panel">
            <div className="ph"><h2>What the numbers say</h2></div>
            <ul style={{ margin: 0, paddingLeft: 18, display: "flex", flexDirection: "column", gap: 8, fontSize: 12.5, color: "#C7D4E5" }}>
              <li>Cloudburst and thunderstorm: 2–4× the PR-AUC of per-pixel gradient boosting at T+2 to T+4 h, so spatial and temporal context adds skill.</li>
              <li>Skill falls with lead time and is near chance at T+6 h. Most T+6 h events show no precursor at issue time.</li>
              <li>Flash flood: roughly tied with gradient boosting, which gets flow accumulation per pixel. Next step is routing the cloudburst head through the DEM Flood Router inside the network.</li>
            </ul>
          </section>
        </div>
      </div>
    </div>
  );
}
