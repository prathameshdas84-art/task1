// Signals tab: the flat list of every forensic signal, colored by layer.

import { SIGNAL_COLORS } from "../lib/constants";

export default function SignalsTab({ result }) {
  return (
            <div className="tab-content">
              <div className="section-title">
                ⚡ All Forensic Signals
                <span className="badge">{result.signals.length}</span>
              </div>
              {result.signals.map((sig, i) => {
                const prefix = Object.keys(SIGNAL_COLORS).find(p => sig.includes(p));
                const color = SIGNAL_COLORS[prefix] || "#888";
                const isAnomaly = sig.includes("⚡") ||
                  (!sig.includes("✓") && !sig.includes("passed") &&
                   !sig.includes("skipped") && !sig.includes("consistent"));
                return (
                  <div key={i} className="signal-row"
                    style={{ borderLeftColor: color }}>
                    <span className="signal-prefix" style={{ color }}>
                      {prefix || ""}
                    </span>
                    <span className="signal-text"
                      style={{ color: isAnomaly ? "#fff" : "#888" }}>
                      {sig.replace(prefix || "", "").trim()}
                    </span>
                  </div>
                );
              })}
            </div>
  );
}
