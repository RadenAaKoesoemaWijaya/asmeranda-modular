"use client";

/**
 * Sidebar navigasi — tetap di setiap halaman.
 * Item disable jika workflow step belum siap.
 */
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useWorkflow } from "@/lib/workflow-store";
import { useAuth } from "@/lib/auth-store";
import { useT } from "@/lib/i18n";

const STEPS = [
  {
    href: "/data-upload",
    key: "nav.upload",
    can: "eda",
    icon: "📂",
    alwaysEnabled: true,
  },
  { href: "/eda", key: "nav.eda", can: "eda", icon: "🔍" },
  {
    href: "/preprocessing",
    key: "nav.preprocessing",
    can: "preprocessing",
    icon: "⚙️",
  },
  {
    href: "/optimization",
    key: "nav.optimization",
    can: "optimization",
    icon: "🔧",
    paradigm: "supervised",
  },
  {
    href: "/training",
    key: "nav.training",
    can: "training",
    icon: "🧠",
    paradigm: "supervised",
  },
  {
    href: "/shap",
    key: "nav.shap",
    can: "shap",
    icon: "📊",
    paradigm: "supervised",
  },
  {
    href: "/lime",
    key: "nav.lime",
    can: "lime",
    icon: "🔬",
    paradigm: "supervised",
  },
  {
    href: "/inference",
    key: "nav.inference",
    can: "inference",
    icon: "🔮",
    alwaysEnabled: true,
  },
  {
    href: "/clustering",
    key: "nav.clustering",
    can: "clustering",
    icon: "🎯",
    paradigm: "unsupervised",
  },
  {
    href: "/timeseries",
    key: "nav.timeseries",
    can: "timeseries",
    icon: "📈",
  },
  {
    href: "/advanced-ml",
    key: "nav.advanced_ml",
    can: "advanced_ml",
    icon: "🚀",
  },
];

export default function Sidebar() {
  const pathname = usePathname();
  const lang = useWorkflow((s) => s.language) || "id";
  const canProceedTo = useWorkflow((s) => s.canProceedTo);
  const problemType = useWorkflow((s) => s.problemType);
  const stateId = useWorkflow((s) => s.stateId);
  const modelId = useWorkflow((s) => s.modelId);
  const setLang = useWorkflow((s) => s.set);
  const datasetId = useWorkflow((s) => s.datasetId);
  const tr = useT(lang);

  const user = useAuth((s) => s.user);
  const logout = useAuth((s) => s.logout);

  const getStepTooltip = (step, enabled) => {
    if (enabled) return "";
    if (!datasetId) return "Unggah dataset terlebih dahulu";
    if (
      !stateId &&
      step.href !== "/eda" &&
      step.href !== "/preprocessing" &&
      step.href !== "/timeseries"
    ) {
      return "Selesaikan tahap Preprocessing terlebih dahulu";
    }
    if (
      step.paradigm === "supervised" &&
      (problemType === "Clustering" || problemType === "Unsupervised")
    ) {
      return "Modul ini hanya untuk Supervised Learning (Klasifikasi / Regresi)";
    }
    if (
      step.paradigm === "unsupervised" &&
      (problemType === "Classification" || problemType === "Regression")
    ) {
      return "Modul Clustering hanya untuk Unsupervised Learning";
    }
    if ((step.href === "/shap" || step.href === "/lime") && !modelId) {
      return "Latih model terlebih dahulu di Pelatihan Model";
    }
    return "Selesaikan langkah sebelumnya terlebih dahulu";
  };

  // Hitung progress berapa langkah sudah selesai
  const completedSteps = STEPS.filter((s) => !s.alwaysEnabled && canProceedTo(s.can)).length;
  const progressPct = Math.round((completedSteps / (STEPS.length - 1)) * 100);

  return (
    <aside className="app-sidebar">
      {/* ── Header / Brand ── */}
      <div className="sidebar-header">
        <div className="sidebar-brand">
          <div className="sidebar-brand-icon">🤖</div>
          <div>
            <div className="sidebar-title">{tr("app.title")}</div>
          </div>
        </div>
        <div className="sidebar-subtitle">{tr("app.subtitle")}</div>

        {/* Progress bar workflow */}
        {datasetId && (
          <div style={{ marginTop: "12px" }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "10px",
                color: "var(--sidebar-muted)",
                marginBottom: "4px",
              }}
            >
              <span>Progres Workflow</span>
              <span>{progressPct}%</span>
            </div>
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* ── Navigation ── */}
      <nav className="sidebar-nav">
        <div className="sidebar-section-label">Workflow</div>
        {STEPS.map((step, idx) => {
          const enabled = step.alwaysEnabled || canProceedTo(step.can);
          const active = pathname === step.href;
          const done = !step.alwaysEnabled && canProceedTo(step.can);
          const tooltip = getStepTooltip(step, enabled);

          return (
            <Link
              key={step.href}
              href={enabled ? step.href : "#"}
              className={[
                "sidebar-link",
                active ? "sidebar-link--active" : "",
                !enabled ? "sidebar-link--disabled" : "",
              ]
                .filter(Boolean)
                .join(" ")}
              onClick={(e) => !enabled && e.preventDefault()}
              title={tooltip}
            >
              <span className="sidebar-link-icon">{step.icon}</span>
              <span>
                {idx + 1}. {tr(step.key)}
              </span>
              {done && !active && (
                <span className="sidebar-link-badge">✓</span>
              )}
              {!enabled && (
                <span
                  className="sidebar-link-badge"
                  style={{ background: "var(--sidebar-muted)" }}
                >
                  🔒
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* ── Footer — User Profile & Language ── */}
      <div className="sidebar-footer">
        {user && (
          <div style={{
            marginBottom: "12px",
            padding: "8px 10px",
            borderRadius: "var(--radius-md, 0.5rem)",
            background: "var(--sidebar-surface, #edf4ff)",
            border: "1px solid var(--sidebar-border, #dfeafc)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", overflow: "hidden" }}>
              <span style={{ fontSize: "14px" }}>👤</span>
              <div style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
                <span style={{
                  fontSize: "12px",
                  fontWeight: "600",
                  color: "var(--sidebar-text, #12314d)",
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}>
                  {user.username || "User"}
                </span>
                <span style={{
                  fontSize: "10px",
                  color: "var(--color-primary-600, #2563eb)",
                  fontWeight: "500",
                  textTransform: "uppercase",
                }}>
                  {user.role || "analyst"}
                </span>
              </div>
            </div>
            <button
              type="button"
              onClick={logout}
              title="Keluar / Logout"
              style={{
                background: "none",
                border: "none",
                cursor: "pointer",
                fontSize: "14px",
                padding: "4px",
                borderRadius: "var(--radius-sm, 0.25rem)",
                color: "var(--color-error-600, #dc2626)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              🚪
            </button>
          </div>
        )}

        <label className="sidebar-lang-label">🌐 Bahasa / Language</label>
        <select
          className="sidebar-lang-select"
          value={lang}
          onChange={(e) => setLang({ language: e.target.value })}
        >
          <option value="id">🇮🇩 Bahasa Indonesia</option>
          <option value="en">🇺🇸 English</option>
        </select>

        <div
          style={{
            marginTop: "12px",
            fontSize: "10px",
            color: "var(--sidebar-muted)",
            textAlign: "center",
            lineHeight: 1.5,
          }}
        >
          © PT. Asmer Sahabat Sukses
        </div>
      </div>
    </aside>
  );
}
