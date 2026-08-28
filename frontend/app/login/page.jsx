"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-store";
import { useWorkflow } from "@/lib/workflow-store";
import { useT } from "@/lib/i18n";

export default function LoginPage() {
  const router = useRouter();
  const lang = useWorkflow((s) => s.language) || "id";
  const tr = useT(lang);

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const login = useAuth((s) => s.login);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!username.trim() || !password.trim()) {
      setErrorMessage("Silakan masukkan username dan password.");
      return;
    }

    setErrorMessage("");
    setIsSubmitting(true);

    try {
      const res = await login(username.trim(), password);
      if (res.success) {
        router.push("/data-upload");
      } else {
        setErrorMessage(res.error || "Username atau password salah.");
      }
    } catch (err) {
      setErrorMessage(err.message || "Terjadi kesalahan pada sistem.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleUseDefaultAdmin = () => {
    setUsername("admin");
    setPassword("Admin@Asmeranda2026!");
    setErrorMessage("");
  };

  return (
    <div style={{
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      minHeight: "85vh",
      padding: "24px",
    }}>
      <div style={{
        width: "100%",
        maxWidth: "440px",
        background: "#ffffff",
        border: "1px solid var(--color-slate-200, #e2e8f0)",
        borderRadius: "var(--radius-xl, 1rem)",
        boxShadow: "var(--shadow-xl, 0 20px 25px -5px rgb(0 0 0 / 0.1))",
        padding: "36px 32px",
      }}>
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: "28px" }}>
          <div style={{
            fontSize: "40px",
            marginBottom: "12px",
            display: "inline-block",
            filter: "drop-shadow(0 4px 6px rgba(37,99,235,0.2))",
          }}>
            🤖
          </div>
          <h1 style={{
            fontSize: "22px",
            fontWeight: "700",
            color: "var(--color-slate-900, #0f172a)",
            marginBottom: "6px",
            letterSpacing: "-0.02em",
          }}>
            {tr("app.title") || "Asmeranda AI"}
          </h1>
          <p style={{
            fontSize: "13px",
            color: "var(--color-slate-500, #64748b)",
            lineHeight: "1.4",
          }}>
            Masuk ke Platform Machine Learning & Analytics Enterprise
          </p>
        </div>

        {/* Error Alert */}
        {errorMessage && (
          <div style={{
            padding: "12px 16px",
            marginBottom: "20px",
            borderRadius: "var(--radius-md, 0.5rem)",
            background: "var(--color-error-50, #fff1f2)",
            border: "1px solid var(--color-error-100, #fee2e2)",
            color: "var(--color-error-700, #b91c1c)",
            fontSize: "13px",
            display: "flex",
            alignItems: "center",
            gap: "8px",
          }}>
            <span>⚠️</span>
            <span>{errorMessage}</span>
          </div>
        )}

        {/* Login Form */}
        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
          <div>
            <label style={{
              display: "block",
              fontSize: "13px",
              fontWeight: "600",
              color: "var(--color-slate-700, #334155)",
              marginBottom: "6px",
            }}>
              Username
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Masukkan username"
              autoComplete="username"
              required
              style={{
                width: "100%",
                padding: "10px 14px",
                fontSize: "14px",
                border: "1px solid var(--color-slate-300, #cbd5e1)",
                borderRadius: "var(--radius-md, 0.5rem)",
                outline: "none",
                transition: "border-color 0.15s, box-shadow 0.15s",
                background: "#f8fafc",
              }}
              onFocus={(e) => {
                e.target.style.borderColor = "var(--color-primary-500, #3b82f6)";
                e.target.style.boxShadow = "var(--shadow-blue, 0 0 0 3px rgb(59 130 246 / 0.15))";
                e.target.style.background = "#ffffff";
              }}
              onBlur={(e) => {
                e.target.style.borderColor = "var(--color-slate-300, #cbd5e1)";
                e.target.style.boxShadow = "none";
                e.target.style.background = "#f8fafc";
              }}
            />
          </div>

          <div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
              <label style={{
                fontSize: "13px",
                fontWeight: "600",
                color: "var(--color-slate-700, #334155)",
              }}>
                Password
              </label>
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                style={{
                  background: "none",
                  border: "none",
                  fontSize: "12px",
                  color: "var(--color-primary-600, #2563eb)",
                  cursor: "pointer",
                  padding: 0,
                }}
              >
                {showPassword ? "Sembunyikan" : "Tampilkan"}
              </button>
            </div>
            <input
              type={showPassword ? "text" : "password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Masukkan password"
              autoComplete="current-password"
              required
              style={{
                width: "100%",
                padding: "10px 14px",
                fontSize: "14px",
                border: "1px solid var(--color-slate-300, #cbd5e1)",
                borderRadius: "var(--radius-md, 0.5rem)",
                outline: "none",
                transition: "border-color 0.15s, box-shadow 0.15s",
                background: "#f8fafc",
              }}
              onFocus={(e) => {
                e.target.style.borderColor = "var(--color-primary-500, #3b82f6)";
                e.target.style.boxShadow = "var(--shadow-blue, 0 0 0 3px rgb(59 130 246 / 0.15))";
                e.target.style.background = "#ffffff";
              }}
              onBlur={(e) => {
                e.target.style.borderColor = "var(--color-slate-300, #cbd5e1)";
                e.target.style.boxShadow = "none";
                e.target.style.background = "#f8fafc";
              }}
            />
          </div>

          <button
            type="submit"
            disabled={isSubmitting}
            style={{
              marginTop: "8px",
              padding: "12px",
              fontSize: "14px",
              fontWeight: "600",
              color: "#ffffff",
              background: isSubmitting ? "var(--color-slate-400, #94a3b8)" : "var(--color-primary-600, #2563eb)",
              border: "none",
              borderRadius: "var(--radius-md, 0.5rem)",
              cursor: isSubmitting ? "not-allowed" : "pointer",
              transition: "background 0.2s, transform 0.1s",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "8px",
            }}
          >
            {isSubmitting ? (
              <>
                <span className="spinner" style={{ width: "16px", height: "16px", borderWidth: "2px" }}></span>
                <span>Memproses...</span>
              </>
            ) : (
              <span>Masuk ke Sistem</span>
            )}
          </button>
        </form>

        {/* Demo Quick Fill Helper */}
        <div style={{
          marginTop: "24px",
          padding: "14px",
          borderRadius: "var(--radius-md, 0.5rem)",
          background: "var(--color-slate-50, #f8fafc)",
          border: "1px dashed var(--color-slate-300, #cbd5e1)",
          textAlign: "center",
        }}>
          <p style={{ fontSize: "12px", color: "var(--color-slate-500, #64748b)", marginBottom: "8px" }}>
            🔑 Akun Default Administrator
          </p>
          <button
            type="button"
            onClick={handleUseDefaultAdmin}
            style={{
              fontSize: "12px",
              fontWeight: "600",
              color: "var(--color-primary-600, #2563eb)",
              background: "var(--color-primary-50, #eff6ff)",
              border: "1px solid var(--color-primary-200, #bfdbfe)",
              padding: "6px 12px",
              borderRadius: "var(--radius-sm, 0.25rem)",
              cursor: "pointer",
            }}
          >
            Gunakan Akun Default (admin)
          </button>
        </div>
      </div>
    </div>
  );
}
