"use client";

import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useAuth, rehydrateAuth } from "@/lib/auth-store";
import { getAuthToken } from "@/lib/api";

export default function AuthGuard({ children }) {
  const router = useRouter();
  const pathname = usePathname();
  const [mounted, setMounted] = useState(false);
  const isAuthenticated = useAuth((s) => s.isAuthenticated);
  const checkAuth = useAuth((s) => s.checkAuth);

  useEffect(() => {
    rehydrateAuth();
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;

    const token = getAuthToken();
    const isLoginPage = pathname === "/login";

    if (!token && !isLoginPage) {
      router.replace("/login");
    } else if (token && isLoginPage) {
      router.replace("/data-upload");
    } else if (token && !isLoginPage) {
      // Validate token validity in background
      checkAuth();
    }
  }, [mounted, pathname, router, checkAuth]);

  if (!mounted) {
    return null;
  }

  const token = getAuthToken();
  const isLoginPage = pathname === "/login";

  if (!token && !isLoginPage) {
    return (
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "60vh",
        flexDirection: "column",
        gap: "16px",
        color: "var(--text-muted, #64748b)"
      }}>
        <div className="spinner" style={{ width: "32px", height: "32px" }}></div>
        <p style={{ fontSize: "14px" }}>Memverifikasi sesi autentikasi...</p>
      </div>
    );
  }

  return children;
}
