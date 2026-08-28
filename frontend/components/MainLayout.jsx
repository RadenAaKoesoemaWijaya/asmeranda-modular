"use client";

/**
 * Layout utama — sidebar di kiri (desktop) atau bawah (mobile).
 * Dilengkapi dengan AuthGuard untuk proteksi rute.
 */
import { useEffect } from "react";
import { usePathname } from "next/navigation";
import Sidebar from "@/components/Sidebar";
import AuthGuard from "@/components/AuthGuard";
import { rehydrateWorkflow } from "@/lib/workflow-store";

export default function MainLayout({ children }) {
  const pathname = usePathname();
  const isLoginPage = pathname === "/login";

  useEffect(() => {
    rehydrateWorkflow();
  }, []);

  if (isLoginPage) {
    return (
      <AuthGuard>
        <main style={{
          width: "100%",
          minHeight: "100vh",
          background: "linear-gradient(135deg, #f8fbff 0%, #eef5ff 100%)",
        }}>
          {children}
        </main>
      </AuthGuard>
    );
  }

  return (
    <div className="app-shell">
      <Sidebar />
      <main className="app-main">
        <AuthGuard>{children}</AuthGuard>
      </main>
    </div>
  );
}
