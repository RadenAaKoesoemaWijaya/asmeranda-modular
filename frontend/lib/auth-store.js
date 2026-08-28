"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";
import { api, getAuthToken, setAuthToken } from "./api";

export const useAuth = create(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      login: async (username, password) => {
        set({ isLoading: true, error: null });
        try {
          const res = await api.auth.login(username, password);
          if (res && res.access_token) {
            set({
              token: res.access_token,
              user: res.user || { username, role: "admin" },
              isAuthenticated: true,
              isLoading: false,
              error: null,
            });
            return { success: true, user: res.user };
          } else {
            throw new Error("Respons login tidak valid");
          }
        } catch (err) {
          const msg = err?.payload?.detail || err?.message || "Login gagal";
          set({ isLoading: false, error: msg, isAuthenticated: false });
          return { success: false, error: msg };
        }
      },

      logout: () => {
        api.auth.logout();
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          error: null,
        });
        if (typeof window !== "undefined") {
          window.location.href = "/login";
        }
      },

      checkAuth: async () => {
        const token = getAuthToken();
        if (!token) {
          set({ user: null, token: null, isAuthenticated: false });
          return false;
        }

        try {
          const profile = await api.auth.me();
          set({
            user: profile,
            token,
            isAuthenticated: true,
          });
          return true;
        } catch (err) {
          // Token expired or invalid
          api.auth.logout();
          set({
            user: null,
            token: null,
            isAuthenticated: false,
          });
          return false;
        }
      },

      setUser: (user) => set({ user, isAuthenticated: !!user }),
    }),
    {
      name: "asmeranda-auth",
      skipHydration: true,
    }
  )
);

export function rehydrateAuth() {
  if (typeof window !== "undefined") {
    useAuth.persist.rehydrate();
  }
}
