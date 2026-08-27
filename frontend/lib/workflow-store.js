"use client";

/**
 * Workflow store - menggantikan st.session_state.
 * Disimpan ke localStorage agar tidak hilang saat refresh.
 */
import { create } from "zustand";
import { persist } from "zustand/middleware";

const DEFAULTS = {
  datasetId: null,
  datasetName: null,
  targetColumn: null,
  problemType: null, // "Classification" | "Regression" | "Clustering" | "Forecasting"
  numericalColumns: [],
  categoricalColumns: [],
  featureNames: [],
  stateId: null, // hasil preprocessing
  nSamplesTrain: 0,
  nSamplesTest: 0,
  nFeatures: 0,
  modelId: null,
  modelType: null,
  metrics: null,
  cvScores: null,
  language: "id",
  clusteringResults: null,
  optimizationResults: null,
  optimizedHyperparams: {}, // { [modelType]: best_params }
  advancedMLResults: null,
};

export const useWorkflow = create(
  persist(
    (set, get) => ({
      ...DEFAULTS,

      set: (patch) => set(patch),
      reset: () => set({ ...DEFAULTS }),
      setOptimizedHyperparams: (modelType, params) =>
        set((state) => ({
          optimizedHyperparams: {
            ...(state.optimizedHyperparams || {}),
            [modelType]: params,
          },
        })),
      resetPreprocessing: () =>
        set({
          stateId: null,
          targetColumn: null,
          problemType: null,
          numericalColumns: [],
          categoricalColumns: [],
          featureNames: [],
          nSamplesTrain: 0,
          nSamplesTest: 0,
          nFeatures: 0,
          modelId: null,
          modelType: null,
          metrics: null,
          cvScores: null,
          clusteringResults: null,
          optimizationResults: null,
          optimizedHyperparams: {},
          advancedMLResults: null,
        }),
      resetTraining: () =>
        set({
          modelId: null,
          modelType: null,
          metrics: null,
          cvScores: null,
          clusteringResults: null,
          optimizationResults: null,
          advancedMLResults: null,
        }),

      // Cek apakah step tertentu siap diakses
      canProceedTo: (step) => {
        const s = get();
        const isSupervised =
          s.problemType === "Classification" || s.problemType === "Regression";
        const isUnsupervised =
          s.problemType === "Clustering" || s.problemType === "Unsupervised";

        switch (step) {
          case "eda":
            return !!s.datasetId;
          case "preprocessing":
            return !!s.datasetId;
          case "optimization":
            return !!s.stateId && isSupervised;
          case "training":
            return !!s.stateId && isSupervised;
          case "shap":
          case "lime":
            return !!s.modelId && isSupervised;
          case "clustering":
            // Clustering aktif untuk Unsupervised, atau jika belum eksplisit tapi preprocessing selesai
            return !!s.stateId && isUnsupervised;
          case "timeseries":
            return !!s.datasetId;
          case "advanced_ml":
            return !!s.stateId;
          default:
            return false;
        }
      },
    }),
    {
      name: "asmeranda-workflow",
      skipHydration: true,
    }
  )
);

/** Panggil sekali di client agar state localStorage termuat tanpa hydration mismatch. */
export function rehydrateWorkflow() {
  if (typeof window !== "undefined") {
    useWorkflow.persist.rehydrate();
  }
}
