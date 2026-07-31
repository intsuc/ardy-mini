// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { useSyncExternalStore } from "react"

type StateListener = () => void

export interface UiAction {
  readonly trigger: () => void
  readonly onTrigger: (
    listener: () => void,
    signal?: AbortSignal
  ) => () => void
}

export type ModelCacheState =
  | "checking"
  | "missing"
  | "downloading"
  | "cancelling"
  | "verifying"
  | "initializing"
  | "ready"
  | "clearing"
  | "error"

export type ModelRuntimeState =
  | "idle"
  | "loading"
  | "ready"
  | "error"

export type ModelCacheErrorOperation =
  | "download"
  | "initialization"
  | "clear"

export interface ModelUiState {
  cache: ModelCacheState
  errorOperation: ModelCacheErrorOperation | null
  runtime: ModelRuntimeState
  cachedFiles: number
  totalFiles: number
  cachedBytes: number
  totalBytes: number
  verifiedFiles: number
  initializationSteps: number
  totalInitializationSteps: number
}

interface ModelAssetProgress {
  cachedFiles: number
  totalFiles: number
  cachedBytes: number
  totalBytes: number
}

export type ModelUiEvent =
  | { type: "reset" }
  | { type: "cache-check-started" }
  | ({ type: "cache-missing" } & ModelAssetProgress)
  | { type: "download-started" }
  | { type: "download-cancel-requested" }
  | ({ type: "download-progress" } & ModelAssetProgress)
  | { type: "download-completed" }
  | {
      type: "verification-progress"
      completedFiles: number
      totalFiles: number
    }
  | { type: "initialization-started" }
  | {
      type: "initialization-progress"
      completedSteps: number
      totalSteps: number
    }
  | ({
      type: "cache-ready"
    } & Pick<ModelAssetProgress, "totalFiles" | "totalBytes">)
  | { type: "clear-started" }
  | { type: "cache-cleared" }
  | {
      type: "cache-error"
      operation: ModelCacheErrorOperation
    }
  | { type: "runtime-loading" }
  | { type: "runtime-ready" }
  | { type: "runtime-idle" }
  | { type: "runtime-error" }

export interface ModelUiControl {
  readonly id: string
  readonly getSnapshot: () => ModelUiState
  readonly subscribe: (listener: StateListener) => () => void
  readonly dispatch: (event: ModelUiEvent) => void
}

export interface ExternalControl<State, Value> {
  readonly id: string
  readonly getSnapshot: () => State
  readonly subscribe: (listener: StateListener) => () => void
  readonly setState: (patch: Partial<State>) => void
  readonly commit: (value: Value) => void
  readonly onCommit: (
    listener: (value: Value) => void,
    signal?: AbortSignal
  ) => () => void
}

function createExternalControl<State extends object, Value>(
  id: string,
  initialState: State,
  stateFromValue: (value: Value, current: State) => Partial<State>
): ExternalControl<State, Value> {
  let state = Object.freeze({ ...initialState }) as State
  const stateListeners = new Set<StateListener>()
  const commitListeners = new Set<(value: Value) => void>()

  const setState = (patch: Partial<State>) => {
    const next = { ...state, ...patch }
    const changed = Object.keys(next).some(
      (key) =>
        next[key as keyof State] !== state[key as keyof State]
    )
    if (!changed) return
    state = Object.freeze(next) as State
    stateListeners.forEach((listener) => listener())
  }

  return {
    id,
    getSnapshot: () => state,
    subscribe: (listener) => {
      stateListeners.add(listener)
      return () => stateListeners.delete(listener)
    },
    setState,
    commit: (value) => {
      setState(stateFromValue(value, state))
      commitListeners.forEach((listener) => listener(value))
    },
    onCommit: (listener, signal) => {
      if (signal?.aborted) return () => {}
      commitListeners.add(listener)
      const unsubscribe = () => {
        commitListeners.delete(listener)
        signal?.removeEventListener("abort", unsubscribe)
      }
      signal?.addEventListener("abort", unsubscribe, { once: true })
      return unsubscribe
    },
  }
}

function createUiAction(): UiAction {
  const listeners = new Set<() => void>()

  return {
    trigger: () => listeners.forEach((listener) => listener()),
    onTrigger: (listener, signal) => {
      if (signal?.aborted) return () => {}
      listeners.add(listener)
      const unsubscribe = () => {
        listeners.delete(listener)
        signal?.removeEventListener("abort", unsubscribe)
      }
      signal?.addEventListener("abort", unsubscribe, { once: true })
      return unsubscribe
    },
  }
}

const INITIAL_MODEL_UI_STATE: ModelUiState = Object.freeze({
  cache: "checking",
  errorOperation: null,
  runtime: "idle",
  cachedFiles: 0,
  totalFiles: 0,
  cachedBytes: 0,
  totalBytes: 0,
  verifiedFiles: 0,
  initializationSteps: 0,
  totalInitializationSteps: 0,
})

function finiteWholeNumber(value: number): number {
  return Number.isFinite(value) ? Math.max(0, Math.floor(value)) : 0
}

function modelAssetProgress(
  progress: ModelAssetProgress
): Pick<
  ModelUiState,
  "cachedBytes" | "cachedFiles" | "totalBytes" | "totalFiles"
> {
  const totalFiles = finiteWholeNumber(progress.totalFiles)
  const totalBytes = finiteWholeNumber(progress.totalBytes)
  return {
    totalFiles,
    totalBytes,
    cachedFiles: Math.min(
      finiteWholeNumber(progress.cachedFiles),
      totalFiles
    ),
    cachedBytes: Math.min(
      finiteWholeNumber(progress.cachedBytes),
      totalBytes
    ),
  }
}

function reduceModelUiState(
  state: ModelUiState,
  event: ModelUiEvent
): ModelUiState {
  switch (event.type) {
    case "reset":
      return INITIAL_MODEL_UI_STATE
    case "cache-check-started":
      return {
        ...state,
        cache: "checking",
        errorOperation: null,
        verifiedFiles: 0,
        initializationSteps: 0,
        totalInitializationSteps: 0,
      }
    case "cache-missing": {
      const progress = modelAssetProgress(event)
      return {
        ...state,
        ...progress,
        cache: "missing",
        errorOperation: null,
        verifiedFiles: 0,
        initializationSteps: 0,
        totalInitializationSteps: 0,
      }
    }
    case "download-started":
      if (
        state.cache !== "missing" &&
        !(
          state.cache === "error" &&
          (state.errorOperation === "download" ||
            state.errorOperation === "initialization")
        )
      ) {
        return state
      }
      return {
        ...state,
        cache: "downloading",
        errorOperation: null,
        verifiedFiles: 0,
        initializationSteps: 0,
        totalInitializationSteps: 0,
      }
    case "download-cancel-requested":
      if (state.cache !== "downloading") return state
      return { ...state, cache: "cancelling" }
    case "download-progress": {
      if (state.cache !== "downloading") return state
      const progress = modelAssetProgress(event)
      return {
        ...state,
        ...progress,
        cachedFiles: Math.min(
          progress.totalFiles,
          Math.max(state.cachedFiles, progress.cachedFiles)
        ),
        cachedBytes: Math.min(
          progress.totalBytes,
          Math.max(state.cachedBytes, progress.cachedBytes)
        ),
      }
    }
    case "download-completed":
      if (state.cache !== "downloading") return state
      return {
        ...state,
        cache: "verifying",
        cachedFiles: state.totalFiles,
        cachedBytes: state.totalBytes,
        verifiedFiles: 0,
        initializationSteps: 0,
        totalInitializationSteps: 0,
      }
    case "verification-progress":
      if (
        state.cache !== "downloading" &&
        state.cache !== "verifying" &&
        state.cache !== "initializing"
      ) {
        return state
      }
      return {
        ...state,
        cache:
          state.cache === "initializing"
            ? "initializing"
            : "verifying",
        totalFiles: finiteWholeNumber(event.totalFiles),
        cachedBytes: state.totalBytes,
        verifiedFiles: Math.min(
          finiteWholeNumber(event.completedFiles),
          finiteWholeNumber(event.totalFiles)
        ),
      }
    case "initialization-started":
      if (
        state.cache !== "downloading" &&
        state.cache !== "verifying" &&
        state.cache !== "ready"
      ) {
        return state
      }
      return {
        ...state,
        cache: "initializing",
        verifiedFiles: 0,
        initializationSteps: 0,
        totalInitializationSteps: 0,
      }
    case "initialization-progress": {
      if (
        state.cache !== "downloading" &&
        state.cache !== "verifying" &&
        state.cache !== "initializing" &&
        state.cache !== "ready"
      ) {
        return state
      }
      const totalInitializationSteps = finiteWholeNumber(
        event.totalSteps
      )
      return {
        ...state,
        cache: "initializing",
        initializationSteps: Math.min(
          finiteWholeNumber(event.completedSteps),
          totalInitializationSteps
        ),
        totalInitializationSteps,
      }
    }
    case "cache-ready": {
      const totalFiles = finiteWholeNumber(event.totalFiles)
      const totalBytes = finiteWholeNumber(event.totalBytes)
      return {
        ...state,
        cache: "ready",
        errorOperation: null,
        cachedFiles: totalFiles,
        totalFiles,
        cachedBytes: totalBytes,
        totalBytes,
      }
    }
    case "clear-started":
      if (
        state.cachedBytes === 0 ||
        state.cache === "checking" ||
        state.cache === "downloading" ||
        state.cache === "cancelling" ||
        state.cache === "verifying" ||
        state.cache === "initializing" ||
        state.cache === "clearing"
      ) {
        return state
      }
      return {
        ...state,
        cache: "clearing",
        errorOperation: null,
      }
    case "cache-cleared":
      if (state.cache !== "clearing") return state
      return {
        ...state,
        cache: "missing",
        errorOperation: null,
        cachedFiles: 0,
        cachedBytes: 0,
        verifiedFiles: 0,
        initializationSteps: 0,
        totalInitializationSteps: 0,
      }
    case "cache-error":
      if (
        (event.operation !== "clear" &&
          state.cache === "clearing") ||
        (event.operation === "clear" && state.cache !== "clearing")
      ) {
        return state
      }
      return {
        ...state,
        cache: "error",
        errorOperation: event.operation,
      }
    case "runtime-loading":
      if (state.runtime === "loading") return state
      return { ...state, runtime: "loading" }
    case "runtime-ready":
      if (state.runtime === "ready") return state
      return { ...state, runtime: "ready" }
    case "runtime-idle":
      if (state.runtime === "idle") return state
      return { ...state, runtime: "idle" }
    case "runtime-error":
      if (state.runtime === "error") return state
      return { ...state, runtime: "error" }
  }
}

function createModelUiControl(): ModelUiControl {
  let state = INITIAL_MODEL_UI_STATE
  const listeners = new Set<StateListener>()

  return {
    id: "model-cache",
    getSnapshot: () => state,
    subscribe: (listener) => {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    dispatch: (event) => {
      const next = Object.freeze(reduceModelUiState(state, event))
      if (next === state) return
      state = next
      listeners.forEach((listener) => listener())
    },
  }
}

export interface CheckedControlState {
  checked: boolean
  disabled: boolean
}

export interface SliderControlState {
  value: number
  min: number
  max: number
  step: number
  disabled: boolean
  ariaValueText?: string
}

export interface PressedControlState {
  pressed: boolean
  disabled: boolean
}

export interface GenerationActionsState {
  primaryLabel: "Start motion" | "Update motion"
  activeLabel:
    | "Starting…"
    | "Updating…"
    | "Regenerating…"
    | null
  primaryDisabled: boolean
  menuDisabled: boolean
  regenerateDisabled: boolean
  newMotionDisabled: boolean
}

export interface DisclosureControlState {
  open: boolean
}

export type PreviewSettingsTab = "motion" | "view"

export interface PreviewSettingsTabState {
  value: PreviewSettingsTab
}

export interface SelectControlState {
  value: string
  disabled: boolean
}

export interface UnsupportedDeviceState {
  open: boolean
  title: string
  description: string
}

function createCheckedControl(
  id: string,
  checked: boolean,
  disabled = false
) {
  return createExternalControl<CheckedControlState, boolean>(
    id,
    { checked, disabled },
    (nextChecked) => ({ checked: nextChecked })
  )
}

function createSliderControl(
  id: string,
  initialState: Omit<SliderControlState, "disabled"> & {
    disabled?: boolean
  }
) {
  return createExternalControl<SliderControlState, number>(
    id,
    { disabled: false, ...initialState },
    (nextValue, current) => ({
      value: Math.max(current.min, Math.min(current.max, nextValue)),
    })
  )
}

export const targetBufferControl = createSliderControl("target-buffer", {
  value: 80,
  min: 40,
  max: 200,
  step: 40,
  ariaValueText: "80 frames",
})

export const timelineControl = createSliderControl("timeline", {
  value: 0,
  min: 0,
  max: 1,
  step: 1,
  disabled: true,
  ariaValueText: "00:00.00 of 00:00.00",
})

export const showVrmControl = createCheckedControl(
  "show-vrm",
  true,
  true
)

export const showSkeletonControl = createCheckedControl(
  "show-skeleton",
  true
)

export const showContactsControl = createCheckedControl(
  "show-contacts",
  true
)

export const showOrientationsControl = createCheckedControl(
  "show-orientations",
  true
)

export const showTrajectoryControl = createCheckedControl(
  "show-trajectory",
  true
)

export const playPauseControl = createExternalControl<
  PressedControlState,
  boolean
>(
  "play-pause",
  { pressed: false, disabled: true },
  (pressed) => ({ pressed })
)

export const previewSettingsControl = createExternalControl<
  DisclosureControlState,
  boolean
>(
  "preview-settings",
  { open: false },
  (open) => ({ open })
)

export const previewSettingsTabControl = createExternalControl<
  PreviewSettingsTabState,
  PreviewSettingsTab
>(
  "preview-settings-tab",
  { value: "view" },
  (value) => ({ value })
)

export const playbackSpeedControl = createExternalControl<
  SelectControlState,
  string
>(
  "playback-speed",
  { value: "1", disabled: true },
  (value) => ({ value })
)

export const unsupportedDeviceControl = createExternalControl<
  UnsupportedDeviceState,
  UnsupportedDeviceState
>(
  "unsupported-device",
  {
    open: false,
    title: "WebGPU is required",
    description:
      "Use a browser and device that support WebGPU, then reload the page.",
  },
  (state) => state
)

export const generationActionsControl = createExternalControl<
  GenerationActionsState,
  GenerationActionsState
>(
  "generation-actions",
  {
    primaryLabel: "Start motion",
    activeLabel: null,
    primaryDisabled: true,
    menuDisabled: true,
    regenerateDisabled: true,
    newMotionDisabled: true,
  },
  (state) => state
)

export const modelUiControl = createModelUiControl()

export const modelDownloadAction = createUiAction()

export const modelDownloadCancelAction = createUiAction()

export const clearModelCacheAction = createUiAction()

export const regenerateMotionAction = createUiAction()

export const startNewMotionAction = createUiAction()

export function useModelUiState(): ModelUiState {
  return useSyncExternalStore(
    modelUiControl.subscribe,
    modelUiControl.getSnapshot,
    modelUiControl.getSnapshot
  )
}

export function useControlState<State, Value>(
  control: ExternalControl<State, Value>
): State {
  return useSyncExternalStore(
    control.subscribe,
    control.getSnapshot,
    control.getSnapshot
  )
}
