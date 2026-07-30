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

export interface ProgressControlState {
  value: number
}

export interface DisclosureControlState {
  open: boolean
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

export const durationControl = createSliderControl("duration", {
  value: 4,
  min: 2,
  max: 10,
  step: 2,
  ariaValueText: "4 seconds",
})

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

export const continuousGenerationControl = createCheckedControl(
  "stream-generation",
  true
)

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

export const loopControl = createExternalControl<
  PressedControlState,
  boolean
>(
  "loop-toggle",
  { pressed: true, disabled: true },
  (pressed) => ({ pressed })
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

function createProgressControl(id: string) {
  return createExternalControl<ProgressControlState, number>(
    id,
    { value: 0 },
    (value) => ({
      value: Math.max(0, Math.min(100, Math.round(value))),
    })
  )
}

export const modelProgressControl =
  createProgressControl("model-progressbar")

export const generationProgressControl =
  createProgressControl("generation-progressbar")

export const removeSavedModelAction = createUiAction()

export function useControlState<State, Value>(
  control: ExternalControl<State, Value>
): State {
  return useSyncExternalStore(
    control.subscribe,
    control.getSnapshot,
    control.getSnapshot
  )
}
