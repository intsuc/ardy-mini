// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it, vi } from "vitest"

import {
  clearModelCacheAction,
  modelDownloadAction,
  modelDownloadCancelAction,
  modelUiControl,
  regenerateMotionAction,
  startNewMotionAction,
  type UiAction,
} from "../ui-control-store"

afterEach(() => {
  modelUiControl.dispatch({ type: "reset" })
})

describe("model startup state machine", () => {
  it("pauses a partial download, resumes it, and prepares the runtime", () => {
    modelUiControl.dispatch({
      type: "cache-missing",
      cachedFiles: 0,
      totalFiles: 5,
      cachedBytes: 0,
      totalBytes: 1_000,
    })
    expect(modelUiControl.getSnapshot()).toMatchObject({
      cache: "missing",
      runtime: "idle",
      cachedFiles: 0,
      totalFiles: 5,
      cachedBytes: 0,
      totalBytes: 1_000,
    })

    modelUiControl.dispatch({ type: "download-started" })
    modelUiControl.dispatch({ type: "runtime-loading" })
    modelUiControl.dispatch({
      type: "download-progress",
      cachedFiles: 2,
      totalFiles: 5,
      cachedBytes: 400,
      totalBytes: 1_000,
    })
    expect(modelUiControl.getSnapshot()).toMatchObject({
      cache: "downloading",
      runtime: "loading",
      cachedFiles: 2,
      cachedBytes: 400,
    })

    modelUiControl.dispatch({ type: "download-cancel-requested" })
    expect(modelUiControl.getSnapshot().cache).toBe("cancelling")

    modelUiControl.dispatch({ type: "runtime-idle" })
    modelUiControl.dispatch({
      type: "cache-missing",
      cachedFiles: 2,
      totalFiles: 5,
      cachedBytes: 400,
      totalBytes: 1_000,
    })
    expect(modelUiControl.getSnapshot()).toMatchObject({
      cache: "missing",
      runtime: "idle",
      cachedFiles: 2,
      totalFiles: 5,
      cachedBytes: 400,
      totalBytes: 1_000,
    })

    modelUiControl.dispatch({ type: "download-started" })
    modelUiControl.dispatch({ type: "runtime-loading" })
    modelUiControl.dispatch({
      type: "download-progress",
      cachedFiles: 5,
      totalFiles: 5,
      cachedBytes: 1_000,
      totalBytes: 1_000,
    })
    expect(modelUiControl.getSnapshot()).toMatchObject({
      cache: "downloading",
      runtime: "loading",
      cachedFiles: 5,
      cachedBytes: 1_000,
    })

    modelUiControl.dispatch({ type: "download-completed" })
    expect(modelUiControl.getSnapshot()).toMatchObject({
      cache: "verifying",
      cachedFiles: 5,
      cachedBytes: 1_000,
    })
    modelUiControl.dispatch({
      type: "initialization-progress",
      completedSteps: 0,
      totalSteps: 4,
    })
    modelUiControl.dispatch({
      type: "verification-progress",
      completedFiles: 3,
      totalFiles: 5,
    })
    expect(modelUiControl.getSnapshot()).toMatchObject({
      cache: "initializing",
      cachedFiles: 5,
      totalFiles: 5,
      cachedBytes: 1_000,
      verifiedFiles: 3,
      initializationSteps: 0,
      totalInitializationSteps: 4,
    })

    modelUiControl.dispatch({
      type: "initialization-progress",
      completedSteps: 1,
      totalSteps: 4,
    })
    modelUiControl.dispatch({
      type: "verification-progress",
      completedFiles: 5,
      totalFiles: 5,
    })
    modelUiControl.dispatch({
      type: "initialization-progress",
      completedSteps: 4,
      totalSteps: 4,
    })

    modelUiControl.dispatch({
      type: "cache-ready",
      totalFiles: 5,
      totalBytes: 1_000,
    })
    modelUiControl.dispatch({ type: "runtime-ready" })
    expect(modelUiControl.getSnapshot()).toEqual({
      cache: "ready",
      errorOperation: null,
      runtime: "ready",
      cachedFiles: 5,
      totalFiles: 5,
      cachedBytes: 1_000,
      totalBytes: 1_000,
      verifiedFiles: 5,
      initializationSteps: 4,
      totalInitializationSteps: 4,
    })
  })

  it("clears persisted files without unloading the ready runtime", () => {
    modelUiControl.dispatch({
      type: "cache-ready",
      totalFiles: 5,
      totalBytes: 1_000,
    })
    modelUiControl.dispatch({ type: "runtime-ready" })

    modelUiControl.dispatch({ type: "clear-started" })
    expect(modelUiControl.getSnapshot()).toMatchObject({
      cache: "clearing",
      runtime: "ready",
      cachedBytes: 1_000,
    })

    modelUiControl.dispatch({ type: "cache-cleared" })
    expect(modelUiControl.getSnapshot()).toEqual({
      cache: "missing",
      errorOperation: null,
      runtime: "ready",
      cachedFiles: 0,
      totalFiles: 5,
      cachedBytes: 0,
      totalBytes: 1_000,
      verifiedFiles: 0,
      initializationSteps: 0,
      totalInitializationSteps: 0,
    })
  })

  it("clears a partial download before the runtime is ready", () => {
    modelUiControl.dispatch({
      type: "cache-missing",
      cachedFiles: 2,
      totalFiles: 5,
      cachedBytes: 400,
      totalBytes: 1_000,
    })

    modelUiControl.dispatch({ type: "clear-started" })
    expect(modelUiControl.getSnapshot()).toMatchObject({
      cache: "clearing",
      runtime: "idle",
      cachedFiles: 2,
      cachedBytes: 400,
    })

    modelUiControl.dispatch({ type: "cache-cleared" })
    expect(modelUiControl.getSnapshot()).toMatchObject({
      cache: "missing",
      runtime: "idle",
      cachedFiles: 0,
      cachedBytes: 0,
      totalFiles: 5,
      totalBytes: 1_000,
    })
  })

  it("rejects operations that conflict with an active download", () => {
    modelUiControl.dispatch({
      type: "cache-missing",
      cachedFiles: 1,
      totalFiles: 5,
      cachedBytes: 200,
      totalBytes: 1_000,
    })
    modelUiControl.dispatch({ type: "download-started" })

    const downloading = modelUiControl.getSnapshot()
    modelUiControl.dispatch({ type: "clear-started" })
    expect(modelUiControl.getSnapshot()).toBe(downloading)

    modelUiControl.dispatch({ type: "download-cancel-requested" })
    const cancelling = modelUiControl.getSnapshot()
    modelUiControl.dispatch({
      type: "download-progress",
      cachedFiles: 5,
      totalFiles: 5,
      cachedBytes: 1_000,
      totalBytes: 1_000,
    })
    modelUiControl.dispatch({ type: "download-cancel-requested" })
    expect(modelUiControl.getSnapshot()).toBe(cancelling)
  })

  it("keeps resumed download progress monotonic while cached files are replayed", () => {
    modelUiControl.dispatch({
      type: "cache-missing",
      cachedFiles: 2,
      totalFiles: 5,
      cachedBytes: 400,
      totalBytes: 1_000,
    })
    modelUiControl.dispatch({ type: "cache-check-started" })
    expect(modelUiControl.getSnapshot()).toMatchObject({
      cache: "checking",
      cachedFiles: 2,
      cachedBytes: 400,
    })

    modelUiControl.dispatch({
      type: "cache-missing",
      cachedFiles: 2,
      totalFiles: 5,
      cachedBytes: 400,
      totalBytes: 1_000,
    })
    modelUiControl.dispatch({ type: "download-started" })
    modelUiControl.dispatch({
      type: "download-progress",
      cachedFiles: 1,
      totalFiles: 5,
      cachedBytes: 100,
      totalBytes: 1_000,
    })
    expect(modelUiControl.getSnapshot()).toMatchObject({
      cache: "downloading",
      cachedFiles: 2,
      cachedBytes: 400,
    })

    modelUiControl.dispatch({
      type: "download-progress",
      cachedFiles: 3,
      totalFiles: 5,
      cachedBytes: 600,
      totalBytes: 1_000,
    })
    expect(modelUiControl.getSnapshot()).toMatchObject({
      cache: "downloading",
      cachedFiles: 3,
      cachedBytes: 600,
    })
  })

  it("stops accepting download progress and cancellation after download completion", () => {
    modelUiControl.dispatch({
      type: "cache-missing",
      cachedFiles: 0,
      totalFiles: 5,
      cachedBytes: 0,
      totalBytes: 1_000,
    })
    modelUiControl.dispatch({ type: "download-started" })
    modelUiControl.dispatch({
      type: "download-progress",
      cachedFiles: 5,
      totalFiles: 5,
      cachedBytes: 1_000,
      totalBytes: 1_000,
    })
    modelUiControl.dispatch({ type: "download-completed" })

    const preparing = modelUiControl.getSnapshot()
    modelUiControl.dispatch({
      type: "download-progress",
      cachedFiles: 1,
      totalFiles: 5,
      cachedBytes: 100,
      totalBytes: 1_000,
    })
    modelUiControl.dispatch({ type: "download-cancel-requested" })

    expect(modelUiControl.getSnapshot()).toBe(preparing)
    expect(preparing).toMatchObject({
      cache: "verifying",
      cachedFiles: 5,
      cachedBytes: 1_000,
    })
  })

  it("distinguishes an initialization retry from a download retry", () => {
    modelUiControl.dispatch({
      type: "cache-missing",
      cachedFiles: 5,
      totalFiles: 5,
      cachedBytes: 1_000,
      totalBytes: 1_000,
    })
    modelUiControl.dispatch({ type: "download-started" })
    modelUiControl.dispatch({ type: "download-completed" })
    modelUiControl.dispatch({
      type: "initialization-progress",
      completedSteps: 0,
      totalSteps: 4,
    })
    modelUiControl.dispatch({
      type: "cache-error",
      operation: "initialization",
    })
    expect(modelUiControl.getSnapshot()).toMatchObject({
      cache: "error",
      errorOperation: "initialization",
    })

    modelUiControl.dispatch({ type: "download-started" })
    expect(modelUiControl.getSnapshot()).toMatchObject({
      cache: "downloading",
      errorOperation: null,
    })
  })
})

const uiActions: readonly (readonly [string, UiAction])[] = [
  ["model download", modelDownloadAction],
  ["model download cancellation", modelDownloadCancelAction],
  ["model cache clearing", clearModelCacheAction],
  ["motion regeneration", regenerateMotionAction],
  ["new motion", startNewMotionAction],
]

describe.each(uiActions)("%s action", (_label, action) => {
  it("notifies active subscribers and supports explicit unsubscribe", () => {
    const listener = vi.fn()
    const unsubscribe = action.onTrigger(listener)

    action.trigger()
    expect(listener).toHaveBeenCalledTimes(1)

    unsubscribe()
    action.trigger()
    expect(listener).toHaveBeenCalledTimes(1)
  })

  it("removes subscriptions when their AbortSignal aborts", () => {
    const listener = vi.fn()
    const controller = new AbortController()
    action.onTrigger(listener, controller.signal)

    action.trigger()
    expect(listener).toHaveBeenCalledTimes(1)

    controller.abort()
    action.trigger()
    expect(listener).toHaveBeenCalledTimes(1)

    const alreadyAborted = new AbortController()
    alreadyAborted.abort()
    action.onTrigger(listener, alreadyAborted.signal)
    action.trigger()
    expect(listener).toHaveBeenCalledTimes(1)
  })
})
