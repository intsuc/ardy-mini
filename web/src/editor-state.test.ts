// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
  normalizeConstraintMarker,
  normalizeEditorState,
  normalizeInitialTransform,
  normalizeWaypoint,
} from "./editor-state";

describe("motion editor state validation", () => {
  it("normalizes initial transform, waypoints, and constraint tracks", () => {
    const state = normalizeEditorState(
      {
        initial_transform: {
          position: [1, 0, -2],
          heading_radians: Math.PI / 2,
        },
        waypoints: [
          {
            id: "turn",
            frame: 4,
            position: [2, 0, 3],
          },
        ],
        constraints: [
          {
            id: "left-hand",
            kind: "transform",
            start_frame: 2,
            end_frame: 5,
            joint_index: 2,
            position: [0, 1, 0],
            orientation: [0, 0, 0, 2],
          },
        ],
        output_visibility: {
          contacts: false,
          orientationAxes: true,
          mesh: true,
          reference: true,
        },
      },
      { frameCount: 8, jointCount: 3 },
    );

    expect(state.initialTransform.position).toEqual([1, 0, -2]);
    expect(state.waypoints[0]).toMatchObject({ id: "turn", frame: 4, enabled: true });
    expect(state.constraints[0].orientation).toEqual([0, 0, 0, 1]);
    expect(state.constraints[0]).toMatchObject({ startFrame: 2, endFrame: 5 });
    expect(state.outputVisibility.contacts).toBe(false);
    expect(state.outputVisibility.orientationAxes).toBe(true);
    expect(state.outputVisibility.skeleton).toBe(true);
    expect(state.outputVisibility.mesh).toBe(true);
    expect(state.outputVisibility.reference).toBe(true);
  });

  it("keeps future timeline targets beyond the current clip", () => {
    const state = normalizeEditorState(
      {
        waypoints: [
          {
            id: "future-root",
            frame: 120,
            position: [4, 0, 2],
          },
        ],
        constraints: [
          {
            id: "future-hand",
            kind: "position",
            startFrame: 100,
            endFrame: 140,
            jointIndex: 2,
            position: [0.5, 1.2, 0],
          },
        ],
      },
      { frameCount: 8, jointCount: 3 },
    );

    expect(state.waypoints[0].frame).toBe(120);
    expect(state.constraints[0]).toMatchObject({
      startFrame: 100,
      endFrame: 140,
    });
  });

  it("rejects invalid editing values before they reach Three.js", () => {
    expect(() =>
      normalizeInitialTransform({ position: [0, Number.NaN, 0], headingRadians: 0 }),
    ).toThrow(/finite/);
    expect(() =>
      normalizeWaypoint(
        { id: "late", frame: 10, position: [0, 0, 0] },
        { frameCount: 10 },
      ),
    ).toThrow(/outside/);
    expect(() =>
      normalizeConstraintMarker(
        {
          id: "bad-joint",
          kind: "position",
          startFrame: 0,
          jointIndex: 3,
          position: [0, 0, 0],
        },
        { frameCount: 1, jointCount: 3 },
      ),
    ).toThrow(/outside/);
    expect(() =>
      normalizeConstraintMarker({
        id: "backwards",
        kind: "position",
        startFrame: 2,
        endFrame: 1,
        position: [0, 0, 0],
      }),
    ).toThrow(/must not precede/);
  });

  it("defaults optional comparison layers to off", () => {
    const state = normalizeEditorState({});
    expect(state.outputVisibility.mesh).toBe(false);
    expect(state.outputVisibility.reference).toBe(false);
  });
});
