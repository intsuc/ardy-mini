// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import { normalizeStructuredMotion } from "./motion-data";
import {
  SESSION_FILE_VERSION,
  createMotionSession,
  decodeMotionJson,
  decodeSessionBinary,
  decodeSessionFile,
  decodeSessionJson,
  encodeMotionCsv,
  encodeMotionJson,
  encodeSessionBinary,
  encodeSessionJson,
  getSessionRestoreMode,
  isContinuationModelCompatible,
  safeDownloadFilename,
} from "./session-format";

function exampleMotion() {
  return normalizeStructuredMotion({
    skeleton: {
      id: "mini",
      name: "Mini",
      jointNames: ["Root", "Foot"],
      parents: [-1, 0],
      rootJointIndex: 0,
      contactJointIndices: [1],
      contactNames: ["Foot"],
    },
    positions: [
      [
        [0, 1, 0],
        [0, 0, 0],
      ],
      [
        [1, 1, 0],
        [1, 0, 0],
      ],
    ],
    normalizedMotion: [
      [0.1, 0.2],
      [0.3, 0.4],
    ],
    localRotations: [
      [
        [0, 0, 0, 1],
        [0, 0, 0, 1],
      ],
      [
        [0, 0, 0, 1],
        [0, 0, 0, 1],
      ],
    ],
    globalRotations: [
      [
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
      ],
      [
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
      ],
    ],
    roots: [
      [0, 0, 0],
      [1, 0, 0],
    ],
    contacts: [[1], [0]],
    fps: 20,
  });
}

function exampleSession() {
  return createMotionSession({
    motion: exampleMotion(),
    editor: {
      initialTransform: { position: [1, 0, 2], headingRadians: 0.5 },
      waypoints: [
        {
          id: "goal",
          frame: 1,
          position: [2, 0, 2],
          enabled: true,
        },
      ],
      constraints: [
        {
          id: "root",
          kind: "root",
          startFrame: 0,
          endFrame: 1,
          jointIndex: 0,
          position: [1, 1, 0],
          enabled: true,
        },
      ],
      outputVisibility: {
        skeleton: true,
        mesh: false,
        reference: false,
        trajectory: true,
        contacts: true,
        orientationAxes: true,
        constraints: true,
        initialTransform: true,
        waypoints: true,
      },
    },
    generationConstraints: [
      {
        id: "runtime-root",
        kind: "root",
        frame: 1,
        values: new Float32Array([0.5, 0, -0.25]),
        mask: new Float32Array([1, 0, 1]),
      },
    ],
    provenance: {
      prompt: "walk forward",
      seed: 42,
      modelId: "ardy-mini",
      createdAt: "2026-07-29T00:00:00.000Z",
    },
    continuation: {
      hybridTokens: Float32Array.from({ length: 2 * 3 }, (_, index) => index + 0.25),
      hybridDim: 3,
      frameCount: 2,
      random: { seed: 42, state: 17, spareNormal: 0.125 },
      initialTranslation: [0, 0, 0],
      initialHeading: 0,
    },
  });
}

function expectRoundTrip(session: ReturnType<typeof exampleSession>): void {
  expect(session.motion.skeleton.jointNames).toEqual(["Root", "Foot"]);
  expect(Array.from(session.motion.positions)).toEqual([0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]);
  expect(session.motion.contacts).toEqual(new Uint8Array([1, 0]));
  expect(session.motion.localRotations?.shape).toEqual([2, 2, 4]);
  expect(session.motion.globalRotations?.shape).toEqual([2, 2, 9]);
  expect(session.editor.waypoints[0].id).toBe("goal");
  expect(session.generationConstraints?.[0].id).toBe("runtime-root");
  expect(session.generationConstraints?.[0].values).toEqual(
    new Float32Array([0.5, 0, -0.25]),
  );
  expect(session.provenance?.prompt).toBe("walk forward");
  expect(session.continuation?.hybridDim).toBe(3);
  expect(Array.from(session.continuation?.hybridTokens ?? [])).toEqual([
    0.25, 1.25, 2.25, 3.25, 4.25, 5.25,
  ]);
}

describe("portable motion sessions", () => {
  it("round-trips the versioned safe JSON format", () => {
    const session = decodeSessionJson(encodeSessionJson(exampleSession()));
    expectRoundTrip(session);
    expect(getSessionRestoreMode(session)).toBe("continuable");
  });

  it("round-trips the compact binary format without pickle", () => {
    const encoded = encodeSessionBinary(exampleSession());
    expect(new TextDecoder().decode(encoded.subarray(0, 8))).toBe("ARDYSES1");
    expectRoundTrip(decodeSessionBinary(encoded));
  });

  it("detects JSON and binary browser files by content", async () => {
    const session = exampleSession();
    const json = await decodeSessionFile(
      new Blob([encodeSessionJson(session)], { type: "application/json" }),
    );
    const encoded = encodeSessionBinary(session);
    const binary = await decodeSessionFile(
      new Blob([new Uint8Array(encoded).buffer], {
        type: "application/vnd.ardy.session",
      }),
    );
    expectRoundTrip(json);
    expectRoundTrip(binary);
    expect(
      getSessionRestoreMode(
        createMotionSession({
          motion: exampleMotion(),
        }),
      ),
    ).toBe("playback-only");
  });

  it("only restores continuation state with its originating model identity", () => {
    const session = exampleSession();
    const matching = { id: "ardy-mini", variant: "core40-minilm" };
    expect(isContinuationModelCompatible(session.provenance, matching)).toBe(
      true,
    );
    expect(getSessionRestoreMode(session, matching)).toBe("continuable");

    expect(
      isContinuationModelCompatible(session.provenance, {
        ...matching,
        id: "another-model",
      }),
    ).toBe(false);
    expect(
      getSessionRestoreMode(session, {
        ...matching,
        id: "another-model",
      }),
    ).toBe("playback-only");

    const variantSession = createMotionSession({
      ...session,
      provenance: {
        ...session.provenance,
        modelVariant: "core40-minilm",
      },
    });
    expect(
      getSessionRestoreMode(variantSession, {
        id: "ardy-mini",
        variant: "different-weights",
      }),
    ).toBe("playback-only");
  });

  it("rejects unsupported versions, corrupted binary, and incomplete continuation state", () => {
    const json = JSON.parse(encodeSessionJson(exampleSession())) as Record<string, unknown>;
    json.version = SESSION_FILE_VERSION + 1;
    expect(() => decodeSessionJson(JSON.stringify(json))).toThrow(/Unsupported session version/);

    const binary = encodeSessionBinary(exampleSession());
    binary[0] = 0;
    expect(() => decodeSessionBinary(binary)).toThrow(/magic/);

    const invalidContinuation = JSON.parse(encodeSessionJson(exampleSession())) as {
      continuation: { hybridTokens: number[]; hybridDim: number };
    };
    invalidContinuation.continuation.hybridTokens.pop();
    expect(() => decodeSessionJson(JSON.stringify(invalidContinuation))).toThrow(
      /complete 3-value tokens/,
    );

    const invalidConstraint = JSON.parse(encodeSessionJson(exampleSession())) as {
      generationConstraints: Array<{ mask: number[] }>;
    };
    invalidConstraint.generationConstraints[0].mask[0] = 2;
    expect(() => decodeSessionJson(JSON.stringify(invalidConstraint))).toThrow(
      /between 0 and 1/,
    );
  });

  it("rejects generation constraints that cannot survive worker validation", () => {
    const source = exampleSession();
    expect(() =>
      createMotionSession({
        ...source,
        generationConstraints: [
          {
            id: "inactive",
            kind: "root",
            frame: 0,
            values: new Float32Array([0, 0, 0]),
            mask: new Float32Array([0.25, 0.25, 0.25]),
          },
        ],
      }),
    ).toThrow(/observe at least one/);

    const active = source.generationConstraints![0];
    expect(() =>
      createMotionSession({
        ...source,
        generationConstraints: [
          { ...active, id: "same" },
          { ...active, id: " same " },
        ],
      }),
    ).toThrow(/duplicated/);
  });
});

describe("motion exports", () => {
  it("round-trips structured JSON and emits position/root/contact CSV columns", () => {
    const motion = decodeMotionJson(encodeMotionJson(exampleMotion()));
    expect(motion.positionsShape).toEqual([2, 2, 3]);
    expect(motion.contacts).toEqual(new Uint8Array([1, 0]));

    const csv = encodeMotionCsv(motion);
    const [header, first, second] = csv.split("\n");
    expect(header).toContain("position:Root:x");
    expect(header).toContain("root:2");
    expect(header).toContain("contact:Foot");
    expect(first.startsWith("0,0,")).toBe(true);
    expect(second.startsWith("1,0.05,")).toBe(true);
  });

  it("sanitizes browser download names", () => {
    expect(safeDownloadFilename('../Walk: "fast".json', "motion.json")).toBe(
      "Walk-fast-.json",
    );
  });
});
