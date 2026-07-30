// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import {
  MeshStandardNodeMaterial,
  type Node,
  Vector2,
} from "three/webgpu";
import { describe, expect, it } from "vitest";

import {
  computeCameraRelativeGroundLayout,
  configureGroundGridMaterial,
  createGroundGridMaterial,
  groundGridPhase,
  updateGroundGridOrigin,
} from "./ground-grid";

interface MutableUniformValue {
  value: unknown;
}

function materialUniforms(
  material: MeshStandardNodeMaterial,
): MutableUniformValue[] {
  const uniforms: MutableUniformValue[] = [];
  material.colorNode?.traverse((node: Node) => {
    if ("isUniformNode" in node && node.isUniformNode === true) {
      uniforms.push(node as unknown as MutableUniformValue);
    }
  });
  return uniforms;
}

describe("camera-relative ground layout", () => {
  it("snaps its center and coverage to the major grid period", () => {
    expect(
      computeCameraRelativeGroundLayout({
        targetX: 7.4,
        targetZ: -7.6,
        cameraFar: 100,
        maxCameraDistance: 12,
      }),
    ).toEqual({
      centerX: 5,
      centerZ: -10,
      halfExtent: 115,
      sideLength: 230,
      phaseX: 0,
      phaseZ: 0,
    });
  });

  it("keeps the grid phase stable after moving by a major period", () => {
    expect(groundGridPhase(2.25, 5)).toBeCloseTo(2.25);
    expect(groundGridPhase(7.25, 5)).toBeCloseTo(2.25);
    expect(groundGridPhase(-2.75, 5)).toBeCloseTo(2.25);
  });

  it("rejects invalid extents and incompatible grid periods", () => {
    expect(() =>
      computeCameraRelativeGroundLayout({
        targetX: 0,
        targetZ: 0,
        cameraFar: 0,
        maxCameraDistance: 12,
      }),
    ).toThrow(/far distance/);
    expect(() =>
      createGroundGridMaterial({
        minorSpacing: 0.6,
        majorSpacing: 5,
      }),
    ).toThrow(/whole-number multiple/);
  });
});

describe("Pristine Grid material", () => {
  it("uses a standard node material with a TSL color graph", () => {
    const material = createGroundGridMaterial();

    expect(material).toBeInstanceOf(MeshStandardNodeMaterial);
    // Node materials ignore Material.dithering in Three.js r185. The viewer
    // applies effective dithering in its final-output pipeline instead.
    expect(material.dithering).toBe(false);
    expect(material.userData.pristineGrid).toBe(true);
    expect(material.colorNode).not.toBeNull();
    expect(material.colorNode?.isNode).toBe(true);
  });

  it("updates only a small periodic phase after the ground is recentered", () => {
    const material = configureGroundGridMaterial(
      new MeshStandardNodeMaterial(),
    );
    updateGroundGridOrigin(material, 1_000_002.25, -1_000_002.75);
    const phase = materialUniforms(material).find(
      (uniform) => uniform.value instanceof Vector2,
    );

    expect(phase?.value).toBeInstanceOf(Vector2);
    expect((phase?.value as Vector2).toArray()).toEqual([2.25, 2.25]);
  });

  it("reuses the TSL graph while updating its uniform configuration", () => {
    const material = createGroundGridMaterial();
    const colorNode = material.colorNode;
    const uniforms = materialUniforms(material);
    const minorSpacingUniform = uniforms.find(
      (uniform) => uniform.value === 0.5,
    );
    const majorSpacingUniform = uniforms.find(
      (uniform) => uniform.value === 5,
    );

    configureGroundGridMaterial(material, {
      minorSpacing: 1,
      majorSpacing: 10,
    });

    expect(material.colorNode).toBe(colorNode);
    expect(minorSpacingUniform?.value).toBe(1);
    expect(majorSpacingUniform?.value).toBe(10);
  });

  it("rebases an existing origin after changing the major period", () => {
    const material = createGroundGridMaterial();

    updateGroundGridOrigin(material, 7.25, -2.75);
    configureGroundGridMaterial(material, {
      majorSpacing: 10,
    });

    const phase = materialUniforms(material).find(
      (uniform) => uniform.value instanceof Vector2,
    );
    expect((phase?.value as Vector2).toArray()).toEqual([7.25, 7.25]);
  });
});
