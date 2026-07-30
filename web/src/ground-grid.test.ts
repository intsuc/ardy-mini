// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import * as THREE from "three";
import { describe, expect, it } from "vitest";

import {
  computeCameraRelativeGroundLayout,
  configureGroundGridMaterial,
  createGroundGridMaterial,
  groundGridPhase,
  updateGroundGridOrigin,
} from "./ground-grid";

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
  it("enables dithering and supplies a stable shader marker", () => {
    const material = createGroundGridMaterial();

    expect(material).toBeInstanceOf(THREE.MeshStandardMaterial);
    expect(material.dithering).toBe(true);
    expect(material.userData.pristineGrid).toBe(true);
    expect(material.customProgramCacheKey()).toContain(
      "ardy-pristine-ground-grid-v1",
    );
  });

  it("injects rebased grid coordinates and both grid scales before lighting", () => {
    const material = createGroundGridMaterial();
    const shader = {
      uniforms: {},
      vertexShader: `
#include <common>
void main() {
  #include <begin_vertex>
}
`,
      fragmentShader: `
#include <common>
void main() {
  vec4 diffuseColor = vec4(1.0);
  #include <color_fragment>
  #include <lights_fragment_begin>
}
`,
    } as unknown as Parameters<typeof material.onBeforeCompile>[0];

    material.onBeforeCompile(
      shader,
      {} as Parameters<typeof material.onBeforeCompile>[1],
    );

    expect(shader.vertexShader).toContain("mat3(modelMatrix) * transformed");
    expect(shader.vertexShader).not.toContain(
      "modelMatrix * vec4(transformed",
    );
    expect(shader.fragmentShader).toContain("float ardyPristineGrid");
    expect(shader.fragmentShader).toContain("ardyGroundMinorSpacing");
    expect(shader.fragmentShader).toContain("ardyGroundMajorSpacing");
    expect(shader.fragmentShader).toContain(
      "ardyGroundGridPosition / ardyGroundMinorSpacing",
    );
    expect(shader.fragmentShader).toContain(
      "ardyGroundGridPosition / ardyGroundMajorSpacing",
    );
    expect(shader.fragmentShader).not.toContain("+ vec2(0.5)");
    expect(shader.fragmentShader.indexOf("float ardyMinorGrid")).toBeLessThan(
      shader.fragmentShader.indexOf("#include <lights_fragment_begin>"),
    );
  });

  it("updates only a small periodic phase after the ground is recentered", () => {
    const material = configureGroundGridMaterial(
      new THREE.MeshStandardMaterial(),
    );
    updateGroundGridOrigin(material, 1_000_002.25, -1_000_002.75);
    const shader = {
      uniforms: {},
      vertexShader:
        "#include <common>\nvoid main(){\n#include <begin_vertex>\n}",
      fragmentShader:
        "#include <common>\nvoid main(){\n#include <color_fragment>\n}",
    } as unknown as Parameters<typeof material.onBeforeCompile>[0];

    material.onBeforeCompile(
      shader,
      {} as Parameters<typeof material.onBeforeCompile>[1],
    );
    const phase = (
      shader.uniforms.ardyGroundGridPhase as { value: THREE.Vector2 }
    ).value;

    expect(phase.toArray()).toEqual([2.25, 2.25]);
  });

  it("reuses the injected shader while updating its uniform configuration", () => {
    const material = createGroundGridMaterial();
    const shader = {
      uniforms: {},
      vertexShader:
        "#include <common>\nvoid main(){\n#include <begin_vertex>\n}",
      fragmentShader:
        "#include <common>\nvoid main(){\n#include <color_fragment>\n}",
    } as unknown as Parameters<typeof material.onBeforeCompile>[0];

    material.onBeforeCompile(
      shader,
      {} as Parameters<typeof material.onBeforeCompile>[1],
    );
    const minorSpacingUniform = shader.uniforms.ardyGroundMinorSpacing;
    const majorSpacingUniform = shader.uniforms.ardyGroundMajorSpacing;

    configureGroundGridMaterial(material, {
      minorSpacing: 1,
      majorSpacing: 10,
    });

    expect(shader.uniforms.ardyGroundMinorSpacing).toBe(minorSpacingUniform);
    expect(shader.uniforms.ardyGroundMajorSpacing).toBe(majorSpacingUniform);
    expect(minorSpacingUniform).toEqual({ value: 1 });
    expect(majorSpacingUniform).toEqual({ value: 10 });
    expect(
      material.customProgramCacheKey().match(/pristine-ground-grid/g),
    ).toHaveLength(1);
  });

  it("rebases an existing origin after changing the major period", () => {
    const material = createGroundGridMaterial();
    const shader = {
      uniforms: {},
      vertexShader:
        "#include <common>\nvoid main(){\n#include <begin_vertex>\n}",
      fragmentShader:
        "#include <common>\nvoid main(){\n#include <color_fragment>\n}",
    } as unknown as Parameters<typeof material.onBeforeCompile>[0];
    material.onBeforeCompile(
      shader,
      {} as Parameters<typeof material.onBeforeCompile>[1],
    );

    updateGroundGridOrigin(material, 7.25, -2.75);
    configureGroundGridMaterial(material, {
      majorSpacing: 10,
    });

    const phase = (
      shader.uniforms.ardyGroundGridPhase as { value: THREE.Vector2 }
    ).value;
    expect(phase.toArray()).toEqual([7.25, 7.25]);
  });
});
