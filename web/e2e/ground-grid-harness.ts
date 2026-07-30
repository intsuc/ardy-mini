// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import * as THREE from "three";

import { createGroundGridMaterial } from "../src/ground-grid";

export interface GroundGridPhaseSamples {
  origin: number[];
  betweenLines: number[];
  nextMinorLine: number[];
  shaderErrors: string[];
}

export async function renderGroundGridPhaseSamples(): Promise<GroundGridPhaseSamples> {
  const size = 1_000;
  const renderer = new THREE.WebGLRenderer();
  renderer.setSize(size, size, false);
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  const target = new THREE.WebGLRenderTarget(size, size);
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 20);
  const material = createGroundGridMaterial({
    baseColor: 0x000000,
    minorColor: 0x00ff00,
    majorColor: 0xff0000,
    minorSpacing: 0.5,
    majorSpacing: 5,
    minorLineWidth: 0.08,
    majorLineWidth: 0.008,
    minorOpacity: 1,
    majorOpacity: 0.5,
  });
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(10, 10),
    material,
  );
  const shaderErrors: string[] = [];

  try {
    renderer.debug.onShaderError = (
      gl,
      program,
      vertexShader,
      fragmentShader,
    ) => {
      shaderErrors.push(
        [
          gl.getProgramInfoLog(program),
          gl.getShaderInfoLog(vertexShader),
          gl.getShaderInfoLog(fragmentShader),
        ]
          .filter(Boolean)
          .join("\n"),
      );
    };
    scene.add(new THREE.AmbientLight(0xffffff, Math.PI));
    ground.rotation.x = -Math.PI / 2;
    scene.add(ground);
    camera.position.set(0, 10, 0);
    camera.up.set(0, 0, -1);
    camera.lookAt(0, 0, 0);

    renderer.setRenderTarget(target);
    await renderer.compileAsync(scene, camera);
    renderer.render(scene, camera);

    const pixel = new Uint8Array(4);
    const sample = (x: number, z = 0.13): number[] => {
      const pixelX = Math.floor(((x + 5) / 10) * size);
      const pixelY = Math.floor(((z + 5) / 10) * size);
      renderer.readRenderTargetPixels(
        target,
        pixelX,
        pixelY,
        1,
        1,
        pixel,
      );
      return [...pixel];
    };

    return {
      // Red over green proves that both scales occupy the world origin.
      origin: sample(0),
      betweenLines: sample(0.25),
      nextMinorLine: sample(0.5),
      shaderErrors,
    };
  } finally {
    target.dispose();
    material.dispose();
    ground.geometry.dispose();
    renderer.dispose();
  }
}
