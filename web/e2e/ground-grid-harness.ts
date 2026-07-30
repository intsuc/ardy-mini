// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import * as THREE from "three/webgpu";

import { createGroundGridMaterial } from "../src/ground-grid";

export interface GroundGridPhaseSamples {
  origin: number[];
  betweenLines: number[];
  nextMinorLine: number[];
  beforeNextMajorLine: number[];
  nextMajorLine: number[];
  afterNextMajorLine: number[];
}

export interface GroundGridObliqueDiagnostics {
  width: number;
  height: number;
  rowMeans: number[];
  movedRowMeans: number[];
  meanAbsoluteDelta: number;
  maximumRowMeanStep: number;
}

function luminance(
  pixels: ArrayLike<number>,
  pixelOffset: number,
): number {
  return (
    pixels[pixelOffset] * 0.2126 +
    pixels[pixelOffset + 1] * 0.7152 +
    pixels[pixelOffset + 2] * 0.0722
  );
}

function requireNativeWebGpu(renderer: THREE.WebGPURenderer): void {
  if (renderer.backend.isWebGPUBackend !== true) {
    throw new Error("Ground-grid GPU regression requires native WebGPU.");
  }
}

/**
 * Renders the production grid at a shallow perspective and then repeats it
 * after a tiny camera rotation. The upper ground band is where the minor grid
 * approaches the pixel footprint and is most susceptible to moire/shimmer.
 */
export async function renderGroundGridObliqueDiagnostics(): Promise<GroundGridObliqueDiagnostics> {
  const width = 800;
  const height = 600;
  const renderer = new THREE.WebGPURenderer({ antialias: false });
  await renderer.init();
  requireNativeWebGpu(renderer);
  renderer.setSize(width, height, false);
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  const target = new THREE.RenderTarget(width, height, {
    format: THREE.RGBAFormat,
    type: THREE.UnsignedByteType,
  });
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);
  const camera = new THREE.PerspectiveCamera(38, width / height, 0.01, 100);
  const material = createGroundGridMaterial({
    baseColor: 0x202020,
    minorColor: 0x686868,
    majorColor: 0x888888,
    minorSpacing: 0.5,
    majorSpacing: 5,
    minorLineWidth: 0.018,
    majorLineWidth: 0.006,
    minorOpacity: 0.44,
    // Isolate the highest-frequency grid so this diagnostic measures minor
    // line shimmer instead of the intentional major-line crossings.
    majorOpacity: 0,
  });
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(200, 200),
    material,
  );

  try {
    scene.add(new THREE.AmbientLight(0xffffff, Math.PI));
    ground.rotation.x = -Math.PI / 2;
    scene.add(ground);
    camera.position.set(3.1, 2.15, 3.4);
    camera.lookAt(0, 0.85, 0);

    renderer.setRenderTarget(target);
    await renderer.compileAsync(scene, camera);

    const renderPixels = async (): Promise<ArrayLike<number>> => {
      renderer.render(scene, camera);
      return renderer.readRenderTargetPixelsAsync(
        target,
        0,
        0,
        width,
        height,
      );
    };

    const initialPixels = await renderPixels();
    camera.position.applyAxisAngle(
      new THREE.Vector3(0, 1, 0),
      THREE.MathUtils.degToRad(0.08),
    );
    camera.lookAt(0, 0.85, 0);
    const movedPixels = await renderPixels();

    // readRenderTargetPixelsAsync is bottom-up. This interval corresponds to
    // the distant, shallow-angle portion of the visible ground.
    const rowStart = Math.floor(height * 0.58);
    const rowEnd = Math.floor(height * 0.8);
    const columnStart = Math.floor(width * 0.15);
    const columnEnd = Math.floor(width * 0.85);
    const rowMeans: number[] = [];
    const movedRowMeans: number[] = [];
    let absoluteDelta = 0;
    let sampleCount = 0;

    for (let y = rowStart; y < rowEnd; y += 1) {
      let initialRow = 0;
      let movedRow = 0;
      for (let x = columnStart; x < columnEnd; x += 1) {
        const offset = (y * width + x) * 4;
        const initial = luminance(initialPixels, offset);
        const moved = luminance(movedPixels, offset);
        initialRow += initial;
        movedRow += moved;
        absoluteDelta += Math.abs(initial - moved);
        sampleCount += 1;
      }
      rowMeans.push(initialRow / (columnEnd - columnStart));
      movedRowMeans.push(movedRow / (columnEnd - columnStart));
    }

    let maximumRowMeanStep = 0;
    for (let index = 1; index < rowMeans.length; index += 1) {
      maximumRowMeanStep = Math.max(
        maximumRowMeanStep,
        Math.abs(rowMeans[index] - rowMeans[index - 1]),
      );
    }

    return {
      width,
      height,
      rowMeans,
      movedRowMeans,
      meanAbsoluteDelta: absoluteDelta / sampleCount,
      maximumRowMeanStep,
    };
  } finally {
    target.dispose();
    material.dispose();
    ground.geometry.dispose();
    renderer.dispose();
  }
}

export async function renderGroundGridPhaseSamples(): Promise<GroundGridPhaseSamples> {
  const size = 1_200;
  const renderer = new THREE.WebGPURenderer({ antialias: false });
  await renderer.init();
  requireNativeWebGpu(renderer);
  renderer.setSize(size, size, false);
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  const target = new THREE.RenderTarget(size, size, {
    format: THREE.RGBAFormat,
    type: THREE.UnsignedByteType,
  });
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-6, 6, 6, -6, 0.1, 20);
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
    new THREE.PlaneGeometry(12, 12),
    material,
  );
  try {
    scene.add(new THREE.AmbientLight(0xffffff, Math.PI));
    ground.rotation.x = -Math.PI / 2;
    scene.add(ground);
    camera.position.set(0, 10, 0);
    camera.up.set(0, 0, -1);
    camera.lookAt(0, 0, 0);

    renderer.setRenderTarget(target);
    await renderer.compileAsync(scene, camera);
    renderer.render(scene, camera);

    const sample = async (x: number, z = 0.13): Promise<number[]> => {
      const pixelX = Math.floor(((x + 6) / 12) * size);
      const pixelY = Math.floor(((z + 6) / 12) * size);
      const pixel = await renderer.readRenderTargetPixelsAsync(
        target,
        pixelX,
        pixelY,
        1,
        1,
      );
      return Array.from(pixel);
    };

    return {
      // Red over green proves that both scales occupy the world origin.
      origin: await sample(0),
      betweenLines: await sample(0.25),
      nextMinorLine: await sample(0.5),
      beforeNextMajorLine: await sample(4.5),
      nextMajorLine: await sample(5),
      afterNextMajorLine: await sample(5.5),
    };
  } finally {
    target.dispose();
    material.dispose();
    ground.geometry.dispose();
    renderer.dispose();
  }
}
