// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import * as THREE from "three/webgpu";
import * as TSL from "three/tsl";

const EIGHT_BIT_QUANTIZATION_LEVELS = 255;

interface DitherNode {
  readonly a: DitherNode;
  readonly rgb: DitherNode;
  readonly xy: DitherNode;
  add(value: unknown): DitherNode;
  div(value: unknown): DitherNode;
  lessThan(value: unknown): DitherNode;
  mul(value: unknown): DitherNode;
  sub(value: unknown): DitherNode;
}

interface DitherPassNode extends DitherNode {
  dispose(): void;
  getTextureNode(name: string): DitherNode;
}

interface DitherTslFacade {
  readonly screenCoordinate: DitherNode;
  interleavedGradientNoise(position: unknown): DitherNode;
  pass(scene: THREE.Scene, camera: THREE.Camera): DitherPassNode;
  renderOutput(
    color: unknown,
    toneMapping: unknown,
    outputColorSpace: unknown,
  ): DitherNode;
  select(
    condition: unknown,
    whenTrue: unknown,
    whenFalse: unknown,
  ): DitherNode;
  vec4(rgb: unknown, alpha: unknown): DitherNode;
}

// As with the procedural ground graph, constraining the TSL surface prevents
// TypeScript from expanding its full recursive overload graph.
const ditherTsl = TSL as unknown as DitherTslFacade;

/**
 * Final-output pipeline for the WebGPU preview.
 *
 * WebGPURenderer performs tone mapping and color-space conversion in a
 * full-screen output pass. Node materials currently ignore Material.dithering,
 * so low-luminance fog gradients otherwise reach the 8-bit canvas as visible
 * bands. Applying a stationary half-LSB dither after those transforms preserves
 * their mean color while breaking up the quantization boundaries.
 */
export class DitheredRenderPipeline {
  readonly scenePass: THREE.PassNode;
  readonly pipeline: THREE.RenderPipeline;

  constructor(
    renderer: THREE.WebGPURenderer,
    scene: THREE.Scene,
    camera: THREE.Camera,
  ) {
    const scenePass = ditherTsl.pass(scene, camera);
    this.scenePass = scenePass as unknown as THREE.PassNode;

    const displayOutput = ditherTsl.renderOutput(
      scenePass,
      renderer.toneMapping,
      renderer.outputColorSpace,
    );
    const foregroundMask = ditherTsl.select(
      scenePass.getTextureNode("depth").lessThan(1),
      1,
      0,
    );
    const dither = ditherTsl.interleavedGradientNoise(
      ditherTsl.screenCoordinate.xy,
    )
      .sub(0.5)
      .div(EIGHT_BIT_QUANTIZATION_LEVELS)
      .mul(foregroundMask);
    const ditheredOutput = ditherTsl.vec4(
      displayOutput.rgb.add(dither),
      displayOutput.a,
    );

    this.pipeline = new THREE.RenderPipeline(
      renderer,
      ditheredOutput as unknown as THREE.Node<"vec4">,
    );
    // displayOutput already performs ACES/sRGB conversion. Disabling the
    // wrapper prevents double tone mapping and double color conversion.
    this.pipeline.outputColorTransform = false;
  }

  render(): void {
    this.pipeline.render();
  }

  dispose(): void {
    this.pipeline.dispose();
    this.scenePass.dispose();
  }
}
