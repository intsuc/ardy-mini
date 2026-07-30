// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import * as THREE from "three";

const GRID_SHADER_CACHE_KEY = "ardy-pristine-ground-grid-v1";

const DEFAULTS = {
  baseColor: "#262626",
  minorColor: "#404040",
  majorColor: "#404040",
  minorSpacing: 0.5,
  majorSpacing: 5,
  minorLineWidth: 0.018,
  majorLineWidth: 0.006,
  minorOpacity: 0.44,
  majorOpacity: 0.7,
} as const;

const GRID_VERTEX_DECLARATIONS = /* glsl */ `
uniform vec2 ardyGroundGridPhase;
varying vec2 ardyGroundGridPosition;
`;

const GRID_VERTEX_POSITION = /* glsl */ `
// Drop model translation before interpolation. The small phase uniform restores
// world-grid alignment without passing large world coordinates through fract().
ardyGroundGridPosition =
  (mat3(modelMatrix) * transformed).xz + ardyGroundGridPhase;
`;

// Adapted from Ben Golus's "The Best Darn Grid Shader (Yet)" and Brandon
// Jones's MIT-licensed WebGPU implementation:
// https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8
// https://github.com/toji/pristine-grid-webgpu
const GRID_FRAGMENT_DECLARATIONS = /* glsl */ `
uniform vec3 ardyGroundMinorColor;
uniform vec3 ardyGroundMajorColor;
uniform float ardyGroundMinorSpacing;
uniform float ardyGroundMajorSpacing;
uniform float ardyGroundMinorLineWidth;
uniform float ardyGroundMajorLineWidth;
uniform float ardyGroundMinorOpacity;
uniform float ardyGroundMajorOpacity;
varying vec2 ardyGroundGridPosition;

float ardyPristineGrid(vec2 uv, vec2 lineWidth) {
  vec4 uvDDXY = vec4(dFdx(uv), dFdy(uv));
  vec2 uvDeriv = vec2(length(uvDDXY.xz), length(uvDDXY.yw));
  vec2 invertLine = step(vec2(0.5), lineWidth);
  vec2 targetWidth = mix(lineWidth, vec2(1.0) - lineWidth, invertLine);
  vec2 drawWidth = min(
    max(targetWidth, uvDeriv),
    vec2(0.5)
  );
  vec2 lineAA = uvDeriv * 1.5;
  vec2 gridUV = abs(fract(uv) * 2.0 - 1.0);
  gridUV = mix(vec2(1.0) - gridUV, gridUV, invertLine);
  vec2 grid = vec2(1.0) - smoothstep(
    drawWidth - lineAA,
    drawWidth + lineAA,
    gridUV
  );
  grid *= clamp(targetWidth / max(drawWidth, vec2(1e-6)), 0.0, 1.0);
  grid = mix(
    grid,
    targetWidth,
    clamp(uvDeriv * 2.0 - 1.0, 0.0, 1.0)
  );
  grid = mix(grid, vec2(1.0) - grid, invertLine);
  return mix(grid.x, 1.0, grid.y);
}
`;

const GRID_FRAGMENT_COLOR = /* glsl */ `
// The non-inverted Pristine Grid branch is already centered on integer UVs.
// Keeping both coordinates unshifted makes every major line replace a minor.
float ardyMinorGrid = ardyPristineGrid(
  ardyGroundGridPosition / ardyGroundMinorSpacing,
  vec2(ardyGroundMinorLineWidth)
);
float ardyMajorGrid = ardyPristineGrid(
  ardyGroundGridPosition / ardyGroundMajorSpacing,
  vec2(ardyGroundMajorLineWidth)
);
diffuseColor.rgb = mix(
  diffuseColor.rgb,
  ardyGroundMinorColor,
  ardyMinorGrid * ardyGroundMinorOpacity
);
diffuseColor.rgb = mix(
  diffuseColor.rgb,
  ardyGroundMajorColor,
  ardyMajorGrid * ardyGroundMajorOpacity
);
`;

type StandardMaterialShader = Parameters<
  THREE.MeshStandardMaterial["onBeforeCompile"]
>[0];

interface GroundGridState {
  origin: THREE.Vector2;
  phase: { value: THREE.Vector2 };
  minorColor: { value: THREE.Color };
  majorColor: { value: THREE.Color };
  minorSpacing: { value: number };
  majorSpacing: { value: number };
  minorLineWidth: { value: number };
  majorLineWidth: { value: number };
  minorOpacity: { value: number };
  majorOpacity: { value: number };
}

const gridStates = new WeakMap<THREE.MeshStandardMaterial, GroundGridState>();

export interface CameraRelativeGroundOptions {
  /** Current orbit target in world-space metres. */
  targetX: number;
  /** Current orbit target in world-space metres. */
  targetZ: number;
  /** Perspective camera far distance in metres. */
  cameraFar: number;
  /** Furthest distance allowed between the camera and orbit target. */
  maxCameraDistance: number;
  /** Extra coverage beyond the camera frustum. */
  padding?: number;
  /** Period used to snap the moving ground origin. */
  majorSpacing?: number;
}

export interface CameraRelativeGroundLayout {
  centerX: number;
  centerZ: number;
  halfExtent: number;
  sideLength: number;
  phaseX: number;
  phaseZ: number;
}

export interface GroundGridMaterialOptions {
  baseColor?: THREE.ColorRepresentation;
  minorColor?: THREE.ColorRepresentation;
  majorColor?: THREE.ColorRepresentation;
  minorSpacing?: number;
  majorSpacing?: number;
  /** Full line width as a fraction of a minor cell. */
  minorLineWidth?: number;
  /** Full line width as a fraction of a major cell. */
  majorLineWidth?: number;
  minorOpacity?: number;
  majorOpacity?: number;
  roughness?: number;
  metalness?: number;
}

function finiteNumber(value: number, label: string): number {
  if (!Number.isFinite(value)) {
    throw new RangeError(`${label} must be finite.`);
  }
  return value;
}

function positiveNumber(value: number, label: string): number {
  finiteNumber(value, label);
  if (value <= 0) {
    throw new RangeError(`${label} must be greater than zero.`);
  }
  return value;
}

function unitInterval(value: number, label: string): number {
  finiteNumber(value, label);
  if (value < 0 || value > 1) {
    throw new RangeError(`${label} must be between zero and one.`);
  }
  return value;
}

function replaceShaderChunk(
  source: string,
  chunk: string,
  addition: string,
): string {
  if (!source.includes(chunk)) {
    throw new Error(`Unable to inject the ground grid after ${chunk}.`);
  }
  return source.replace(chunk, `${chunk}\n${addition}`);
}

/**
 * Returns a small, non-negative phase suitable for shader arithmetic.
 *
 * Repeating the origin by a whole major period returns the same phase, so a
 * camera-relative mesh can be recentered without the grid swimming.
 */
export function groundGridPhase(origin: number, majorSpacing = 5): number {
  finiteNumber(origin, "Ground origin");
  positiveNumber(majorSpacing, "Major grid spacing");
  const phase = origin % majorSpacing;
  if (phase === 0) return 0;
  return phase < 0 ? phase + majorSpacing : phase;
}

/**
 * Computes a camera-relative ground quad that covers the camera far plane.
 *
 * The center and extent are snapped to the major grid period. The mesh can
 * therefore move with OrbitControls while its procedural grid stays fixed in
 * world space.
 */
export function computeCameraRelativeGroundLayout({
  targetX,
  targetZ,
  cameraFar,
  maxCameraDistance,
  padding = 0,
  majorSpacing = 5,
}: CameraRelativeGroundOptions): CameraRelativeGroundLayout {
  finiteNumber(targetX, "Ground target X");
  finiteNumber(targetZ, "Ground target Z");
  positiveNumber(cameraFar, "Camera far distance");
  finiteNumber(maxCameraDistance, "Maximum camera distance");
  finiteNumber(padding, "Ground padding");
  positiveNumber(majorSpacing, "Major grid spacing");
  if (maxCameraDistance < 0) {
    throw new RangeError("Maximum camera distance cannot be negative.");
  }
  if (padding < 0) {
    throw new RangeError("Ground padding cannot be negative.");
  }

  const centerX = Math.round(targetX / majorSpacing) * majorSpacing;
  const centerZ = Math.round(targetZ / majorSpacing) * majorSpacing;
  const minimumHalfExtent = cameraFar + maxCameraDistance + padding;
  const halfExtent =
    Math.ceil(minimumHalfExtent / majorSpacing) * majorSpacing;

  return {
    centerX,
    centerZ,
    halfExtent,
    sideLength: halfExtent * 2,
    phaseX: groundGridPhase(centerX, majorSpacing),
    phaseZ: groundGridPhase(centerZ, majorSpacing),
  };
}

function resolveState(
  options: GroundGridMaterialOptions,
  current?: GroundGridState,
): GroundGridState {
  const minorSpacing = positiveNumber(
    options.minorSpacing ??
      current?.minorSpacing.value ??
      DEFAULTS.minorSpacing,
    "Minor grid spacing",
  );
  const majorSpacing = positiveNumber(
    options.majorSpacing ??
      current?.majorSpacing.value ??
      DEFAULTS.majorSpacing,
    "Major grid spacing",
  );
  const periodRatio = majorSpacing / minorSpacing;
  if (Math.abs(periodRatio - Math.round(periodRatio)) > 1e-6) {
    throw new RangeError(
      "Major grid spacing must be a whole-number multiple of minor spacing.",
    );
  }

  const minorLineWidth = unitInterval(
    options.minorLineWidth ??
      current?.minorLineWidth.value ??
      DEFAULTS.minorLineWidth,
    "Minor grid line width",
  );
  const majorLineWidth = unitInterval(
    options.majorLineWidth ??
      current?.majorLineWidth.value ??
      DEFAULTS.majorLineWidth,
    "Major grid line width",
  );
  const minorOpacity = unitInterval(
    options.minorOpacity ??
      current?.minorOpacity.value ??
      DEFAULTS.minorOpacity,
    "Minor grid opacity",
  );
  const majorOpacity = unitInterval(
    options.majorOpacity ??
      current?.majorOpacity.value ??
      DEFAULTS.majorOpacity,
    "Major grid opacity",
  );
  if (current) {
    if (options.minorColor !== undefined) {
      current.minorColor.value.set(options.minorColor);
    }
    if (options.majorColor !== undefined) {
      current.majorColor.value.set(options.majorColor);
    }
    current.minorSpacing.value = minorSpacing;
    current.majorSpacing.value = majorSpacing;
    current.phase.value.set(
      groundGridPhase(current.origin.x, majorSpacing),
      groundGridPhase(current.origin.y, majorSpacing),
    );
    current.minorLineWidth.value = minorLineWidth;
    current.majorLineWidth.value = majorLineWidth;
    current.minorOpacity.value = minorOpacity;
    current.majorOpacity.value = majorOpacity;
    return current;
  }
  return {
    origin: new THREE.Vector2(),
    phase: { value: new THREE.Vector2() },
    minorColor: {
      value: new THREE.Color(options.minorColor ?? DEFAULTS.minorColor),
    },
    majorColor: {
      value: new THREE.Color(options.majorColor ?? DEFAULTS.majorColor),
    },
    minorSpacing: { value: minorSpacing },
    majorSpacing: { value: majorSpacing },
    minorLineWidth: { value: minorLineWidth },
    majorLineWidth: { value: majorLineWidth },
    minorOpacity: { value: minorOpacity },
    majorOpacity: { value: majorOpacity },
  };
}

function injectGroundGrid(
  shader: StandardMaterialShader,
  state: GroundGridState,
): void {
  shader.uniforms.ardyGroundGridPhase = state.phase;
  shader.uniforms.ardyGroundMinorColor = state.minorColor;
  shader.uniforms.ardyGroundMajorColor = state.majorColor;
  shader.uniforms.ardyGroundMinorSpacing = state.minorSpacing;
  shader.uniforms.ardyGroundMajorSpacing = state.majorSpacing;
  shader.uniforms.ardyGroundMinorLineWidth = state.minorLineWidth;
  shader.uniforms.ardyGroundMajorLineWidth = state.majorLineWidth;
  shader.uniforms.ardyGroundMinorOpacity = state.minorOpacity;
  shader.uniforms.ardyGroundMajorOpacity = state.majorOpacity;

  shader.vertexShader = replaceShaderChunk(
    shader.vertexShader,
    "#include <common>",
    GRID_VERTEX_DECLARATIONS,
  );
  shader.vertexShader = replaceShaderChunk(
    shader.vertexShader,
    "#include <begin_vertex>",
    GRID_VERTEX_POSITION,
  );
  shader.fragmentShader = replaceShaderChunk(
    shader.fragmentShader,
    "#include <common>",
    GRID_FRAGMENT_DECLARATIONS,
  );
  shader.fragmentShader = replaceShaderChunk(
    shader.fragmentShader,
    "#include <color_fragment>",
    GRID_FRAGMENT_COLOR,
  );
}

/**
 * Adds the Pristine Grid shader to an existing standard material.
 *
 * The grid is mixed into `diffuseColor` before Three.js applies lighting,
 * shadows, fog, tone mapping, and dithering.
 */
export function configureGroundGridMaterial(
  material: THREE.MeshStandardMaterial,
  options: GroundGridMaterialOptions = {},
): THREE.MeshStandardMaterial {
  const current = gridStates.get(material);
  const state = resolveState(options, current);
  gridStates.set(material, state);

  if (options.baseColor !== undefined) {
    material.color.set(options.baseColor);
  }
  if (options.roughness !== undefined) {
    material.roughness = unitInterval(options.roughness, "Ground roughness");
  }
  if (options.metalness !== undefined) {
    material.metalness = unitInterval(options.metalness, "Ground metalness");
  }
  material.dithering = true;
  material.userData.pristineGrid = true;

  if (!current) {
    const previousOnBeforeCompile = material.onBeforeCompile.bind(material);
    const previousCacheKey = material.customProgramCacheKey.bind(material);
    material.onBeforeCompile = (shader, renderer): void => {
      previousOnBeforeCompile(shader, renderer);
      injectGroundGrid(shader, state);
    };
    material.customProgramCacheKey = (): string =>
      `${previousCacheKey()}|${GRID_SHADER_CACHE_KEY}`;
  }

  material.needsUpdate = true;
  return material;
}

/** Creates the one-pass, shadow-receiving ground material. */
export function createGroundGridMaterial(
  options: GroundGridMaterialOptions = {},
): THREE.MeshStandardMaterial {
  const material = new THREE.MeshStandardMaterial({
    color: options.baseColor ?? DEFAULTS.baseColor,
    roughness: options.roughness ?? 1,
    metalness: options.metalness ?? 0,
  });
  return configureGroundGridMaterial(material, options);
}

/**
 * Updates the rebased shader origin after moving the camera-relative mesh.
 */
export function updateGroundGridOrigin(
  material: THREE.MeshStandardMaterial,
  originX: number,
  originZ: number,
): void {
  const state = gridStates.get(material);
  if (!state) {
    throw new TypeError("Ground material has not been configured.");
  }
  state.origin.set(originX, originZ);
  state.phase.value.set(
    groundGridPhase(originX, state.majorSpacing.value),
    groundGridPhase(originZ, state.majorSpacing.value),
  );
}
