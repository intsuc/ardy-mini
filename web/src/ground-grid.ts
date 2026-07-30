// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import {
  Color,
  type ColorRepresentation,
  MeshStandardNodeMaterial,
  type Node,
  Vector2,
} from "three/webgpu";
import * as TSL from "three/tsl";

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

type GridNodeType =
  | "color"
  | "float"
  | "mat4"
  | "vec2"
  | "vec3"
  | "vec4";

/**
 * A deliberately small facade over the TSL API used by this shader.
 *
 * The full TSL overload graph is extremely expensive for TypeScript to infer
 * through a long procedural shader. Keeping the internal graph structurally
 * typed avoids multi-gigabyte type-checks while the public material boundary
 * remains Three.js's `Node` type.
 */
interface GridNode<T extends GridNodeType> {
  readonly nodeType: T;
  readonly rgb: GridNode<"vec3">;
  readonly x: GridNode<"float">;
  readonly xz: GridNode<"vec2">;
  readonly y: GridNode<"float">;
  readonly yw: GridNode<"vec2">;
}

interface MutableGridUniform<
  T extends GridNodeType,
  TValue,
> extends GridNode<T> {
  value: TValue;
}

interface GridTslFacade {
  readonly materialColor: GridNode<"vec4">;
  readonly modelWorldMatrix: GridNode<"mat4">;
  readonly positionLocal: GridNode<"vec3">;
  abs<T extends GridNodeType>(value: unknown): GridNode<T>;
  add<T extends GridNodeType>(left: unknown, right: unknown): GridNode<T>;
  clamp<T extends GridNodeType>(
    value: unknown,
    low: unknown,
    high: unknown,
  ): GridNode<T>;
  dFdx<T extends GridNodeType>(value: unknown): GridNode<T>;
  dFdy<T extends GridNodeType>(value: unknown): GridNode<T>;
  div<T extends GridNodeType>(left: unknown, right: unknown): GridNode<T>;
  fract<T extends GridNodeType>(value: unknown): GridNode<T>;
  length(value: unknown): GridNode<"float">;
  max<T extends GridNodeType>(left: unknown, right: unknown): GridNode<T>;
  min<T extends GridNodeType>(left: unknown, right: unknown): GridNode<T>;
  mix<T extends GridNodeType>(
    left: unknown,
    right: unknown,
    factor: unknown,
  ): GridNode<T>;
  mul<T extends GridNodeType>(left: unknown, right: unknown): GridNode<T>;
  smoothstep<T extends GridNodeType>(
    low: unknown,
    high: unknown,
    value: unknown,
  ): GridNode<T>;
  step<T extends GridNodeType>(
    edge: unknown,
    value: unknown,
  ): GridNode<T>;
  sub<T extends GridNodeType>(left: unknown, right: unknown): GridNode<T>;
  uniform<T extends GridNodeType, TValue>(
    value: TValue,
  ): MutableGridUniform<T, TValue>;
  vec2(x?: unknown, y?: unknown): GridNode<"vec2">;
  vec4(x?: unknown, y?: unknown, z?: unknown, w?: unknown): GridNode<"vec4">;
}

const gridTsl = TSL as unknown as GridTslFacade;

// Adapted from Ben Golus's "The Best Darn Grid Shader (Yet)" and Brandon
// Jones's MIT-licensed WebGPU implementation:
// https://bgolus.medium.com/the-best-darn-grid-shader-yet-727f9278b9d8
// https://github.com/toji/pristine-grid-webgpu
function pristineGrid(
  uv: GridNode<"vec2">,
  lineWidth: GridNode<"vec2">,
): GridNode<"float"> {
  const uvDx = gridTsl.dFdx<"vec2">(uv);
  const uvDy = gridTsl.dFdy<"vec2">(uv);
  const uvDDXY = gridTsl.vec4(uvDx, uvDy);
  const xDerivative = gridTsl.length(uvDDXY.xz);
  const yDerivative = gridTsl.length(uvDDXY.yw);
  const uvDeriv = gridTsl.vec2(xDerivative, yDerivative);

  const halfWidth = gridTsl.vec2(0.5);
  const invertLine = gridTsl.step<"vec2">(halfWidth, lineWidth);
  const oneMinusLineWidth = gridTsl.sub<"vec2">(
    gridTsl.vec2(1),
    lineWidth,
  );
  const targetWidth = gridTsl.mix<"vec2">(
    lineWidth,
    oneMinusLineWidth,
    invertLine,
  );
  const derivativeWidth = gridTsl.max<"vec2">(targetWidth, uvDeriv);
  const drawWidth = gridTsl.min<"vec2">(derivativeWidth, halfWidth);
  const lineAA = gridTsl.mul<"vec2">(uvDeriv, 1.5);

  const repeatedUV = gridTsl.fract<"vec2">(uv);
  const doubledUV = gridTsl.mul<"vec2">(repeatedUV, 2);
  const centeredUV = gridTsl.sub<"vec2">(doubledUV, 1);
  const absoluteUV = gridTsl.abs<"vec2">(centeredUV);
  const invertedUV = gridTsl.sub<"vec2">(
    gridTsl.vec2(1),
    absoluteUV,
  );
  const selectedUV = gridTsl.mix<"vec2">(
    invertedUV,
    absoluteUV,
    invertLine,
  );

  const lowerAA = gridTsl.sub<"vec2">(drawWidth, lineAA);
  const upperAA = gridTsl.add<"vec2">(drawWidth, lineAA);
  const antialiased = gridTsl.smoothstep<"vec2">(
    lowerAA,
    upperAA,
    selectedUV,
  );
  const initialGrid = gridTsl.sub<"vec2">(
    gridTsl.vec2(1),
    antialiased,
  );

  const safeDrawWidth = gridTsl.max<"vec2">(
    drawWidth,
    gridTsl.vec2(1e-6),
  );
  const widthRatio = gridTsl.div<"vec2">(targetWidth, safeDrawWidth);
  const widthFade = gridTsl.clamp<"vec2">(widthRatio, 0, 1);
  const widthAdjusted = gridTsl.mul<"vec2">(initialGrid, widthFade);

  const doubledDerivative = gridTsl.mul<"vec2">(uvDeriv, 2);
  const derivativeFadeInput = gridTsl.sub<"vec2">(
    doubledDerivative,
    1,
  );
  const derivativeFade = gridTsl.clamp<"vec2">(
    derivativeFadeInput,
    0,
    1,
  );
  const distantGrid = gridTsl.mix<"vec2">(
    widthAdjusted,
    targetWidth,
    derivativeFade,
  );

  const invertedGrid = gridTsl.sub<"vec2">(
    gridTsl.vec2(1),
    distantGrid,
  );
  const selectedGrid = gridTsl.mix<"vec2">(
    distantGrid,
    invertedGrid,
    invertLine,
  );

  const combinedGrid = gridTsl.mix<"float">(
    selectedGrid.x,
    1,
    selectedGrid.y,
  );
  return combinedGrid;
}

interface GroundGridState {
  origin: Vector2;
  phase: MutableGridUniform<"vec2", Vector2>;
  minorColor: MutableGridUniform<"color", Color>;
  majorColor: MutableGridUniform<"color", Color>;
  minorSpacing: MutableGridUniform<"float", number>;
  majorSpacing: MutableGridUniform<"float", number>;
  minorLineWidth: MutableGridUniform<"float", number>;
  majorLineWidth: MutableGridUniform<"float", number>;
  minorOpacity: MutableGridUniform<"float", number>;
  majorOpacity: MutableGridUniform<"float", number>;
}

const gridStates = new WeakMap<MeshStandardNodeMaterial, GroundGridState>();

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
  baseColor?: ColorRepresentation;
  minorColor?: ColorRepresentation;
  majorColor?: ColorRepresentation;
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
    origin: new Vector2(),
    phase: gridTsl.uniform<"vec2", Vector2>(new Vector2()),
    minorColor: gridTsl.uniform<"color", Color>(
      new Color(options.minorColor ?? DEFAULTS.minorColor),
    ),
    majorColor: gridTsl.uniform<"color", Color>(
      new Color(options.majorColor ?? DEFAULTS.majorColor),
    ),
    minorSpacing: gridTsl.uniform<"float", number>(minorSpacing),
    majorSpacing: gridTsl.uniform<"float", number>(majorSpacing),
    minorLineWidth: gridTsl.uniform<"float", number>(minorLineWidth),
    majorLineWidth: gridTsl.uniform<"float", number>(majorLineWidth),
    minorOpacity: gridTsl.uniform<"float", number>(minorOpacity),
    majorOpacity: gridTsl.uniform<"float", number>(majorOpacity),
  };
}

function createGroundGridColorNode(
  state: GroundGridState,
): Node<"vec3"> {
  // Use a direction (w = 0) so model translation is discarded. The small
  // periodic phase restores world-grid alignment without feeding large world
  // coordinates into fract().
  const localPosition = gridTsl.vec4(gridTsl.positionLocal, 0);
  const untranslatedWorldPosition = gridTsl.mul<"vec4">(
    gridTsl.modelWorldMatrix,
    localPosition,
  );
  const gridPosition = gridTsl.add<"vec2">(
    untranslatedWorldPosition.xz,
    state.phase,
  );
  const minorUV = gridTsl.div<"vec2">(
    gridPosition,
    state.minorSpacing,
  );
  const minorWidth = gridTsl.vec2(state.minorLineWidth);
  const minorGrid = pristineGrid(
    minorUV,
    minorWidth,
  );
  const majorUV = gridTsl.div<"vec2">(
    gridPosition,
    state.majorSpacing,
  );
  const majorWidth = gridTsl.vec2(state.majorLineWidth);
  const majorGrid = pristineGrid(
    majorUV,
    majorWidth,
  );

  const minorBlend = gridTsl.mul<"float">(
    minorGrid,
    state.minorOpacity,
  );
  const minorColor = gridTsl.mix<"vec3">(
    gridTsl.materialColor.rgb,
    state.minorColor,
    minorBlend,
  );
  const majorBlend = gridTsl.mul<"float">(
    majorGrid,
    state.majorOpacity,
  );
  const groundColor = gridTsl.mix<"vec3">(
    minorColor,
    state.majorColor,
    majorBlend,
  );
  return groundColor as unknown as Node<"vec3">;
}

/**
 * Adds the Pristine Grid TSL graph to an existing standard node material.
 *
 * The grid is mixed into the material color before Three.js applies lighting,
 * shadows, fog, and tone mapping. Display-space dithering is handled by the
 * viewer's final-output pipeline.
 */
export function configureGroundGridMaterial(
  material: MeshStandardNodeMaterial,
  options: GroundGridMaterialOptions = {},
): MeshStandardNodeMaterial {
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
  material.userData.pristineGrid = true;

  if (!current) {
    const colorNode: Node<"vec3"> = createGroundGridColorNode(state);
    material.colorNode = colorNode;
  }

  material.needsUpdate = true;
  return material;
}

/** Creates the one-pass, shadow-receiving ground material. */
export function createGroundGridMaterial(
  options: GroundGridMaterialOptions = {},
): MeshStandardNodeMaterial {
  const material = new MeshStandardNodeMaterial({
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
  material: MeshStandardNodeMaterial,
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
