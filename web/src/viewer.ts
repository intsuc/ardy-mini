// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import * as THREE from "three/webgpu";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { VRM } from "@pixiv/three-vrm";

import {
  DEFAULT_EDITOR_STATE,
  DEFAULT_OUTPUT_VISIBILITY,
  normalizeConstraintMarker,
  normalizeEditorState,
  normalizeInitialTransform,
  normalizeWaypoint,
  type InitialTransform,
  type MotionConstraintMarker,
  type MotionEditorState,
  type MotionWaypoint,
  type Vector3Tuple,
  type ViewerOutputVisibility,
} from "./editor-state";
import {
  CORE27_FOOT_CONTACT_JOINTS,
  CORE27_JOINT_COUNT,
  CORE27_PARENTS,
  CORE27_SKELETON,
  DEFAULT_MOTION_FPS,
  normalizeGroundedNeutralPose,
  normalizeSkeletonMetadata,
  normalizeStructuredMotion,
  toFiniteFloat32Array,
  type RotationTrack,
  type SkeletonMetadata,
  type StructuredMotionResult,
} from "./motion-data";
import {
  CORE27_VRM_BINDINGS,
  createVrmRetargetPlan,
  retargetMotionFrame,
  toVrmPosition,
  type VrmRetargetFrame,
  type VrmRetargetPlan,
} from "./vrm-retarget";
import {
  computeCameraRelativeGroundLayout,
  createGroundGridMaterial,
  updateGroundGridOrigin,
} from "./ground-grid";
import { DitheredRenderPipeline } from "./dithered-render-pipeline";
import {
  loadVrmAvatar,
  type LoadedVrmAvatar,
  type VrmModelInfo,
} from "./vrm-loader";

export {
  CORE27_FOOT_CONTACT_JOINTS,
  CORE27_JOINT_COUNT,
  CORE27_PARENTS,
  CORE27_SKELETON,
  normalizeSkeletonMetadata,
  normalizeStructuredMotion,
  isJointInContact,
} from "./motion-data";
export type {
  RotationFormat,
  RotationTrack,
  SkeletonMetadata,
  StructuredMotionResult,
} from "./motion-data";
export {
  DEFAULT_EDITOR_STATE,
  DEFAULT_INITIAL_TRANSFORM,
  DEFAULT_OUTPUT_VISIBILITY,
  normalizeConstraintMarker,
  normalizeEditorState,
  normalizeInitialTransform,
  normalizeOutputVisibility,
  normalizeWaypoint,
} from "./editor-state";
export type {
  InitialTransform,
  MotionConstraintKind,
  MotionConstraintMarker,
  MotionEditorState,
  MotionWaypoint,
  QuaternionTuple,
  Vector3Tuple,
  ViewerOutputVisibility,
} from "./editor-state";

export type MotionClip = StructuredMotionResult;
export type { VrmModelInfo } from "./vrm-loader";

/**
 * sRGB counterparts of the dark neutral tokens from shadcn preset buFzUhO.
 * Three.js does not parse CSS custom properties or OKLCH color strings.
 */
const VIEWER_COLORS = {
  background: "#0a0a0a",
  foreground: "#fafafa",
  primary: "#e5e5e5",
  mutedForeground: "#a1a1a1",
  chart1: "#d4d4d4",
  chart4: "#404040",
  chart5: "#262626",
  destructive: "#ff6467",
} as const;

const GROUND_MINOR_SPACING = 0.5;
const GROUND_MAJOR_SPACING = 5;
const GROUND_Y = -0.006;
const CAMERA_MOVE_STEPS_PER_SECOND = 12;
const CAMERA_MOVE_MAX_ELAPSED_SECONDS = 0.05;
const CAMERA_RESET_MIN_DURATION_MS = 360;
const CAMERA_RESET_MAX_DURATION_MS = 700;
const CAMERA_RESET_MS_PER_WORLD_UNIT = 48;

export interface PlaybackState {
  frame: number;
  frameCount: number;
  fps: number;
  playing: boolean;
  speed: number;
}

export type PlaybackListener = (state: PlaybackState) => void;
export type EditorStateListener = (state: MotionEditorState) => void;
export type GroundClickListener = (point: Vector3Tuple) => void;
export type ViewerOutputKey = keyof ViewerOutputVisibility;

export interface SkeletonInstanceCounts {
  joints: number;
  bones: number;
}

export interface ResetCameraOptions {
  /** Smoothly restore the composed view. Reduced-motion mode always jumps. */
  animated?: boolean;
}

interface SetMotionBaseOptions {
  /** Whether playback should continue after installing the clip. */
  playing?: boolean;
  /** Reset the orbit camera for a genuinely new presentation. */
  resetCamera?: boolean;
}

export type SetMotionOptions =
  | (SetMotionBaseOptions & {
      /** Start a new presentation at this frame. */
      frame?: number;
      preserveContinuity?: false;
    })
  | (SetMotionBaseOptions & {
      /**
       * Keep the exact internal playhead, playback clock, and VRM spring
       * simulation while replacing a clip with its streamed continuation.
       */
      preserveContinuity: true;
      frame?: never;
    });

export interface MotionContinuityContext {
  requested: boolean;
  hasPreviousClip: boolean;
  skeletonChanged: boolean;
  previousFrame: number;
  nextFrame: number;
}

/**
 * Secondary motion is safe to preserve only when the clip update does not
 * change either the rig or the playhead. An actual seek must reinitialize the
 * simulation so stale spring positions are not carried to another pose.
 */
export function canPreserveMotionContinuity({
  requested,
  hasPreviousClip,
  skeletonChanged,
  previousFrame,
  nextFrame,
}: MotionContinuityContext): boolean {
  return (
    requested &&
    hasPreviousClip &&
    !skeletonChanged &&
    Math.abs(previousFrame - nextFrame) <= 1e-6
  );
}

/**
 * Converts held-key time into a frame-rate-independent camera distance.
 *
 * The cap prevents a delayed animation frame from producing a large jump
 * after the tab or main thread resumes.
 */
export function cameraMovementDistance(
  orbitDistance: number,
  elapsedSeconds: number,
): number {
  if (!Number.isFinite(orbitDistance) || orbitDistance <= 0) {
    throw new RangeError("Camera orbit distance must be positive.");
  }
  if (!Number.isFinite(elapsedSeconds) || elapsedSeconds < 0) {
    throw new RangeError("Camera movement time cannot be negative.");
  }
  const step = THREE.MathUtils.clamp(orbitDistance * 0.05, 0.08, 0.4);
  return (
    step *
    CAMERA_MOVE_STEPS_PER_SECOND *
    Math.min(elapsedSeconds, CAMERA_MOVE_MAX_ELAPSED_SECONDS)
  );
}

function cameraResetDuration(distance: number): number {
  return THREE.MathUtils.clamp(
    CAMERA_RESET_MIN_DURATION_MS +
      distance * CAMERA_RESET_MS_PER_WORLD_UNIT,
    CAMERA_RESET_MIN_DURATION_MS,
    CAMERA_RESET_MAX_DURATION_MS,
  );
}

function cameraResetProgress(progress: number): number {
  const normalized = THREE.MathUtils.clamp(progress, 0, 1);
  return normalized < 0.5
    ? 4 * normalized * normalized * normalized
    : 1 - Math.pow(-2 * normalized + 2, 3) / 2;
}

function shortestAngleDelta(from: number, to: number): number {
  return THREE.MathUtils.euclideanModulo(
    to - from + Math.PI,
    Math.PI * 2,
  ) - Math.PI;
}

export function isWebGpuRendererBackend(renderer: {
  readonly isWebGPURenderer?: unknown;
  readonly backend?: unknown;
}): boolean {
  return (
    renderer.isWebGPURenderer === true &&
    (renderer.backend as { readonly isWebGPUBackend?: unknown } | undefined)
      ?.isWebGPUBackend === true
  );
}

function requireNativeWebGpuBackend(renderer: THREE.WebGPURenderer): void {
  // WebGPURenderer installs a WebGL2 fallback internally and currently exposes
  // no public force-WebGPU option. This application already requires WebGPU,
  // so reject initialization instead of silently changing rendering backends.
  (
    renderer as unknown as {
      _getFallback: ((error: unknown) => unknown) | null;
    }
  )._getFallback = null;
}

/** Counts used by the instanced skeleton layers for a dynamic skeleton. */
export function skeletonInstanceCounts(
  skeleton: SkeletonMetadata,
): SkeletonInstanceCounts {
  return {
    joints: skeleton.jointNames.length,
    bones: skeleton.parents.reduce(
      (count, parent) => count + (parent === -1 ? 0 : 1),
      0,
    ),
  };
}

/**
 * Backwards-compatible entry point used by main.ts. It now also accepts
 * dynamic skeleton metadata and the richer optional result tracks.
 */
export function normalizeMotionClip(input: unknown): MotionClip {
  return normalizeStructuredMotion(input);
}

export function frameAfterElapsed(
  startFrame: number,
  elapsedSeconds: number,
  fps: number,
  speed: number,
  frameCount: number,
  loop: boolean,
): { frame: number; ended: boolean } {
  if (frameCount <= 1) return { frame: 0, ended: true };
  const candidate = Math.max(0, startFrame + elapsedSeconds * fps * speed);
  if (loop) {
    return { frame: candidate % frameCount, ended: false };
  }
  if (candidate >= frameCount - 1) {
    return { frame: frameCount - 1, ended: true };
  }
  return { frame: candidate, ended: false };
}

function sameSkeleton(left: SkeletonMetadata, right: SkeletonMetadata): boolean {
  if (
    left.rootJointIndex !== right.rootJointIndex ||
    left.jointNames.length !== right.jointNames.length ||
    left.parents.length !== right.parents.length ||
    left.contactJointIndices.length !== right.contactJointIndices.length
  ) {
    return false;
  }
  return (
    left.jointNames.every(
      (jointName, index) => jointName === right.jointNames[index],
    ) &&
    left.parents.every((parent, index) => parent === right.parents[index]) &&
    left.contactJointIndices.every((joint, index) => joint === right.contactJointIndices[index])
  );
}

function disposeObject(root: THREE.Object3D): void {
  root.traverse((object) => {
    const candidate = object as THREE.Mesh | THREE.Line;
    if ("geometry" in candidate && candidate.geometry) candidate.geometry.dispose();
    if ("material" in candidate && candidate.material) {
      const materials = Array.isArray(candidate.material) ? candidate.material : [candidate.material];
      materials.forEach((material) => material.dispose());
    }
  });
}

function clearGroup(group: THREE.Group): void {
  for (const child of [...group.children]) {
    group.remove(child);
    disposeObject(child);
  }
}

function normalizeTrajectoryPoints(input: unknown): Float32Array {
  const values = toFiniteFloat32Array(input, "Trajectory points");
  if (values.length === 0 || values.length % 3 !== 0) {
    throw new RangeError("Trajectory points must contain N × 3 values.");
  }
  return values;
}

interface CameraResetTransition {
  readonly startedAt: number;
  readonly duration: number;
  readonly startTarget: THREE.Vector3;
  readonly endTarget: THREE.Vector3;
  readonly startOffset: THREE.Spherical;
  readonly endOffset: THREE.Spherical;
  readonly azimuthDelta: number;
  readonly finalNear: number;
  readonly finalFar: number;
}

export class SkeletonViewer {
  private readonly canvas: HTMLCanvasElement;
  private readonly scene: THREE.Scene;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly renderer: THREE.WebGPURenderer;
  private readonly renderPipeline: DitheredRenderPipeline;
  private readonly controls: OrbitControls;
  private joints!: THREE.InstancedMesh;
  private bones!: THREE.InstancedMesh;
  private readonly trajectory: THREE.Line;
  private readonly orientationAxesGroup = new THREE.Group();
  private readonly constraintGroup = new THREE.Group();
  private readonly initialTransformGroup = new THREE.Group();
  private readonly waypointGroup = new THREE.Group();
  private readonly vrmRoot = new THREE.Group();
  private readonly shadowRig = new THREE.Group();
  private readonly shadowLight = new THREE.DirectionalLight(
    VIEWER_COLORS.primary,
    4.2,
  );
  private readonly shadowLightTarget = new THREE.Object3D();
  private readonly groundMaterial = createGroundGridMaterial({
    baseColor: VIEWER_COLORS.chart5,
    minorColor: VIEWER_COLORS.chart4,
    majorColor: VIEWER_COLORS.chart4,
    minorSpacing: GROUND_MINOR_SPACING,
    majorSpacing: GROUND_MAJOR_SPACING,
    roughness: 1,
    metalness: 0,
  });
  private readonly groundSurface = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 1),
    this.groundMaterial,
  );
  private readonly resizeObserver: ResizeObserver;
  private readonly jointTransform = new THREE.Object3D();
  private readonly boneTransform = new THREE.Object3D();
  private readonly currentPoint = new THREE.Vector3();
  private readonly parentPoint = new THREE.Vector3();
  private readonly direction = new THREE.Vector3();
  private readonly midpoint = new THREE.Vector3();
  private readonly upAxis = new THREE.Vector3(0, 1, 0);
  private readonly regularJointColor = new THREE.Color(VIEWER_COLORS.primary);
  private readonly contactJointColor = new THREE.Color(VIEWER_COLORS.destructive);
  private readonly rotationA = new THREE.Quaternion();
  private readonly rotationB = new THREE.Quaternion();
  private readonly rotationMatrix = new THREE.Matrix4();
  private readonly groundRaycaster = new THREE.Raycaster();
  private readonly groundPointer = new THREE.Vector2();
  private readonly groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  private readonly groundIntersection = new THREE.Vector3();
  private readonly cameraForward = new THREE.Vector3();
  private readonly cameraRight = new THREE.Vector3();
  private readonly cameraOffset = new THREE.Vector3();
  private readonly cameraResetTarget = new THREE.Vector3();
  private readonly cameraResetOffset = new THREE.Vector3();

  private skeleton: SkeletonMetadata = CORE27_SKELETON;
  private contactChannelByJoint = new Int32Array();
  private localWorldRotations: THREE.Quaternion[] = [];
  private localRotationResolved = new Uint8Array();
  private neutralPose: MotionClip | null = null;
  private clip: MotionClip | null = null;
  private vrm: VRM | null = null;
  private vrmUtils: LoadedVrmAvatar["utils"] | null = null;
  private vrmRetargetPlan: VrmRetargetPlan | null = null;
  private vrmTargetHipsHeight = 1;
  private vrmVisible = true;
  private vrmLoadRevision = 0;
  private customTrajectory: Float32Array | null = null;
  private orientationJointIndices: number[] = [];
  private orientationAxes: THREE.AxesHelper[] = [];
  private orientationAxisSize = 0.13;
  private outputVisibility: ViewerOutputVisibility = { ...DEFAULT_OUTPUT_VISIBILITY };
  private initialTransform: InitialTransform = DEFAULT_EDITOR_STATE.initialTransform;
  private waypoints: MotionWaypoint[] = [];
  private constraints: MotionConstraintMarker[] = [];
  private frameCursor = 0;
  private playing = false;
  private loop = true;
  private speed = 1;
  private reducedMotion = false;
  private cameraFollowX = 0;
  private cameraFollowZ = 0;
  private hasCameraFollowAnchor = false;
  private lastAnimationTime: number | null = null;
  private lastReportedFrame = -1;
  private animationFrame = 0;
  private resizeFrame = 0;
  private groundSideLength = 0;
  private cameraMovementForward = 0;
  private cameraMovementRight = 0;
  private lastCameraMovementTime: number | null = null;
  private cameraResetTransition: CameraResetTransition | null = null;
  private rendererInitialized = false;
  private rendererReady = false;
  private disposed = false;
  private needsRender = true;
  private pageVisible = !document.hidden;
  private playbackListener: PlaybackListener | null = null;
  private editorListener: EditorStateListener | null = null;
  private groundClickListener: GroundClickListener | null = null;
  private groundPickPointer:
    | {
        pointerId: number;
        startX: number;
        startY: number;
        maxDistanceSquared: number;
      }
    | null = null;

  private readonly invalidate = (): void => {
    this.needsRender = true;
    this.scheduleFrame();
  };

  private readonly handleCameraInteractionStart = (): void => {
    this.cancelCameraReset();
  };

  private readonly handleVisibilityChange = (): void => {
    this.pageVisible = !document.hidden;
    this.lastAnimationTime = null;
    if (this.pageVisible) {
      this.vrm?.springBoneManager?.reset();
      this.invalidate();
    } else {
      this.cameraMovementForward = 0;
      this.cameraMovementRight = 0;
      this.lastCameraMovementTime = null;
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = 0;
    }
  };

  private readonly handleGroundPointerDown = (event: PointerEvent): void => {
    if (!this.groundClickListener) return;
    if (!event.isPrimary || event.button !== 0) {
      // A second pointer indicates a pinch/orbit gesture rather than placement.
      this.groundPickPointer = null;
      return;
    }
    this.groundPickPointer = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      maxDistanceSquared: 0,
    };
  };

  private readonly handleGroundPointerMove = (event: PointerEvent): void => {
    const pointer = this.groundPickPointer;
    if (!pointer || pointer.pointerId !== event.pointerId) return;
    const deltaX = event.clientX - pointer.startX;
    const deltaY = event.clientY - pointer.startY;
    pointer.maxDistanceSquared = Math.max(
      pointer.maxDistanceSquared,
      deltaX * deltaX + deltaY * deltaY,
    );
  };

  private readonly handleGroundPointerUp = (event: PointerEvent): void => {
    const pointer = this.groundPickPointer;
    if (!pointer || pointer.pointerId !== event.pointerId) return;
    this.handleGroundPointerMove(event);
    this.groundPickPointer = null;
    const listener = this.groundClickListener;
    if (!listener || pointer.maxDistanceSquared > 6 * 6) return;

    const bounds = this.canvas.getBoundingClientRect();
    if (bounds.width <= 0 || bounds.height <= 0) return;
    this.groundPointer.set(
      ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
      -((event.clientY - bounds.top) / bounds.height) * 2 + 1,
    );
    this.groundRaycaster.setFromCamera(this.groundPointer, this.camera);
    const hit = this.groundRaycaster.ray.intersectPlane(
      this.groundPlane,
      this.groundIntersection,
    );
    if (!hit) return;
    listener([hit.x, 0, hit.z]);
  };

  private readonly handleGroundPointerCancel = (event: PointerEvent): void => {
    if (this.groundPickPointer?.pointerId === event.pointerId) {
      this.groundPickPointer = null;
    }
  };

  static async create(canvas: HTMLCanvasElement): Promise<SkeletonViewer> {
    const viewer = new SkeletonViewer(canvas);
    try {
      await viewer.renderer.init();
      viewer.rendererInitialized = true;
      if (!isWebGpuRendererBackend(viewer.renderer)) {
        throw new Error(
          "Three.js initialized a WebGL fallback instead of the required WebGPU backend.",
        );
      }
      // Build the complete scene/output pipeline inside the initialization
      // boundary so setup failures reach the unavailable-preview UI instead
      // of surfacing from a later animation frame.
      viewer.renderPipeline.render();
      viewer.rendererReady = true;
      viewer.invalidate();
      return viewer;
    } catch (error) {
      viewer.dispose();
      throw error;
    }
  }

  private constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.canvas.style.touchAction = "none";
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(VIEWER_COLORS.background);
    this.scene.fog = new THREE.FogExp2(VIEWER_COLORS.background, 0.035);

    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
    this.camera.position.set(3.1, 2.15, 3.4);

    this.renderer = new THREE.WebGPURenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    requireNativeWebGpuBackend(this.renderer);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderPipeline = new DitheredRenderPipeline(
      this.renderer,
      this.scene,
      this.camera,
    );

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 0.85, 0);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 12;
    this.controls.update();
    this.controls.addEventListener("change", this.invalidate);
    this.controls.addEventListener("start", this.handleCameraInteractionStart);
    this.canvas.addEventListener("pointerdown", this.handleGroundPointerDown);
    this.canvas.addEventListener("pointermove", this.handleGroundPointerMove);
    this.canvas.addEventListener("pointerup", this.handleGroundPointerUp);
    this.canvas.addEventListener("pointercancel", this.handleGroundPointerCancel);

    const hemisphere = new THREE.HemisphereLight(
      VIEWER_COLORS.foreground,
      VIEWER_COLORS.chart5,
      2.1,
    );
    this.scene.add(hemisphere);
    this.shadowRig.name = "shadow-follow-rig";
    this.shadowLight.name = "shadow-key-light";
    this.shadowLightTarget.name = "shadow-key-light-target";
    this.shadowLight.position.set(-3, 6, 4);
    this.shadowLightTarget.position.set(0, 0.9, 0);
    this.shadowLight.target = this.shadowLightTarget;
    this.shadowLight.castShadow = true;
    this.shadowLight.shadow.mapSize.set(1024, 1024);
    this.shadowLight.shadow.bias = -0.0002;
    this.shadowLight.shadow.normalBias = 0.015;
    this.shadowLight.shadow.radius = 2;
    const shadowCamera = this.shadowLight.shadow.camera;
    shadowCamera.left = -4;
    shadowCamera.right = 4;
    shadowCamera.top = 4;
    shadowCamera.bottom = -4;
    shadowCamera.near = 0.1;
    shadowCamera.far = 16;
    shadowCamera.updateProjectionMatrix();
    this.shadowRig.add(this.shadowLight, this.shadowLightTarget);
    this.scene.add(this.shadowRig);
    const rimLight = new THREE.DirectionalLight(VIEWER_COLORS.mutedForeground, 2.2);
    rimLight.position.set(5, 2, -4);
    this.scene.add(rimLight);

    this.groundSurface.name = "camera-relative-ground-grid";
    this.groundSurface.rotation.x = -Math.PI / 2;
    this.groundSurface.position.y = GROUND_Y;
    this.groundSurface.receiveShadow = true;
    this.groundSurface.frustumCulled = false;
    this.scene.add(this.groundSurface);
    this.updateGroundSurfaceLayout();

    this.createSkeletonMeshes(this.skeleton);
    this.trajectory = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({
        color: VIEWER_COLORS.chart1,
        transparent: true,
        opacity: 0.5,
      }),
    );
    this.scene.add(
      this.trajectory,
      this.orientationAxesGroup,
      this.constraintGroup,
      this.initialTransformGroup,
      this.waypointGroup,
      this.vrmRoot,
    );
    this.joints.visible = false;
    this.bones.visible = false;
    this.trajectory.visible = false;
    this.orientationAxesGroup.visible = false;
    this.constraintGroup.visible = false;
    this.buildInitialTransformMarker();
    this.setOrientationAxes("all", this.orientationAxisSize);
    this.applyOutputVisibility();

    this.resizeObserver = new ResizeObserver(() => {
      if (this.resizeFrame) return;
      this.resizeFrame = requestAnimationFrame(() => {
        this.resizeFrame = 0;
        this.resize();
      });
    });
    this.resizeObserver.observe(canvas.parentElement ?? canvas);
    document.addEventListener("visibilitychange", this.handleVisibilityChange);
    this.resize();
    this.scheduleFrame();
  }

  set onPlaybackChange(listener: PlaybackListener | null) {
    this.playbackListener = listener;
    this.emitPlaybackState(true);
  }

  set onEditorStateChange(listener: EditorStateListener | null) {
    this.editorListener = listener;
    if (listener) listener(this.getEditorState());
  }

  /**
   * Receive click/tap positions on the y=0 ground plane. Pointer travel above
   * six CSS pixels is treated as an orbit gesture and does not emit a point.
   */
  set onGroundClick(listener: GroundClickListener | null) {
    this.groundClickListener = listener;
    if (!listener) this.groundPickPointer = null;
  }

  setSkeleton(skeleton: unknown): void {
    const normalized = normalizeSkeletonMetadata(skeleton);
    if (sameSkeleton(this.skeleton, normalized)) {
      this.skeleton = normalized;
      return;
    }
    this.neutralPose = null;
    if (this.clip) this.clearMotion();
    this.skeleton = normalized;
    this.rebuildVrmRetargetPlan();
    this.createSkeletonMeshes(normalized);
    this.setOrientationAxes("all", this.orientationAxisSize);
    this.constraints = this.constraints.filter(
      (marker) =>
        marker.jointIndex === undefined ||
        marker.jointIndex < normalized.jointNames.length,
    );
    this.buildConstraintMarkers();
    this.applyOutputVisibility();
    this.invalidate();
    this.emitEditorState();
  }

  getSkeleton(): SkeletonMetadata {
    return this.skeleton;
  }

  /**
   * Install a model-provided rest pose without creating playable motion.
   * The pose remains visible while the playback state stays at zero frames.
   */
  setNeutralPose(skeleton: unknown, neutralJoints: unknown): void {
    this.neutralPose = normalizeGroundedNeutralPose(
      skeleton,
      neutralJoints,
    );
    if (this.clip) return;
    this.showNeutralPose();
    this.invalidate();
  }

  /**
   * Load a local VRM 0.x/1.0 avatar. Heavy loader code is fetched only after
   * this method is called, so the default skeleton preview stays lightweight.
   */
  async loadVrm(file: File): Promise<VrmModelInfo> {
    const revision = ++this.vrmLoadRevision;
    const loaded = await loadVrmAvatar(file);
    if (revision !== this.vrmLoadRevision) {
      loaded.utils.deepDispose(loaded.vrm.scene);
      throw new DOMException("VRM loading was superseded.", "AbortError");
    }

    let hipsHeight: number;
    let retargetPlan: VrmRetargetPlan;
    try {
      const hips = loaded.vrm.humanoid.getNormalizedBoneNode("hips");
      if (!hips) {
        throw new TypeError("The VRM does not expose its required hips bone.");
      }
      hipsHeight =
        loaded.vrm.humanoid.normalizedRestPose.hips?.position?.[1] ??
        hips.position.y;
      if (!Number.isFinite(hipsHeight) || hipsHeight <= 0) {
        throw new RangeError("The VRM has an invalid humanoid scale.");
      }

      retargetPlan = this.buildVrmRetargetPlan(loaded.vrm, hipsHeight);
      const initialFrame = this.clip
        ? retargetMotionFrame(this.clip, this.frameCursor, retargetPlan)
        : null;
      loaded.vrm.humanoid.resetNormalizedPose();
      if (initialFrame) {
        this.applyVrmRetargetFrame(loaded.vrm, initialFrame);
      }
      loaded.vrm.springBoneManager?.reset();
      loaded.vrm.update(0);
      await this.renderer.compileAsync(
        loaded.vrm.scene,
        this.camera,
        this.scene,
      );
      if (revision !== this.vrmLoadRevision) {
        throw new DOMException("VRM loading was superseded.", "AbortError");
      }
    } catch (error) {
      loaded.utils.deepDispose(loaded.vrm.scene);
      throw error;
    }

    // All fallible validation happens above. Keep the current avatar alive until
    // the replacement is ready, then swap the staged model synchronously.
    this.disposeVrmAvatar();
    this.vrm = loaded.vrm;
    this.vrmUtils = loaded.utils;
    this.vrmRetargetPlan = retargetPlan;
    this.vrmTargetHipsHeight = hipsHeight;
    this.vrmRoot.position.set(0, 0, 0);
    this.vrmRoot.add(loaded.vrm.scene);
    this.vrmRoot.visible = this.vrmVisible;
    this.updateShadowAnchorForFrame(this.frameCursor);
    this.invalidate();
    return loaded.info;
  }

  clearVrm(): void {
    this.vrmLoadRevision += 1;
    this.disposeVrmAvatar();
    this.updateShadowAnchorForFrame(this.frameCursor);
    this.invalidate();
  }

  setVrmVisible(visible: boolean): void {
    this.vrmVisible = visible;
    this.vrmRoot.visible = visible && this.vrm !== null;
    if (this.vrmRoot.visible) this.vrm?.springBoneManager?.reset();
    this.updateShadowAnchorForFrame(this.frameCursor);
    this.invalidate();
  }

  setMotion(
    clip: MotionClip,
    options: SetMotionOptions = {},
  ): void {
    const hadPreviousClip = this.clip !== null;
    const previousFrame = this.frameCursor;
    const previousPlaying = this.playing;
    const skeletonChanged = !sameSkeleton(this.skeleton, clip.skeleton);
    const requestedContinuity = options.preserveContinuity === true;
    const nextFrame = THREE.MathUtils.clamp(
      requestedContinuity && hadPreviousClip
        ? previousFrame
        : (options.frame ?? 0),
      0,
      clip.frameCount - 1,
    );
    const preserveContinuity = canPreserveMotionContinuity({
      requested: requestedContinuity,
      hasPreviousClip: hadPreviousClip,
      skeletonChanged,
      previousFrame,
      nextFrame,
    });
    this.clip = clip;
    if (skeletonChanged) {
      this.skeleton = clip.skeleton;
      this.createSkeletonMeshes(this.skeleton);
      this.setOrientationAxes("all", this.orientationAxisSize);
    } else {
      this.skeleton = clip.skeleton;
    }
    this.rebuildVrmRetargetPlan();
    this.frameCursor = nextFrame;
    this.playing =
      (options.playing ?? true) &&
      !this.reducedMotion &&
      clip.frameCount > 1;
    if (
      !preserveContinuity ||
      !previousPlaying ||
      !this.playing
    ) {
      this.lastAnimationTime = null;
    }
    this.lastReportedFrame = -1;
    this.buildTrajectory();
    if (!preserveContinuity) {
      this.hasCameraFollowAnchor = false;
      this.vrm?.humanoid.resetNormalizedPose();
    }
    this.updatePose(nextFrame);
    if (!preserveContinuity) {
      this.vrm?.springBoneManager?.reset();
    }
    this.applyOutputVisibility();
    if (options.resetCamera ?? true) this.resetCamera();
    this.invalidate();
    this.emitPlaybackState(true);
  }

  clearMotion(): void {
    this.cancelCameraReset();
    this.clip = null;
    this.playing = false;
    this.frameCursor = 0;
    this.lastAnimationTime = null;
    this.trajectory.visible = false;
    this.orientationAxesGroup.visible = false;
    this.constraintGroup.visible = false;
    this.resetVrmPose();
    this.vrmRoot.position.set(0, 0, 0);
    this.hasCameraFollowAnchor = false;
    this.updateShadowAnchor(0, 0);
    this.showNeutralPose();
    this.invalidate();
    this.emitPlaybackState(true);
  }

  setPlaying(playing: boolean): void {
    if (!this.clip || this.clip.frameCount < 2) return;
    const nextPlaying = playing && !this.reducedMotion;
    if (
      nextPlaying &&
      this.frameCursor >= this.clip.frameCount - 1 &&
      !this.loop
    ) {
      this.frameCursor = 0;
    }
    this.playing = nextPlaying;
    this.lastAnimationTime = null;
    this.invalidate();
    this.emitPlaybackState(true);
  }

  togglePlaying(): void {
    this.setPlaying(!this.playing);
  }

  setLoop(loop: boolean): void {
    this.loop = loop;
    this.emitPlaybackState(true);
  }

  setSpeed(speed: number): void {
    if (!Number.isFinite(speed) || speed <= 0) throw new RangeError("Playback speed must be positive.");
    this.speed = speed;
    this.emitPlaybackState(true);
  }

  setReducedMotion(reduced: boolean): void {
    if (reduced) this.stopCameraInertia();
    this.reducedMotion = reduced;
    this.controls.enableDamping = !reduced;
    if (reduced) this.finishCameraReset();
    if (reduced && this.playing) {
      this.playing = false;
      this.emitPlaybackState(true);
    }
    this.lastAnimationTime = null;
    this.controls.update();
    this.invalidate();
  }

  orbit(horizontalSteps: number, verticalSteps: number): void {
    this.cancelCameraReset();
    const increment = Math.PI / 24;
    if (horizontalSteps) this.controls.rotateLeft(horizontalSteps * increment);
    if (verticalSteps) this.controls.rotateUp(verticalSteps * increment);
    this.invalidate();
  }

  zoom(direction: "in" | "out"): void {
    this.cancelCameraReset();
    const scale = 0.85;
    if (direction === "in") {
      this.controls.dollyIn(scale);
    } else {
      this.controls.dollyOut(scale);
    }
    this.invalidate();
  }

  moveCamera(forwardSteps: number, rightSteps: number): void {
    if (!Number.isFinite(forwardSteps) || !Number.isFinite(rightSteps)) {
      throw new RangeError("Camera movement steps must be finite.");
    }
    if (forwardSteps === 0 && rightSteps === 0) return;
    this.cancelCameraReset();
    this.controls.update();
    const step = THREE.MathUtils.clamp(
      this.controls.getDistance() * 0.05,
      0.08,
      0.4,
    );
    if (!this.translateCamera(forwardSteps, rightSteps, step)) return;
    this.invalidate();
  }

  /**
   * Starts or updates continuous view-relative keyboard movement.
   *
   * OrbitControls keeps azimuth independently from polar angle, so movement
   * remains defined at the top-down pole and follows horizontal orbiting there.
   */
  setCameraMovement(forward: number, right: number): void {
    if (!Number.isFinite(forward) || !Number.isFinite(right)) {
      throw new RangeError("Camera movement axes must be finite.");
    }
    if (
      forward === this.cameraMovementForward &&
      right === this.cameraMovementRight
    ) {
      return;
    }
    if (forward !== 0 || right !== 0) this.cancelCameraReset();
    const wasMoving =
      this.cameraMovementForward !== 0 || this.cameraMovementRight !== 0;
    this.cameraMovementForward = forward;
    this.cameraMovementRight = right;
    const isMoving = forward !== 0 || right !== 0;
    if (!isMoving) {
      this.lastCameraMovementTime = null;
    } else if (!wasMoving) {
      this.lastCameraMovementTime = performance.now();
    }
    this.invalidate();
  }

  private translateCamera(
    forward: number,
    right: number,
    distance: number,
  ): boolean {
    if (distance <= 0) return false;
    const azimuth = this.controls.getAzimuthalAngle();
    this.cameraForward.set(
      -Math.sin(azimuth),
      0,
      -Math.cos(azimuth),
    );
    this.cameraRight.set(
      Math.cos(azimuth),
      0,
      -Math.sin(azimuth),
    );
    this.cameraOffset
      .copy(this.cameraForward)
      .multiplyScalar(forward)
      .addScaledVector(this.cameraRight, right);
    if (this.cameraOffset.lengthSq() === 0) return false;
    this.cameraOffset.normalize().multiplyScalar(distance);
    this.camera.position.add(this.cameraOffset);
    this.controls.target.add(this.cameraOffset);
    return true;
  }

  seek(frame: number): void {
    if (!this.clip) return;
    this.frameCursor = THREE.MathUtils.clamp(frame, 0, this.clip.frameCount - 1);
    this.lastAnimationTime = null;
    this.updatePose(this.frameCursor);
    this.vrm?.springBoneManager?.reset();
    this.invalidate();
    this.emitPlaybackState(true);
  }

  setTrajectory(points: Float32Array | readonly Vector3Tuple[] | null): void {
    this.customTrajectory =
      points === null
        ? null
        : normalizeTrajectoryPoints(
            Array.isArray(points) ? points.map((point) => [...point]) : points,
          );
    this.buildTrajectory();
    this.applyOutputVisibility();
    this.invalidate();
  }

  setOrientationAxes(jointIndices: readonly number[] | "all", size = 0.13): void {
    if (!Number.isFinite(size) || size <= 0) throw new RangeError("Orientation axis size must be positive.");
    const indices =
      jointIndices === "all"
        ? Array.from({ length: this.skeleton.jointNames.length }, (_, index) => index)
        : [...jointIndices];
    if (
      new Set(indices).size !== indices.length ||
      indices.some(
        (joint) => !Number.isInteger(joint) || joint < 0 || joint >= this.skeleton.jointNames.length,
      )
    ) {
      throw new RangeError("Orientation axis joints must be unique valid skeleton indices.");
    }
    this.orientationJointIndices = indices;
    this.orientationAxisSize = size;
    clearGroup(this.orientationAxesGroup);
    this.orientationAxes = indices.map(() => {
      const axes = new THREE.AxesHelper(size);
      axes.renderOrder = 3;
      this.orientationAxesGroup.add(axes);
      return axes;
    });
    if (this.clip) this.updateOrientationAxes(this.frameCursor);
    this.applyOutputVisibility();
    this.invalidate();
  }

  setInitialTransform(transform: InitialTransform): void {
    this.initialTransform = normalizeInitialTransform(transform);
    this.buildInitialTransformMarker();
    this.invalidate();
    this.emitEditorState();
  }

  setWaypoints(waypoints: readonly MotionWaypoint[]): void {
    const normalized = waypoints.map((waypoint) =>
      normalizeWaypoint(waypoint),
    );
    if (new Set(normalized.map((waypoint) => waypoint.id)).size !== normalized.length) {
      throw new RangeError("Waypoint ids must be unique.");
    }
    this.waypoints = normalized;
    this.buildWaypointMarkers();
    this.invalidate();
    this.emitEditorState();
  }

  upsertWaypoint(waypoint: MotionWaypoint): void {
    const normalized = normalizeWaypoint(waypoint);
    const index = this.waypoints.findIndex((item) => item.id === normalized.id);
    if (index === -1) {
      this.waypoints = [...this.waypoints, normalized];
    } else {
      this.waypoints = this.waypoints.map((item, itemIndex) =>
        itemIndex === index ? normalized : item,
      );
    }
    this.buildWaypointMarkers();
    this.invalidate();
    this.emitEditorState();
  }

  removeWaypoint(id: string): boolean {
    const next = this.waypoints.filter((waypoint) => waypoint.id !== id);
    if (next.length === this.waypoints.length) return false;
    this.waypoints = next;
    this.buildWaypointMarkers();
    this.invalidate();
    this.emitEditorState();
    return true;
  }

  setConstraintMarkers(markers: readonly MotionConstraintMarker[]): void {
    const normalized = markers.map((marker) =>
      normalizeConstraintMarker(marker, {
        jointCount: this.skeleton.jointNames.length,
      }),
    );
    if (new Set(normalized.map((marker) => marker.id)).size !== normalized.length) {
      throw new RangeError("Constraint ids must be unique.");
    }
    this.constraints = normalized;
    this.buildConstraintMarkers();
    this.updateConstraintMarkers(Math.floor(this.frameCursor));
    this.invalidate();
    this.emitEditorState();
  }

  setOutputVisibility(visibility: Partial<ViewerOutputVisibility>): void {
    const next = { ...this.outputVisibility };
    const keys: ViewerOutputKey[] = [
      "skeleton",
      "mesh",
      "reference",
      "trajectory",
      "contacts",
      "orientationAxes",
      "constraints",
      "initialTransform",
      "waypoints",
    ];
    for (const key of keys) {
      if (visibility[key] !== undefined) next[key] = Boolean(visibility[key]);
    }
    this.outputVisibility = next;
    this.applyOutputVisibility();
    if (this.clip) this.updatePose(this.frameCursor);
    this.invalidate();
    this.emitEditorState();
  }

  setOutputVisible(output: ViewerOutputKey, visible: boolean): void {
    if (!(output in DEFAULT_OUTPUT_VISIBILITY)) {
      throw new RangeError(`Unknown viewer output "${String(output)}".`);
    }
    this.setOutputVisibility({ [output]: visible });
  }

  applyEditorState(state: MotionEditorState): void {
    const normalized = normalizeEditorState(state, {
      jointCount: this.skeleton.jointNames.length,
    });
    this.initialTransform = normalized.initialTransform;
    this.waypoints = [...normalized.waypoints];
    this.constraints = [...normalized.constraints];
    this.outputVisibility = { ...normalized.outputVisibility };
    this.buildInitialTransformMarker();
    this.buildWaypointMarkers();
    this.buildConstraintMarkers();
    this.updateConstraintMarkers(Math.floor(this.frameCursor));
    this.applyOutputVisibility();
    if (this.clip) this.updatePose(this.frameCursor);
    this.invalidate();
    this.emitEditorState();
  }

  getEditorState(): MotionEditorState {
    return normalizeEditorState(
      {
        initialTransform: this.initialTransform,
        waypoints: this.waypoints,
        constraints: this.constraints,
        outputVisibility: this.outputVisibility,
      },
      {
        jointCount: this.skeleton.jointNames.length,
      },
    );
  }

  resetCamera({ animated = true }: ResetCameraOptions = {}): void {
    const target = new THREE.Vector3();
    const position = new THREE.Vector3();
    let near = this.camera.near;
    let far = this.camera.far;

    if (!this.clip) {
      position.set(3.1, 2.15, 3.4);
      target.set(0, 0.85, 0);
      this.hasCameraFollowAnchor = false;
    } else {
      const box = new THREE.Box3();
      const point = new THREE.Vector3();
      const frame0 = Math.min(
        Math.floor(this.frameCursor),
        this.clip.frameCount - 1,
      );
      const frame1 = Math.min(frame0 + 1, this.clip.frameCount - 1);
      const alpha =
        frame1 === frame0 ? 0 : this.frameCursor - frame0;
      this.pointAt(
        point,
        this.skeleton.rootJointIndex,
        frame0,
        frame1,
        alpha,
      );
      const rootX = point.x;
      const rootZ = point.z;
      for (let joint = 0; joint < this.skeleton.jointNames.length; joint += 1) {
        this.pointAt(point, joint, frame0, frame1, alpha);
        box.expandByPoint(point);
      }
      if (box.isEmpty()) return;
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const distance = Math.max(
        2.5,
        Math.max(size.x, size.y, size.z) * 1.85,
      );
      const targetY = Math.max(0.7, center.y);
      target.set(rootX, targetY, rootZ);
      position.set(
        rootX + distance * 0.78,
        targetY + distance * 0.48,
        rootZ + distance,
      );
      this.cameraFollowX = rootX;
      this.cameraFollowZ = rootZ;
      this.hasCameraFollowAnchor = true;
      near = Math.max(0.01, distance / 200);
      far = Math.max(100, distance * 20);
    }

    this.applyCameraReset(position, target, near, far, animated);
  }

  private applyCameraReset(
    position: THREE.Vector3,
    target: THREE.Vector3,
    near: number,
    far: number,
    animated: boolean,
  ): void {
    this.cancelCameraReset();
    this.stopCameraInertia();
    const positionDistance = this.camera.position.distanceTo(position);
    const targetDistance = this.controls.target.distanceTo(target);
    const travelDistance = Math.max(positionDistance, targetDistance);
    if (
      !animated ||
      this.reducedMotion ||
      travelDistance <= Number.EPSILON
    ) {
      this.camera.position.copy(position);
      this.controls.target.copy(target);
      this.camera.near = near;
      this.camera.far = far;
      this.camera.updateProjectionMatrix();
      this.controls.update();
      this.invalidate();
      return;
    }

    const startOffset = new THREE.Spherical().setFromVector3(
      this.camera.position.clone().sub(this.controls.target),
    );
    // OrbitControls retains the last meaningful heading at the polar axis.
    startOffset.theta = this.controls.getAzimuthalAngle();
    const endOffset = new THREE.Spherical().setFromVector3(
      position.clone().sub(target),
    );
    this.camera.near = Math.min(this.camera.near, near);
    this.camera.far = Math.max(this.camera.far, far);
    this.camera.updateProjectionMatrix();
    this.cameraResetTransition = {
      startedAt: performance.now(),
      duration: cameraResetDuration(travelDistance),
      startTarget: this.controls.target.clone(),
      endTarget: target.clone(),
      startOffset,
      endOffset,
      azimuthDelta: shortestAngleDelta(startOffset.theta, endOffset.theta),
      finalNear: near,
      finalFar: far,
    };
    this.invalidate();
  }

  private cancelCameraReset(): void {
    this.cameraResetTransition = null;
  }

  /**
   * Clear OrbitControls' pending damping without exposing its private deltas
   * or rendering the fully-applied inertial pose for a frame.
   */
  private stopCameraInertia(): void {
    if (!this.controls.enableDamping) return;
    const position = this.camera.position.clone();
    const target = this.controls.target.clone();
    this.controls.enableDamping = false;
    this.controls.update();
    this.camera.position.copy(position);
    this.controls.target.copy(target);
    this.controls.update();
    this.controls.enableDamping = true;
  }

  private finishCameraReset(): void {
    const transition = this.cameraResetTransition;
    if (!transition) return;
    this.cameraResetTransition = null;
    this.controls.target.copy(transition.endTarget);
    this.cameraResetOffset.setFromSpherical(transition.endOffset);
    this.camera.position
      .copy(transition.endTarget)
      .add(this.cameraResetOffset);
    this.camera.near = transition.finalNear;
    this.camera.far = transition.finalFar;
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.invalidate();
  }

  getPlaybackState(): PlaybackState {
    return {
      frame: Math.floor(this.frameCursor),
      frameCount: this.clip?.frameCount ?? 0,
      fps: this.clip?.fps ?? DEFAULT_MOTION_FPS,
      playing: this.playing,
      speed: this.speed,
    };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.rendererReady = false;
    cancelAnimationFrame(this.animationFrame);
    cancelAnimationFrame(this.resizeFrame);
    this.resizeObserver.disconnect();
    document.removeEventListener("visibilitychange", this.handleVisibilityChange);
    this.canvas.removeEventListener("pointerdown", this.handleGroundPointerDown);
    this.canvas.removeEventListener("pointermove", this.handleGroundPointerMove);
    this.canvas.removeEventListener("pointerup", this.handleGroundPointerUp);
    this.canvas.removeEventListener("pointercancel", this.handleGroundPointerCancel);
    this.controls.removeEventListener("change", this.invalidate);
    this.controls.removeEventListener(
      "start",
      this.handleCameraInteractionStart,
    );
    this.controls.dispose();
    this.vrmLoadRevision += 1;
    this.disposeVrmAvatar();
    this.disposeSkeletonMeshes();
    this.scene.remove(this.groundSurface, this.shadowRig);
    this.groundSurface.geometry.dispose();
    this.groundMaterial.dispose();
    this.shadowLight.shadow.dispose();
    this.trajectory.geometry.dispose();
    (this.trajectory.material as THREE.Material).dispose();
    [
      this.orientationAxesGroup,
      this.constraintGroup,
      this.initialTransformGroup,
      this.waypointGroup,
    ].forEach(clearGroup);
    this.renderPipeline.dispose();
    if (this.rendererInitialized) this.renderer.dispose();
  }

  private readonly animate = (time: number): void => {
    this.animationFrame = 0;
    if (!this.pageVisible) return;

    let poseChanged = false;
    let elapsedSeconds = 0;
    if (this.playing && this.clip) {
      if (this.lastAnimationTime !== null) {
        elapsedSeconds = Math.min((time - this.lastAnimationTime) / 1000, 0.25);
        const next = frameAfterElapsed(
          this.frameCursor,
          elapsedSeconds,
          this.clip.fps,
          this.speed,
          this.clip.frameCount,
          this.loop,
        );
        this.frameCursor = next.frame;
        if (next.ended && !this.loop) this.playing = false;
        this.updatePose(this.frameCursor);
        poseChanged = true;
        this.emitPlaybackState(next.ended);
      }
      this.lastAnimationTime = time;
    } else {
      this.lastAnimationTime = null;
    }

    const cameraResetChanged = this.updateCameraReset(time);
    const controlsChanged = this.controls.update();
    const cameraMoved = this.updateCameraMovement(time);
    if (
      this.vrm &&
      this.vrmRoot.visible &&
      (this.needsRender || poseChanged || controlsChanged)
    ) {
      this.vrm.update(elapsedSeconds);
    }
    if (
      this.needsRender ||
      poseChanged ||
      controlsChanged ||
      cameraMoved ||
      cameraResetChanged
    ) {
      this.updateGroundSurfaceLayout();
      this.renderPipeline.render();
      this.needsRender = false;
    }
    if (
      this.playing ||
      controlsChanged ||
      this.hasCameraMovement() ||
      this.cameraResetTransition
    ) {
      this.scheduleFrame();
    }
  };

  private updateCameraReset(time: number): boolean {
    const transition = this.cameraResetTransition;
    if (!transition) return false;
    const rawProgress =
      (time - transition.startedAt) / transition.duration;
    const progress = THREE.MathUtils.clamp(rawProgress, 0, 1);
    const eased = cameraResetProgress(progress);
    this.cameraResetTarget.lerpVectors(
      transition.startTarget,
      transition.endTarget,
      eased,
    );
    const radius = THREE.MathUtils.lerp(
      transition.startOffset.radius,
      transition.endOffset.radius,
      eased,
    );
    const phi = THREE.MathUtils.lerp(
      transition.startOffset.phi,
      transition.endOffset.phi,
      eased,
    );
    const theta =
      transition.startOffset.theta + transition.azimuthDelta * eased;
    this.cameraResetOffset.setFromSphericalCoords(radius, phi, theta);
    this.controls.target.copy(this.cameraResetTarget);
    this.camera.position
      .copy(this.cameraResetTarget)
      .add(this.cameraResetOffset);

    if (progress >= 1) {
      this.cameraResetTransition = null;
      this.controls.target.copy(transition.endTarget);
      this.cameraResetOffset.setFromSpherical(transition.endOffset);
      this.camera.position
        .copy(transition.endTarget)
        .add(this.cameraResetOffset);
      this.camera.near = transition.finalNear;
      this.camera.far = transition.finalFar;
      this.camera.updateProjectionMatrix();
    }
    return true;
  }

  private hasCameraMovement(): boolean {
    return (
      this.cameraMovementForward !== 0 ||
      this.cameraMovementRight !== 0
    );
  }

  private updateCameraMovement(time: number): boolean {
    if (!this.hasCameraMovement()) {
      this.lastCameraMovementTime = null;
      return false;
    }
    const previousTime = this.lastCameraMovementTime;
    this.lastCameraMovementTime = time;
    if (previousTime === null) return false;
    const elapsedSeconds = Math.max(0, (time - previousTime) / 1_000);
    const distance = cameraMovementDistance(
      this.controls.getDistance(),
      elapsedSeconds,
    );
    return this.translateCamera(
      this.cameraMovementForward,
      this.cameraMovementRight,
      distance,
    );
  }

  private scheduleFrame(): void {
    if (
      !this.animationFrame &&
      this.pageVisible &&
      this.rendererReady &&
      !this.disposed
    ) {
      this.animationFrame = requestAnimationFrame(this.animate);
    }
  }

  private resize(): void {
    const parent = this.canvas.parentElement;
    const width = Math.max(1, parent?.clientWidth ?? this.canvas.clientWidth);
    const height = Math.max(1, parent?.clientHeight ?? this.canvas.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.invalidate();
  }

  private createSkeletonMeshes(skeleton: SkeletonMetadata): void {
    if (this.joints && this.bones) this.disposeSkeletonMeshes();
    const { joints: jointCount, bones: boneCount } =
      skeletonInstanceCounts(skeleton);
    this.contactChannelByJoint = new Int32Array(jointCount);
    this.contactChannelByJoint.fill(-1);
    skeleton.contactJointIndices.forEach((joint, channel) => {
      this.contactChannelByJoint[joint] = channel;
    });
    this.localWorldRotations = Array.from({ length: jointCount }, () => new THREE.Quaternion());
    this.localRotationResolved = new Uint8Array(jointCount);
    this.joints = new THREE.InstancedMesh(
      new THREE.SphereGeometry(0.027, 18, 12),
      new THREE.MeshStandardMaterial({
        color: VIEWER_COLORS.foreground,
        roughness: 0.38,
        metalness: 0.12,
      }),
      jointCount,
    );
    this.joints.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.joints.castShadow = true;
    this.joints.frustumCulled = false;
    this.bones = new THREE.InstancedMesh(
      new THREE.CapsuleGeometry(0.014, 1, 4, 8),
      new THREE.MeshStandardMaterial({
        color: VIEWER_COLORS.mutedForeground,
        roughness: 0.5,
        metalness: 0.08,
      }),
      Math.max(1, boneCount),
    );
    this.bones.count = boneCount;
    this.bones.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.bones.castShadow = true;
    this.bones.frustumCulled = false;
    this.scene.add(this.joints, this.bones);
  }

  private disposeSkeletonMeshes(): void {
    for (const mesh of [this.joints, this.bones]) {
      if (mesh) {
        this.scene.remove(mesh);
        mesh.geometry.dispose();
        (mesh.material as THREE.Material).dispose();
      }
    }
  }

  private pointAt(
    target: THREE.Vector3,
    joint: number,
    frame0: number,
    frame1: number,
    alpha: number,
  ): void {
    const clip = this.clip;
    if (!clip) return;
    this.pointAtClip(target, clip, joint, frame0, frame1, alpha);
  }

  private pointAtClip(
    target: THREE.Vector3,
    clip: MotionClip,
    joint: number,
    frame0: number,
    frame1: number,
    alpha: number,
  ): void {
    const componentsPerFrame = clip.skeleton.jointNames.length * 3;
    const jointOffset = joint * 3;
    const offset0 = frame0 * componentsPerFrame + jointOffset;
    const offset1 = frame1 * componentsPerFrame + jointOffset;
    target.set(
      THREE.MathUtils.lerp(clip.positions[offset0], clip.positions[offset1], alpha),
      THREE.MathUtils.lerp(clip.positions[offset0 + 1], clip.positions[offset1 + 1], alpha),
      THREE.MathUtils.lerp(clip.positions[offset0 + 2], clip.positions[offset1 + 2], alpha),
    );
  }

  private updateSkeletonPose(
    clip: MotionClip,
    frame0: number,
    frame1: number,
    alpha: number,
  ): void {
    for (let joint = 0; joint < this.skeleton.jointNames.length; joint += 1) {
      this.pointAtClip(
        this.currentPoint,
        clip,
        joint,
        frame0,
        frame1,
        alpha,
      );
      this.jointTransform.position.copy(this.currentPoint);
      this.jointTransform.scale.setScalar(joint === this.skeleton.rootJointIndex ? 1.25 : 1);
      this.jointTransform.quaternion.identity();
      this.jointTransform.updateMatrix();
      this.joints.setMatrixAt(joint, this.jointTransform.matrix);

      const contactChannel = this.contactChannelByJoint[joint];
      const inContact =
        this.outputVisibility.contacts &&
        contactChannel >= 0 &&
        Boolean(
          clip.contacts?.[
            frame0 * this.skeleton.contactJointIndices.length + contactChannel
          ],
        );
      this.joints.setColorAt(joint, inContact ? this.contactJointColor : this.regularJointColor);
    }

    let boneIndex = 0;
    for (let joint = 0; joint < this.skeleton.jointNames.length; joint += 1) {
      const parent = this.skeleton.parents[joint];
      if (parent === -1) continue;
      this.pointAtClip(
        this.currentPoint,
        clip,
        joint,
        frame0,
        frame1,
        alpha,
      );
      this.pointAtClip(
        this.parentPoint,
        clip,
        parent,
        frame0,
        frame1,
        alpha,
      );
      this.direction.subVectors(this.currentPoint, this.parentPoint);
      const length = Math.max(this.direction.length(), 0.0001);
      this.midpoint.addVectors(this.currentPoint, this.parentPoint).multiplyScalar(0.5);
      this.boneTransform.position.copy(this.midpoint);
      this.boneTransform.quaternion.setFromUnitVectors(this.upAxis, this.direction.normalize());
      this.boneTransform.scale.set(1, Math.max(0.001, length - 0.028), 1);
      this.boneTransform.updateMatrix();
      this.bones.setMatrixAt(boneIndex, this.boneTransform.matrix);
      boneIndex += 1;
    }

    this.joints.instanceMatrix.needsUpdate = true;
    if (this.joints.instanceColor) this.joints.instanceColor.needsUpdate = true;
    this.bones.instanceMatrix.needsUpdate = true;
  }

  private updatePose(frameCursor: number): void {
    const clip = this.clip;
    if (!clip) return;
    const frame0 = Math.min(Math.floor(frameCursor), clip.frameCount - 1);
    const frame1 = Math.min(frame0 + 1, clip.frameCount - 1);
    const alpha = frame1 === frame0 ? 0 : frameCursor - frame0;
    this.updateSkeletonPose(clip, frame0, frame1, alpha);
    this.updateOrientationAxes(frameCursor);
    this.updateConstraintMarkers(frame0);
    const vrmFrame = this.updateVrmPose(frameCursor);
    this.updateShadowAnchorForFrame(frameCursor, vrmFrame);
  }

  private showNeutralPose(): void {
    const pose = this.neutralPose;
    if (!pose) {
      this.applyOutputVisibility();
      return;
    }

    if (!sameSkeleton(this.skeleton, pose.skeleton)) {
      this.skeleton = pose.skeleton;
      this.createSkeletonMeshes(this.skeleton);
      this.setOrientationAxes("all", this.orientationAxisSize);
      this.rebuildVrmRetargetPlan();
    } else {
      this.skeleton = pose.skeleton;
    }
    this.updateSkeletonPose(pose, 0, 0, 0);
    this.applyOutputVisibility();
  }

  private updateShadowAnchorForFrame(
    frameCursor: number,
    updatedVrmFrame?: VrmRetargetFrame | null,
  ): void {
    const clip = this.clip;
    if (!clip) {
      this.updateShadowAnchor(0, 0);
      return;
    }
    const frame0 = Math.min(Math.floor(frameCursor), clip.frameCount - 1);
    const frame1 = Math.min(frame0 + 1, clip.frameCount - 1);
    const alpha = frame1 === frame0 ? 0 : frameCursor - frame0;
    this.pointAt(
      this.currentPoint,
      this.skeleton.rootJointIndex,
      frame0,
      frame1,
      alpha,
    );
    const rootX = this.currentPoint.x;
    const rootZ = this.currentPoint.z;
    const vrmFrame =
      updatedVrmFrame === undefined &&
      this.vrm &&
      this.vrmRetargetPlan
        ? retargetMotionFrame(clip, frameCursor, this.vrmRetargetPlan)
        : updatedVrmFrame;
    if (vrmFrame && this.vrmRetargetPlan) {
      this.pointAt(
        this.parentPoint,
        this.vrmRetargetPlan.hipsSourceJointIndex,
        frame0,
        frame1,
        alpha,
      );
      const displayedHips = toVrmPosition(
        vrmFrame.hipsPosition,
        this.vrmRetargetPlan.metaVersion,
      );
      this.vrmRoot.position.set(
        this.parentPoint.x - displayedHips[0],
        0,
        this.parentPoint.z - displayedHips[2],
      );
    }
    this.updateShadowAnchor(rootX, rootZ);
    this.followCamera(rootX, rootZ);
  }

  private updateShadowAnchor(x: number, z: number): void {
    this.shadowRig.position.set(x, 0, z);
  }

  private updateGroundSurfaceLayout(): void {
    const layout = computeCameraRelativeGroundLayout({
      targetX: this.controls.target.x,
      targetZ: this.controls.target.z,
      cameraFar: this.camera.far,
      maxCameraDistance: this.controls.maxDistance,
      padding: GROUND_MAJOR_SPACING,
      majorSpacing: GROUND_MAJOR_SPACING,
    });
    if (
      this.groundSurface.position.x === layout.centerX &&
      this.groundSurface.position.z === layout.centerZ &&
      this.groundSideLength === layout.sideLength
    ) {
      return;
    }
    this.groundSurface.position.set(layout.centerX, GROUND_Y, layout.centerZ);
    this.groundSurface.scale.set(layout.sideLength, layout.sideLength, 1);
    this.groundSideLength = layout.sideLength;
    updateGroundGridOrigin(
      this.groundMaterial,
      layout.centerX,
      layout.centerZ,
    );
  }

  private followCamera(x: number, z: number): void {
    if (!this.hasCameraFollowAnchor) {
      this.cameraFollowX = x;
      this.cameraFollowZ = z;
      this.hasCameraFollowAnchor = true;
      return;
    }
    const deltaX = x - this.cameraFollowX;
    const deltaZ = z - this.cameraFollowZ;
    this.cameraFollowX = x;
    this.cameraFollowZ = z;
    if (deltaX === 0 && deltaZ === 0) return;
    const transition = this.cameraResetTransition;
    if (transition) {
      transition.startTarget.x += deltaX;
      transition.startTarget.z += deltaZ;
      transition.endTarget.x += deltaX;
      transition.endTarget.z += deltaZ;
    }
    this.camera.position.x += deltaX;
    this.camera.position.z += deltaZ;
    this.controls.target.x += deltaX;
    this.controls.target.z += deltaZ;
  }

  private rebuildVrmRetargetPlan(): void {
    const vrm = this.vrm;
    if (!vrm) {
      this.vrmRetargetPlan = null;
      return;
    }
    this.vrmRetargetPlan = this.buildVrmRetargetPlan(
      vrm,
      this.vrmTargetHipsHeight,
    );
  }

  private buildVrmRetargetPlan(
    vrm: VRM,
    targetHipsHeight: number,
  ): VrmRetargetPlan {
    const presentBones = CORE27_VRM_BINDINGS.flatMap(({ targetBone }) =>
      vrm.humanoid.getNormalizedBoneNode(targetBone)
        ? [targetBone]
        : [],
    );
    return createVrmRetargetPlan(this.skeleton, {
      presentBones,
      targetHipsHeight,
      metaVersion: vrm.meta.metaVersion,
    });
  }

  private resetVrmPose(): void {
    const vrm = this.vrm;
    if (!vrm) return;
    vrm.humanoid.resetNormalizedPose();
    vrm.springBoneManager?.reset();
  }

  private updateVrmPose(frameCursor: number): VrmRetargetFrame | null {
    const vrm = this.vrm;
    const clip = this.clip;
    const plan = this.vrmRetargetPlan;
    if (!vrm || !clip || !plan) return null;
    const frame = retargetMotionFrame(clip, frameCursor, plan);
    this.applyVrmRetargetFrame(vrm, frame);
    return frame;
  }

  private applyVrmRetargetFrame(vrm: VRM, frame: VrmRetargetFrame): void {
    const hips = vrm.humanoid.getNormalizedBoneNode("hips");
    hips?.position.fromArray(frame.hipsPosition);
    for (const { targetBone, rotation } of frame.rotations) {
      vrm.humanoid
        .getNormalizedBoneNode(targetBone)
        ?.quaternion.fromArray(rotation);
    }
  }

  private disposeVrmAvatar(): void {
    const vrm = this.vrm;
    if (vrm) {
      this.vrmRoot.remove(vrm.scene);
      this.vrmUtils?.deepDispose(vrm.scene);
    }
    this.vrm = null;
    this.vrmUtils = null;
    this.vrmRetargetPlan = null;
    this.vrmRoot.position.set(0, 0, 0);
    this.vrmRoot.visible = false;
  }

  private rotationAt(
    track: RotationTrack,
    frameCursor: number,
    joint: number,
    target: THREE.Quaternion,
  ): THREE.Quaternion {
    const frame0 = Math.min(Math.floor(frameCursor), track.shape[0] - 1);
    const frame1 = Math.min(frame0 + 1, track.shape[0] - 1);
    const alpha = frame1 === frame0 ? 0 : frameCursor - frame0;
    this.readRotation(track, frame0, joint, target);
    if (alpha > 0) {
      this.readRotation(track, frame1, joint, this.rotationB);
      target.slerp(this.rotationB, alpha);
    }
    return target.normalize();
  }

  private readRotation(
    track: RotationTrack,
    frame: number,
    joint: number,
    target: THREE.Quaternion,
  ): void {
    const components = track.shape[2];
    const offset = (frame * track.shape[1] + joint) * components;
    if (components === 4) {
      target.fromArray(track.values, offset);
      return;
    }
    const value = track.values;
    this.rotationMatrix.set(
      value[offset],
      value[offset + 1],
      value[offset + 2],
      0,
      value[offset + 3],
      value[offset + 4],
      value[offset + 5],
      0,
      value[offset + 6],
      value[offset + 7],
      value[offset + 8],
      0,
      0,
      0,
      0,
      1,
    );
    target.setFromRotationMatrix(this.rotationMatrix);
  }

  private updateOrientationAxes(frameCursor: number): void {
    const clip = this.clip;
    if (
      !clip ||
      !this.outputVisibility.orientationAxes ||
      this.orientationAxes.length === 0
    ) {
      return;
    }
    const frame0 = Math.min(Math.floor(frameCursor), clip.frameCount - 1);
    const frame1 = Math.min(frame0 + 1, clip.frameCount - 1);
    const alpha = frame1 === frame0 ? 0 : frameCursor - frame0;
    this.localRotationResolved.fill(0);
    const resolveLocalWorld = (joint: number): THREE.Quaternion => {
      if (this.localRotationResolved[joint]) return this.localWorldRotations[joint];
      const local = this.rotationAt(
        clip.localRotations!,
        frameCursor,
        joint,
        this.localWorldRotations[joint],
      );
      const parent = this.skeleton.parents[joint];
      if (parent !== -1) local.premultiply(resolveLocalWorld(parent));
      this.localRotationResolved[joint] = 1;
      return local;
    };

    this.orientationJointIndices.forEach((joint, axisIndex) => {
      const axes = this.orientationAxes[axisIndex];
      this.pointAt(axes.position, joint, frame0, frame1, alpha);
      if (clip.globalRotations) {
        this.rotationAt(clip.globalRotations, frameCursor, joint, this.rotationA);
        axes.quaternion.copy(this.rotationA);
      } else if (clip.localRotations) {
        axes.quaternion.copy(resolveLocalWorld(joint));
      } else {
        axes.quaternion.identity();
      }
    });
  }

  private buildTrajectory(): void {
    const points: THREE.Vector3[] = [];
    if (this.customTrajectory) {
      for (let offset = 0; offset < this.customTrajectory.length; offset += 3) {
        points.push(
          new THREE.Vector3(
            this.customTrajectory[offset],
            this.customTrajectory[offset + 1],
            this.customTrajectory[offset + 2],
          ),
        );
      }
    } else if (this.clip) {
      const jointCount = this.skeleton.jointNames.length;
      for (let frame = 0; frame < this.clip.frameCount; frame += 1) {
        const offset = (frame * jointCount + this.skeleton.rootJointIndex) * 3;
        points.push(
          new THREE.Vector3(
            this.clip.positions[offset],
            0.008,
            this.clip.positions[offset + 2],
          ),
        );
      }
    }
    this.trajectory.geometry.dispose();
    this.trajectory.geometry = new THREE.BufferGeometry().setFromPoints(points);
    this.trajectory.visible = this.outputVisibility.trajectory && points.length > 1;
  }

  private buildInitialTransformMarker(): void {
    clearGroup(this.initialTransformGroup);
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.1, 0.13, 28),
      new THREE.MeshBasicMaterial({
        color: VIEWER_COLORS.chart1,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.82,
      }),
    );
    ring.rotation.x = -Math.PI / 2;
    const arrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(0, 0.012, 0),
      0.34,
      VIEWER_COLORS.chart1,
      0.09,
      0.045,
    );
    this.initialTransformGroup.add(ring, arrow);
    this.initialTransformGroup.position.fromArray(this.initialTransform.position);
    this.initialTransformGroup.rotation.y = this.initialTransform.headingRadians;
    this.initialTransformGroup.visible = this.outputVisibility.initialTransform;
  }

  private buildWaypointMarkers(): void {
    clearGroup(this.waypointGroup);
    for (const waypoint of this.waypoints) {
      const marker = new THREE.Group();
      marker.name = `waypoint:${waypoint.id}`;
      marker.position.fromArray(waypoint.position);
      marker.visible = waypoint.enabled;
      const pin = new THREE.Mesh(
        new THREE.SphereGeometry(0.055, 14, 10),
        new THREE.MeshBasicMaterial({ color: VIEWER_COLORS.primary }),
      );
      marker.add(pin);
      if (waypoint.headingRadians !== undefined) {
        marker.rotation.y = waypoint.headingRadians;
        marker.add(
          new THREE.ArrowHelper(
            new THREE.Vector3(0, 0, 1),
            new THREE.Vector3(0, 0.01, 0),
            0.24,
            VIEWER_COLORS.primary,
            0.07,
            0.035,
          ),
        );
      }
      this.waypointGroup.add(marker);
    }
    this.waypointGroup.visible = this.outputVisibility.waypoints;
  }

  private buildConstraintMarkers(): void {
    clearGroup(this.constraintGroup);
    for (const marker of this.constraints) {
      const markerGroup = new THREE.Group();
      markerGroup.name = `constraint:${marker.id}`;
      markerGroup.userData.marker = marker;
      if (marker.position) markerGroup.position.fromArray(marker.position);
      if (marker.orientation) markerGroup.quaternion.fromArray(marker.orientation);
      const color = marker.color ?? VIEWER_COLORS.destructive;
      if (marker.position || marker.jointIndex !== undefined) {
        markerGroup.add(
          new THREE.Mesh(
            new THREE.IcosahedronGeometry(0.065, 1),
            new THREE.MeshBasicMaterial({
              color,
              transparent: true,
              opacity: 0.88,
              wireframe: marker.kind === "trajectory",
            }),
          ),
        );
      }
      if (marker.orientation) markerGroup.add(new THREE.AxesHelper(0.18));
      this.constraintGroup.add(markerGroup);
    }
    this.constraintGroup.visible = this.outputVisibility.constraints && Boolean(this.clip);
  }

  private updateConstraintMarkers(frame: number): void {
    const clip = this.clip;
    for (const child of this.constraintGroup.children) {
      const marker = child.userData.marker as MotionConstraintMarker;
      const active =
        marker.enabled &&
        this.outputVisibility.constraints &&
        frame >= marker.startFrame &&
        frame <= marker.endFrame;
      child.visible = active;
      if (
        active &&
        clip &&
        marker.jointIndex !== undefined &&
        marker.position === undefined &&
        marker.jointIndex < this.skeleton.jointNames.length
      ) {
        const integerFrame = Math.min(frame, clip.frameCount - 1);
        this.pointAt(child.position, marker.jointIndex, integerFrame, integerFrame, 0);
      }
    }
  }

  private applyOutputVisibility(): void {
    const hasClip = Boolean(this.clip);
    const hasSkeletonPose = hasClip || this.neutralPose !== null;
    this.joints.visible = hasSkeletonPose && this.outputVisibility.skeleton;
    this.bones.visible = hasSkeletonPose && this.outputVisibility.skeleton;
    this.trajectory.visible =
      this.outputVisibility.trajectory &&
      (this.customTrajectory !== null || Boolean(this.clip)) &&
      (this.trajectory.geometry.getAttribute("position")?.count ?? 0) > 1;
    this.orientationAxesGroup.visible =
      hasClip && this.outputVisibility.orientationAxes && this.orientationAxes.length > 0;
    this.constraintGroup.visible = hasClip && this.outputVisibility.constraints;
    this.initialTransformGroup.visible = this.outputVisibility.initialTransform;
    this.waypointGroup.visible = this.outputVisibility.waypoints;
  }

  private emitPlaybackState(force: boolean): void {
    if (!this.playbackListener) return;
    const state = this.getPlaybackState();
    if (!force && state.frame === this.lastReportedFrame) return;
    this.lastReportedFrame = state.frame;
    this.playbackListener(state);
  }

  private emitEditorState(): void {
    this.editorListener?.(this.getEditorState());
  }
}
