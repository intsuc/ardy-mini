// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

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
  normalizeSkeletonMetadata,
  normalizeStructuredMotion,
  toFiniteFloat32Array,
  type RotationTrack,
  type SkeletonMetadata,
  type StructuredMotionResult,
} from "./motion-data";

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

/**
 * This optional display is a joint-driven capsule/sphere body proxy. It is
 * intentionally lightweight and is not an SMPL body mesh or a skinning result.
 */
export const BODY_PROXY_DESCRIPTION =
  "Joint-driven capsule and sphere body proxy (not an SMPL body mesh).";

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

/** Counts used by each instanced skeleton/proxy layer for a dynamic skeleton. */
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

/** Map the primary clip playhead to the same wall-clock time in a reference. */
export function referenceFrameAtPlayhead(
  playheadFrame: number,
  primaryFps: number,
  referenceFps: number,
  referenceFrameCount: number,
): number {
  if (
    !Number.isFinite(playheadFrame) ||
    !Number.isFinite(primaryFps) ||
    !Number.isFinite(referenceFps) ||
    primaryFps <= 0 ||
    referenceFps <= 0 ||
    !Number.isSafeInteger(referenceFrameCount) ||
    referenceFrameCount < 1
  ) {
    throw new RangeError("Reference playback timing must contain valid finite values.");
  }
  return Math.min(
    Math.max(0, (playheadFrame / primaryFps) * referenceFps),
    referenceFrameCount - 1,
  );
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
    left.parents.length !== right.parents.length ||
    left.contactJointIndices.length !== right.contactJointIndices.length
  ) {
    return false;
  }
  return (
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

export class SkeletonViewer {
  private readonly canvas: HTMLCanvasElement;
  private readonly scene: THREE.Scene;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly controls: OrbitControls;
  private joints!: THREE.InstancedMesh;
  private bones!: THREE.InstancedMesh;
  private proxyJoints!: THREE.InstancedMesh;
  private proxyBones!: THREE.InstancedMesh;
  private referenceJoints!: THREE.InstancedMesh;
  private referenceBones!: THREE.InstancedMesh;
  private readonly trajectory: THREE.Line;
  private readonly orientationAxesGroup = new THREE.Group();
  private readonly constraintGroup = new THREE.Group();
  private readonly initialTransformGroup = new THREE.Group();
  private readonly waypointGroup = new THREE.Group();
  private readonly resizeObserver: ResizeObserver;
  private readonly jointTransform = new THREE.Object3D();
  private readonly boneTransform = new THREE.Object3D();
  private readonly currentPoint = new THREE.Vector3();
  private readonly parentPoint = new THREE.Vector3();
  private readonly direction = new THREE.Vector3();
  private readonly midpoint = new THREE.Vector3();
  private readonly upAxis = new THREE.Vector3(0, 1, 0);
  private readonly regularJointColor = new THREE.Color("#b9f55a");
  private readonly contactJointColor = new THREE.Color("#c877ff");
  private readonly rotationA = new THREE.Quaternion();
  private readonly rotationB = new THREE.Quaternion();
  private readonly rotationMatrix = new THREE.Matrix4();
  private readonly groundRaycaster = new THREE.Raycaster();
  private readonly groundPointer = new THREE.Vector2();
  private readonly groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  private readonly groundIntersection = new THREE.Vector3();

  private skeleton: SkeletonMetadata = CORE27_SKELETON;
  private contactChannelByJoint = new Int32Array();
  private localWorldRotations: THREE.Quaternion[] = [];
  private localRotationResolved = new Uint8Array();
  private clip: MotionClip | null = null;
  private referenceClip: MotionClip | null = null;
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
  private lastAnimationTime: number | null = null;
  private lastReportedFrame = -1;
  private animationFrame = 0;
  private resizeFrame = 0;
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

  private readonly handleVisibilityChange = (): void => {
    this.pageVisible = !document.hidden;
    this.lastAnimationTime = null;
    if (this.pageVisible) {
      this.invalidate();
    } else {
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

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.canvas.style.touchAction = "none";
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#071013");
    this.scene.fog = new THREE.FogExp2("#071013", 0.035);

    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
    this.camera.position.set(3.1, 2.15, 3.4);

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 0.85, 0);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 12;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.update();
    this.controls.addEventListener("change", this.invalidate);
    this.canvas.addEventListener("pointerdown", this.handleGroundPointerDown);
    this.canvas.addEventListener("pointermove", this.handleGroundPointerMove);
    this.canvas.addEventListener("pointerup", this.handleGroundPointerUp);
    this.canvas.addEventListener("pointercancel", this.handleGroundPointerCancel);

    const hemisphere = new THREE.HemisphereLight("#d9f8ff", "#173026", 2.1);
    this.scene.add(hemisphere);
    const keyLight = new THREE.DirectionalLight("#ecffd1", 4.2);
    keyLight.position.set(-3, 6, 4);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(1024, 1024);
    this.scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight("#55d8dd", 2.2);
    rimLight.position.set(5, 2, -4);
    this.scene.add(rimLight);

    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(11, 96),
      new THREE.MeshStandardMaterial({
        color: "#0c1719",
        roughness: 0.92,
        metalness: 0.03,
      }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.006;
    floor.receiveShadow = true;
    this.scene.add(floor);

    const grid = new THREE.GridHelper(18, 36, "#345357", "#183033");
    const gridMaterial = grid.material as THREE.LineBasicMaterial;
    gridMaterial.transparent = true;
    gridMaterial.opacity = 0.48;
    this.scene.add(grid);

    this.createSkeletonMeshes(this.skeleton);
    this.trajectory = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({
        color: "#6cffbd",
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
    );
    this.joints.visible = false;
    this.bones.visible = false;
    this.proxyJoints.visible = false;
    this.proxyBones.visible = false;
    this.referenceJoints.visible = false;
    this.referenceBones.visible = false;
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
    if (this.clip) this.clearMotion();
    if (
      this.referenceClip &&
      !sameSkeleton(this.referenceClip.skeleton, normalized)
    ) {
      this.referenceClip = null;
    }
    this.skeleton = normalized;
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

  setMotion(
    clip: MotionClip,
    autoplay = true,
    resetCamera = true,
  ): void {
    const skeletonChanged = !sameSkeleton(this.skeleton, clip.skeleton);
    if (
      this.referenceClip &&
      !sameSkeleton(this.referenceClip.skeleton, clip.skeleton)
    ) {
      this.referenceClip = null;
    }
    this.clip = clip;
    if (skeletonChanged) {
      this.skeleton = clip.skeleton;
      this.createSkeletonMeshes(this.skeleton);
      this.setOrientationAxes("all", this.orientationAxisSize);
    } else {
      this.skeleton = clip.skeleton;
    }
    this.frameCursor = 0;
    this.playing = autoplay && !this.reducedMotion && clip.frameCount > 1;
    this.lastAnimationTime = null;
    this.lastReportedFrame = -1;
    this.buildTrajectory();
    this.updatePose(0);
    this.applyOutputVisibility();
    if (resetCamera) this.resetCamera();
    this.invalidate();
    this.emitPlaybackState(true);
  }

  /**
   * Overlay a comparison clip at the same wall-clock playhead as the primary
   * clip. The reference uses a warm wireframe skeleton so it is distinguishable
   * by both color and shape.
   */
  setReferenceMotion(clip: MotionClip | null): void {
    if (clip === null) {
      this.referenceClip = null;
      this.applyOutputVisibility();
      this.invalidate();
      return;
    }
    if (this.clip && !sameSkeleton(this.clip.skeleton, clip.skeleton)) {
      throw new RangeError(
        "Reference motion skeleton must match the primary motion skeleton.",
      );
    }
    if (!this.clip && !sameSkeleton(this.skeleton, clip.skeleton)) {
      this.skeleton = clip.skeleton;
      this.createSkeletonMeshes(this.skeleton);
      this.setOrientationAxes("all", this.orientationAxisSize);
    }
    this.referenceClip = clip;
    this.updateReferencePose(this.frameCursor);
    this.applyOutputVisibility();
    this.invalidate();
  }

  clearMotion(): void {
    this.clip = null;
    this.playing = false;
    this.frameCursor = 0;
    this.lastAnimationTime = null;
    this.joints.visible = false;
    this.bones.visible = false;
    this.proxyJoints.visible = false;
    this.proxyBones.visible = false;
    this.referenceJoints.visible = false;
    this.referenceBones.visible = false;
    this.trajectory.visible = false;
    this.orientationAxesGroup.visible = false;
    this.constraintGroup.visible = false;
    this.invalidate();
    this.emitPlaybackState(true);
  }

  setPlaying(playing: boolean): void {
    if (!this.clip || this.clip.frameCount < 2) return;
    if (playing && this.frameCursor >= this.clip.frameCount - 1 && !this.loop) {
      this.frameCursor = 0;
    }
    this.playing = playing;
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
    this.reducedMotion = reduced;
    this.controls.enableDamping = !reduced;
    if (reduced && this.playing) {
      this.playing = false;
      this.emitPlaybackState(true);
    }
    this.lastAnimationTime = null;
    this.controls.update();
    this.invalidate();
  }

  orbit(horizontalSteps: number, verticalSteps: number): void {
    const increment = Math.PI / 24;
    if (horizontalSteps) this.controls.rotateLeft(horizontalSteps * increment);
    if (verticalSteps) this.controls.rotateUp(verticalSteps * increment);
    this.invalidate();
  }

  zoom(direction: "in" | "out"): void {
    const scale = 0.85;
    if (direction === "in") {
      this.controls.dollyIn(scale);
    } else {
      this.controls.dollyOut(scale);
    }
    this.invalidate();
  }

  seek(frame: number): void {
    if (!this.clip) return;
    this.frameCursor = THREE.MathUtils.clamp(frame, 0, this.clip.frameCount - 1);
    this.lastAnimationTime = null;
    this.updatePose(this.frameCursor);
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

  resetCamera(): void {
    if (!this.clip) {
      this.camera.position.set(3.1, 2.15, 3.4);
      this.controls.target.set(0, 0.85, 0);
      this.controls.update();
      this.invalidate();
      return;
    }

    const box = new THREE.Box3();
    const point = new THREE.Vector3();
    const componentsPerFrame = this.skeleton.jointNames.length * 3;
    for (
      let frame = 0;
      frame < this.clip.frameCount;
      frame += Math.max(1, Math.floor(this.clip.frameCount / 80))
    ) {
      const frameOffset = frame * componentsPerFrame;
      for (let joint = 0; joint < this.skeleton.jointNames.length; joint += 1) {
        point.fromArray(this.clip.positions, frameOffset + joint * 3);
        box.expandByPoint(point);
      }
    }
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const distance = Math.max(2.5, Math.max(size.x, size.y, size.z) * 1.85);
    this.controls.target.set(center.x, Math.max(0.7, center.y), center.z);
    this.camera.position.set(
      center.x + distance * 0.78,
      center.y + distance * 0.48,
      center.z + distance,
    );
    this.camera.near = Math.max(0.01, distance / 200);
    this.camera.far = Math.max(100, distance * 20);
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
    cancelAnimationFrame(this.animationFrame);
    cancelAnimationFrame(this.resizeFrame);
    this.resizeObserver.disconnect();
    document.removeEventListener("visibilitychange", this.handleVisibilityChange);
    this.canvas.removeEventListener("pointerdown", this.handleGroundPointerDown);
    this.canvas.removeEventListener("pointermove", this.handleGroundPointerMove);
    this.canvas.removeEventListener("pointerup", this.handleGroundPointerUp);
    this.canvas.removeEventListener("pointercancel", this.handleGroundPointerCancel);
    this.controls.removeEventListener("change", this.invalidate);
    this.controls.dispose();
    this.disposeSkeletonMeshes();
    this.trajectory.geometry.dispose();
    (this.trajectory.material as THREE.Material).dispose();
    [
      this.orientationAxesGroup,
      this.constraintGroup,
      this.initialTransformGroup,
      this.waypointGroup,
    ].forEach(clearGroup);
    this.renderer.dispose();
  }

  private readonly animate = (time: number): void => {
    this.animationFrame = 0;
    if (!this.pageVisible) return;

    let poseChanged = false;
    if (this.playing && this.clip) {
      if (this.lastAnimationTime !== null) {
        const elapsedSeconds = Math.min((time - this.lastAnimationTime) / 1000, 0.25);
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

    const controlsChanged = this.controls.update();
    if (this.needsRender || poseChanged || controlsChanged) {
      this.renderer.render(this.scene, this.camera);
      this.needsRender = false;
    }
    if (this.playing || controlsChanged) this.scheduleFrame();
  };

  private scheduleFrame(): void {
    if (!this.animationFrame && this.pageVisible) {
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
        color: "#ffffff",
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
        color: "#67c6c4",
        roughness: 0.5,
        metalness: 0.08,
      }),
      Math.max(1, boneCount),
    );
    this.bones.count = boneCount;
    this.bones.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.bones.castShadow = true;
    this.bones.frustumCulled = false;
    this.proxyJoints = new THREE.InstancedMesh(
      new THREE.SphereGeometry(0.064, 14, 10),
      new THREE.MeshStandardMaterial({
        color: "#78989d",
        roughness: 0.82,
        metalness: 0,
        transparent: true,
        opacity: 0.4,
        depthWrite: false,
      }),
      jointCount,
    );
    this.proxyJoints.name = "body-proxy-joints";
    this.proxyJoints.userData.description = BODY_PROXY_DESCRIPTION;
    this.proxyJoints.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.proxyJoints.frustumCulled = false;
    this.proxyJoints.renderOrder = 1;
    this.proxyBones = new THREE.InstancedMesh(
      new THREE.CapsuleGeometry(0.043, 1, 4, 8),
      new THREE.MeshStandardMaterial({
        color: "#5f7d83",
        roughness: 0.86,
        metalness: 0,
        transparent: true,
        opacity: 0.36,
        depthWrite: false,
      }),
      Math.max(1, boneCount),
    );
    this.proxyBones.name = "body-proxy-bones";
    this.proxyBones.userData.description = BODY_PROXY_DESCRIPTION;
    this.proxyBones.count = boneCount;
    this.proxyBones.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.proxyBones.frustumCulled = false;
    this.proxyBones.renderOrder = 1;

    this.referenceJoints = new THREE.InstancedMesh(
      new THREE.IcosahedronGeometry(0.035, 1),
      new THREE.MeshBasicMaterial({
        color: "#ffad66",
        wireframe: true,
        transparent: true,
        opacity: 0.9,
        depthWrite: false,
      }),
      jointCount,
    );
    this.referenceJoints.name = "reference-motion-joints";
    this.referenceJoints.userData.description =
      "Reference motion overlay (warm wireframe comparison).";
    this.referenceJoints.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.referenceJoints.frustumCulled = false;
    this.referenceJoints.renderOrder = 2;
    this.referenceBones = new THREE.InstancedMesh(
      new THREE.CapsuleGeometry(0.012, 1, 3, 6),
      new THREE.MeshBasicMaterial({
        color: "#ffe0bd",
        wireframe: true,
        transparent: true,
        opacity: 0.78,
        depthWrite: false,
      }),
      Math.max(1, boneCount),
    );
    this.referenceBones.name = "reference-motion-bones";
    this.referenceBones.userData.description =
      "Reference motion overlay (warm wireframe comparison).";
    this.referenceBones.count = boneCount;
    this.referenceBones.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.referenceBones.frustumCulled = false;
    this.referenceBones.renderOrder = 2;
    this.scene.add(
      this.proxyJoints,
      this.proxyBones,
      this.referenceJoints,
      this.referenceBones,
      this.joints,
      this.bones,
    );
  }

  private disposeSkeletonMeshes(): void {
    for (const mesh of [
      this.joints,
      this.bones,
      this.proxyJoints,
      this.proxyBones,
      this.referenceJoints,
      this.referenceBones,
    ]) {
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

  private updatePose(frameCursor: number): void {
    const clip = this.clip;
    if (!clip) return;
    const frame0 = Math.min(Math.floor(frameCursor), clip.frameCount - 1);
    const frame1 = Math.min(frame0 + 1, clip.frameCount - 1);
    const alpha = frame1 === frame0 ? 0 : frameCursor - frame0;
    const updateProxy = this.outputVisibility.mesh;
    for (let joint = 0; joint < this.skeleton.jointNames.length; joint += 1) {
      this.pointAt(this.currentPoint, joint, frame0, frame1, alpha);
      this.jointTransform.position.copy(this.currentPoint);
      this.jointTransform.scale.setScalar(joint === this.skeleton.rootJointIndex ? 1.25 : 1);
      this.jointTransform.quaternion.identity();
      this.jointTransform.updateMatrix();
      this.joints.setMatrixAt(joint, this.jointTransform.matrix);
      if (updateProxy) {
        this.proxyJoints.setMatrixAt(joint, this.jointTransform.matrix);
      }

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
      this.pointAt(this.currentPoint, joint, frame0, frame1, alpha);
      this.pointAt(this.parentPoint, parent, frame0, frame1, alpha);
      this.direction.subVectors(this.currentPoint, this.parentPoint);
      const length = Math.max(this.direction.length(), 0.0001);
      this.midpoint.addVectors(this.currentPoint, this.parentPoint).multiplyScalar(0.5);
      this.boneTransform.position.copy(this.midpoint);
      this.boneTransform.quaternion.setFromUnitVectors(this.upAxis, this.direction.normalize());
      this.boneTransform.scale.set(1, Math.max(0.001, length - 0.028), 1);
      this.boneTransform.updateMatrix();
      this.bones.setMatrixAt(boneIndex, this.boneTransform.matrix);
      if (updateProxy) {
        this.boneTransform.scale.set(
          1,
          Math.max(0.001, length - 0.086),
          1,
        );
        this.boneTransform.updateMatrix();
        this.proxyBones.setMatrixAt(boneIndex, this.boneTransform.matrix);
      }
      boneIndex += 1;
    }

    this.joints.instanceMatrix.needsUpdate = true;
    if (this.joints.instanceColor) this.joints.instanceColor.needsUpdate = true;
    this.bones.instanceMatrix.needsUpdate = true;
    if (updateProxy) {
      this.proxyJoints.instanceMatrix.needsUpdate = true;
      this.proxyBones.instanceMatrix.needsUpdate = true;
    }
    this.updateReferencePose(frameCursor);
    this.updateOrientationAxes(frameCursor);
    this.updateConstraintMarkers(frame0);
  }

  private updateReferencePose(primaryFrameCursor: number): void {
    const primary = this.clip;
    const reference = this.referenceClip;
    if (!primary || !reference || !this.outputVisibility.reference) return;
    const referenceCursor = referenceFrameAtPlayhead(
      primaryFrameCursor,
      primary.fps,
      reference.fps,
      reference.frameCount,
    );
    const frame0 = Math.min(
      Math.floor(referenceCursor),
      reference.frameCount - 1,
    );
    const frame1 = Math.min(frame0 + 1, reference.frameCount - 1);
    const alpha = frame1 === frame0 ? 0 : referenceCursor - frame0;

    for (
      let joint = 0;
      joint < reference.skeleton.jointNames.length;
      joint += 1
    ) {
      this.pointAtClip(
        this.currentPoint,
        reference,
        joint,
        frame0,
        frame1,
        alpha,
      );
      this.jointTransform.position.copy(this.currentPoint);
      this.jointTransform.quaternion.identity();
      this.jointTransform.scale.setScalar(
        joint === reference.skeleton.rootJointIndex ? 1.18 : 1,
      );
      this.jointTransform.updateMatrix();
      this.referenceJoints.setMatrixAt(joint, this.jointTransform.matrix);
    }

    let boneIndex = 0;
    for (
      let joint = 0;
      joint < reference.skeleton.jointNames.length;
      joint += 1
    ) {
      const parent = reference.skeleton.parents[joint];
      if (parent === -1) continue;
      this.pointAtClip(
        this.currentPoint,
        reference,
        joint,
        frame0,
        frame1,
        alpha,
      );
      this.pointAtClip(
        this.parentPoint,
        reference,
        parent,
        frame0,
        frame1,
        alpha,
      );
      this.direction.subVectors(this.currentPoint, this.parentPoint);
      const length = Math.max(this.direction.length(), 0.0001);
      this.midpoint
        .addVectors(this.currentPoint, this.parentPoint)
        .multiplyScalar(0.5);
      this.boneTransform.position.copy(this.midpoint);
      this.boneTransform.quaternion.setFromUnitVectors(
        this.upAxis,
        this.direction.normalize(),
      );
      this.boneTransform.scale.set(
        1,
        Math.max(0.001, length - 0.024),
        1,
      );
      this.boneTransform.updateMatrix();
      this.referenceBones.setMatrixAt(
        boneIndex,
        this.boneTransform.matrix,
      );
      boneIndex += 1;
    }
    this.referenceJoints.instanceMatrix.needsUpdate = true;
    this.referenceBones.instanceMatrix.needsUpdate = true;
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
        color: "#6cffbd",
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
      "#6cffbd",
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
        new THREE.MeshBasicMaterial({ color: "#ffd56a" }),
      );
      marker.add(pin);
      if (waypoint.headingRadians !== undefined) {
        marker.rotation.y = waypoint.headingRadians;
        marker.add(
          new THREE.ArrowHelper(
            new THREE.Vector3(0, 0, 1),
            new THREE.Vector3(0, 0.01, 0),
            0.24,
            "#ffd56a",
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
      const color = marker.color ?? "#ff8f70";
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
    const hasReference = hasClip && Boolean(this.referenceClip);
    this.joints.visible = hasClip && this.outputVisibility.skeleton;
    this.bones.visible = hasClip && this.outputVisibility.skeleton;
    this.proxyJoints.visible = hasClip && this.outputVisibility.mesh;
    this.proxyBones.visible = hasClip && this.outputVisibility.mesh;
    this.referenceJoints.visible =
      hasReference && this.outputVisibility.reference;
    this.referenceBones.visible =
      hasReference && this.outputVisibility.reference;
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
