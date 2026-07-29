// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export const CORE27_JOINT_COUNT = 27;
export const CORE27_PARENTS = Object.freeze([
  -1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 9, 10, 10, 4, 13, 14, 15, 16, 16, 0, 19, 20, 21, 0, 23, 24, 25,
]);
export const CORE27_FOOT_CONTACT_JOINTS = Object.freeze([25, 26, 21, 22]);

const POSITION_COMPONENTS_PER_FRAME = CORE27_JOINT_COUNT * 3;
const BONE_COUNT = CORE27_JOINT_COUNT - 1;
const DEFAULT_FPS = 20;

export interface MotionClip {
  positions: Float32Array;
  contacts?: Uint8Array;
  frameCount: number;
  fps: number;
}

export interface PlaybackState {
  frame: number;
  frameCount: number;
  fps: number;
  playing: boolean;
  speed: number;
}

export type PlaybackListener = (state: PlaybackState) => void;

type MotionPayload = {
  posedJoints?: unknown;
  posed_joints?: unknown;
  positions?: unknown;
  footContacts?: unknown;
  foot_contacts?: unknown;
  frameCount?: unknown;
  numFrames?: unknown;
  fps?: unknown;
  motion?: unknown;
};

function isArrayBufferView(value: unknown): value is ArrayBufferView {
  return ArrayBuffer.isView(value);
}

function toFloat32Array(value: unknown, label: string): Float32Array {
  if (value instanceof Float32Array) return value;
  if (value instanceof ArrayBuffer) return new Float32Array(value);
  if (isArrayBufferView(value)) {
    const numbers = Array.from(value as unknown as ArrayLike<number>, Number);
    return Float32Array.from(numbers);
  }
  if (Array.isArray(value)) {
    const flattened: number[] = [];
    const visit = (item: unknown): void => {
      if (Array.isArray(item)) {
        item.forEach(visit);
      } else if (typeof item === "number") {
        flattened.push(item);
      } else {
        throw new TypeError(`${label} contains a non-numeric value.`);
      }
    };
    visit(value);
    return Float32Array.from(flattened);
  }
  throw new TypeError(`${label} must be a Float32Array, ArrayBuffer, or numeric array.`);
}

function toUint8Array(value: unknown, label: string): Uint8Array {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (isArrayBufferView(value)) {
    return Uint8Array.from(Array.from(value as unknown as ArrayLike<number>, Number), (item) =>
      item > 0.5 ? 1 : 0,
    );
  }
  if (Array.isArray(value)) {
    const flattened: number[] = [];
    const visit = (item: unknown): void => {
      if (Array.isArray(item)) {
        item.forEach(visit);
      } else if (typeof item === "number" || typeof item === "boolean") {
        flattened.push(Number(item));
      } else {
        throw new TypeError(`${label} contains a non-numeric value.`);
      }
    };
    visit(value);
    return Uint8Array.from(flattened, (item) => (item > 0.5 ? 1 : 0));
  }
  throw new TypeError(`${label} must be a typed array, ArrayBuffer, or numeric array.`);
}

function finitePositiveNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : fallback;
}

/**
 * Normalize the small set of motion-result spellings used by the Python API
 * and browser worker into the single rendering contract.
 */
export function normalizeMotionClip(input: unknown): MotionClip {
  if (!input || typeof input !== "object") {
    throw new TypeError("The inference worker returned an invalid motion result.");
  }

  let payload = input as MotionPayload;
  if (payload.motion && typeof payload.motion === "object") {
    payload = payload.motion as MotionPayload;
  }

  const rawPositions = payload.posedJoints ?? payload.posed_joints ?? payload.positions;
  const positions = toFloat32Array(rawPositions, "Motion joint positions");
  if (positions.length === 0 || positions.length % POSITION_COMPONENTS_PER_FRAME !== 0) {
    throw new RangeError(
      `Motion joint positions must contain T × ${CORE27_JOINT_COUNT} × 3 values; received ${positions.length}.`,
    );
  }
  for (let index = 0; index < positions.length; index += 1) {
    if (!Number.isFinite(positions[index])) {
      throw new RangeError(`Motion joint positions contain a non-finite value at index ${index}.`);
    }
  }

  const inferredFrameCount = positions.length / POSITION_COMPONENTS_PER_FRAME;
  const statedFrameCount = Number(payload.frameCount ?? payload.numFrames ?? inferredFrameCount);
  if (!Number.isInteger(statedFrameCount) || statedFrameCount !== inferredFrameCount) {
    throw new RangeError(
      `Motion frame count ${String(payload.frameCount ?? payload.numFrames)} does not match ${inferredFrameCount} frames of data.`,
    );
  }

  const rawContacts = payload.footContacts ?? payload.foot_contacts;
  const contacts = rawContacts === undefined ? undefined : toUint8Array(rawContacts, "Foot contacts");
  if (contacts && contacts.length !== inferredFrameCount * 4) {
    throw new RangeError(`Foot contacts must contain T × 4 values; received ${contacts.length}.`);
  }

  return {
    positions,
    contacts,
    frameCount: inferredFrameCount,
    fps: finitePositiveNumber(payload.fps, DEFAULT_FPS),
  };
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

export class SkeletonViewer {
  private readonly canvas: HTMLCanvasElement;
  private readonly scene: THREE.Scene;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly controls: OrbitControls;
  private readonly joints: THREE.InstancedMesh;
  private readonly bones: THREE.InstancedMesh;
  private readonly trajectory: THREE.Line;
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

  private clip: MotionClip | null = null;
  private frameCursor = 0;
  private playing = false;
  private loop = true;
  private speed = 1;
  private lastAnimationTime: number | null = null;
  private lastReportedFrame = -1;
  private animationFrame = 0;
  private resizeFrame = 0;
  private needsRender = true;
  private pageVisible = !document.hidden;
  private listener: PlaybackListener | null = null;

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

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
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

    this.joints = new THREE.InstancedMesh(
      new THREE.SphereGeometry(0.027, 18, 12),
      new THREE.MeshStandardMaterial({
        color: "#ffffff",
        roughness: 0.38,
        metalness: 0.12,
      }),
      CORE27_JOINT_COUNT,
    );
    this.joints.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.joints.castShadow = true;
    this.joints.frustumCulled = false;
    this.scene.add(this.joints);

    this.bones = new THREE.InstancedMesh(
      new THREE.CapsuleGeometry(0.014, 1, 4, 8),
      new THREE.MeshStandardMaterial({
        color: "#67c6c4",
        roughness: 0.5,
        metalness: 0.08,
      }),
      BONE_COUNT,
    );
    this.bones.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.bones.castShadow = true;
    this.bones.frustumCulled = false;
    this.scene.add(this.bones);

    this.trajectory = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({
        color: "#6cffbd",
        transparent: true,
        opacity: 0.32,
      }),
    );
    this.scene.add(this.trajectory);

    this.joints.visible = false;
    this.bones.visible = false;
    this.trajectory.visible = false;

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
    this.listener = listener;
    this.emitPlaybackState(true);
  }

  setMotion(clip: MotionClip, autoplay = true): void {
    this.clip = clip;
    this.frameCursor = 0;
    this.playing = autoplay && clip.frameCount > 1;
    this.lastAnimationTime = null;
    this.lastReportedFrame = -1;
    this.joints.visible = true;
    this.bones.visible = true;
    this.buildTrajectory(clip);
    this.updatePose(0);
    this.resetCamera();
    this.invalidate();
    this.emitPlaybackState(true);
  }

  clearMotion(): void {
    this.clip = null;
    this.playing = false;
    this.frameCursor = 0;
    this.lastAnimationTime = null;
    this.joints.visible = false;
    this.bones.visible = false;
    this.trajectory.visible = false;
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
    this.controls.enableDamping = !reduced;
    if (reduced) this.lastAnimationTime = null;
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
    for (let frame = 0; frame < this.clip.frameCount; frame += Math.max(1, Math.floor(this.clip.frameCount / 80))) {
      const rootOffset = frame * POSITION_COMPONENTS_PER_FRAME;
      for (let joint = 0; joint < CORE27_JOINT_COUNT; joint += 1) {
        const index = rootOffset + joint * 3;
        point.fromArray(this.clip.positions, index);
        box.expandByPoint(point);
      }
    }
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const distance = Math.max(2.5, Math.max(size.x, size.y, size.z) * 1.85);
    this.controls.target.set(center.x, Math.max(0.7, center.y), center.z);
    this.camera.position.set(center.x + distance * 0.78, center.y + distance * 0.48, center.z + distance);
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
      fps: this.clip?.fps ?? DEFAULT_FPS,
      playing: this.playing,
      speed: this.speed,
    };
  }

  dispose(): void {
    cancelAnimationFrame(this.animationFrame);
    cancelAnimationFrame(this.resizeFrame);
    this.resizeObserver.disconnect();
    document.removeEventListener("visibilitychange", this.handleVisibilityChange);
    this.controls.removeEventListener("change", this.invalidate);
    this.controls.dispose();
    this.joints.geometry.dispose();
    (this.joints.material as THREE.Material).dispose();
    this.bones.geometry.dispose();
    (this.bones.material as THREE.Material).dispose();
    this.trajectory.geometry.dispose();
    (this.trajectory.material as THREE.Material).dispose();
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

  private updatePose(frameCursor: number): void {
    const clip = this.clip;
    if (!clip) return;
    const frame0 = Math.min(Math.floor(frameCursor), clip.frameCount - 1);
    const frame1 = Math.min(frame0 + 1, clip.frameCount - 1);
    const alpha = frame1 === frame0 ? 0 : frameCursor - frame0;
    const offset0 = frame0 * POSITION_COMPONENTS_PER_FRAME;
    const offset1 = frame1 * POSITION_COMPONENTS_PER_FRAME;

    const setPoint = (target: THREE.Vector3, joint: number): void => {
      const jointOffset = joint * 3;
      const x0 = clip.positions[offset0 + jointOffset];
      const y0 = clip.positions[offset0 + jointOffset + 1];
      const z0 = clip.positions[offset0 + jointOffset + 2];
      target.set(
        THREE.MathUtils.lerp(x0, clip.positions[offset1 + jointOffset], alpha),
        THREE.MathUtils.lerp(y0, clip.positions[offset1 + jointOffset + 1], alpha),
        THREE.MathUtils.lerp(z0, clip.positions[offset1 + jointOffset + 2], alpha),
      );
    };

    const contactsOffset = frame0 * 4;
    for (let joint = 0; joint < CORE27_JOINT_COUNT; joint += 1) {
      setPoint(this.currentPoint, joint);
      this.jointTransform.position.copy(this.currentPoint);
      this.jointTransform.scale.setScalar(joint === 0 ? 1.25 : 1);
      this.jointTransform.quaternion.identity();
      this.jointTransform.updateMatrix();
      this.joints.setMatrixAt(joint, this.jointTransform.matrix);

      const contactIndex = CORE27_FOOT_CONTACT_JOINTS.indexOf(joint);
      const inContact = contactIndex >= 0 && Boolean(clip.contacts?.[contactsOffset + contactIndex]);
      this.joints.setColorAt(joint, inContact ? this.contactJointColor : this.regularJointColor);
    }

    let boneIndex = 0;
    for (let joint = 1; joint < CORE27_JOINT_COUNT; joint += 1) {
      const parent = CORE27_PARENTS[joint];
      setPoint(this.currentPoint, joint);
      setPoint(this.parentPoint, parent);
      this.direction.subVectors(this.currentPoint, this.parentPoint);
      const length = Math.max(this.direction.length(), 0.0001);
      this.midpoint.addVectors(this.currentPoint, this.parentPoint).multiplyScalar(0.5);
      this.boneTransform.position.copy(this.midpoint);
      this.boneTransform.quaternion.setFromUnitVectors(this.upAxis, this.direction.normalize());
      // CapsuleGeometry is one unit plus two radii; the small correction keeps
      // adjacent joint spheres and bones visually connected.
      this.boneTransform.scale.set(1, Math.max(0.001, length - 0.028), 1);
      this.boneTransform.updateMatrix();
      this.bones.setMatrixAt(boneIndex, this.boneTransform.matrix);
      boneIndex += 1;
    }

    this.joints.instanceMatrix.needsUpdate = true;
    if (this.joints.instanceColor) this.joints.instanceColor.needsUpdate = true;
    this.bones.instanceMatrix.needsUpdate = true;
  }

  private buildTrajectory(clip: MotionClip): void {
    const points: THREE.Vector3[] = [];
    for (let frame = 0; frame < clip.frameCount; frame += 1) {
      const offset = frame * POSITION_COMPONENTS_PER_FRAME;
      points.push(new THREE.Vector3(clip.positions[offset], 0.008, clip.positions[offset + 2]));
    }
    this.trajectory.geometry.dispose();
    this.trajectory.geometry = new THREE.BufferGeometry().setFromPoints(points);
    this.trajectory.visible = points.length > 1;
  }

  private emitPlaybackState(force: boolean): void {
    if (!this.listener) return;
    const state = this.getPlaybackState();
    if (!force && state.frame === this.lastReportedFrame) return;
    this.lastReportedFrame = state.frame;
    this.listener(state);
  }
}
