// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

export type MotionConstraintKind =
  | "root"
  | "full-body"
  | "left-hand"
  | "right-hand"
  | "left-foot"
  | "right-foot";

/**
 * A persisted constraint marker expressed in normalized ARDY motion features.
 *
 * The text-only browser runtime does not feed these values into ONNX. Sessions
 * retain them so imported edits and post-processing metadata remain readable.
 */
export interface MotionConstraint {
  id: string;
  kind: MotionConstraintKind;
  frame: number;
  endFrame?: number;
  values: Float32Array;
  mask: Float32Array;
}
