// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

export const LOOP_CONTROL_STATE_EVENT = "ardy:loop-control-state";
export const LOOP_CONTROL_CHANGE_EVENT = "ardy:loop-control-change";

export interface LoopControlState {
  pressed?: boolean;
  disabled?: boolean;
}
