// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import * as React from "react"

import { Checkbox } from "@/components/ui/checkbox"
import { Progress } from "@/components/ui/progress"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import {
  type CheckedControlState,
  type ExternalControl,
  type ProgressControlState,
  type SliderControlState,
  useControlState,
} from "@/ui-control-store"

type CheckedControl = ExternalControl<CheckedControlState, boolean>
type ProgressControl = ExternalControl<ProgressControlState, number>
type SliderControl = ExternalControl<SliderControlState, number>

type BoundCheckboxProps = Omit<
  React.ComponentProps<typeof Checkbox>,
  "checked" | "defaultChecked" | "disabled" | "id" | "onCheckedChange"
> & {
  control: CheckedControl
}

function BoundCheckbox({
  control,
  ...props
}: BoundCheckboxProps) {
  const state = useControlState(control)

  return (
    <Checkbox
      {...props}
      id={control.id}
      checked={state.checked}
      disabled={state.disabled}
      onCheckedChange={control.commit}
    />
  )
}

type BoundSwitchProps = Omit<
  React.ComponentProps<typeof Switch>,
  "checked" | "defaultChecked" | "disabled" | "id" | "onCheckedChange"
> & {
  control: CheckedControl
}

function BoundSwitch({
  control,
  ...props
}: BoundSwitchProps) {
  const state = useControlState(control)

  return (
    <Switch
      {...props}
      id={control.id}
      checked={state.checked}
      disabled={state.disabled}
      onCheckedChange={control.commit}
    />
  )
}

type BoundProgressProps = Omit<
  React.ComponentProps<typeof Progress>,
  "id" | "value"
> & {
  control: ProgressControl
}

function BoundProgress({
  control,
  ...props
}: BoundProgressProps) {
  const state = useControlState(control)

  return (
    <Progress
      {...props}
      id={control.id}
      value={state.value}
    />
  )
}

type BoundSliderProps = Omit<
  React.ComponentProps<typeof Slider>,
  | "aria-valuetext"
  | "defaultValue"
  | "disabled"
  | "id"
  | "max"
  | "min"
  | "onValueChange"
  | "step"
  | "value"
> & {
  control: SliderControl
}

function BoundSlider({
  control,
  ...props
}: BoundSliderProps) {
  const state = useControlState(control)

  return (
    <Slider
      {...props}
      id={control.id}
      value={state.value}
      min={state.min}
      max={state.max}
      step={state.step}
      disabled={state.disabled}
      aria-valuetext={state.ariaValueText}
      onValueChange={(value) => {
        if (typeof value === "number") control.commit(value)
      }}
    />
  )
}

export { BoundCheckbox, BoundProgress, BoundSlider, BoundSwitch }
