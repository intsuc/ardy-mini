// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from "react"
import {
  IconCameraRotate,
  IconPlayerPause,
  IconPlayerPlay,
  IconRefresh,
  IconRepeat,
  IconUpload,
  IconX,
} from "@tabler/icons-react"

import {
  Alert,
  AlertAction,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert"
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { Badge } from "@/components/ui/badge"
import { Button, ButtonLink } from "@/components/ui/button"
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyTitle,
} from "@/components/ui/empty"
import {
  Field,
  FieldDescription,
  FieldError,
  FieldGroup,
  FieldLabel,
  FieldLegend,
  FieldSet,
} from "@/components/ui/field"
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
  InputGroupText,
  InputGroupTextarea,
} from "@/components/ui/input-group"
import { Label } from "@/components/ui/label"
import {
  NativeSelect,
  NativeSelectOption,
} from "@/components/ui/native-select"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Spinner } from "@/components/ui/spinner"
import { Toggle } from "@/components/ui/toggle"
import {
  BoundCheckbox,
  BoundProgress,
  BoundSlider,
  BoundSwitch,
} from "@/components/control-bindings"
import {
  PROMPT_EXAMPLES,
  PROMPT_EXAMPLE_EVENT,
} from "@/prompt-examples"
import { bootstrap } from "@/main"
import {
  continuousGenerationControl,
  durationControl,
  generationProgressControl,
  loopControl,
  modelProgressControl,
  previewSettingsControl,
  removeSavedModelAction,
  showContactsControl,
  showOrientationsControl,
  showSkeletonControl,
  showTrajectoryControl,
  showVrmControl,
  targetBufferControl,
  timelineControl,
  useControlState,
} from "@/ui-control-store"

function EmptyFieldError({
  id,
}: {
  id: string
}) {
  return (
    <FieldError id={id} aria-live="polite">
      <span />
    </FieldError>
  )
}

function selectPrompt(prompt: string) {
  document.dispatchEvent(
    new CustomEvent<string>(PROMPT_EXAMPLE_EVENT, {
      detail: prompt,
    })
  )
}

const PROMPT_SELECT_ITEMS = PROMPT_EXAMPLES.map(({ label, prompt }) => ({
  label,
  value: prompt,
}))

function PromptExampleSelect() {
  return (
    <Field className="min-w-0">
      <FieldLabel id="prompt-example-label">
        Example prompt
      </FieldLabel>
      <Select
        items={PROMPT_SELECT_ITEMS}
        onValueChange={(value) => {
          if (typeof value === "string") selectPrompt(value)
        }}
      >
        <SelectTrigger
          id="prompt-example"
          className="w-full min-w-0"
          aria-labelledby="prompt-example-label"
        >
          <SelectValue placeholder="Choose an example" />
        </SelectTrigger>
        <SelectContent alignItemWithTrigger={false} align="start">
          <SelectGroup>
            <SelectLabel>Motion examples</SelectLabel>
            {PROMPT_EXAMPLES.map((example) => (
              <SelectItem
                key={example.label}
                value={example.prompt}
              >
                {example.label}
              </SelectItem>
            ))}
          </SelectGroup>
        </SelectContent>
      </Select>
    </Field>
  )
}

function AppHeader() {
  return (
    <header className="relative z-10 grid min-h-15 grid-cols-[auto_minmax(0,1fr)] items-center gap-4 border-b bg-background px-3 py-2 max-[760px]:grid-cols-1 max-[520px]:gap-2 max-[520px]:px-2.5">
      <strong className="text-sm leading-tight">
        ARDY Mini
      </strong>

      <div
        className="flex min-w-0 items-center justify-end gap-1.5 overflow-hidden max-[760px]:justify-start max-[760px]:overflow-x-auto"
        id="model-runtime-status"
        aria-label="Model and runtime status"
      >
        <Badge variant="outline" id="privacy-badge">
          <span
            className="size-1.5 shrink-0 rounded-full bg-primary"
            aria-hidden="true"
          />
          Local
        </Badge>
        <Separator
          orientation="vertical"
          role="none"
          aria-hidden="true"
        />
        <Badge variant="outline" id="gpu-badge">
          <span
            className="size-1.5 shrink-0 rounded-full bg-muted-foreground data-[state=available]:bg-primary"
            id="gpu-dot"
            aria-hidden="true"
          />
          <span id="gpu-label">Checking WebGPU</span>
        </Badge>
        <Separator
          orientation="vertical"
          role="none"
          aria-hidden="true"
        />
        <Badge variant="outline" id="isolation-badge">
          <span
            className="size-1.5 shrink-0 rounded-full bg-muted-foreground data-[state=available]:bg-primary"
            id="isolation-dot"
            aria-hidden="true"
          />
          <span id="isolation-label">Checking WASM threads</span>
        </Badge>
        <span className="sr-only" id="model-runtime-detail">
          Model, WebGPU, and WebAssembly runtime readiness.
        </span>
      </div>
    </header>
  )
}

function ModelSection() {
  const importModelRef = useRef<HTMLButtonElement>(null)
  const removeDialogCancelRef = useRef<HTMLButtonElement>(null)
  const focusImportAfterCloseRef = useRef(false)
  const [removeDialogOpen, setRemoveDialogOpen] = useState(false)

  return (
    <section
      className="model-setup flex flex-col gap-3 border-b p-3"
      aria-labelledby="model-step-title"
    >
      <h2
        className="text-xs font-medium text-muted-foreground"
        id="model-step-title"
      >
        Model
      </h2>

      <Card id="model-card" size="sm">
        <CardHeader>
          <CardTitle className="flex min-w-0 items-center gap-2">
            <span
              className="size-1.5 shrink-0 rounded-full bg-muted-foreground group-data-[state=ready]/card:bg-primary"
              aria-hidden="true"
            />
            <span className="truncate" id="model-title">
              Model pack required
            </span>
          </CardTitle>
          <CardDescription id="model-detail">
            Choose an exported Core40 browser-pack folder.
          </CardDescription>
          <CardAction>
            <Badge
              variant="outline"
              id="model-state"
              data-state="missing"
            >
              Not loaded
            </Badge>
          </CardAction>
          <div
            className="col-span-full mt-1 grid grid-cols-[minmax(3rem,1fr)_auto] items-center gap-1.5"
            id="model-progress"
            hidden
          >
            <BoundProgress
              control={modelProgressControl}
              aria-label="Model loading progress"
            />
            <span
              className="text-xs text-muted-foreground"
              id="model-progress-label"
            />
          </div>
        </CardHeader>
        <CardFooter id="model-setup-help">
          <CardDescription>
            Select <code>artifacts/browser/core40</code> (about 1.4 GiB,
            four ONNX graphs). The pack is stored only in this browser.{" "}
            <ButtonLink
              variant="link"
              size="xs"
              href="https://github.com/intsuc/ardy-mini#fully-in-browser-minilm-demo"
              target="_blank"
              rel="noreferrer"
            >
              Export instructions
            </ButtonLink>
          </CardDescription>
        </CardFooter>
      </Card>

      <div className="grid min-w-0 gap-1.5 min-[360px]:grid-cols-[minmax(0,1fr)_auto]">
        <Button
          ref={importModelRef}
          id="import-model"
          className="min-w-0"
          variant="secondary"
          size="lg"
          type="button"
        >
          <span id="import-model-label">Choose model pack</span>
        </Button>
        <input
          id="model-file-input"
          type="file"
          multiple
          hidden
          aria-hidden="true"
        />
        <AlertDialog
          open={removeDialogOpen}
          onOpenChange={(open) => setRemoveDialogOpen(open)}
        >
          <AlertDialogTrigger
            render={
              <Button
                id="remove-model"
                className="min-w-0"
                variant="destructive"
                size="lg"
                type="button"
                hidden
              >
                Remove saved pack
              </Button>
            }
          />
          <AlertDialogContent
            size="sm"
            initialFocus={removeDialogCancelRef}
            finalFocus={() => {
              if (!focusImportAfterCloseRef.current) return true
              focusImportAfterCloseRef.current = false
              return importModelRef.current
            }}
          >
            <AlertDialogHeader>
              <AlertDialogTitle>
                Remove the saved model pack?
              </AlertDialogTitle>
              <AlertDialogDescription>
                This unloads the model and removes its browser copy.
                Generated motion remains available in the preview.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel ref={removeDialogCancelRef}>
                Cancel
              </AlertDialogCancel>
              <AlertDialogAction
                id="confirm-remove-model"
                variant="destructive"
                onClick={() => {
                  focusImportAfterCloseRef.current = true
                  setRemoveDialogOpen(false)
                  removeSavedModelAction.trigger()
                }}
              >
                Remove saved pack
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>

      <Alert
        id="model-error-banner"
        variant="destructive"
        tabIndex={-1}
        hidden
      >
        <AlertTitle id="model-error-title">
          Model import failed
        </AlertTitle>
        <AlertDescription id="model-error-message" />
        <AlertAction>
          <Button
            id="dismiss-model-error"
            variant="ghost"
            size="icon-lg"
            type="button"
            aria-label="Dismiss model error"
          >
            <IconX data-icon="inline-start" aria-hidden="true" />
          </Button>
        </AlertAction>
      </Alert>
    </section>
  )
}

function PromptSection() {
  return (
    <section
      className="flex flex-col gap-3 border-b p-3"
      aria-labelledby="prompt-step-title"
    >
      <div className="flex items-center justify-between gap-3">
        <h2
          className="text-xs font-medium text-muted-foreground"
          id="prompt-step-title"
        >
          Prompt
        </h2>
      </div>

      <FieldGroup>
        <Field>
          <FieldLabel htmlFor="prompt">
            Motion description
          </FieldLabel>
          <InputGroup>
            <InputGroupTextarea
              id="prompt"
              name="prompt"
              rows={3}
              maxLength={280}
              spellCheck
              autoComplete="off"
              placeholder="A person walks forward, then waves with their right hand."
              aria-describedby="prompt-count prompt-error"
              required
            />
            <InputGroupAddon align="block-end">
              <InputGroupText id="prompt-count">
                0 / 280
              </InputGroupText>
            </InputGroupAddon>
          </InputGroup>
          <EmptyFieldError id="prompt-error" />
        </Field>
      </FieldGroup>

      <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-end gap-1.5">
        <PromptExampleSelect />
        <Button
          id="apply-prompt"
          variant="secondary"
          type="button"
        >
          Apply live
        </Button>
      </div>
    </section>
  )
}

function ClipSection() {
  const continuousGeneration = useControlState(
    continuousGenerationControl
  )

  return (
    <section
      className="flex flex-col gap-3 border-b p-3"
      aria-labelledby="basic-settings-title"
    >
      <div className="flex items-center justify-between gap-3">
        <h2
          className="text-xs font-medium text-muted-foreground"
          id="basic-settings-title"
        >
          Clip
        </h2>
        <output
          id="duration-output"
          className="text-xs text-muted-foreground"
        >
          4 seconds
        </output>
      </div>

      <FieldGroup>
        <Field>
          <FieldLabel id="duration-label">Duration</FieldLabel>
          <BoundSlider
            control={durationControl}
            aria-labelledby="duration-label"
          />
          <div
            className="flex justify-between text-xs text-muted-foreground"
            aria-hidden="true"
          >
            <span>2s</span>
            <span>10s</span>
          </div>
        </Field>

        <FieldGroup className="grid grid-cols-2 gap-2.5 max-[520px]:grid-cols-1">
          <Field>
            <FieldLabel htmlFor="seed">Seed</FieldLabel>
            <InputGroup>
              <InputGroupInput
                id="seed"
                name="seed"
                type="number"
                min="0"
                max="4294967295"
                step="1"
                inputMode="numeric"
                defaultValue="2"
                aria-describedby="seed-error"
              />
              <InputGroupAddon align="inline-end">
                <InputGroupButton
                  id="randomize-seed"
                  size="icon-xs"
                  aria-label="Choose a random seed"
                  title="Random seed"
                >
                  <IconRefresh
                    data-icon="inline-start"
                    aria-hidden="true"
                  />
                </InputGroupButton>
              </InputGroupAddon>
            </InputGroup>
            <EmptyFieldError id="seed-error" />
          </Field>

          <Field>
            <FieldLabel htmlFor="backend">Backend</FieldLabel>
            <NativeSelect
              id="backend"
              name="backend"
              aria-describedby="backend-help"
              defaultValue="auto"
            >
              <NativeSelectOption value="auto">
                Auto
              </NativeSelectOption>
              <NativeSelectOption value="webgpu">
                WebGPU
              </NativeSelectOption>
              <NativeSelectOption value="wasm">
                WASM
              </NativeSelectOption>
            </NativeSelect>
            <FieldDescription
              className="sr-only"
              id="backend-help"
            >
              Auto prefers WebGPU and falls back to WebAssembly.
            </FieldDescription>
          </Field>
        </FieldGroup>

        <Field
          data-disabled={continuousGeneration.disabled || undefined}
          orientation="horizontal"
        >
          <FieldLabel htmlFor={continuousGenerationControl.id}>
            Continuous generation
          </FieldLabel>
          <BoundSwitch control={continuousGenerationControl} />
        </Field>

        <Field>
          <div className="flex items-center justify-between gap-3">
            <FieldLabel id="target-buffer-label">
              Target buffer
            </FieldLabel>
            <output
              className="text-xs text-muted-foreground"
              id="target-buffer-output"
            >
              80 frames
            </output>
          </div>
          <BoundSlider
            control={targetBufferControl}
            aria-labelledby="target-buffer-label"
          />
          <div
            className="flex justify-between text-xs text-muted-foreground"
            aria-hidden="true"
          >
            <span>40</span>
            <span>200 frames</span>
          </div>
        </Field>
      </FieldGroup>
    </section>
  )
}

function GenerationActionsSection() {
  return (
    <section
      className="sticky bottom-0 z-10 flex flex-col gap-3 border-b bg-background p-3 max-[760px]:static"
      aria-label="Generation actions"
    >
      <Card
        id="generation-progress"
        size="sm"
        data-state="idle"
      >
        <CardHeader>
          <CardTitle className="truncate" id="generation-stage">
            Waiting for model
          </CardTitle>
          <CardAction>
            <span
              className="text-xs text-muted-foreground tabular-nums"
              id="generation-percent"
            >
              —
            </span>
          </CardAction>
          <CardDescription
            className="sr-only"
            id="generate-help"
          >
            Load a model pack to enable generation.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <BoundProgress
            control={generationProgressControl}
            aria-label="Generation progress"
          />
          <span className="sr-only" id="generation-note">
            The first run may compile GPU pipelines.
          </span>
        </CardContent>
      </Card>

      <Button
        id="generate"
        size="lg"
        type="submit"
        aria-describedby="generate-help"
        aria-keyshortcuts="Control+Enter Meta+Enter"
        disabled
      >
        <Spinner
          id="generate-spinner"
          className="hidden"
          data-icon="inline-start"
          aria-hidden="true"
        />
        <span id="generate-label">
          Generate motion
        </span>
      </Button>

      <div className="flex flex-wrap items-center gap-1.5 [&>*]:flex-auto">
        <Button
          id="restart-generation"
          variant="secondary"
          type="button"
        >
          Restart
        </Button>
        <Button
          id="restart-from-now"
          variant="secondary"
          type="button"
        >
          Restart from now
        </Button>
        <Button
          id="cancel-generation"
          className="invisible data-[state=active]:visible"
          variant="destructive"
          type="button"
          data-state="idle"
          aria-hidden="true"
          tabIndex={-1}
          disabled
        >
          Cancel
        </Button>
      </div>
    </section>
  )
}

function GenerationMessages() {
  return (
    <Alert
      className="m-3"
      id="error-banner"
      variant="destructive"
      tabIndex={-1}
      hidden
    >
      <AlertTitle id="error-title">Generation failed</AlertTitle>
      <AlertDescription id="error-message" />
      <AlertAction>
        <Button
          id="dismiss-error"
          variant="ghost"
          size="icon-lg"
          type="button"
          aria-label="Dismiss generation error"
        >
          <IconX data-icon="inline-start" aria-hidden="true" />
        </Button>
      </AlertAction>
    </Alert>
  )
}

function LoopToggle() {
  const state = useControlState(loopControl)

  return (
    <Toggle
      id="loop-toggle"
      variant="outline"
      size="lg"
      pressed={state.pressed}
      disabled={state.disabled}
      onPressedChange={loopControl.commit}
      aria-label="Loop playback"
    >
      <IconRepeat data-icon="inline-start" aria-hidden="true" />
    </Toggle>
  )
}

function ViewportPanel() {
  return (
    <section
      className="viewport-panel"
      id="viewport-panel"
      aria-labelledby="preview-title"
    >
      <div className="flex min-h-15 items-center justify-between gap-3 border-b bg-background px-3 py-2.5">
        <div>
          <p className="text-xs text-muted-foreground">Output</p>
          <h2 className="text-sm font-medium" id="preview-title">
            3D preview
          </h2>
        </div>
        <div className="flex min-w-0 flex-wrap items-center justify-end gap-1.5">
          <Badge
            variant="outline"
            id="motion-badge"
          >
            No motion
          </Badge>
          <Badge
            variant="outline"
            id="runtime-metric"
            hidden
          >
            <strong id="runtime-value">—</strong>
          </Badge>
        </div>
      </div>

      <PreviewSettingsSection />

      <div className="viewport" id="viewport">
        <canvas
          id="motion-canvas"
          aria-label="Interactive 3D motion preview"
          aria-describedby="viewer-help"
          aria-keyshortcuts="W A S D Space ArrowLeft ArrowRight Shift+ArrowLeft Shift+ArrowRight Shift+ArrowUp Shift+ArrowDown Home"
          tabIndex={0}
        >
          Interactive Core27 skeleton preview.
        </canvas>
        <p className="sr-only" id="viewer-help">
          Drag or swipe to orbit and scroll or pinch to zoom. With the
          preview focused, press Space to play or pause, Left and Right
          Arrow to seek, W A S D to move the camera, Shift plus Arrow keys
          to orbit, Plus or Minus to zoom, and Home to reset the camera.
        </p>

        <Empty
          className="pointer-events-none absolute top-1/2 left-1/2 w-[min(22rem,calc(100%_-_2rem))] -translate-x-1/2 -translate-y-1/2"
          id="empty-state"
        >
          <EmptyHeader>
            <EmptyTitle>No motion loaded</EmptyTitle>
            <EmptyDescription>
              Load the Core40 model, enter a prompt, then generate.
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      </div>

      <div
        className="playback-bar"
        id="playback-bar"
        role="group"
        aria-label="Playback controls"
      >
        <Button
          id="play-pause"
          className="group/play"
          size="icon-lg"
          type="button"
          aria-label="Play motion"
          disabled
        >
          <IconPlayerPlay
            className="group-data-[playing=true]/play:hidden"
            data-icon="inline-start"
            aria-hidden="true"
          />
          <IconPlayerPause
            className="hidden group-data-[playing=true]/play:block"
            data-icon="inline-start"
            aria-hidden="true"
          />
        </Button>
        <span
          className="min-w-15 text-xs tabular-nums"
          id="current-time"
        >
          00:00.00
        </span>
        <BoundSlider
          control={timelineControl}
          className="min-w-12 max-[520px]:col-start-2"
          aria-label="Motion timeline"
        />
        <span
          className="min-w-15 text-xs text-muted-foreground tabular-nums"
          id="total-time"
        >
          00:00.00
        </span>
        <Label className="speed-control">
          <span className="sr-only">Playback speed</span>
          <NativeSelect
            id="playback-speed"
            aria-label="Playback speed"
            defaultValue="1"
            disabled
          >
            <NativeSelectOption value="0.5">0.5×</NativeSelectOption>
            <NativeSelectOption value="1">1×</NativeSelectOption>
            <NativeSelectOption value="2">2×</NativeSelectOption>
          </NativeSelect>
        </Label>
        <LoopToggle />
        <Button
          id="reset-camera"
          variant="ghost"
          size="icon-lg"
          type="button"
          aria-label="Reset camera"
        >
          <IconCameraRotate
            data-icon="inline-start"
            aria-hidden="true"
          />
        </Button>
      </div>
    </section>
  )
}

function VrmAvatarSection() {
  const showVrm = useControlState(showVrmControl)

  return (
    <section
      className="grid gap-3"
      aria-labelledby="vrm-avatar-title"
    >
      <div className="flex items-center justify-between gap-3">
        <h3
          className="text-xs font-medium text-muted-foreground"
          id="vrm-avatar-title"
        >
          VRM avatar
        </h3>
        <Badge id="vrm-state" variant="outline" data-state="missing">
          Optional
        </Badge>
      </div>

      <Card id="vrm-card" size="sm">
        <CardHeader>
          <CardTitle className="truncate" id="vrm-name">
            No avatar loaded
          </CardTitle>
          <CardDescription id="vrm-detail">
            Load a VRM 0.x or 1.0 file for local preview.
          </CardDescription>
        </CardHeader>
        <CardFooter className="grid grid-cols-2 gap-1.5">
          <Button
            id="import-vrm"
            variant="secondary"
            type="button"
          >
            <IconUpload data-icon="inline-start" aria-hidden="true" />
            <span id="import-vrm-label">Load VRM</span>
          </Button>
          <input
            id="vrm-file-input"
            type="file"
            accept=".vrm,model/gltf-binary,application/octet-stream"
            hidden
            aria-hidden="true"
          />
          <Button
            id="remove-vrm"
            variant="destructive"
            type="button"
            disabled
          >
            Remove avatar
          </Button>
        </CardFooter>
      </Card>

      <Field
        orientation="horizontal"
        data-disabled={showVrm.disabled || undefined}
      >
        <FieldLabel htmlFor={showVrmControl.id}>
          Show VRM avatar
        </FieldLabel>
        <BoundSwitch control={showVrmControl} />
      </Field>

      <Alert
        id="vrm-error-banner"
        variant="destructive"
        tabIndex={-1}
        hidden
      >
        <AlertTitle>VRM import failed</AlertTitle>
        <AlertDescription id="vrm-error-message" />
        <AlertAction>
          <Button
            id="dismiss-vrm-error"
            variant="ghost"
            size="icon-lg"
            type="button"
            aria-label="Dismiss VRM error"
          >
            <IconX data-icon="inline-start" aria-hidden="true" />
          </Button>
        </AlertAction>
      </Alert>
    </section>
  )
}

function DisplayControls() {
  return (
    <FieldSet>
      <FieldLegend variant="label">Display</FieldLegend>
      <FieldGroup
        data-slot="checkbox-group"
        className="grid grid-cols-2 max-[520px]:grid-cols-1"
      >
        {([
          [showSkeletonControl, "Skeleton"],
          [showContactsControl, "Foot contacts"],
          [showOrientationsControl, "Orientations"],
          [showTrajectoryControl, "Root trajectory"],
        ] as const).map(([control, label]) => (
          <Field
            orientation="horizontal"
            key={control.id}
          >
            <BoundCheckbox control={control} />
            <FieldLabel htmlFor={control.id}>{label}</FieldLabel>
          </Field>
        ))}
      </FieldGroup>
    </FieldSet>
  )
}

function PreviewSettingsSection() {
  const state = useControlState(previewSettingsControl)

  return (
    <Accordion
      id={previewSettingsControl.id}
      className="border-b px-3"
      value={state.open ? ["view-settings"] : []}
      onValueChange={(value) =>
        previewSettingsControl.commit(value.includes("view-settings"))
      }
    >
      <AccordionItem value="view-settings">
        <AccordionTrigger id="preview-settings-trigger">
          View settings
        </AccordionTrigger>
        <AccordionContent keepMounted className="grid gap-3 pb-3">
          <VrmAvatarSection />
          <Separator />
          <DisplayControls />
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  )
}

function GenerationPanel() {
  return (
    <aside
      className="panel generation-panel"
      id="generator-panel"
      aria-labelledby="generation-panel-title"
      tabIndex={-1}
    >
      <div className="sticky top-0 z-10 flex min-h-15 items-center gap-3 border-b bg-background px-3 py-2.5 max-[760px]:static">
        <div>
          <p className="text-xs text-muted-foreground">Input</p>
          <h1
            className="text-sm font-medium"
            id="generation-panel-title"
          >
            Motion generation
          </h1>
        </div>
      </div>

      <form id="generation-form" noValidate>
        <ModelSection />

        <PromptSection />

        <ClipSection />

        <GenerationActionsSection />

        <GenerationMessages />
      </form>
    </aside>
  )
}

export function App() {
  useEffect(() => bootstrap(), [])

  return (
    <div id="app">
      <a className="skip-link" href="#prompt">
        Skip to generation controls
      </a>

      <AppHeader />

      <p
        className="sr-only"
        id="app-status"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      />

      <main className="workspace">
        <GenerationPanel />

        <ViewportPanel />
      </main>
    </div>
  )
}

export default App
