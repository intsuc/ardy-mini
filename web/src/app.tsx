// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from "react"
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
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
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
} from "@/components/ui/field"
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
  InputGroupText,
  InputGroupTextarea,
} from "@/components/ui/input-group"
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
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import { Spinner } from "@/components/ui/spinner"
import {
  PROMPT_EXAMPLES,
  PROMPT_EXAMPLE_EVENT,
} from "@/prompt-examples"
import { bootstrap } from "@/main"

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

function PromptExampleSelect() {
  return (
    <Field className="min-w-0 gap-1.5">
      <FieldLabel id="prompt-example-label">
        Example prompt
      </FieldLabel>
      <Select onValueChange={selectPrompt}>
        <SelectTrigger
          id="prompt-example"
          className="w-full min-w-0"
          aria-labelledby="prompt-example-label"
        >
          <SelectValue placeholder="Choose an example" />
        </SelectTrigger>
        <SelectContent position="popper" align="start">
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
          decorative
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
          decorative
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
  return (
    <section
      className="model-setup flex flex-col gap-3 border-b p-3"
      aria-labelledby="model-step-title"
    >
      <div className="flex items-center justify-between gap-3">
        <h2
          className="text-xs font-medium text-muted-foreground"
          id="model-step-title"
        >
          Model
        </h2>
        <Badge variant="secondary">
          Core40
        </Badge>
      </div>

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
            <Progress
              id="model-progressbar"
              value={0}
              aria-label="Model loading progress"
            />
            <span
              className="text-xs text-muted-foreground"
              id="model-progress-label"
            />
          </div>
        </CardHeader>
        <CardFooter id="model-setup-help">
          <p className="setup-note">
            Select <code>artifacts/browser/core40</code> (about 1.4 GiB,
            four ONNX graphs). The pack is stored only in this browser.{" "}
            <a
              href="https://github.com/intsuc/ardy-mini#fully-in-browser-minilm-demo"
              target="_blank"
              rel="noreferrer"
            >
              Export instructions
            </a>
          </p>
        </CardFooter>
      </Card>

      <div className="grid min-w-0 gap-1.5 min-[360px]:grid-cols-[minmax(0,1fr)_auto]">
        <Button
          id="import-model"
          className="w-full min-w-0"
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
        <Button
          id="remove-model"
          className="w-full min-w-0"
          variant="destructive"
          size="lg"
          type="button"
          hidden
        >
          Remove saved pack
        </Button>
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
        <Field className="field-group">
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
          htmlFor="duration"
        >
          4 seconds
        </output>
      </div>

      <FieldGroup>
        <Field className="field-group">
          <FieldLabel htmlFor="duration">Duration</FieldLabel>
          <input
            id="duration"
            name="duration"
            type="range"
            min="2"
            max="10"
            step="2"
            defaultValue="4"
            aria-label="Duration"
            aria-valuetext="4 seconds"
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
          <Field className="field-group">
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

          <Field className="field-group">
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

        <label
          className="flex min-h-11 items-center justify-between gap-2.5"
          htmlFor="stream-generation"
        >
          <span className="text-xs font-medium">
            Continuous generation
          </span>
          <input
            id="stream-generation"
            type="checkbox"
            role="switch"
            defaultChecked
          />
        </label>

        <Field className="field-group">
          <div className="flex items-center justify-between gap-3">
            <FieldLabel htmlFor="target-buffer">
              Target buffer
            </FieldLabel>
            <output
              className="text-xs text-muted-foreground"
              htmlFor="target-buffer"
            >
              80 frames
            </output>
          </div>
          <input
            id="target-buffer"
            type="range"
            min="40"
            max="200"
            step="40"
            defaultValue="80"
            aria-label="Target generation buffer in frames"
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
          <Progress
            aria-label="Generation progress"
            id="generation-progressbar"
            value={0}
          />
          <span className="sr-only" id="generation-note">
            The first run may compile GPU pipelines.
          </span>
        </CardContent>
      </Card>

      <Button
        id="generate"
        className="w-full"
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
        <span
          className="text-xs opacity-70"
          id="button-shortcut"
          aria-hidden="true"
        >
          Ctrl ↵
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
    <>
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

      <details className="settings-details" id="runtime-settings">
        <summary>Runtime notes</summary>
        <div className="details-body">
          <p>
            Prompts, model files, avatars, and generated motion stay
            on this device.
          </p>
          <a
            href="./notices/THIRD_PARTY_MODELS_AND_DATA.md"
            target="_blank"
            rel="noreferrer"
          >
            Model and software notices
          </a>
        </div>
      </details>
    </>
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
          aria-label="Interactive 3D skeleton preview"
          aria-describedby="viewer-help"
          aria-keyshortcuts="Space ArrowLeft ArrowRight Shift+ArrowLeft Shift+ArrowRight Shift+ArrowUp Shift+ArrowDown Home"
          tabIndex={0}
        >
          Interactive Core27 skeleton preview.
        </canvas>
        <p className="sr-only" id="viewer-help">
          Drag or swipe to orbit and scroll or pinch to zoom. With the
          preview focused, press Space to play or pause, Left and Right
          Arrow to seek, Shift plus Arrow keys to orbit, Plus or Minus to
          zoom, and Home to reset the camera.
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
        <input
          id="timeline"
          className="timeline-slider"
          type="range"
          min="0"
          max="1"
          step="1"
          defaultValue="0"
          aria-label="Motion timeline"
          disabled
        />
        <span
          className="min-w-15 text-xs text-muted-foreground tabular-nums"
          id="total-time"
        >
          00:00.00
        </span>
        <label className="speed-control">
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
        </label>
        <Button
          id="loop-toggle"
          className="aria-pressed:bg-accent aria-pressed:text-accent-foreground"
          variant="outline"
          size="icon-lg"
          type="button"
          aria-pressed="true"
          aria-label="Loop playback"
          disabled
        >
          <IconRepeat data-icon="inline-start" aria-hidden="true" />
        </Button>
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

      <label
        className="flex min-h-11 items-center gap-2.5 text-xs font-medium"
        htmlFor="show-vrm"
      >
        <input id="show-vrm" type="checkbox" defaultChecked disabled />
        <span>Show VRM avatar</span>
      </label>

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
    <section className="grid gap-3" aria-labelledby="display-title">
      <h3
        className="text-xs font-medium text-muted-foreground"
        id="display-title"
      >
        Display
      </h3>
      <div className="grid grid-cols-2 max-[520px]:grid-cols-1">
        <label
          className="flex min-h-11 items-center gap-2.5 text-xs font-medium"
          htmlFor="show-skeleton"
        >
          <input id="show-skeleton" type="checkbox" defaultChecked />
          <span>Skeleton</span>
        </label>
        <label
          className="flex min-h-11 items-center gap-2.5 text-xs font-medium"
          htmlFor="show-contacts"
        >
          <input id="show-contacts" type="checkbox" defaultChecked />
          <span>Foot contacts</span>
        </label>
        <label
          className="flex min-h-11 items-center gap-2.5 text-xs font-medium"
          htmlFor="show-orientations"
        >
          <input id="show-orientations" type="checkbox" />
          <span>Orientations</span>
        </label>
        <label
          className="flex min-h-11 items-center gap-2.5 text-xs font-medium"
          htmlFor="show-trajectory"
        >
          <input id="show-trajectory" type="checkbox" defaultChecked />
          <span>Root trajectory</span>
        </label>
      </div>
    </section>
  )
}

function PreviewSettingsSection() {
  return (
    <details className="settings-details" id="preview-settings">
      <summary>View settings</summary>
      <div className="details-body grid gap-3">
        <VrmAvatarSection />
        <Separator />
        <DisplayControls />
      </div>
    </details>
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
      <div className="sticky top-0 z-10 flex min-h-15 items-center justify-between gap-3 border-b bg-background px-3 py-2.5 max-[760px]:static">
        <div>
          <p className="text-xs text-muted-foreground">Input</p>
          <h1
            className="text-sm font-medium"
            id="generation-panel-title"
          >
            Motion generation
          </h1>
        </div>
        <Badge variant="outline">20 FPS</Badge>
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
