// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import {
  IconCameraRotate,
  IconPlayerPause,
  IconPlayerPlay,
  IconRefresh,
  IconRepeat,
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
import { Input } from "@/components/ui/input"
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
import { Textarea } from "@/components/ui/textarea"
import {
  PROMPT_EXAMPLES,
  PROMPT_EXAMPLE_EVENT,
} from "@/prompt-examples"

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
          className="min-h-11 w-full min-w-0"
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
                className="min-h-11"
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
    <header className="relative z-10 grid min-h-15 grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-4 border-b bg-background px-3 py-2 max-[760px]:grid-cols-[minmax(0,1fr)_auto] max-[520px]:gap-2 max-[520px]:px-2.5">
      <a
        className="inline-flex min-h-11 items-center gap-2.5 text-foreground no-underline"
        href="./"
        aria-label="Reload ARDY Mini"
      >
        <Badge
          variant="outline"
          className="size-8 p-0"
          aria-hidden="true"
        >
          A
        </Badge>
        <span>
          <strong className="block text-sm leading-tight">
            ARDY Mini
          </strong>
          <small className="mt-0.5 block text-xs leading-none text-muted-foreground max-[520px]:hidden">
            Core40 browser runtime
          </small>
        </span>
      </a>

      <div
        className="flex min-w-0 items-center justify-center gap-1.5 overflow-hidden max-[760px]:col-span-full max-[760px]:row-start-2 max-[760px]:justify-start max-[760px]:overflow-x-auto"
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

      <nav
        className="flex items-center justify-end gap-1.5"
        aria-label="Session"
      >
        <Button
          id="new-session"
          className="min-h-11"
          variant="ghost"
          size="sm"
          type="button"
        >
          New
        </Button>
        <Button
          id="import-session"
          className="min-h-11"
          variant="ghost"
          size="sm"
          type="button"
        >
          Import
        </Button>
        <input
          id="session-file-input"
          type="file"
          accept=".json,.ardysession,application/json,application/vnd.ardy.session"
          hidden
          aria-hidden="true"
        />
        <Button
          id="export-session"
          className="min-h-11"
          variant="ghost"
          size="sm"
          type="button"
        >
          Export
        </Button>
      </nav>
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
            <div
              className="h-1 w-full overflow-hidden bg-secondary"
              id="model-progressbar"
              role="progressbar"
              aria-label="Model loading progress"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={0}
            >
              <span
                className="block h-full w-full origin-left scale-x-0 bg-primary transition-transform"
                id="model-progress-fill"
              />
            </div>
            <span
              className="text-xs text-muted-foreground"
              id="model-progress-label"
            />
          </div>
        </CardHeader>
        <CardFooter>
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
          className="min-h-11 w-full min-w-0"
          variant="secondary"
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
          className="min-h-11 w-full min-w-0"
          variant="destructive"
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
            className="min-h-11 min-w-11"
            variant="ghost"
            size="icon-sm"
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
        <Badge
          id="prompt-count"
          variant="outline"
          aria-hidden="true"
        >
          0 / 280
        </Badge>
      </div>

      <FieldGroup>
        <Field className="field-group">
          <FieldLabel htmlFor="prompt">
            Motion description
          </FieldLabel>
          <Textarea
            id="prompt"
            name="prompt"
            rows={3}
            maxLength={280}
            spellCheck
            autoComplete="off"
            placeholder="A person walks forward, then waves with their right hand."
            aria-describedby="prompt-error"
            required
          />
          <EmptyFieldError id="prompt-error" />
        </Field>
      </FieldGroup>

      <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-end gap-1.5">
        <PromptExampleSelect />
        <Button
          id="apply-prompt"
          className="min-h-11"
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
            <div className="grid grid-cols-[minmax(0,1fr)_2.75rem] gap-1.5">
              <Input
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
              <Button
                id="randomize-seed"
                type="button"
                className="min-h-11 min-w-11"
                variant="outline"
                size="icon"
                aria-label="Choose a random seed"
                title="Random seed"
              >
                <IconRefresh
                  data-icon="inline-start"
                  aria-hidden="true"
                />
              </Button>
            </div>
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
          <span>
            <span className="block text-xs font-medium">
              Continuous generation
            </span>
            <small className="mt-0.5 block text-xs text-muted-foreground">
              Keep the target buffer filled during playback.
            </small>
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
        className="gap-1.5 py-2"
        size="sm"
        data-state="idle"
      >
        <CardHeader className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-2 px-3">
          <CardTitle
            className="truncate text-xs"
            id="generation-stage"
          >
            Waiting for model
          </CardTitle>
          <span
            className="text-xs text-muted-foreground tabular-nums"
            id="generation-percent"
          >
            —
          </span>
          <CardDescription
            className="sr-only"
            id="generate-help"
          >
            Load a model pack to enable generation.
          </CardDescription>
        </CardHeader>
        <CardContent className="px-3">
          <div
            className="h-1 w-full overflow-hidden bg-secondary"
            role="progressbar"
            aria-label="Generation progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={0}
            id="generation-progressbar"
          >
            <span
              className="block h-full w-full origin-left scale-x-0 bg-primary transition-transform"
              id="generation-progress-fill"
            />
          </div>
          <span className="sr-only" id="generation-note">
            The first run may compile GPU pipelines.
          </span>
        </CardContent>
      </Card>

      <Button
        id="generate"
        className="min-h-11 w-full"
        type="submit"
        aria-describedby="generate-help"
        aria-keyshortcuts="Control+Enter Meta+Enter"
        disabled
      >
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
          className="min-h-11"
          variant="secondary"
          type="button"
        >
          Restart
        </Button>
        <Button
          id="restart-from-now"
          className="min-h-11"
          variant="secondary"
          type="button"
        >
          Restart from now
        </Button>
        <Button
          id="cancel-generation"
          className="invisible min-h-11 data-[state=active]:visible"
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
            className="min-h-11 min-w-11"
            variant="ghost"
            size="icon-sm"
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
            Prompts, constraints, and generated motion stay on this
            device.
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
        <div className="flex items-center gap-1.5">
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
          <Badge
            variant="outline"
            id="correction-metric"
            aria-label="Motion correction metrics"
            hidden
          >
            root <strong id="root-error-value">—</strong> · slide{" "}
            <strong id="foot-slide-value">—</strong>
          </Badge>
        </div>
      </div>

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

        <Card
          className="absolute top-1/2 left-1/2 z-10 min-w-60 -translate-x-1/2 -translate-y-1/2"
          id="loading-overlay"
          size="sm"
          aria-hidden="true"
          hidden
        >
          <CardHeader className="grid grid-cols-[auto_minmax(0,1fr)] items-center gap-x-3">
            <Spinner className="loading-indicator" />
            <div className="flex flex-col gap-0.5">
              <CardTitle id="loading-title">
                Generating motion
              </CardTitle>
              <CardDescription id="loading-detail">
                Encoding prompt…
              </CardDescription>
            </div>
          </CardHeader>
        </Card>

        <Badge
          variant="outline"
          className="pointer-events-none absolute right-3 bottom-3 max-[520px]:hidden"
          id="camera-hint"
        >
          Drag: orbit · Wheel: zoom · Home: reset
        </Badge>
      </div>

      <div
        className="playback-bar"
        id="playback-bar"
        role="group"
        aria-label="Playback controls"
      >
        <Button
          id="play-pause"
          className="group/play min-h-11 min-w-11"
          size="icon"
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
            className="[&_select]:min-h-11"
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
          className="is-active min-h-11 min-w-11 aria-pressed:bg-accent aria-pressed:text-accent-foreground"
          variant="outline"
          size="icon"
          type="button"
          aria-pressed="true"
          aria-label="Loop playback"
          disabled
        >
          <IconRepeat data-icon="inline-start" aria-hidden="true" />
        </Button>
        <Button
          id="reset-camera"
          className="min-h-11 min-w-11"
          variant="ghost"
          size="icon"
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

function InitialTransformSection() {
  return (
    <details className="tool-section" open>
      <summary>Initial transform</summary>
      <div className="tool-section__body">
        <FieldGroup className="grid grid-cols-3 gap-2.5">
          <Field className="field-group">
            <FieldLabel htmlFor="initial-x">X (m)</FieldLabel>
            <Input
              id="initial-x"
              type="number"
              defaultValue="0"
              step="0.1"
              inputMode="decimal"
            />
          </Field>
          <Field className="field-group">
            <FieldLabel htmlFor="initial-z">Z (m)</FieldLabel>
            <Input
              id="initial-z"
              type="number"
              defaultValue="0"
              step="0.1"
              inputMode="decimal"
            />
          </Field>
          <Field className="field-group">
            <FieldLabel htmlFor="initial-heading">
              Heading
            </FieldLabel>
            <Input
              id="initial-heading"
              type="number"
              defaultValue="0"
              min="-180"
              max="180"
              step="1"
              inputMode="decimal"
            />
          </Field>
        </FieldGroup>
      </div>
    </details>
  )
}

function GuidanceSection() {
  return (
    <details className="tool-section" open>
      <summary>Guidance and planning</summary>
      <div className="tool-section__body">
        <FieldGroup className="grid grid-cols-2 gap-2.5 max-[520px]:grid-cols-1">
          <Field className="field-group">
            <FieldLabel htmlFor="text-cfg">Text CFG</FieldLabel>
            <Input
              id="text-cfg"
              type="number"
              min="0"
              max="10"
              step="0.1"
              defaultValue="3.5"
            />
          </Field>
          <Field className="field-group">
            <FieldLabel htmlFor="constraint-cfg">
              Constraint CFG
            </FieldLabel>
            <Input
              id="constraint-cfg"
              type="number"
              min="0"
              max="10"
              step="0.1"
              defaultValue="1"
            />
          </Field>
        </FieldGroup>
        <FieldGroup className="grid grid-cols-2 gap-2.5 max-[520px]:grid-cols-1">
          <Field className="field-group">
            <FieldLabel htmlFor="history-frames">
              History frames
            </FieldLabel>
            <Input
              id="history-frames"
              type="number"
              min="0"
              max="40"
              step="1"
              defaultValue="40"
            />
          </Field>
          <Field className="field-group">
            <FieldLabel htmlFor="future-crop">Future crop</FieldLabel>
            <Input
              id="future-crop"
              type="number"
              min="0"
              max="120"
              step="1"
              defaultValue="80"
            />
          </Field>
        </FieldGroup>
        <FieldGroup className="grid grid-cols-2 gap-2.5 max-[520px]:grid-cols-1">
          <Field className="field-group">
            <FieldLabel htmlFor="replan-buffer">
              Replan buffer
            </FieldLabel>
            <Input
              id="replan-buffer"
              type="number"
              min="1"
              max="80"
              step="1"
              defaultValue="20"
            />
          </Field>
          <Field className="field-group">
            <FieldLabel htmlFor="replan-threshold">
              Threshold
            </FieldLabel>
            <Input
              id="replan-threshold"
              type="number"
              min="1"
              max="80"
              step="1"
              defaultValue="10"
            />
          </Field>
        </FieldGroup>
      </div>
    </details>
  )
}

function ConstraintsSection() {
  return (
    <details className="tool-section" open>
      <summary>Constraints</summary>
      <div className="tool-section__body">
        <div
          className="constraint-timeline"
          id="constraint-timeline"
          role="group"
          aria-label="Constraint timeline tracks"
        >
          <div className="constraint-ruler" aria-hidden="true">
            <span>Track</span>
            <span>0</span>
            <span>40</span>
            <span>80</span>
            <span>120</span>
          </div>
          <button
            id="constraint-track-full-body"
            className="constraint-track"
            type="button"
            data-track="full-body"
            aria-pressed="false"
          >
            <span>Full Body</span>
            <span className="track-lane" />
          </button>
          <button
            id="constraint-track-root"
            className="constraint-track"
            type="button"
            data-track="root"
            aria-pressed="false"
          >
            <span>Root</span>
            <span className="track-lane" />
          </button>
          <button
            id="constraint-track-left-hand"
            className="constraint-track"
            type="button"
            data-track="left-hand"
            aria-pressed="false"
          >
            <span>L Hand</span>
            <span className="track-lane" />
          </button>
          <button
            id="constraint-track-right-hand"
            className="constraint-track"
            type="button"
            data-track="right-hand"
            aria-pressed="false"
          >
            <span>R Hand</span>
            <span className="track-lane" />
          </button>
          <button
            id="constraint-track-left-foot"
            className="constraint-track"
            type="button"
            data-track="left-foot"
            aria-pressed="false"
          >
            <span>L Foot</span>
            <span className="track-lane" />
          </button>
          <button
            id="constraint-track-right-foot"
            className="constraint-track"
            type="button"
            data-track="right-foot"
            aria-pressed="false"
          >
            <span>R Foot</span>
            <span className="track-lane" />
          </button>
        </div>

        <FieldGroup className="grid grid-cols-3 gap-2.5 max-[520px]:grid-cols-1">
          <Field className="field-group">
            <FieldLabel htmlFor="constraint-type">Value</FieldLabel>
            <NativeSelect
              id="constraint-type"
              defaultValue="position"
            >
              <NativeSelectOption value="position">
                Position
              </NativeSelectOption>
              <NativeSelectOption value="rotation">
                Rotation
              </NativeSelectOption>
              <NativeSelectOption value="pose">
                Pose
              </NativeSelectOption>
            </NativeSelect>
          </Field>
          <Field className="field-group">
            <FieldLabel htmlFor="constraint-frame">Start</FieldLabel>
            <Input
              id="constraint-frame"
              type="number"
              min="0"
              step="1"
              defaultValue="0"
            />
          </Field>
          <Field className="field-group">
            <FieldLabel htmlFor="constraint-end-frame">End</FieldLabel>
            <Input
              id="constraint-end-frame"
              type="number"
              min="0"
              step="1"
              defaultValue="0"
            />
          </Field>
        </FieldGroup>
        <div className="flex flex-wrap items-center gap-1.5 [&>*]:flex-auto">
          <Button
            id="add-constraint"
            className="min-h-11"
            variant="secondary"
            type="button"
          >
            Add
          </Button>
          <Button
            id="delete-constraint"
            className="min-h-11"
            variant="destructive"
            type="button"
          >
            Delete selected
          </Button>
          <Button
            id="clear-constraints"
            className="min-h-11"
            variant="destructive"
            type="button"
          >
            Clear all
          </Button>
        </div>
      </div>
    </details>
  )
}

function RootControlSection() {
  return (
    <details className="tool-section">
      <summary>Root control</summary>
      <div className="tool-section__body">
        <label
          className="flex min-h-11 items-center justify-between gap-2.5"
          htmlFor="waypoint-mode"
        >
          <span>
            <span className="block text-xs font-medium">
              Waypoint placement
            </span>
            <small className="mt-0.5 block text-xs text-muted-foreground">
              Click the ground plane to add root targets.
            </small>
          </span>
          <input id="waypoint-mode" type="checkbox" role="switch" />
        </label>
        <FieldGroup className="grid grid-cols-2 gap-2.5 max-[520px]:grid-cols-1">
          <Field className="field-group">
            <FieldLabel htmlFor="waypoint-interval">
              Interval
            </FieldLabel>
            <Input
              id="waypoint-interval"
              type="number"
              min="1"
              max="200"
              step="1"
              defaultValue="20"
            />
          </Field>
          <label
            className="flex min-h-11 items-center gap-2.5 text-xs font-medium"
            htmlFor="waypoint-dense"
          >
            <input id="waypoint-dense" type="checkbox" />
            <span>Dense trajectory</span>
          </label>
        </FieldGroup>
        <Button
          id="add-waypoint"
          className="min-h-11 w-full"
          variant="secondary"
          type="button"
        >
          Add waypoint at playhead
        </Button>

        <Separator />
        <div className="text-xs font-medium">Target velocity</div>
        <p className="text-xs text-muted-foreground">
          Blends from the current root velocity over 2 seconds.
        </p>
        <FieldGroup className="grid grid-cols-2 gap-2.5 max-[520px]:grid-cols-1">
          <Field className="field-group">
            <FieldLabel htmlFor="target-velocity">
              Speed (m/s)
            </FieldLabel>
            <Input
              id="target-velocity"
              type="number"
              min="0"
              max="10"
              step="0.1"
              defaultValue="0"
            />
          </Field>
          <Field className="field-group">
            <FieldLabel htmlFor="target-heading">
              Heading
            </FieldLabel>
            <Input
              id="target-heading"
              type="number"
              min="-180"
              max="180"
              step="1"
              defaultValue="0"
            />
          </Field>
        </FieldGroup>
        <Button
          id="apply-target-velocity"
          className="min-h-11 w-full"
          variant="secondary"
          type="button"
        >
          Apply velocity
        </Button>
      </div>
    </details>
  )
}

function PostprocessSection() {
  return (
    <details className="tool-section">
      <summary>Postprocess</summary>
      <div className="tool-section__body">
        <label
          className="flex min-h-11 items-center justify-between gap-2.5"
          htmlFor="postprocess-enabled"
        >
          <span>
            <span className="block text-xs font-medium">
              Motion correction
            </span>
            <small className="mt-0.5 block text-xs text-muted-foreground">
              Refine contacts and constraint adherence after decoding.
            </small>
          </span>
          <input
            id="postprocess-enabled"
            type="checkbox"
            role="switch"
          />
        </label>
        <FieldGroup className="grid grid-cols-2 gap-2.5 max-[520px]:grid-cols-1">
          <Field className="field-group">
            <FieldLabel htmlFor="root-height-margin">
              Root margin
            </FieldLabel>
            <Input
              id="root-height-margin"
              type="number"
              min="0"
              max="1"
              step="0.01"
              defaultValue="0.04"
            />
          </Field>
          <Field className="field-group">
            <FieldLabel htmlFor="contact-threshold">
              Contact threshold
            </FieldLabel>
            <Input
              id="contact-threshold"
              type="number"
              min="0"
              max="1"
              step="0.01"
              defaultValue="0.5"
            />
          </Field>
        </FieldGroup>
      </div>
    </details>
  )
}

function DisplaySection() {
  return (
    <details className="tool-section" open>
      <summary>Display</summary>
      <div className="tool-section__body grid grid-cols-2 max-[520px]:grid-cols-1">
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
        <label
          className="flex min-h-11 items-center gap-2.5 text-xs font-medium"
          htmlFor="show-mesh"
        >
          <input id="show-mesh" type="checkbox" />
          <span>Body proxy</span>
        </label>
        <label
          className="flex min-h-11 items-center gap-2.5 text-xs font-medium"
          htmlFor="show-reference"
        >
          <input id="show-reference" type="checkbox" />
          <span>Reference motion</span>
        </label>
        <Button
          id="import-reference"
          className="col-span-full min-h-11 w-full max-[520px]:col-span-1"
          variant="secondary"
          type="button"
        >
          Import reference
        </Button>
        <input
          id="reference-file-input"
          type="file"
          accept=".json,.ardysession,application/json,application/vnd.ardy.session"
          hidden
          aria-hidden="true"
        />
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

function InspectorPanel() {
  return (
    <aside
      className="panel inspector-panel"
      aria-labelledby="inspector-title"
    >
      <div className="sticky top-0 z-10 flex min-h-15 items-center justify-between gap-3 border-b bg-background px-3 py-2.5 max-[760px]:static">
        <div>
          <p className="text-xs text-muted-foreground">Control</p>
          <h2 className="text-sm font-medium" id="inspector-title">
            Motion parameters
          </h2>
        </div>
        <Button
          id="export-motion"
          className="min-h-11"
          variant="secondary"
          type="button"
        >
          Export motion
        </Button>
      </div>

      <InitialTransformSection />

      <GuidanceSection />

      <ConstraintsSection />

      <RootControlSection />

      <PostprocessSection />

      <DisplaySection />
    </aside>
  )
}

export function App() {
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

        <InspectorPanel />
      </main>
    </div>
  )
}

export default App
