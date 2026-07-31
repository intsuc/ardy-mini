// SPDX-FileCopyrightText: Copyright (c) 2026 intsuc
// SPDX-License-Identifier: Apache-2.0

import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react"
import { createPortal } from "react-dom"
import {
  IconCameraRotate,
  IconChevronDown,
  IconPlayerPause,
  IconPlayerPlay,
  IconRefresh,
  IconSettings,
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
import { Button } from "@/components/ui/button"
import {
  ButtonGroup,
  ButtonGroupSeparator,
} from "@/components/ui/button-group"
import {
  Card,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Field,
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
  InputGroupTextarea,
} from "@/components/ui/input-group"
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
  ComboboxTrigger,
} from "@/components/ui/combobox"
import {
  Drawer,
  DrawerContent,
  DrawerTitle,
  DrawerTrigger,
} from "@/components/ui/drawer"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Popover,
  PopoverContent,
  PopoverTitle,
  PopoverTrigger,
} from "@/components/ui/popover"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Spinner } from "@/components/ui/spinner"
import {
  Progress,
  ProgressLabel,
  ProgressValue,
} from "@/components/ui/progress"
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  BoundCheckbox,
  BoundSlider,
} from "@/components/control-bindings"
import {
  DEFAULT_PROMPT,
  matchesPromptExample,
  PROMPT_EXAMPLES,
  PROMPT_EXAMPLE_EVENT,
} from "@/prompt-examples"
import {
  resolveModelTermsUrl,
  staticSpaceVariable,
} from "@/deployment-config"
import { bootstrap } from "@/main"
import {
  clearModelCacheAction,
  generationActionsControl,
  modelDownloadAction,
  modelDownloadCancelAction,
  modelUiControl,
  playbackSpeedControl,
  playPauseControl,
  previewSettingsControl,
  previewSettingsTabControl,
  regenerateMotionAction,
  showContactsControl,
  showOrientationsControl,
  showSkeletonControl,
  showTrajectoryControl,
  showVrmControl,
  targetBufferControl,
  timelineControl,
  unsupportedDeviceControl,
  useControlState,
  useModelUiState,
  startNewMotionAction,
} from "@/ui-control-store"

function EmptyFieldError({
  id,
}: {
  id: string
}) {
  return <FieldError id={id} aria-live="polite" forceMount />
}

function selectPrompt(prompt: string) {
  document.dispatchEvent(
    new CustomEvent<string>(PROMPT_EXAMPLE_EVENT, {
      detail: prompt,
    })
  )
}

const PLAYBACK_SPEED_OPTIONS = [
  { label: "0.5×", value: "0.5" },
  { label: "1×", value: "1" },
  { label: "2×", value: "2" },
]

const MOBILE_LAYOUT_QUERY = "(max-width: 840px)"

function setPreviewSettingsOpen(open: boolean) {
  previewSettingsControl.commit(open)
}

function useStablePortal(
  host: HTMLElement | null,
  className: string
): HTMLElement | null {
  const [container] = useState(() => {
    const element = document.createElement("div")
    element.className = className
    return element
  })
  const [connected, setConnected] = useState(false)

  useLayoutEffect(() => {
    if (!host) {
      setConnected(false)
      return
    }
    if (container.parentElement !== host) {
      host.append(container)
    }
    setConnected(true)
  }, [container, host])

  useEffect(
    () => () => {
      container.remove()
    },
    [container]
  )

  return connected ? container : null
}

function PromptExampleCombobox() {
  return (
    <Combobox
      items={PROMPT_EXAMPLES}
      itemToStringLabel={(example) => example.label}
      itemToStringValue={(example) => example.prompt}
      filter={matchesPromptExample}
      autoHighlight
      onValueChange={(value) => {
        if (value) selectPrompt(value.prompt)
      }}
    >
      <ComboboxTrigger
        render={
          <InputGroupButton
            type="button"
            aria-label="Choose an example prompt"
          />
        }
      >
        Examples
      </ComboboxTrigger>
      <ComboboxContent className="w-72 min-w-0">
        <ComboboxInput
          id="prompt-example"
          placeholder="Search examples"
          autoComplete="off"
          aria-label="Search example prompts"
          showTrigger={false}
        />
        <ComboboxEmpty>No examples found.</ComboboxEmpty>
        <ComboboxList>
          {(example) => (
            <ComboboxItem key={example.prompt} value={example}>
              {example.label}
            </ComboboxItem>
          )}
        </ComboboxList>
      </ComboboxContent>
    </Combobox>
  )
}

function VrmDropTarget() {
  return (
    <Alert
      className="pointer-events-none fixed top-3 left-1/2 z-50 w-[min(24rem,calc(100%_-_1.5rem))] -translate-x-1/2 shadow-lg"
      id="vrm-drop-target"
      hidden
    >
      <IconUpload aria-hidden="true" />
      <AlertTitle>Drop VRM avatar</AlertTitle>
      <AlertDescription>
        Release the file to load or replace the current avatar.
      </AlertDescription>
    </Alert>
  )
}

function VrmLoadingStatus() {
  return (
    <Alert
      className="pointer-events-none absolute bottom-3 left-1/2 w-[min(20rem,calc(100%_-_1.5rem))] -translate-x-1/2 shadow-lg"
      id="vrm-loading-status"
      role="status"
      aria-live="polite"
      aria-atomic="true"
      hidden
    >
      <Spinner role="presentation" aria-hidden="true" />
      <AlertTitle>Loading VRM avatar</AlertTitle>
      <AlertDescription className="truncate" id="vrm-loading-file" />
    </Alert>
  )
}

function UnsupportedDeviceDialog() {
  const state = useControlState(unsupportedDeviceControl)
  const contentRef = useRef<HTMLDivElement>(null)

  return (
    <AlertDialog
      open={state.open}
      onOpenChange={(open, eventDetails) => {
        if (!open) eventDetails.cancel()
      }}
    >
      <AlertDialogContent
        ref={contentRef}
        initialFocus={contentRef}
        size="sm"
        tabIndex={-1}
      >
        <AlertDialogHeader>
          <AlertDialogTitle>{state.title}</AlertDialogTitle>
          <AlertDialogDescription>
            {state.description}
          </AlertDialogDescription>
        </AlertDialogHeader>
      </AlertDialogContent>
    </AlertDialog>
  )
}

function formatModelBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "—"
  const units = ["B", "KiB", "MiB", "GiB"]
  const exponent = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1
  )
  const value = bytes / 1024 ** exponent
  return `${value >= 10 || exponent === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[exponent]}`
}

function modelByteProgress(
  state: ReturnType<typeof modelUiControl.getSnapshot>
): number | null {
  if (state.totalBytes <= 0) return null
  return Math.min(
    100,
    Math.round((state.cachedBytes / state.totalBytes) * 100)
  )
}

function modelPreparationProgress(
  state: ReturnType<typeof modelUiControl.getSnapshot>
): number {
  const totalSteps =
    state.totalFiles + state.totalInitializationSteps
  if (totalSteps <= 0) return 0
  const completedSteps =
    state.verifiedFiles + state.initializationSteps
  return Math.min(
    100,
    Math.round((completedSteps / totalSteps) * 100)
  )
}

function preferredStartupDialogFocus(): HTMLElement | true {
  return document.getElementById("motion-canvas") ?? true
}

function ModelStartupDialog({
  appReady,
}: {
  appReady: boolean
}) {
  const state = useModelUiState()
  const unsupported = useControlState(unsupportedDeviceControl)
  const contentRef = useRef<HTMLDivElement>(null)
  const downloadSize =
    state.totalBytes > 0
      ? formatModelBytes(state.totalBytes)
      : null
  const open = !appReady && !unsupported.open
  const hasPartialDownload =
    state.cachedBytes > 0 && state.cachedBytes < state.totalBytes
  const hasCompleteDownload =
    state.totalBytes > 0 && state.cachedBytes >= state.totalBytes
  const awaitingConsent =
    state.cache === "missing" ||
    state.cache === "error"
  const downloading = state.cache === "downloading"
  const cancelling = state.cache === "cancelling"
  const verifying = state.cache === "verifying"
  const initializing = state.cache === "initializing"
  const clearing = state.cache === "clearing"
  const checking = state.cache === "checking"
  const downloadError =
    state.cache === "error" && state.errorOperation === "download"
  const initializationError =
    state.cache === "error" &&
    state.errorOperation === "initialization"
  const clearError =
    state.cache === "error" && state.errorOperation === "clear"
  const startupError = downloadError || initializationError
  const showModelNotice = awaitingConsent && !clearError
  const modelTermsUrl = resolveModelTermsUrl({
    buildTermsValue: import.meta.env.VITE_MODEL_TERMS_URL,
    buildValue: import.meta.env.VITE_MODEL_BASE_URL,
    pageUrl: globalThis.location.href,
    spaceTermsValue:
      staticSpaceVariable("ARDY_MODEL_TERMS_URL") ?? undefined,
    spaceValue:
      staticSpaceVariable("ARDY_MODEL_BASE_URL") ?? undefined,
  })

  let title = "Starting ARDY Mini"
  let description =
    "Preparing the model for WebGPU. This may take a moment."

  if (checking) {
    title = "Checking this browser"
    description =
      "Checking device support and model files already stored here."
  } else if (awaitingConsent && !startupError && !clearError) {
    title = hasCompleteDownload
      ? "Resume model setup?"
      : hasPartialDownload
        ? "Resume model download?"
        : "Download model files?"
    description = hasCompleteDownload
      ? "All model files are already stored in this browser. Resume preparing them for WebGPU."
      : hasPartialDownload
        ? `${formatModelBytes(state.cachedBytes)} of ${formatModelBytes(state.totalBytes)} is already stored in this browser. Resume downloading the remaining files.`
        : downloadSize
          ? `ARDY Mini needs a ${downloadSize} model download. The files will be stored in this browser so future visits can start faster.`
          : "ARDY Mini needs to download its model files. They will be stored in this browser so future visits can start faster."
  } else if (startupError) {
    title = downloadError
      ? "Model download failed"
      : "ARDY Mini couldn’t start"
    description = downloadError
      ? "The model files could not be downloaded. Check your connection and try again; downloaded data will be reused."
      : "The downloaded model could not be verified or initialized for WebGPU. Try preparing it again."
  } else if (clearError) {
    title = "Couldn’t clear cached files"
    description =
      "The partial model download could not be removed. Try clearing it again or resume the download."
  } else if (downloading) {
    title = "Downloading model files"
    description =
      "Keep this tab open. Downloaded data is stored in this browser and verified before use."
  } else if (cancelling) {
    title = "Stopping download"
    description =
      "Downloaded data is being kept so the download can resume later."
  } else if (clearing) {
    title = "Clearing cached files"
    description =
      "Removing the partial model download from this browser."
  } else if (verifying) {
    title = "Preparing model"
    description =
      "Verifying the model files before they are used."
  } else if (initializing) {
    title = "Preparing model"
    description =
      "Verifying model files and starting the WebGPU inference runtime."
  }

  return (
    <AlertDialog
      open={open}
      onOpenChange={(nextOpen, eventDetails) => {
        if (!nextOpen) eventDetails.cancel()
      }}
    >
      <AlertDialogContent
        ref={contentRef}
        initialFocus={contentRef}
        finalFocus={preferredStartupDialogFocus}
        tabIndex={-1}
        aria-busy={
          checking ||
          downloading ||
          cancelling ||
          clearing ||
          verifying ||
          initializing
        }
      >
        <AlertDialogHeader>
          <AlertDialogTitle>{title}</AlertDialogTitle>
          <AlertDialogDescription>
            {description}
            {showModelNotice ? (
              <>
                {" Built with Meta Llama 3. By continuing, you acknowledge the "}
                <a
                  href={modelTermsUrl}
                  target="_blank"
                  rel="noreferrer"
                >
                  model terms and intended-use limits
                </a>{" "}
                that apply. © 2026 intsuc ·{" "}
                <a href="mailto:i@intsuc.dev">i@intsuc.dev</a>
              </>
            ) : null}
          </AlertDialogDescription>
        </AlertDialogHeader>
        {downloading ? (
          <Progress
            id="model-download-progress"
            value={modelByteProgress(state)}
            getAriaValueText={() =>
              state.totalBytes > 0
                ? `${formatModelBytes(state.cachedBytes)} of ${formatModelBytes(state.totalBytes)}`
                : "Downloading model files"
            }
          >
            <ProgressLabel>Download progress</ProgressLabel>
            <ProgressValue>
              {(_formattedValue, value) =>
                value === null ? "—" : `${value}%`
              }
            </ProgressValue>
          </Progress>
        ) : null}
        {verifying ||
        initializing ||
        (state.cache === "ready" && state.runtime !== "ready") ? (
          <Progress
            id="model-preparation-progress"
            aria-label="Model preparation progress"
            value={modelPreparationProgress(state)}
          />
        ) : null}
        {checking || cancelling || clearing ? (
          <Spinner
            className="justify-self-center"
            aria-label={title}
          />
        ) : null}
        {awaitingConsent ? (
          <AlertDialogFooter>
            {hasPartialDownload ? (
              <Button
                id="clear-partial-model-cache"
                variant="outline"
                type="button"
                onClick={() => {
                  modelUiControl.dispatch({ type: "clear-started" })
                  clearModelCacheAction.trigger()
                }}
              >
                Clear partial download
              </Button>
            ) : null}
            <Button
              id="confirm-model-download"
              type="button"
              onClick={() => modelDownloadAction.trigger()}
            >
              {startupError
                ? "Try again"
                : hasCompleteDownload
                  ? "Resume setup"
                  : hasPartialDownload
                    ? "Resume download"
                    : "Download model"}
            </Button>
          </AlertDialogFooter>
        ) : downloading ? (
          <AlertDialogFooter>
            <Button
              id="cancel-model-download"
              variant="outline"
              type="button"
              onClick={() => modelDownloadCancelAction.trigger()}
            >
              Cancel download
            </Button>
          </AlertDialogFooter>
        ) : null}
      </AlertDialogContent>
    </AlertDialog>
  )
}

function ModelCacheControl() {
  const state = useModelUiState()
  const actionRef = useRef<HTMLButtonElement>(null)
  const clearCancelRef = useRef<HTMLButtonElement>(null)
  const [clearDialogOpen, setClearDialogOpen] = useState(false)
  const canClear =
    state.runtime === "ready" &&
    state.cachedBytes > 0 &&
    state.cache !== "clearing"
  const cleared =
    state.runtime === "ready" &&
    state.cachedBytes === 0 &&
    state.cache === "missing"
  const actionLabel =
    state.cache === "clearing"
      ? "Clearing…"
      : cleared
        ? "Cache cleared"
        : state.cache === "error" &&
            state.errorOperation === "clear"
          ? "Retry clearing model cache"
          : "Clear model cache…"

  return (
    <AlertDialog
      open={clearDialogOpen}
      onOpenChange={setClearDialogOpen}
    >
      <AlertDialogTrigger
        render={
          <Button
            ref={actionRef}
            id="clear-model-cache"
            className="w-full"
            variant="outline"
            type="button"
            disabled={!canClear}
          />
        }
      >
        {state.cache === "clearing" ? (
          <Spinner data-icon="inline-start" aria-hidden="true" />
        ) : null}
        {actionLabel}
      </AlertDialogTrigger>
      <AlertDialogContent
        size="sm"
        initialFocus={clearCancelRef}
        finalFocus={actionRef}
      >
        <AlertDialogHeader>
          <AlertDialogTitle>Clear cached model files?</AlertDialogTitle>
          <AlertDialogDescription>
            This removes ARDY Mini model files stored by this browser.
            ARDY Mini will keep working in this tab, but the files must
            be downloaded again on your next visit.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel ref={clearCancelRef}>
            Cancel
          </AlertDialogCancel>
          <AlertDialogAction
            id="confirm-clear-model-cache"
            variant="destructive"
            onClick={() => {
              modelUiControl.dispatch({ type: "clear-started" })
              setClearDialogOpen(false)
              clearModelCacheAction.trigger()
            }}
          >
            Clear model cache
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

function PromptComposer() {
  const actions = useControlState(generationActionsControl)
  const visibleLabel =
    actions.activeLabel ?? actions.primaryLabel

  return (
    <section className="prompt-composer border-t bg-background p-2">
      <form id="generation-form" noValidate>
        <Field>
          <FieldLabel className="sr-only" id="prompt-label" htmlFor="prompt">
            Motion description
          </FieldLabel>
          <InputGroup>
            <InputGroupTextarea
              id="prompt"
              name="prompt"
              defaultValue={DEFAULT_PROMPT}
              rows={2}
              maxLength={280}
              spellCheck
              autoComplete="off"
              placeholder="A person walks forward, then waves with their right hand."
              aria-describedby="prompt-error generate-help"
              required
            />
            <InputGroupAddon align="block-end" className="border-t">
              <PromptExampleCombobox />
              <ButtonGroup
                className="ml-auto"
                aria-label="Motion generation actions"
              >
                <Button
                  id="generate"
                  variant="default"
                  size="sm"
                  type="submit"
                  aria-describedby="generate-help"
                  aria-keyshortcuts="Control+Enter Meta+Enter"
                  aria-busy={actions.activeLabel !== null}
                  disabled={actions.primaryDisabled}
                >
                  {actions.activeLabel !== null ? (
                    <Spinner
                      data-icon="inline-start"
                      aria-hidden="true"
                    />
                  ) : null}
                  <span>{visibleLabel}</span>
                </Button>
                <ButtonGroupSeparator />
                <DropdownMenu>
                  <DropdownMenuTrigger
                    id="generation-actions-menu"
                    render={
                      <Button
                        variant="default"
                        size="icon-sm"
                        type="button"
                        aria-label="More motion generation actions"
                        disabled={actions.menuDisabled}
                      />
                    }
                  >
                    <IconChevronDown aria-hidden="true" />
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuGroup>
                      <DropdownMenuItem
                        id="restart-from-now"
                        disabled={actions.regenerateDisabled}
                        onClick={() => regenerateMotionAction.trigger()}
                      >
                        Regenerate from current time
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        id="restart-generation"
                        disabled={actions.newMotionDisabled}
                        onClick={() => startNewMotionAction.trigger()}
                      >
                        Start new motion
                      </DropdownMenuItem>
                    </DropdownMenuGroup>
                  </DropdownMenuContent>
                </DropdownMenu>
              </ButtonGroup>
            </InputGroupAddon>
          </InputGroup>
          <EmptyFieldError id="prompt-error" />
          <span className="sr-only" id="generate-help">
            Download and prepare the model to enable generation.
          </span>
        </Field>
      </form>
    </section>
  )
}

function MotionSettingsSection() {
  return (
    <FieldGroup>
      <Field>
        <FieldLabel htmlFor="seed">Seed</FieldLabel>
        <InputGroup>
          <InputGroupInput
            id="seed"
            name="seed"
            form="generation-form"
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
              <IconRefresh aria-hidden="true" />
            </InputGroupButton>
          </InputGroupAddon>
        </InputGroup>
        <EmptyFieldError id="seed-error" />
      </Field>

      <Field>
        <div className="flex items-center justify-between gap-3">
          <FieldLabel id="target-buffer-label">
            Buffer ahead
          </FieldLabel>
          <output
            className="text-xs text-muted-foreground"
            id="target-buffer-output"
          >
            4 seconds
          </output>
        </div>
        <BoundSlider
          control={targetBufferControl}
          aria-labelledby="target-buffer-label"
          thumbAlignment="center"
        />
        <div
          className="flex justify-between text-xs text-muted-foreground"
          aria-hidden="true"
        >
          <span>2s</span>
          <span>10s</span>
        </div>
      </Field>
    </FieldGroup>
  )
}

function PlaybackSpeedSelect() {
  const state = useControlState(playbackSpeedControl)

  return (
    <Select
      items={PLAYBACK_SPEED_OPTIONS}
      value={state.value}
      disabled={state.disabled}
      onValueChange={(value) => {
        if (typeof value === "string") playbackSpeedControl.commit(value)
      }}
    >
      <SelectTrigger
        id={playbackSpeedControl.id}
        className="max-[520px]:col-start-3"
        aria-label="Playback speed"
      >
        <SelectValue />
      </SelectTrigger>
      <SelectContent
        className="min-w-0"
        align="end"
        alignItemWithTrigger={false}
      >
        <SelectGroup>
          {PLAYBACK_SPEED_OPTIONS.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectGroup>
      </SelectContent>
    </Select>
  )
}

function PlayPauseButton() {
  const state = useControlState(playPauseControl)
  const label = state.pressed ? "Pause motion" : "Play motion"

  return (
    <Tooltip>
      <TooltipTrigger render={<span className="inline-flex" />}>
        <Button
          id={playPauseControl.id}
          className="group/play"
          size="icon-lg"
          type="button"
          data-playing={state.pressed}
          aria-label={label}
          disabled={state.disabled}
        >
          <IconPlayerPlay
            className="group-data-[playing=true]/play:hidden"
            aria-hidden="true"
          />
          <IconPlayerPause
            className="hidden group-data-[playing=true]/play:block"
            aria-hidden="true"
          />
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

function ViewportPanel({
  isMobile,
}: {
  isMobile: boolean
}) {
  return (
    <section
      className="viewport-panel"
      id="viewport-panel"
      aria-labelledby="motion-preview-title"
    >
      <h2 className="sr-only" id="motion-preview-title">
        Motion preview
      </h2>

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
        <VrmLoadingStatus />
        <div className="preview-overlay-controls">
          <p
            className="preview-diagnostics bg-background/80 px-2 py-1 text-xs text-muted-foreground shadow-sm ring-1 ring-border/50 tabular-nums backdrop-blur-sm"
            id="preview-diagnostics"
            aria-hidden="true"
          />
          <div className="preview-overlay-actions">
            <Tooltip>
              <TooltipTrigger
                render={<span className="inline-flex" />}
              >
                <Button
                  id="reset-camera"
                  variant="outline"
                  size="icon-lg"
                  type="button"
                  aria-label="Reset camera"
                  aria-keyshortcuts="Home"
                >
                  <IconCameraRotate aria-hidden="true" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="bottom">
                Reset camera
              </TooltipContent>
            </Tooltip>
            <SettingsSection isMobile={isMobile} />
          </div>
        </div>
      </div>

      <PromptComposer />

      <div
        className="playback-bar"
        id="playback-bar"
        role="group"
        aria-label="Playback controls"
      >
        <PlayPauseButton />
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
        <PlaybackSpeedSelect />
      </div>
    </section>
  )
}

function VrmAvatarSection() {
  const showVrm = useControlState(showVrmControl)

  return (
    <FieldSet>
      <FieldLegend variant="label">VRM avatar</FieldLegend>

      <Card id="vrm-card" size="sm">
        <CardHeader className="min-w-0">
          <CardTitle className="truncate" id="vrm-name">
            No avatar loaded
          </CardTitle>
          <CardDescription
            className="min-w-0 [overflow-wrap:anywhere]"
            id="vrm-detail"
          >
            Load a VRM 0.x or 1.0 file.
          </CardDescription>
        </CardHeader>
        <CardFooter className="grid grid-cols-2 gap-1.5">
          <Button
            id="import-vrm"
            variant="secondary"
            type="button"
          >
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
        <BoundCheckbox control={showVrmControl} />
        <FieldLabel htmlFor={showVrmControl.id}>
          Show VRM avatar
        </FieldLabel>
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
    </FieldSet>
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

function ViewSettingsFields() {
  return (
    <div className="grid gap-3">
      <VrmAvatarSection />
      <Separator />
      <DisplayControls />
    </div>
  )
}

const SETTINGS_TABS_LIST = (
  <TabsList className="grid w-full shrink-0 grid-cols-2">
    <TabsTrigger value="view">View</TabsTrigger>
    <TabsTrigger value="motion">Motion</TabsTrigger>
  </TabsList>
)

function SettingsFields({
  tabsAtBottom = false,
}: {
  tabsAtBottom?: boolean
}) {
  const activeTab = useControlState(previewSettingsTabControl)

  return (
    <div
      className={
        tabsAtBottom
          ? "flex min-h-0 flex-1 flex-col"
          : "flex flex-col gap-3"
      }
    >
      <Tabs
        className={tabsAtBottom ? "min-h-0 flex-1 gap-3" : undefined}
        value={activeTab.value}
        onValueChange={(value) => {
          if (value === "motion" || value === "view") {
            previewSettingsTabControl.commit(value)
          }
        }}
      >
        {tabsAtBottom ? null : SETTINGS_TABS_LIST}
        <TabsContent
          className={tabsAtBottom ? "min-h-0 overflow-y-auto" : undefined}
          value="view"
          keepMounted
        >
          <ViewSettingsFields />
        </TabsContent>
        <TabsContent
          className={tabsAtBottom ? "min-h-0 overflow-y-auto" : undefined}
          value="motion"
          keepMounted
        >
          <MotionSettingsSection />
        </TabsContent>
        {tabsAtBottom ? (
          <>
            <Separator />
            <ModelCacheControl />
            {SETTINGS_TABS_LIST}
          </>
        ) : null}
      </Tabs>
      {tabsAtBottom ? null : (
        <>
          <Separator />
          <ModelCacheControl />
        </>
      )}
    </div>
  )
}

function SettingsSection({
  isMobile,
}: {
  isMobile: boolean
}) {
  const state = useControlState(previewSettingsControl)
  const [stagingHost, setStagingHost] = useState<HTMLDivElement | null>(null)
  const drawerContentRef = useRef<HTMLDivElement>(null)
  const [popoverHost, setPopoverHost] = useState<HTMLDivElement | null>(null)
  const [drawerHost, setDrawerHost] = useState<HTMLDivElement | null>(null)
  const portalContainer = useStablePortal(
    (isMobile ? drawerHost : popoverHost) ?? stagingHost,
    "settings-portal"
  )

  return (
    <>
      <div
        ref={setStagingHost}
        className="portal-staging"
        aria-hidden="true"
        inert
      />
      {isMobile ? (
        <Drawer
          open={state.open}
          onOpenChange={setPreviewSettingsOpen}
          swipeDirection="down"
          showSwipeHandle
        >
          <Tooltip>
            <TooltipTrigger render={<span className="inline-flex" />}>
              <DrawerTrigger
                id="settings-trigger"
                render={
                  <Button
                    variant="outline"
                    size="icon-lg"
                    type="button"
                    aria-label="Settings"
                  />
                }
              >
                <IconSettings aria-hidden="true" />
              </DrawerTrigger>
            </TooltipTrigger>
            <TooltipContent side="bottom">Settings</TooltipContent>
          </Tooltip>
          <DrawerContent
            ref={drawerContentRef}
            id={previewSettingsControl.id}
            initialFocus={drawerContentRef}
            tabIndex={-1}
          >
            <DrawerTitle className="sr-only">Settings</DrawerTitle>
            <div
              ref={setDrawerHost}
              className="flex min-h-0 flex-1 flex-col p-4"
            />
          </DrawerContent>
        </Drawer>
      ) : (
        <Popover
          open={state.open}
          onOpenChange={setPreviewSettingsOpen}
          triggerId="settings-trigger"
        >
          <Tooltip>
            <TooltipTrigger render={<span className="inline-flex" />}>
              <PopoverTrigger
                id="settings-trigger"
                render={
                  <Button
                    variant="outline"
                    size="icon-lg"
                    type="button"
                    aria-label="Settings"
                  />
                }
              >
                <IconSettings aria-hidden="true" />
              </PopoverTrigger>
            </TooltipTrigger>
            <TooltipContent side="bottom">Settings</TooltipContent>
          </Tooltip>
          <PopoverContent
            id={previewSettingsControl.id}
            align="end"
            side="bottom"
            keepMounted
          >
            <PopoverTitle className="sr-only">Settings</PopoverTitle>
            <div ref={setPopoverHost} />
          </PopoverContent>
        </Popover>
      )}
      {portalContainer
        ? createPortal(
            <SettingsFields tabsAtBottom={isMobile} />,
            portalContainer
          )
        : null}
    </>
  )
}

export function App() {
  const modelState = useModelUiState()
  const [isMobile, setIsMobile] = useState(
    () => window.matchMedia(MOBILE_LAYOUT_QUERY).matches
  )
  const [appReady, setAppReady] = useState(
    () =>
      modelState.runtime === "ready" &&
      modelState.cache === "ready"
  )

  useEffect(() => {
    const mediaQuery = window.matchMedia(MOBILE_LAYOUT_QUERY)
    const updateLayout = () => {
      setIsMobile(mediaQuery.matches)
      previewSettingsControl.commit(false)
    }
    mediaQuery.addEventListener("change", updateLayout)
    return () => mediaQuery.removeEventListener("change", updateLayout)
  }, [])

  useEffect(() => {
    if (
      modelState.runtime === "ready" &&
      modelState.cache === "ready"
    ) {
      setAppReady(true)
    }
  }, [modelState.cache, modelState.runtime])

  useEffect(() => {
    let cleanup: (() => void) | undefined
    const frame = window.requestAnimationFrame(() => {
      cleanup = bootstrap()
    })
    return () => {
      window.cancelAnimationFrame(frame)
      cleanup?.()
    }
  }, [])

  return (
    <TooltipProvider delay={250}>
      <div id="app">
        <a
          className="skip-link"
          href="#motion-canvas"
          hidden={!appReady}
        >
          Skip to motion preview
        </a>

        <p
          className="sr-only"
          id="app-status"
          role="status"
          aria-live="polite"
          aria-atomic="true"
        />

        <VrmDropTarget />
        <UnsupportedDeviceDialog />
        <ModelStartupDialog appReady={appReady} />

        <main
          className="workspace"
          data-ready={appReady}
          aria-hidden={!appReady || undefined}
          inert={!appReady}
        >
          <h1 className="sr-only">ARDY browser motion workspace</h1>

          <ViewportPanel isMobile={isMobile} />
        </main>
      </div>
    </TooltipProvider>
  )
}

export default App
