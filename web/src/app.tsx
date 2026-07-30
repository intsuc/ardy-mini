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
  IconLayoutSidebar,
  IconCameraRotate,
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
import { Badge } from "@/components/ui/badge"
import { Button, ButtonLink } from "@/components/ui/button"
import {
  Card,
  CardAction,
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
  InputGroupText,
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
  DrawerDescription,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from "@/components/ui/drawer"
import {
  Popover,
  PopoverContent,
  PopoverHeader,
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
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  BoundCheckbox,
  BoundProgress,
  BoundSlider,
} from "@/components/control-bindings"
import {
  DEFAULT_PROMPT,
  matchesPromptExample,
  PROMPT_EXAMPLES,
  PROMPT_EXAMPLE_EVENT,
} from "@/prompt-examples"
import { bootstrap } from "@/main"
import {
  generationProgressControl,
  modelProgressControl,
  playbackSpeedControl,
  playPauseControl,
  previewSettingsControl,
  removeSavedModelAction,
  showContactsControl,
  showOrientationsControl,
  showSkeletonControl,
  showTrajectoryControl,
  showVrmControl,
  targetBufferControl,
  timelineControl,
  unsupportedDeviceControl,
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

const PLAYBACK_SPEED_OPTIONS = [
  { label: "0.5×", value: "0.5" },
  { label: "1×", value: "1" },
  { label: "2×", value: "2" },
]

const MOBILE_LAYOUT_QUERY = "(max-width: 840px)"

function useStablePortal(
  host: HTMLElement | null,
  className: string
) {
  const [container] = useState(() => {
    const element = document.createElement("div")
    element.className = className
    return element
  })

  useLayoutEffect(() => {
    if (host && container.parentElement !== host) {
      host.append(container)
    }
  }, [container, host])

  useEffect(
    () => () => {
      container.remove()
    },
    [container]
  )

  return container
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
      <ComboboxContent>
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
      <h3
        className="text-xs font-medium text-muted-foreground"
        id="model-step-title"
      >
        Model
      </h3>

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
            Choose the exported Core40 .tar.gz model pack.
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
            Select{" "}
            <code>
              artifacts/browser/ardy-minilm-core40-browser-v1.tar.gz
            </code>
            . The compressed pack is stored only in this browser.{" "}
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
          accept=".tar.gz,application/gzip"
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

function PromptComposer() {
  return (
    <section className="prompt-composer border-t bg-background p-2">
      <Field>
        <InputGroup>
          <InputGroupTextarea
            id="prompt"
            name="prompt"
            form="generation-form"
            defaultValue={DEFAULT_PROMPT}
            rows={2}
            maxLength={280}
            spellCheck
            autoComplete="off"
            placeholder="A person walks forward, then waves with their right hand."
            aria-describedby="prompt-count prompt-error generate-help"
            required
          />
          <InputGroupAddon align="block-start" className="border-b">
            <FieldLabel id="prompt-label" htmlFor="prompt">
              Motion description
            </FieldLabel>
            <InputGroupText className="ml-auto" id="prompt-count">
              {DEFAULT_PROMPT.length} / 280
            </InputGroupText>
          </InputGroupAddon>
          <InputGroupAddon align="block-end" className="border-t">
            <PromptExampleCombobox />
            <InputGroupButton
              id="generate"
              className="ml-auto"
              variant="default"
              size="sm"
              type="submit"
              form="generation-form"
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
              <span id="generate-label">Start motion</span>
            </InputGroupButton>
          </InputGroupAddon>
        </InputGroup>
        <EmptyFieldError id="prompt-error" />
      </Field>

      <div
        className="generation-status mt-1.5 grid min-h-9 grid-cols-[minmax(0,1fr)_auto] items-center gap-x-2"
        id="generation-progress"
        data-state="idle"
        aria-busy="false"
      >
        <div className="flex min-w-0 items-center gap-2">
          <span
            className="truncate text-xs text-muted-foreground"
            id="generation-stage"
          >
            Load a model to start
          </span>
          <span
            className="shrink-0 text-xs text-muted-foreground tabular-nums"
            id="generation-percent"
          >
            —
          </span>
        </div>
        <Button
          id="cancel-generation"
          className="invisible data-[state=active]:visible"
          variant="destructive"
          size="xs"
          type="button"
          data-state="idle"
          aria-hidden="true"
          tabIndex={-1}
          disabled
        >
          Cancel
        </Button>
        <BoundProgress
          control={generationProgressControl}
          className="col-span-full"
          aria-label="Generation progress"
        />
        <span className="sr-only" id="generate-help">
          Load a model pack to enable generation.
        </span>
      </div>
    </section>
  )
}

function MotionSettingsSection() {
  return (
    <section
      className="flex flex-col gap-3 border-b p-3"
      aria-labelledby="new-motion-settings-title"
    >
      <div className="flex items-center justify-between gap-3">
        <h3
          className="text-xs font-medium text-muted-foreground"
          id="new-motion-settings-title"
        >
          New motion
        </h3>
      </div>

      <FieldGroup>
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
    </section>
  )
}

function SessionActionsSection() {
  return (
    <section
      className="flex flex-col gap-3 border-b p-3"
      aria-labelledby="session-actions-title"
    >
      <h3
        className="text-xs font-medium text-muted-foreground"
        id="session-actions-title"
      >
        Session
      </h3>
      <Button
        id="restart-generation"
        variant="secondary"
        type="button"
      >
        Start new motion
      </Button>
      <Button
        id="restart-from-now"
        variant="outline"
        type="button"
      >
        Regenerate from here
      </Button>
    </section>
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
        className="speed-control"
        aria-label="Playback speed"
      >
        <SelectValue />
      </SelectTrigger>
      <SelectContent align="end" alignItemWithTrigger={false}>
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

function SidebarToggle({
  expanded,
  onExpandedChange,
}: {
  expanded: boolean
  onExpandedChange: (expanded: boolean) => void
}) {
  const label = expanded
    ? "Hide motion controls"
    : "Show motion controls"

  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <span
            className="sidebar-toggle-anchor inline-flex"
          />
        }
      >
        <Button
          id="sidebar-toggle"
          variant="outline"
          size="icon-lg"
          type="button"
          aria-label={label}
          aria-controls="generator-panel"
          aria-expanded={expanded}
          onClick={(event) => {
            event.currentTarget.focus()
            onExpandedChange(!expanded)
          }}
        >
          <IconLayoutSidebar aria-hidden="true" />
        </Button>
      </TooltipTrigger>
      <TooltipContent side="bottom">{label}</TooltipContent>
    </Tooltip>
  )
}

function MotionControlsDrawer({
  open,
  onOpenChange,
  hostRef,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  hostRef: (element: HTMLDivElement | null) => void
}) {
  const contentRef = useRef<HTMLDivElement>(null)

  return (
    <Drawer
      open={open}
      onOpenChange={onOpenChange}
      swipeDirection="down"
      showSwipeHandle
    >
      <DrawerTrigger
        render={
          <Button
            variant="outline"
            size="icon-lg"
            type="button"
            aria-label="Motion controls"
          />
        }
      >
        <IconLayoutSidebar aria-hidden="true" />
      </DrawerTrigger>
      <DrawerContent
        ref={contentRef}
        initialFocus={contentRef}
        tabIndex={-1}
      >
        <DrawerHeader>
          <DrawerTitle>Motion controls</DrawerTitle>
          <DrawerDescription>
            Model, buffer, seed, and session actions.
          </DrawerDescription>
        </DrawerHeader>
        <div
          ref={hostRef}
          className="motion-controls-drawer-body flex-1 overflow-y-auto"
        />
      </DrawerContent>
    </Drawer>
  )
}

function ViewportPanel({
  isMobile,
  sidebarExpanded,
  onSidebarExpandedChange,
  motionDrawerOpen,
  onMotionDrawerOpenChange,
  motionDrawerHostRef,
}: {
  isMobile: boolean
  sidebarExpanded: boolean
  onSidebarExpandedChange: (expanded: boolean) => void
  motionDrawerOpen: boolean
  onMotionDrawerOpenChange: (open: boolean) => void
  motionDrawerHostRef: (element: HTMLDivElement | null) => void
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
        <div className="preview-overlay-controls">
          {isMobile ? (
            <MotionControlsDrawer
              open={motionDrawerOpen}
              onOpenChange={onMotionDrawerOpenChange}
              hostRef={motionDrawerHostRef}
            />
          ) : (
            <SidebarToggle
              expanded={sidebarExpanded}
              onExpandedChange={onSidebarExpandedChange}
            />
          )}
          <PreviewSettingsSection
            isMobile={isMobile}
            onMobileOpen={() => onMotionDrawerOpenChange(false)}
          />
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
            >
              <IconCameraRotate aria-hidden="true" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Reset camera</TooltipContent>
        </Tooltip>
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
      <h3
        className="text-xs font-medium text-muted-foreground"
        id="vrm-avatar-title"
      >
        VRM avatar
      </h3>

      <Card id="vrm-card" size="sm">
        <CardHeader>
          <CardTitle className="truncate" id="vrm-name">
            No avatar loaded
          </CardTitle>
          <CardDescription id="vrm-detail">
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

function ViewSettingsFields() {
  return (
    <div className="grid gap-3">
      <VrmAvatarSection />
      <Separator />
      <DisplayControls />
    </div>
  )
}

function PreviewSettingsSection({
  isMobile,
  onMobileOpen,
}: {
  isMobile: boolean
  onMobileOpen: () => void
}) {
  const state = useControlState(previewSettingsControl)
  const [stagingHost, setStagingHost] = useState<HTMLDivElement | null>(null)
  const drawerContentRef = useRef<HTMLDivElement>(null)
  const [popoverHost, setPopoverHost] = useState<HTMLDivElement | null>(null)
  const [drawerHost, setDrawerHost] = useState<HTMLDivElement | null>(null)
  const portalContainer = useStablePortal(
    (isMobile ? drawerHost : popoverHost) ?? stagingHost,
    "view-settings-portal"
  )

  const handleOpenChange = (open: boolean) => {
    if (isMobile && open) onMobileOpen()
    previewSettingsControl.commit(open)
  }

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
          onOpenChange={handleOpenChange}
          swipeDirection="down"
          showSwipeHandle
        >
          <DrawerTrigger
            id="preview-settings-trigger"
            render={
              <Button
                variant="outline"
                size="icon-lg"
                type="button"
                aria-label="View settings"
              />
            }
          >
            <IconSettings aria-hidden="true" />
          </DrawerTrigger>
          <DrawerContent
            ref={drawerContentRef}
            id={previewSettingsControl.id}
            initialFocus={drawerContentRef}
            tabIndex={-1}
          >
            <DrawerHeader>
              <DrawerTitle>View settings</DrawerTitle>
              <DrawerDescription>
                Avatar and preview display options.
              </DrawerDescription>
            </DrawerHeader>
            <div
              ref={setDrawerHost}
              className="flex-1 overflow-y-auto p-4"
            />
          </DrawerContent>
        </Drawer>
      ) : (
        <Popover
          open={state.open}
          onOpenChange={handleOpenChange}
          triggerId="preview-settings-trigger"
        >
          <PopoverTrigger
            id="preview-settings-trigger"
            render={
              <Button
                variant="outline"
                size="icon-lg"
                type="button"
                aria-label="View settings"
              />
            }
          >
            <IconSettings aria-hidden="true" />
          </PopoverTrigger>
          <PopoverContent
            id={previewSettingsControl.id}
            align="end"
            side="bottom"
            keepMounted
          >
            <PopoverHeader>
              <PopoverTitle>View settings</PopoverTitle>
            </PopoverHeader>
            <div ref={setPopoverHost} />
          </PopoverContent>
        </Popover>
      )}
      {createPortal(<ViewSettingsFields />, portalContainer)}
    </>
  )
}

function GenerationPanel() {
  return (
    <aside
      className="panel generation-panel"
      id="generator-panel"
      aria-labelledby="motion-controls-title"
      tabIndex={-1}
    >
      <h2 className="sr-only" id="motion-controls-title">
        Motion controls
      </h2>

      <form id="generation-form" noValidate>
        <ModelSection />

        <MotionSettingsSection />

        <SessionActionsSection />
      </form>
    </aside>
  )
}

export function App() {
  const [isMobile, setIsMobile] = useState(
    () => window.matchMedia(MOBILE_LAYOUT_QUERY).matches
  )
  const [sidebarExpanded, setSidebarExpanded] = useState(true)
  const [motionDrawerOpen, setMotionDrawerOpen] = useState(false)
  const [desktopMotionHost, setDesktopMotionHost] =
    useState<HTMLDivElement | null>(null)
  const [drawerMotionHost, setDrawerMotionHost] =
    useState<HTMLDivElement | null>(null)
  const [stagingMotionHost, setStagingMotionHost] =
    useState<HTMLDivElement | null>(null)
  const motionControlsPortal = useStablePortal(
    (isMobile ? drawerMotionHost : desktopMotionHost) ??
      stagingMotionHost,
    "generation-panel-portal"
  )

  const handleMotionDrawerOpenChange = (open: boolean) => {
    if (open) previewSettingsControl.commit(false)
    setMotionDrawerOpen(open)
  }

  useEffect(() => {
    const mediaQuery = window.matchMedia(MOBILE_LAYOUT_QUERY)
    const updateLayout = () => {
      setIsMobile(mediaQuery.matches)
      if (!mediaQuery.matches) setMotionDrawerOpen(false)
    }
    mediaQuery.addEventListener("change", updateLayout)
    return () => mediaQuery.removeEventListener("change", updateLayout)
  }, [])

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
        <a className="skip-link" href="#motion-canvas">
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

        <main
          className="workspace"
          data-sidebar={sidebarExpanded ? "expanded" : "collapsed"}
        >
          <h1 className="sr-only">ARDY browser motion workspace</h1>

          <div
            ref={setDesktopMotionHost}
            className="generation-panel-host"
          />

          <ViewportPanel
            isMobile={isMobile}
            sidebarExpanded={sidebarExpanded}
            onSidebarExpandedChange={setSidebarExpanded}
            motionDrawerOpen={motionDrawerOpen}
            onMotionDrawerOpenChange={handleMotionDrawerOpenChange}
            motionDrawerHostRef={setDrawerMotionHost}
          />
        </main>

        <div
          ref={setStagingMotionHost}
          className="portal-staging"
          aria-hidden="true"
          inert
        />
        {createPortal(<GenerationPanel />, motionControlsPortal)}
      </div>
    </TooltipProvider>
  )
}

export default App
