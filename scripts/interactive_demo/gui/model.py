# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive-demo GUI: Model tab (split from create_gui)."""

from ..common import *  # noqa: F401,F403


class GuiModelMixin:
    def _build_model_tab(self, client, client_id, tab_group, g, timeline, default_prompt):
        with tab_group.add_tab("Model", viser.Icon.SETTINGS):
            # Two-step model picker: pick the skeleton, then the generation
            # horizon available for it; the resolved model name is displayed
            # below and used by the Load Model button.
            _grouped = models_by_skeleton()
            _skeleton_labels = {
                "core": "Core",
                "g1": "G1",
                "soma": "SOMA",
                "other": "Other",
            }
            _label_to_skeleton = {v: k for k, v in _skeleton_labels.items()}
            _skeleton_order = [s for s in ("core", "g1", "soma", "other") if s in _grouped]
            _skeleton_order += [s for s in _grouped if s not in _skeleton_order]

            def _default_horizon_label(skeleton: str) -> str:
                labels = list(_grouped[skeleton])
                default = str(DEFAULT_HORIZON.get(skeleton, ""))
                return default if default in labels else labels[0]

            _init_skeleton = "core" if "core" in _grouped else _skeleton_order[0]
            _init_horizon = _default_horizon_label(_init_skeleton)

            g.gui_skeleton_dropdown = client.gui.add_dropdown(
                "Skeleton",
                options=[_skeleton_labels.get(s, s) for s in _skeleton_order],
                initial_value=_skeleton_labels.get(_init_skeleton, _init_skeleton),
            )
            g.gui_horizon_dropdown = client.gui.add_dropdown(
                "Horizon",
                options=list(_grouped[_init_skeleton]),
                initial_value=_init_horizon,
                hint="Generation horizon of the model, in frames.",
            )
            # The resolved model is displayed as markdown (not an input) and
            # tracked in _chosen for the Load Model button.
            _chosen = {"model": _grouped[_init_skeleton][_init_horizon]}
            g.gui_chosen_model_md = client.gui.add_markdown(f"**Model:** `{_chosen['model']}`")

            def _selected_skeleton() -> str:
                label = g.gui_skeleton_dropdown.value
                return _label_to_skeleton.get(label, label)

            def _sync_chosen_model() -> None:
                options = _grouped[_selected_skeleton()]
                label = g.gui_horizon_dropdown.value
                if label in options:
                    _chosen["model"] = options[label]
                    g.gui_chosen_model_md.content = f"**Model:** `{_chosen['model']}`"

            @g.gui_skeleton_dropdown.on_update
            def _(event: viser.GuiEvent) -> None:
                skeleton = _selected_skeleton()
                g.gui_horizon_dropdown.options = list(_grouped[skeleton])
                g.gui_horizon_dropdown.value = _default_horizon_label(skeleton)
                _sync_chosen_model()

            @g.gui_horizon_dropdown.on_update
            def _(event: viser.GuiEvent) -> None:
                _sync_chosen_model()

            g.gui_compile_mode = client.gui.add_dropdown(
                "Acceleration",
                options=[
                    "None",
                    "ONNX-TRT (fp16)",
                    "ONNX-TRT (fp32)",
                    "torch.compile",
                ],
                initial_value="ONNX-TRT (fp16)" if self.compile_model else "None",
            )
            _text_encoder_options = [
                "cuda / bfloat16",
                "cuda / float32",
                "cpu / bfloat16",
                "cpu / float32",
            ]
            _text_encoder_initial = "cuda / bfloat16" if torch.cuda.is_available() else "cpu / bfloat16"
            g.gui_text_encoder_mode = client.gui.add_dropdown(
                "Text Encoder",
                options=_text_encoder_options,
                initial_value=_text_encoder_initial,
                hint="Set device + precision for the text encoder. Switching to cpu releases CUDA memory.",
            )

            @g.gui_text_encoder_mode.on_update
            def _(event: viser.GuiEvent) -> None:
                encoder = self.text_encoder
                notify_client = event.client
                if encoder is None:
                    if notify_client:
                        notify_client.add_notification(
                            title="No text encoder loaded",
                            body="Demo was started without an in-process text encoder; nothing to update.",
                            auto_close_seconds=2.0,
                            color="orange",
                        )
                    return
                device_str, dtype_str = [s.strip() for s in g.gui_text_encoder_mode.value.split("/")]
                dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32
                try:
                    encoder.to(device=device_str, dtype=dtype)
                    # Drop lingering Python refs to the old tensors, then return
                    # cached CUDA blocks to the driver. synchronize() ensures
                    # all in-flight ops on the old buffers have completed.
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                except Exception as e:
                    if notify_client:
                        notify_client.add_notification(
                            title="Text encoder update failed",
                            body=str(e),
                            auto_close_seconds=4.0,
                            color="red",
                        )
                    return
                if notify_client:
                    notify_client.add_notification(
                        title="Text encoder updated",
                        body=g.gui_text_encoder_mode.value,
                        auto_close_seconds=2.0,
                        color="blue",
                    )

            g.gui_load_model_button = client.gui.add_button("Load Model")
            g.gui_skeleton_name_text = client.gui.add_text("Loaded Skeleton", initial_value="", disabled=True)
            with client.gui.add_folder("Advanced", expand_by_default=False):
                g.gui_diffusion_steps_slider = client.gui.add_number(
                    "Denoising Steps",
                    initial_value=100,
                    disabled=True,
                )
                g.gui_history_crop_length = client.gui.add_slider(
                    "History Crop Length",
                    min=0,
                    max=200,
                    step=1,
                    initial_value=DEFAULT_HISTORY_CROP_LENGTH,
                )
                g.gui_future_crop_length = client.gui.add_slider(
                    "Future Crop Length",
                    min=0,
                    max=500,
                    step=1,
                    initial_value=160,
                )

                @g.gui_future_crop_length.on_update
                def _(_) -> None:
                    if not self.client_active(client_id):
                        return
                    session = self.client_sessions[client_id]
                    # Distant-constraint hiding uses future_crop as its cutoff, so
                    # moving the crop shifts which constraints are hidden. Refresh
                    # visibility only when that toggle is active.
                    if not session.gui_elements.gui_viz_hide_distant_constraints_checkbox.value:
                        return
                    if session.frame_idx >= 0:
                        self.set_frame(client_id, session.frame_idx)
                g.gui_replan_buffer_size = client.gui.add_number(
                    "Replan Buffer",
                    initial_value=DEFAULT_REPLAN_BUFFER_SIZE,
                    min=0,
                    max=100,
                    step=1,
                )
                g.gui_replan_trigger_thresh = client.gui.add_number(
                    "Replan Trigger Thresh",
                    initial_value=DEFAULT_REPLAN_TRIGGER_THRESH,
                    min=0,
                    max=100,
                    step=1,
                )
                with client.gui.add_folder("Classifier-Free Guidance", expand_by_default=False):
                    g.gui_cfg_text_weight = client.gui.add_slider(
                        "Text Weight",
                        min=0.0,
                        max=10.0,
                        step=0.1,
                        initial_value=2.0,
                        hint="Guidance weight for the text prompt.",
                    )
                    g.gui_cfg_constraint_weight = client.gui.add_slider(
                        "Constraint Weight",
                        min=0.0,
                        max=10.0,
                        step=0.1,
                        initial_value=2.0,
                        hint="Guidance weight for kinematic constraints.",
                    )
                g.gui_seed = client.gui.add_number("Seed", initial_value=2)
                g.gui_num_samples = client.gui.add_number("Num Samples", initial_value=1, disabled=True)

                @g.gui_seed.on_update
                def _(event: viser.GuiEvent) -> None:
                    seed_everything(g.gui_seed.value)
                    if event.client:
                        event.client.add_notification(
                            title="Seed updated",
                            body="Random seed has been updated.",
                            auto_close_seconds=1.0,
                            color="blue",
                        )

            @g.gui_load_model_button.on_click
            def _(event: viser.GuiEvent) -> None:
                g.gui_load_model_button.disabled = True
                g.gui_load_model_button.label = "Loading..."
                loading_notif = None
                try:
                    if event.client:
                        loading_notif = event.client.add_notification(
                            title="Loading model...",
                            body="Please wait while the model loads.",
                            loading=True,
                            with_close_button=False,
                        )
                        # Flush so the toast appears promptly. (viser keys
                        # notification "show" and "update" messages separately in
                        # its send buffer, so the body updates below can't replace
                        # an unsent "show" anymore.)
                        event.client.flush()

                    # Surface each loading phase (download / TRT export / engine
                    # build) on the on-screen notification as load_model progresses.
                    # Flush after each update so every phase is actually transmitted
                    # instead of being collapsed by the next one.
                    def report_progress(message: str) -> None:
                        if loading_notif is not None:
                            loading_notif.body = message
                            if event.client:
                                event.client.flush()

                    # Load model for this client
                    success = self.load_model(
                        client_id,
                        _chosen["model"],
                        progress=report_progress,
                    )

                    if event.client:
                        if success:
                            session = self.client_sessions[client_id]
                            text_feat, _ = session.model.text_encoder([g.gui_prompt_text.value])
                            session.text_embedding = text_feat.to(self.device)
                            session.gui_elements.gui_active_prompt_label.content = (
                                f"**Active Prompt:** {g.gui_prompt_text.value}"
                            )
                            self.restart(client_id)
                            loading_notif.title = "Model loaded"
                            loading_notif.body = "Model loaded successfully!"
                            loading_notif.color = "green"
                        else:
                            loading_notif.title = "Model load failed"
                            loading_notif.body = "Failed to load model. Check console for details."
                            loading_notif.color = "red"
                except Exception:
                    if loading_notif is not None:
                        loading_notif.title = "Model load failed"
                        loading_notif.body = "Failed to load model. Check console for details."
                        loading_notif.color = "red"
                    raise
                finally:
                    # Finalize the toast here so an exception anywhere above
                    # can't leave a permanent spinner with no close button.
                    if loading_notif is not None:
                        loading_notif.loading = False
                        loading_notif.with_close_button = True
                        loading_notif.auto_close_seconds = 3.0
                    g.gui_load_model_button.label = "Load Model"
                    g.gui_load_model_button.disabled = False

        #
        # IO tab
        #
