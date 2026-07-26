# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Part of InteractiveTimelineDemo (split for readability)."""

import base64
import functools

from .common import *  # noqa: F401,F403
from .gui.instructions import QUICK_START_MD

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


@functools.lru_cache(maxsize=None)
def _titlebar_logo_data_uri(filename: str) -> str:
    """Read a titlebar logo PNG from assets/ once and cache it as a base64 data URI."""
    with open(os.path.join(_ASSETS_DIR, filename), "rb") as f:
        encoded = base64.standard_b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class ClientMixin:
    def on_client_connect(self, client: viser.ClientHandle) -> None:
        """Initialize GUI and state for each new client."""
        print(f"Client {client.client_id} connected")

        self.setup_scene(client)

        # Initialize constraint tracks (skeleton will be set when model is loaded)
        constraint_tracks = {
            "Full-Body": FullbodyKeyframeSet(
                name="Full-Body",
                server=client,
                skeleton=None,
            ),
            "End-Effectors": EEJointsKeyframeSet(
                name="End-Effectors",
                server=client,
                skeleton=None,
            ),
            "2D Root": RootKeyframe2DSet(
                name="2D Root",
                server=client,
                skeleton=None,
            ),
        }

        # Create GUI elements and setup timeline (timeline setup is inside create_gui)
        gui_elements, timeline_tracks, timeline_data = self.create_gui(client, constraint_tracks)

        # setup_scene() already configured the theme before the dark-mode
        # checkbox existed; re-apply it now with the checkbox's uuid so the
        # titlebar hosts the dark-mode toggle.
        self.configure_theme(
            client,
            dark_mode=gui_elements.gui_dark_mode_checkbox.value,
            dark_mode_checkbox_uuid=gui_elements.gui_dark_mode_checkbox.uuid,
        )

        # Initialize session state
        session = ClientSession(
            client=client,
            gui_elements=gui_elements,
            constraints=constraint_tracks,
            timeline_data=timeline_data,
        )

        self.client_sessions[client.client_id] = session

        # Setup transform control gizmo for initial body pose
        self.setup_transform_gizmo(client.client_id)

        # Create click plane for waypoint input (needs to be after session is created)
        click_plane = self.setup_click_plane(client)
        session.click_plane = click_plane

        # Setup timeline callbacks
        self.setup_timeline_callbacks(client)

        # Welcome modal: shown once per browser (dismissal persisted client-side
        # via save_choice, e.g. localStorage) so returning users skip it. Placed
        # before the model-loading notification below so it's on screen while
        # the model loads.
        with client.gui.add_modal(
            "Welcome — Quick Start",
            save_choice="ardy.demo.quick_start_ack",
            size="xl",
            show_close_button=True,
        ) as modal:
            client.gui.add_markdown(QUICK_START_MD)
            client.gui.add_button("Got it (don't remind me again)").on_click(lambda _event: modal.close())

        # Load the default model (local folder if CHECKPOINTS_DIR is set, else
        # HF). Surface each loading phase on an on-screen notification — the
        # same way the manual "Load Model" button does — so the initial load
        # isn't a silent wait. (Note: on a machine where the model is already
        # HF-cached and its TRT engines are pre-built, the download/export
        # phases are instant or skipped, so only the phases that actually run
        # are shown.)
        loading_notif = client.add_notification(
            title="Loading model...",
            body=f"Loading '{DEFAULT_MODEL_DIR}'...",
            loading=True,
            with_close_button=False,
        )
        # Flush so the toast appears promptly even while the outgoing buffer
        # is busy with the initial GUI/scene sync. (viser keys notification
        # "show" and "update" messages separately in its send buffer, so the
        # body updates below can't replace an unsent "show" anymore.)
        client.flush()

        def report_progress(message: str) -> None:
            loading_notif.body = message
            client.flush()

        try:
            self.load_model(client.client_id, DEFAULT_MODEL_DIR, progress=report_progress)

            # Initialize text embedding and generate initial motion
            if session.model is not None:
                report_progress("Generating initial motion...")
                seed_everything(gui_elements.gui_seed.value)
                text_feat, _ = session.model.text_encoder([gui_elements.gui_prompt_text.value])
                session.text_embedding = text_feat.to(self.device)
                # Generate initial motion
                self.restart(client.client_id)
                loading_notif.title = "Model loaded"
                loading_notif.body = "Model loaded successfully!"
                loading_notif.color = "green"
            else:
                loading_notif.title = "Model load failed"
                loading_notif.body = "Failed to load model. Check console for details."
                loading_notif.color = "red"
        except Exception:
            loading_notif.title = "Model load failed"
            loading_notif.body = "Failed to load model. Check console for details."
            loading_notif.color = "red"
            raise
        finally:
            # Finalize the toast here so an exception anywhere above can't
            # leave a permanent spinner with no close button.
            loading_notif.loading = False
            loading_notif.with_close_button = True
            loading_notif.auto_close_seconds = 3.0

        # Start playback thread for this client
        session.playback_thread = threading.Thread(
            target=self.run_client_playback,
            args=(client.client_id,),
            daemon=True,
        )
        session.playback_thread.start()
        print(f"Started playback thread for client {client.client_id}")

    def setup_scene(self, client: viser.ClientHandle):
        """Setup the 3D scene for a client."""
        self.configure_theme(client)

        # Add grid
        client.scene.add_grid(
            "/grid",
            width=self.floor_len,
            height=self.floor_len,
            wxyz=viser.transforms.SO3.from_x_radians(-np.pi / 2.0).wxyz,
            position=(0.0, 0.0001, 0.0),
            fade_distance=self.floor_len,
            section_color=LIGHT_THEME["grid"],
            infinite_grid=True,
        )

        # Add checkerboard
        # add_checkerboard(
        #     client,
        #     grid_size=20,
        #     square_size=2.0,
        #     plane_thickness=0.01,
        #     color1=(230, 230, 230),
        #     color2=(40, 40, 40),
        # )

    def setup_transform_gizmo(self, client_id: int):
        """Setup a transform control gizmo for initial body pose configuration."""
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        client = session.client

        # Initialize default values if not set
        if session.init_global_translation is None:
            session.init_global_translation = np.zeros(3, dtype=np.float32)
        if session.init_first_heading_angle is None:
            session.init_first_heading_angle = 0.0

        # Create transform control gizmo
        transform_gizmo = client.scene.add_transform_controls(
            name="/init_body_transform",
            scale=0.5,
            visible=False,
        )

        # Store gizmo reference
        session.transform_gizmo = transform_gizmo

        # Blue marker showing the initial position + facing direction, built
        # from the same arrow-mesh helper used for velocity arrows (see
        # characters.py). skeleton=None is fine: VelocityArrowMesh never reads
        # it outside its constructor.
        session.start_direction_marker = VelocityArrowMesh(
            name=f"start_direction_{client_id}",
            server=client,
            skeleton=None,
            color=(0, 0, 255),
        )
        self._update_start_direction_marker(client_id)

        # Add callback to update session when gizmo is moved
        @transform_gizmo.on_update
        def _(_) -> None:
            if not self.client_active(client_id):
                return
            session = self.client_sessions[client_id]

            # Update translation
            position = np.array(transform_gizmo.position)
            position[1] = 0.0
            session.init_global_translation = position

            # Update heading from rotation (extract Z-axis rotation)
            wxyz = np.array(transform_gizmo.wxyz)
            rotation = viser.transforms.SO3(wxyz)
            # Convert to Euler angles (assuming Y-up)
            # For simplicity, we extract the yaw angle
            rotation_matrix = rotation.as_matrix()
            # Yaw angle from rotation matrix (angle around Y axis)
            angle = np.arctan2(rotation_matrix[0, 2], rotation_matrix[2, 2])
            session.init_first_heading_angle = float(angle)

            print(f"[Transform] Updated init pose: translation={position}, heading={np.degrees(angle):.1f}°")

            self._update_start_direction_marker(client_id)

    def _update_start_direction_marker(self, client_id: int) -> None:
        """Sync the blue start-direction marker with the initial body transform.

        The forward direction mirrors how the gizmo's heading feeds generation:
        the gizmo's yaw is extracted as angle = atan2(R[0, 2], R[2, 2]) above,
        which is exactly the angle of R_y(angle) applied to ardy's canonical
        +Z-forward rest pose (see the "z-forward" convention noted in
        ardy/viz/viser_utils.py). That same scalar is passed straight through
        to generation as init_first_heading_angle (generation.py), so the
        marker's forward vector — (sin(angle), 0, cos(angle)) — points exactly
        where the character will initially face.
        """
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        marker = session.start_direction_marker
        if marker is None:
            return
        heading = session.init_first_heading_angle if session.init_first_heading_angle is not None else 0.0
        forward = np.array([np.sin(heading), 0.0, np.cos(heading)], dtype=np.float32)
        marker.update(
            # VelocityArrowMesh sizes the arrow as |root_velocity| / 4.0, so
            # scale the unit forward vector up to reach the desired length.
            root_velocity=forward * (START_DIRECTION_MARKER_LENGTH * 4.0),
            root_pos=session.init_global_translation,
            visible=session.gui_elements.gui_show_start_direction_checkbox.value,
        )

    def setup_click_plane(self, client: viser.ClientHandle):
        """Setup a transparent click plane for waypoint input.

        The plane sits slightly BELOW the infinite grid (y=-0.01 vs grid y=0.0001), so the grid
        renders in front of it and stays visible while waypoint mode is on. The on_click handler
        still projects clicks onto the y=0 floor. Using a real mesh (instead of a scene-level
        pointer event) keeps three.js OrbitControls intact — left-drag still orbits, right-drag
        still pans, scroll still zooms.
        """
        floor_len = self.floor_len
        click_plane = client.scene.add_mesh_simple(
            "/click_plane",
            vertices=np.array(
                [
                    [-floor_len, -0.01, -floor_len],
                    [floor_len, -0.01, -floor_len],
                    [floor_len, -0.01, floor_len],
                    [-floor_len, -0.01, floor_len],
                ]
            ),
            faces=np.array([[2, 1, 0], [3, 2, 0]]),
            color=(255, 255, 255),
            opacity=0.0,
            visible=False,
        )

        @click_plane.on_click
        def _(event: viser.ScenePointerEvent) -> None:
            event_client = event.client
            assert event_client is not None

            if not self.client_active(event_client.client_id):
                return
            session = self.client_sessions[event_client.client_id]

            if session.waypoint_mode and event.ray_origin is not None and event.ray_direction is not None:
                ray_origin = event.ray_origin
                ray_direction = event.ray_direction
                if abs(ray_direction[1]) > 1e-6:
                    t = -ray_origin[1] / ray_direction[1]
                    if t > 0:
                        x = ray_origin[0] + t * ray_direction[0]
                        z = ray_origin[2] + t * ray_direction[2]
                        self.add_waypoint(event_client.client_id, x, z)

        return click_plane

    def setup_timeline_callbacks(self, client: viser.ClientHandle):
        """Setup timeline callbacks for keyframe and interval management."""
        client_id = client.client_id

        # Check if timeline is available
        if not hasattr(client, "timeline"):
            print("Timeline not available, skipping timeline callbacks")
            return

        @client.timeline.on_frame_change
        def handle_timeline_frame_change(new_frame_idx: int):
            """Update the frame when the user clicks on the timeline."""
            self.set_frame(client_id, new_frame_idx, trigger_by_gui_timeline=True)

        @client.timeline.on_keyframe_add
        def _(keyframe_id: str, track_id: str, frame: int):
            """Called when a keyframe is added to a track."""
            if not self.client_active(client_id):
                return
            session = self.client_sessions[client_id]
            with session.timeline_data["keyframe_update_lock"]:
                constraint_type = session.timeline_data["tracks"][track_id]["name"]
                joint_names = None
                if constraint_type in [
                    "Left Hand",
                    "Right Hand",
                    "Left Foot",
                    "Right Foot",
                ]:
                    joint_names = [constraint_type.replace(" ", "")]
                    constraint_type = "End-Effectors"
                self.add_constraint_callback(client_id, keyframe_id, constraint_type, (frame, frame), joint_names)
                session.timeline_data["keyframes"][keyframe_id] = {
                    "frame": frame,
                    "track_id": track_id,
                }

        @client.timeline.on_interval_add
        def handle_interval_add(interval_id: str, track_id: str, start_frame: int, end_frame: int):
            """Called when an interval is added to a track."""
            if not self.client_active(client_id):
                return
            session = self.client_sessions[client_id]
            with session.timeline_data["keyframe_update_lock"]:
                constraint_type = session.timeline_data["tracks"][track_id]["name"]
                joint_names = None
                if constraint_type in [
                    "Left Hand",
                    "Right Hand",
                    "Left Foot",
                    "Right Foot",
                ]:
                    joint_names = [constraint_type.replace(" ", "")]
                    constraint_type = "End-Effectors"
                self.add_constraint_callback(
                    client_id,
                    interval_id,
                    constraint_type,
                    (start_frame, end_frame),
                    joint_names,
                )
                session.timeline_data["intervals"][interval_id] = {
                    "track_id": track_id,
                    "start_frame_idx": start_frame,
                    "end_frame_idx": end_frame,
                }

        @client.timeline.on_keyframe_delete
        def handle_keyframe_delete(keyframe_id: str):
            """Called when a keyframe is deleted."""
            if not self.client_active(client_id):
                return
            session = self.client_sessions[client_id]
            with session.timeline_data["keyframe_update_lock"]:
                if keyframe_id not in session.timeline_data["keyframes"]:
                    return
                keyframe_data = session.timeline_data["keyframes"][keyframe_id]
                track_id = keyframe_data["track_id"]
                constraint_type = session.timeline_data["tracks"][track_id]["name"]
                cur_frame = keyframe_data["frame"]
                if constraint_type in [
                    "Left Hand",
                    "Right Hand",
                    "Left Foot",
                    "Right Foot",
                ]:
                    constraint_type = "End-Effectors"
                self.remove_constraint_callback(client_id, keyframe_id, constraint_type, (cur_frame, cur_frame))
                del session.timeline_data["keyframes"][keyframe_id]

        @client.timeline.on_interval_delete
        def handle_interval_delete(interval_id: str):
            """Called when an interval is deleted."""
            if not self.client_active(client_id):
                return
            session = self.client_sessions[client_id]
            with session.timeline_data["keyframe_update_lock"]:
                if interval_id not in session.timeline_data["intervals"]:
                    return
                interval_data = session.timeline_data["intervals"][interval_id]
                track_id = interval_data["track_id"]
                constraint_type = session.timeline_data["tracks"][track_id]["name"]
                if constraint_type in [
                    "Left Hand",
                    "Right Hand",
                    "Left Foot",
                    "Right Foot",
                ]:
                    constraint_type = "End-Effectors"
                self.remove_constraint_callback(
                    client_id,
                    interval_id,
                    constraint_type,
                    (interval_data["start_frame_idx"], interval_data["end_frame_idx"]),
                )
                del session.timeline_data["intervals"][interval_id]

    def on_client_disconnect(self, client: viser.ClientHandle) -> None:
        """Clean up when client disconnects."""
        print(f"Client {client.client_id} disconnected")
        client_id = client.client_id

        if client_id in self.client_sessions:
            session = self.client_sessions[client_id]
            # Signal playback thread to stop
            session.stop_playback = True
            # Wait for thread to finish (with timeout)
            if session.playback_thread is not None and session.playback_thread.is_alive():
                session.playback_thread.join(timeout=1.0)
                print(f"Stopped playback thread for client {client_id}")
            del self.client_sessions[client_id]

    def client_active(self, client_id: int) -> bool:
        """Check if a client session is active."""
        return client_id in self.client_sessions

    def configure_theme(
        self,
        client: viser.ClientHandle,
        dark_mode: bool = False,
        dark_mode_checkbox_uuid: str | None = None,
    ):
        """Configure the UI theme."""
        client.gui.set_panel_label("ARDY")
        titlebar_content = viser.theme.TitlebarConfig(
            buttons=(
                viser.theme.TitlebarButton(
                    text="Project Page",
                    icon="Description",
                    href="https://research.nvidia.com/labs/sil/projects/ardy/",
                ),
                viser.theme.TitlebarButton(
                    text="GitHub",
                    icon="GitHub",
                    href="https://github.com/nv-tlabs/ardy",
                ),
            ),
            image=viser.theme.TitlebarImage(
                image_url_light=_titlebar_logo_data_uri("nvidia_logo.png"),
                image_url_dark=_titlebar_logo_data_uri("nvidia_logo_dark.png"),
                image_alt="NVIDIA",
                href="https://www.nvidia.com/",
            ),
            title_text="ARDY",
        )
        client.gui.configure_theme(
            titlebar_content=titlebar_content,
            control_layout="floating",
            control_width="large",
            dark_mode=dark_mode,
            show_logo=False,
            show_share_button=False,
            titlebar_dark_mode_checkbox_uuid=dark_mode_checkbox_uuid,
            brand_color=(152, 189, 255),
        )
