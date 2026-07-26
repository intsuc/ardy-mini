# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive-demo GUI: Text tab (split from create_gui)."""

from ..common import *  # noqa: F401,F403


class GuiTextMixin:
    def _build_text_tab(self, client, client_id, tab_group, g, timeline, default_prompt):
        with tab_group.add_tab("Text", viser.Icon.WRITING):
            # Prompt controls
            prompt_choices = PRESET_PROMPTS

            prompt_buttons = []
            with client.gui.add_folder("Prompt List", expand_by_default=False):
                for prompt in prompt_choices:
                    btn = client.gui.add_button(prompt, hint=prompt)
                    prompt_buttons.append(btn)

            g.gui_prompt_text = client.gui.add_text("Prompt", default_prompt)
            g.gui_update_text_button = client.gui.add_button("Update Text Prompt", color="green")

            # Prompt button handlers
            def make_prompt_handler(btn):
                def handler(event: viser.GuiEvent) -> None:
                    if not self.client_active(client_id):
                        return
                    session = self.client_sessions[client_id]
                    session.gui_elements.gui_prompt_text.value = btn.label
                    threading.Thread(
                        target=self.on_text_prompt_update,
                        args=(client_id,),
                        daemon=True,
                    ).start()

                return handler

            for btn in prompt_buttons:
                btn.on_click(make_prompt_handler(btn))

            @g.gui_update_text_button.on_click
            def _(_) -> None:
                threading.Thread(target=self.on_text_prompt_update, args=(client_id,), daemon=True).start()

        #
        # Generate tab
        #
