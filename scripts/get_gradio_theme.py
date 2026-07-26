# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gradio as gr


def get_gradio_theme(remove_gradio_footer=False):
    theme = gr.themes.Base(
        primary_hue="blue",
        text_size=gr.themes.Size(lg="16px", md="14px", sm="12px", xl="22px", xs="10px", xxl="35px", xxs="9px"),
        font=[
            gr.themes.GoogleFont("Source Sans Pro"),
            "BlinkMacSystemFont",
            "Segoe UI",
            "Roboto",
        ],
    ).set(
        body_text_color="*neutral_900",
        body_text_color_subdued="*neutral_500",
        body_text_color_subdued_dark="*neutral_500",
    )

    css = """
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700;900&display=swap');

        /* Base text */
        body, .gradio-container {
          font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen-Sans, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif !important;
          font-size: 16px !important;
        }

        h1 {
          // font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
          font-weight: 700 !important;
          font-size: 2.75rem !important;
          // margin: 0px;
          padding: 1.5rem 0px 0px 0px;
          // line-height: 1.2;
        }
        h2 {
          // font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
          font-weight: 600 !important;
          font-size: 1.5rem !important;
        }
    """
    if remove_gradio_footer:
        css += """
        footer {
        display: none !important;
        }
        """
    return theme, css
