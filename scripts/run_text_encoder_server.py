# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import gradio as gr
import numpy as np
from get_gradio_theme import get_gradio_theme

from ardy.model.load_model import load_text_encoder

os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "YES")

DEFAULT_TEXT = "A person walks and falls to the ground."
DEFAULT_SERVER_NAME = "0.0.0.0"
DEFAULT_SERVER_PORT = 9550
DEFAULT_TMP_FOLDER = "/tmp/text_encoder/"


class DemoWrapper:
    def __init__(self, text_encoder, tmp_folder):
        self.text_encoder = text_encoder
        self.tmp_folder = tmp_folder

    def __call__(self, text, filename, progress=gr.Progress()):
        # Compute text embedding
        tensor, length = self.text_encoder(text)
        embedding = tensor[:length]
        embedding = embedding.cpu().numpy()

        # Save text embedding
        path = os.path.join(self.tmp_folder, filename)
        np.save(path, embedding)

        output_title = gr.Markdown(visible=True)
        output_text = gr.Markdown(visible=True, value=f"Text: {text}")
        download = gr.DownloadButton(visible=True, value=path)
        return download, output_title, output_text


def download_file():
    return gr.DownloadButton()


def _get_env(name, default):
    return os.getenv(name, default)


def parse_args():
    parser = argparse.ArgumentParser(description="Run text encoder Gradio server.")
    parser.add_argument(
        "--host",
        default=_get_env("GRADIO_SERVER_NAME", DEFAULT_SERVER_NAME),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(_get_env("GRADIO_SERVER_PORT", DEFAULT_SERVER_PORT)),
    )
    parser.add_argument(
        "--tmp-folder",
        default=_get_env("TEXT_ENCODER_TMP_FOLDER", DEFAULT_TMP_FOLDER),
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Uses fp32 for the text encoder rather than the default bfloat16.",
    )
    parser.add_argument(
        "--device",
        default=_get_env("TEXT_ENCODER_DEVICE", None),
        help='Device for the text encoder, e.g. "cpu", "cuda" or "cuda:1". Defaults to cuda if available, else cpu.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    theme, css = get_gradio_theme()
    os.makedirs(args.tmp_folder, exist_ok=True)
    text_encoder = load_text_encoder(mode="local", fp32=args.fp32, device=args.device)
    demo_wrapper_fn = DemoWrapper(text_encoder, args.tmp_folder)

    with gr.Blocks(title="Text encoder", css=css, theme=theme) as demo:
        gr.Markdown("# Text encoder: LLM2Vec")
        gr.Markdown("## Description")
        gr.Markdown("Get a embeddings from a text.")

        gr.Markdown("## Inputs")
        with gr.Row():
            text = gr.Textbox(
                placeholder="Type the motion you want to generate with a sentence",
                show_label=True,
                label="Text prompt",
                value=DEFAULT_TEXT,
                type="text",
            )
        with gr.Row(scale=3):
            with gr.Column(scale=1):
                btn = gr.Button("Encode", variant="primary")
            with gr.Column(scale=1):
                clear = gr.Button("Clear", variant="secondary")
            with gr.Column(scale=3):
                pass

        output_title = gr.Markdown("## Outputs", visible=False)
        output_text = gr.Markdown("", visible=False)
        with gr.Row(scale=3):
            with gr.Column(scale=1):
                download = gr.DownloadButton("Download", variant="primary", visible=False)
            with gr.Column(scale=4):
                pass

        filename = gr.Textbox(
            visible=False,
            value="embedding.npy",
        )

        def clear_fn():
            return [
                gr.DownloadButton(visible=False),
                gr.Markdown(visible=False),
                gr.Markdown(visible=False),
            ]

        outputs = [download, output_title, output_text]

        gr.on(
            triggers=[text.submit, btn.click],
            fn=clear_fn,
            inputs=None,
            outputs=outputs,
        ).then(
            fn=demo_wrapper_fn,
            inputs=[text, filename],
            outputs=outputs,
        )

        download.click(
            fn=download_file,
            inputs=None,
            outputs=[download],
        )
        clear.click(fn=clear_fn, inputs=None, outputs=outputs)

    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
