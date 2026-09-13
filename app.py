import json
import os
import tempfile
from pathlib import Path

import gradio as gr

from src.pipeline import process_path


def extract_policy(pdf_file):
    if not pdf_file:
        return "Please upload a PDF."

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / Path(pdf_file).name
        output_dir = Path(temp_dir) / "output"

        input_path.write_bytes(Path(pdf_file).read_bytes())
        batch = process_path(input_path, output_dir)

        output_file = output_dir / f"{input_path.stem}.json"

        if output_file.exists():
            return json.dumps(
                json.loads(output_file.read_text(encoding="utf-8")),
                indent=2,
                ensure_ascii=False,
            )

        return (
            f"Completed\n"
            f"Succeeded: {batch.succeeded_count}\n"
            f"Low quality: {batch.low_quality_count}\n"
            f"Failed: {batch.failed_count}"
        )


app = gr.Interface(
    fn=extract_policy,
    inputs=gr.File(
        label="Upload policy PDF",
        file_types=[".pdf"],
        type="filepath",
    ),
    outputs=gr.Code(label="Extracted JSON", language="json"),
    title="GMC Policy Extractor",
)

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000)),
)
