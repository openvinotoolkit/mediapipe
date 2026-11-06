"""End-to-end classification example test with a local identity model."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
from python.runfiles import runfiles


class ClassificationTest(unittest.TestCase):
    def test_video_inference_and_frame_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (16, 16)
            )
            self.assertTrue(writer.isOpened(), "MJPG video encoder unavailable")
            try:
                for _ in range(3):
                    writer.write(np.full((16, 16, 3), 128, dtype=np.uint8))
            finally:
                writer.release()

            graph = root / "graph.pbtxt"
            graph.write_text(
                'input_stream: "input_tensors"\n'
                'output_stream: "output_tensors"\n'
                'node { calculator: "OpenVINOSessionCalculator"\n'
                'output_side_packet: "SESSION:session"\n'
                'node_options {\n'
                '[type.googleapis.com/mediapipe.OpenVINOSessionCalculatorOptions] {\n'
                f'model_path: {json.dumps(Path(OPTIONS.model).absolute().as_posix())}\n'
                'device: "CPU" shape: "(1,3,16,16)" num_infer_requests: 1\n'
                '} } }\n'
                'node { calculator: "OpenVINOInferenceCalculator"\n'
                'input_side_packet: "SESSION:session"\n'
                'input_stream: "TENSORS:input_tensors"\n'
                'output_stream: "TENSORS:output_tensors" }\n',
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    str(Path(OPTIONS.binary).resolve()),
                    f"--calculator_graph_config_file={graph}",
                    f"--input_video_path={video}",
                    "--input_width=16",
                    "--input_height=16",
                    "--max_frames=2",
                    "--top_k=0",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("frame 0 output[0]", result.stdout)
            self.assertIn("frame 1 output[0]", result.stdout)
            self.assertNotIn("frame 2 output[0]", result.stdout)
            self.assertRegex(result.stdout, r"shape=\[1,\s*3,\s*16,\s*16\]")
            self.assertIn("values: 128 128 128", result.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    OPTIONS, remaining = parser.parse_known_args()
    runtime_files = runfiles.Create()
    OPTIONS.binary = runtime_files.Rlocation(OPTIONS.binary)
    OPTIONS.model = runtime_files.Rlocation(OPTIONS.model)
    unittest.main(argv=[sys.argv[0]] + remaining)