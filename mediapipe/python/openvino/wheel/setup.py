import os
from pathlib import Path
import shutil

from setuptools import Distribution, Extension, find_namespace_packages, setup
from setuptools.command.build_ext import build_ext


class CopyPrebuiltExtension(build_ext):
    def build_extension(self, extension):
        source = Path(os.environ["OPENVINO_BINDINGS"])
        if not source.is_file():
            raise RuntimeError(f"Prebuilt binding not found: {source}")
        destination = Path(self.get_ext_fullpath(extension.name))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


requirements_path = Path(os.environ["OPENVINO_REQUIREMENTS"])
requirements = [
    line.strip()
    for line in requirements_path.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]

setup(
    name="mediapipe-openvino-demo",
    version="0.1.0",
    description="MediaPipe OpenVINO video classification demo",
    packages=find_namespace_packages(where="src", include=["mediapipe.python.openvino*"]),
    package_dir={"": "src"},
    ext_modules=[
        Extension(
            "mediapipe.python.openvino._openvino_bindings",
            sources=[],
        ),
    ],
    cmdclass={"build_ext": CopyPrebuiltExtension},
    distclass=BinaryDistribution,
    install_requires=requirements,
    python_requires=">=3.12,<3.13",
    entry_points={
        "console_scripts": [
            "openvino-classification-demo=mediapipe.python.openvino.openvino_classification_demo:main",
        ],
    },
    zip_safe=False,
)