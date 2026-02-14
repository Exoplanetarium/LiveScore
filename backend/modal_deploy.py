import modal

MODEL_URL = "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1=0.9186.pth?download=1"
MODEL_PATH = "/root/piano_transcription_inference_data/note_F1=0.9677_pedal_F1=0.9186.pth"

app = modal.App("livescore-gpu")
image = (
    modal.Image.debian_slim()
    .pip_install(
        "annotated-types",
        "anyio",
        "audiolab",
        "audioop-lts",
        "audioread",
        "av",
        "certifi",
        "cffi",
        "charset-normalizer",
        "click",
        "colorama",
        "contourpy",
        "cycler",
        "Cython",
        "decorator",
        "fastapi",
        "filelock",
        "fonttools",
        "fsspec",
        "h11",
        "hmmlearn",
        "humanize",
        "idna",
        "Jinja2",
        "joblib",
        "kiwisolver",
        "lazy_loader",
        "librosa",
        "llvmlite",
        "MarkupSafe",
        "matplotlib",
        "mido",
        "mpmath",
        "msgpack",
        "networkx",
        "noisereduce",
        "numba",
        "numpy",
        "packaging",
        "piano-transcription-inference",
        "pillow",
        "platformdirs",
        "pooch",
        "pretty_midi",
        "pycparser",
        "pydantic",
        "pydantic_core",
        "pyinstrument",
        "pyparsing",
        "python-dateutil",
        "python-multipart",
        "requests",
        "scikit-learn",
        "scipy",
        "setuptools",
        "six",
        "smart_open",
        "sniffio",
        "sounddevice",
        "soundfile",
        "soxr",
        "standard-aifc",
        "standard-chunk",
        "standard-sunau",
        "starlette",
        "sympy",
        "threadpoolctl",
        "torch",
        "torchlibrosa",
        "tqdm",
        "typing-inspection",
        "typing_extensions",
        "urllib3",
        "uvicorn[standard]",
        "vulture",
        "wheel",
        "wrapt",
    )
    .apt_install("ffmpeg")
)
image = image.run_commands(
        "apt-get update && apt-get install -y wget",
        f"mkdir -p $(dirname {MODEL_PATH})",
        f"wget -O {MODEL_PATH} {MODEL_URL}"
    )
image = image.add_local_file("main.py", "/root/main.py")
image = image.add_local_file("detect_note.py", "/root/detect_note.py")
image = image.add_local_file("gpu_ops.py", "/root/gpu_ops.py")
# Add rhythm ML model (only essential files, not training data)
image = image.add_local_file("rhythm_training/__init__.py", "/root/rhythm_training/__init__.py")
image = image.add_local_file("rhythm_training/rhythm_model.py", "/root/rhythm_training/rhythm_model.py")
image = image.add_local_file("rhythm_training/rhythm_model.npz", "/root/rhythm_training/rhythm_model.npz")


@app.function(
    image=image,
    gpu="L40S",
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/root")
    from main import app as fastapi 
    return fastapi
