import modal

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
    "apt-get update && apt-get install -y wget"
)
image = image.add_local_file("main.py", "/root/main.py")
image = image.add_local_file("detect_note.py", "/root/detect_note.py")


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
