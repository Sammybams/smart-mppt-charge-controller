"""Development server entry point."""

import uvicorn


def run() -> None:
    uvicorn.run("smart_mppt.api:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()

