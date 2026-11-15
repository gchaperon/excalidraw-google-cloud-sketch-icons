"""Settings of the build, read from pipeline/config.json."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError

PATH = Path(__file__).resolve().parent / "config.json"


class _Model(BaseModel):
    # Unknown keys are errors (they are usually typos), and settings never change.
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Library(_Model):
    """One output library, made from one of Google's icon zips."""

    id: str = Field(description="Short id: build folder and key in sources.lock.json")
    title: str = Field(description="Heading in the release notes")
    zip: str = Field(description="URL of Google's icon zip")
    root: str = Field(description="Folder in the zip that holds one folder per icon")
    file: str = Field(description="File name of the library in the release")
    names: dict[str, str] = Field(
        default_factory=dict, description="Display names for icon folders, by folder name"
    )


class Config(_Model):
    """All settings of the build."""

    repo: str = Field(description="GitHub repository (owner/name) with the previous state")
    model: str = Field(description="OpenRouter model id")
    rounds: PositiveInt = Field(description="Model calls per icon")
    workers: PositiveInt = Field(description="Icons processed in parallel")
    libraries: list[Library]


def load(path: Path | None = None) -> Config:
    """Read and check the config file (default: PATH).

    Raises:
        SystemExit: If a key is missing or unknown, or a value has the wrong type.
    """
    path = path or PATH
    try:
        return Config.model_validate_json(path.read_text())
    except ValidationError as e:
        raise SystemExit(f"invalid {path.name}:\n{e}") from None
