"""Central configuration — loaded from environment / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── AI services ──────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(..., alias="ANTHROPIC_API_KEY")
    replicate_api_token: str = Field(default="", alias="REPLICATE_API_TOKEN")
    runway_api_token: str = Field(default="", alias="RUNWAY_API_TOKEN")

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: Path = Field(default=Path("./output"), alias="OUTPUT_DIR")
    profiles_dir: Path = Field(default=Path("./profiles"), alias="PROFILES_DIR")
    render_quality: str = Field(default="production", alias="RENDER_QUALITY")

    # ── Spotify Canvas ────────────────────────────────────────────────────────
    canvas_fps: int = Field(default=30, alias="CANVAS_FPS")
    canvas_duration: int = Field(default=8, alias="CANVAS_DURATION_SECONDS")

    # ── YouTube Visualizer ────────────────────────────────────────────────────
    visualizer_fps: int = Field(default=60, alias="VISUALIZER_FPS")
    visualizer_width: int = Field(default=1920, alias="VISUALIZER_WIDTH")
    visualizer_height: int = Field(default=1080, alias="VISUALIZER_HEIGHT")

    # ── Web server ────────────────────────────────────────────────────────────
    port: int = Field(default=7860, alias="PORT")

    @property
    def is_draft(self) -> bool:
        return self.render_quality == "draft"


# Singleton — imported everywhere as `from artwork_machine.config import settings`
settings = Settings()  # type: ignore[call-arg]
