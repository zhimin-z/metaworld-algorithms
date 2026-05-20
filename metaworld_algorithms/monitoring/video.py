import re
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from metaworld_algorithms.types import Agent, GymVectorEnv


@dataclass(frozen=True)
class RecordedVideo:
    path: Path
    task_name: str
    success: bool
    episode_return: float
    episode_length: int
    num_frames: int
    env_index: int
    episode_index: int

    @property
    def caption(self) -> str:
        status = "SUCCESS" if self.success else "FAILURE"
        return f"{self.task_name} {status}"


@dataclass(frozen=True)
class RecordingConfig:
    enabled: bool = False
    every_n_evaluations: int = 5
    record_final: bool = True
    episodes_per_task: int = 1
    fps: int = 10
    frame_stride: int = 5
    width: int = 256
    height: int = 256
    overlay_tail_frames: int = 20
    output_subdir: str = "videos"

    def __post_init__(self) -> None:
        if self.enabled:
            missing_packages = [
                package
                for module_name, package in (
                    ("imageio", "imageio[ffmpeg]"),
                    ("imageio_ffmpeg", "imageio[ffmpeg]"),
                    ("PIL", "pillow"),
                )
                if find_spec(module_name) is None
            ]
            if missing_packages:
                unique_missing_packages = sorted(set(missing_packages))
                missing = ", ".join(unique_missing_packages)
                raise RuntimeError(
                    "Video recording requires optional recording dependencies. "
                    f"Missing: {missing}. Install with an accelerator extra plus recording, "
                    'for example `uv pip install -e ".[cuda12,recording]"`.'
                )


def should_record_videos(
    config: RecordingConfig, evaluation_index: int, *, is_final: bool = False
) -> bool:
    if not config.enabled:
        return False
    if is_final:
        return config.record_final
    return evaluation_index % config.every_n_evaluations == 0


def record_agent_videos(
    envs: GymVectorEnv,
    agent: Agent,
    out_dir: str | Path,
    step: int,
    config: RecordingConfig | None = None,
) -> list[RecordedVideo]:
    config = config or RecordingConfig(enabled=True)

    video_dir = Path(out_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    task_names = [str(task_name) for task_name in envs.get_attr("task_name")]
    obs, _ = envs.reset()
    agent.reset(np.ones(envs.num_envs, dtype=np.bool_))

    frames_by_env: list[list[np.ndarray]] = [[] for _ in range(envs.num_envs)]
    returns = np.zeros(envs.num_envs, dtype=np.float64)
    lengths = np.zeros(envs.num_envs, dtype=np.int64)
    episodes_recorded = np.zeros(envs.num_envs, dtype=np.int64)
    videos: list[RecordedVideo] = []

    while np.any(episodes_recorded < config.episodes_per_task):
        active = episodes_recorded < config.episodes_per_task
        if np.any((lengths % config.frame_stride == 0) & active):
            rendered_frames = envs.render()
            if rendered_frames is None:
                raise RuntimeError(
                    "Rendered envs returned no frames. Spawn video envs with "
                    "render_mode='rgb_array'."
                )
            for env_index, frame in enumerate(rendered_frames):
                should_capture = (
                    active[env_index] and lengths[env_index] % config.frame_stride == 0
                )
                if should_capture:
                    frames_by_env[env_index].append(
                        _prepare_frame(frame, width=config.width, height=config.height)
                    )

        actions = agent.eval_action(obs)
        obs, rewards, terminations, truncations, infos = envs.step(actions)
        dones = np.logical_or(terminations, truncations)
        returns[active] += rewards[active]
        lengths[active] += 1

        for env_index, done in enumerate(dones):
            if not done or episodes_recorded[env_index] >= config.episodes_per_task:
                continue

            success = _episode_success(infos, env_index)
            episode_return = _episode_return(infos, env_index, returns[env_index])
            episode_length = _episode_length(infos, env_index, lengths[env_index])
            frames = _overlay_outcome(
                frames_by_env[env_index],
                success=success,
                tail_frames=config.overlay_tail_frames,
            )
            episode_index = int(episodes_recorded[env_index])
            task_name = task_names[env_index]
            path = video_dir / _video_filename(
                step=step,
                env_index=env_index,
                episode_index=episode_index,
                task_name=task_name,
                success=success,
            )
            _write_video(path, frames, fps=config.fps)
            videos.append(
                RecordedVideo(
                    path=path,
                    task_name=task_name,
                    success=success,
                    episode_return=float(episode_return),
                    episode_length=int(episode_length),
                    num_frames=len(frames),
                    env_index=env_index,
                    episode_index=episode_index,
                )
            )

            episodes_recorded[env_index] += 1
            frames_by_env[env_index] = []
            returns[env_index] = 0.0
            lengths[env_index] = 0

        agent.reset(dones)

    return videos


def _prepare_frame(frame: np.ndarray, *, width: int, height: int) -> np.ndarray:
    from PIL import Image

    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA frame, got shape {frame.shape}.")

    image = Image.fromarray(frame)
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image)


def _overlay_outcome(
    frames: list[np.ndarray], *, success: bool, tail_frames: int
) -> list[np.ndarray]:
    if not frames:
        raise RuntimeError("Cannot write a video with no rendered frames.")
    if tail_frames == 0:
        return frames

    from PIL import Image, ImageDraw, ImageFont

    label = "SUCCESS" if success else "FAILURE"
    fill = (10, 120, 70) if success else (170, 40, 40)
    overlay_start = max(0, len(frames) - tail_frames)
    rendered_frames = list(frames)

    for index in range(overlay_start, len(rendered_frames)):
        image = Image.fromarray(rendered_frames[index]).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        padding = 8
        draw.rectangle(
            (
                padding,
                padding,
                padding * 3 + text_width,
                padding * 3 + text_height,
            ),
            fill=fill,
        )
        draw.text((padding * 2, padding * 2), label, fill=(255, 255, 255), font=font)
        rendered_frames[index] = np.asarray(image)

    return rendered_frames


def _write_video(path: Path, frames: list[np.ndarray], *, fps: int) -> None:
    import imageio.v2 as imageio

    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


def _episode_success(infos: dict, env_index: int) -> bool:
    final_info = infos.get("final_info")
    if final_info is not None and "success" in final_info:
        return bool(final_info["success"][env_index])
    if "success" in infos:
        return bool(infos["success"][env_index])
    return False


def _episode_return(infos: dict, env_index: int, fallback: float) -> float:
    final_info = infos.get("final_info")
    if final_info is not None and "episode" in final_info:
        episode = final_info["episode"]
        if "r" in episode:
            return float(episode["r"][env_index])
    if "episode" in infos and "r" in infos["episode"]:
        return float(infos["episode"]["r"][env_index])
    return float(fallback)


def _episode_length(infos: dict, env_index: int, fallback: int) -> int:
    final_info = infos.get("final_info")
    if final_info is not None and "episode" in final_info:
        episode = final_info["episode"]
        if "l" in episode:
            return int(episode["l"][env_index])
    if "episode" in infos and "l" in infos["episode"]:
        return int(infos["episode"]["l"][env_index])
    return int(fallback)


def _video_filename(
    *,
    step: int,
    env_index: int,
    episode_index: int,
    task_name: str,
    success: bool,
) -> str:
    safe_task_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", task_name).strip("-")
    status = "success" if success else "failure"
    return (
        f"step-{step:012d}__env-{env_index:02d}__episode-{episode_index:02d}"
        f"__{safe_task_name}__{status}.mp4"
    )
