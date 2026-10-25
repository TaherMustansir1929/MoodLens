import platform
import shutil
import subprocess


def install_ffmpeg():
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"Using FFmpeg executable: {ffmpeg_path}")
        return _verify_ffmpeg()

    print("FFmpeg was not found on PATH. Starting installation...")

    if platform.system() == "Windows":
        print(
            "Automatic FFmpeg installation is not supported on Windows. "
            "Install FFmpeg and add its bin directory to PATH, then rerun training."
        )
        return False

    try:
        subprocess.check_call(
            [
                "wget",
                "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
                "-O",
                "/tmp/ffmpeg.tar.xz",
            ]
        )

        subprocess.check_call(["tar", "-xf", "/tmp/ffmpeg.tar.xz", "-C", "/tmp/"])

        result = subprocess.run(
            ["find", "/tmp", "-name", "ffmpeg", "-type", "f"],
            capture_output=True,
            text=True,
        )
        ffmpeg_path = result.stdout.strip()

        subprocess.check_call(["cp", ffmpeg_path, "/usr/local/bin/ffmpeg"])

        subprocess.check_call(["chmod", "+x", "/usr/local/bin/ffmpeg"])

        print("Installed static FFmpeg binary successfully")
    except Exception as e:
        print(f"Failed to install static FFmpeg: {e}")

    return _verify_ffmpeg()


def _verify_ffmpeg():
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=True
        )
        print("FFmpeg version:")
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg installation verification failed")
        return False
