import json
import boto3
import os
import logging
import base64
import uuid
import stat
import subprocess
import time
from tempfile import NamedTemporaryFile
from typing import List, Dict, Any
from PIL import Image
from botocore.config import Config

FFMPEG_LAYER_PATH = "/opt/bin/ffmpeg"
FFMPEG_LAYER_PATH_LOCAL = "ffmpeg"
ENVIRONMENT = "local"  # production | local
BUCKET_NAME = os.environ.get("BUCKET_NAME", "photos-processing")

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class VideoUploader:
    def __init__(self, event: Dict[str, Any]):
        print("Initializing VideoUploader...")
        self.event = event
        self.ffmpeg_path = self.setup_ffmpeg()
        self.s3_client = self.configure_s3_client()
        self.temp_files: List[str] = []
        self.start_time = time.time()
        self.files = self.parse_event_files()
        print(f"Found {len(self.files)} files to process.")

    def setup_ffmpeg(self) -> str:
        print("Setting up FFmpeg...")
        if ENVIRONMENT == "local":
            return FFMPEG_LAYER_PATH_LOCAL

        if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
            tmp_ffmpeg = "/tmp/ffmpeg"
            if not os.path.exists(tmp_ffmpeg):
                os.system(f"cp {FFMPEG_LAYER_PATH} {tmp_ffmpeg}")
                os.chmod(tmp_ffmpeg, stat.S_IEXEC | stat.S_IREAD)
            return tmp_ffmpeg

        os.chmod(FFMPEG_LAYER_PATH, stat.S_IEXEC | stat.S_IREAD)
        return FFMPEG_LAYER_PATH

    def configure_s3_client(self):
        print("Configuring S3 client...")
        config = Config(
            max_pool_connections=25, retries={"max_attempts": 2}, tcp_keepalive=True
        )
        return boto3.client(
            "s3",
            aws_access_key_id="AKIA4RCAORASWOCO5BQU",
            aws_secret_access_key="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",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            config=config,
        )

    def parse_event_files(self) -> List[Dict[str, Any]]:
        print("Parsing event files...")
        try:
            payload = json.loads(self.event["body"])
            files = []

            temp_dir = "/tmp" if ENVIRONMENT != "local" else "./temp"
            files_dir = os.path.join(temp_dir, "uploaded_files")
            if not os.path.exists(files_dir):
                os.makedirs(files_dir)

            for file_data in payload.get("files", []):
                file_content = file_data["content"]
                decoded_content = base64.b64decode(file_content)

                file_name = file_data.get("fileName", f"{uuid.uuid4()}.png")
                mimetype = file_data.get("contentType", "image/png")

                file_path = os.path.join(files_dir, file_name)
                with open(file_path, "wb") as temp_file:
                    temp_file.write(decoded_content)

                if not os.path.exists(file_path):
                    raise Exception(f"File {file_path} was not created successfully.")

                self.temp_files.append(file_path)

                files.append(
                    {
                        "file_path": file_path,
                        "mimetype": mimetype,
                        "original_name": file_name,
                    }
                )
            return files
        except Exception as e:
            logger.error(f"Error parsing event files: {str(e)}")
            raise

    def process_video(self, input_path: str) -> str:
        print(f"Processing video: {input_path}")
        output_path = f"/tmp/{uuid.uuid4()}_processed.mp4"
        try:
            duration_cmd = f'{self.ffmpeg_path} -i "{input_path}" 2>&1 | grep Duration'
            duration_output = subprocess.check_output(duration_cmd, shell=True).decode()
            duration = self.extract_duration(duration_output)
            start_time = max(0, duration - 3)
            cmd = f'{self.ffmpeg_path} -i "{input_path}" -ss {start_time} -t 3 -c copy "{output_path}"'
            subprocess.check_call(cmd, shell=True)
            return output_path
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            raise

    @staticmethod
    def extract_duration(duration_output: str) -> float:
        print(f"Extracting duration from: {duration_output}")
        try:
            time_parts = duration_output.split(",")[0].strip().split(" ")[1].split(":")
            hours, minutes, seconds = map(float, time_parts)
            return hours * 3600 + minutes * 60 + seconds
        except Exception:
            return 0.0

    def optimize_image(self, input_path: str) -> bytes:
        print(f"Optimizing image: {input_path}")
        try:
            with Image.open(input_path) as img:
                if img.mode in ("RGBA", "LA") or (
                    img.mode == "P" and "transparency" in img.info
                ):
                    img = img.convert("RGB")

                img.thumbnail((1920, 1080))
                output = NamedTemporaryFile(delete=False, suffix=".jpg")
                img.save(output.name, "JPEG", quality=80)

                self.temp_files.append(output.name)

                return output.name
        except Exception as e:
            print(f"Error optimizing image: {str(e)}")
            raise

    def upload_to_s3(self, file_path: str, key: str) -> str:
        print(f"Uploading {file_path} to S3 with key {key}")
        try:
            self.s3_client.upload_file(file_path, BUCKET_NAME, key)
            return key
        except Exception as e:
            print(f"Error uploading to S3: {str(e)}")
            raise

    def cleanup(self):
        print("Cleaning up temporary files...")
        for temp_file in self.temp_files:
            try:
                os.remove(temp_file)
            except OSError:
                pass

    def process(self) -> Dict[str, Any]:
        try:
            print("Starting processing...")
            results = []
            for file in self.files:
                file_path = file["file_path"]
                mimetype = file["mimetype"]
                key = f"uploads/{uuid.uuid4()}_{file['original_name']}"

                if mimetype.startswith("video/"):
                    processed_video = self.process_video(file_path)
                    s3_key = self.upload_to_s3(processed_video, key)
                    results.append({"type": "video", "s3_key": s3_key})
                elif mimetype.startswith("image/"):
                    optimized_image = self.optimize_image(file_path)
                    s3_key = self.upload_to_s3(optimized_image, key)
                    results.append({"type": "image", "s3_key": s3_key})

            return {
                "statusCode": 200,
                "body": json.dumps({"success": True, "results": results}),
            }
        except Exception as e:
            print(f"Error in process: {str(e)}")
            return {"statusCode": 500, "body": f"Error: {str(e)}"}
        # finally:
        # self.cleanup()


def lambda_handler(event, context):
    processor = VideoUploader(event)
    return processor.process()
