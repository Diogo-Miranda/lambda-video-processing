import boto3
import subprocess
import os
import stat
import time
import datetime
from tempfile import NamedTemporaryFile
from typing import List, Dict, Any
import concurrent.futures
from botocore.config import Config
from PIL import Image

FFMPEG_LAYER_PATH = '/opt/bin/ffmpeg'
FFMPEG_LAYER_PATH_WITH_DRAW_TEXT = '/opt/bin/ffmpeg2'
FFMPEG_LAYER_PATH_LOCAL = "ffmpeg"
ENVIRONMENT = 'local'  # production | local
BUCKET_NAME = 'photos-processing'
OUTPUT_BUCKET_NAME = 'retrospet-photos-users'

MAX_WORKERS_PROCESS_CLIPS = 10
MAX_WORKERS_PROCESS_IMAGES = 10

class VideoProcessor:
    def __init__(self, event: Dict[str, Any]):
        print("Initializing VideoProcessor...")
        self.event = event
        self.ffmpeg_path = self.setup_ffmpeg()
        self.ffmpeg_with_draw_text_path = self.setup_ffmpeg_with_draw_text()
        self.s3_client = self.configure_s3_client()
        self.temp_files: List[str] = []
        self.start_time = time.time()

        # Ensure the temporary directory exists
        temp_dir = '/tmp' if ENVIRONMENT != 'local' else './temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Download font file from S3
        self.font_path = os.path.join(temp_dir, 'kaarna-regular.otf')
        try:
            self.s3_client.download_file(BUCKET_NAME, 'font/kaarna-regular.otf', self.font_path)
            os.chmod(self.font_path, 0o644)
            print(f"Font downloaded successfully to {self.font_path}")
        except Exception as e:
            print(f"Error downloading font: {str(e)}")
            raise

        self.temp_files.append(self.font_path)
        self.images = self.get_images_from_event()
        print(f"Found {len(self.images)} images to process")

    def get_images_from_event(self) -> List[str]:
        try:
            print("Getting images from event...")
            bucket_uri = self.event['videoConfig']['uploadedResources']['bucketKey']
            
            if '//' in bucket_uri:
                bucket_name = bucket_uri.split('//')[1].split('/')[0]
                prefix = '/'.join(bucket_uri.split('//')[1].split('/')[1:])
            else:
                parts = bucket_uri.split('/', 1)
                bucket_name = parts[0]
                prefix = parts[1] if len(parts) > 1 else ''

            print(f"Parsed bucket name: {bucket_name}, prefix: {prefix}")
            
            temp_dir = '/tmp' if ENVIRONMENT != 'local' else './temp'
            images_dir = os.path.join(temp_dir, 'user_images')
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)

            print(f"Listing objects from bucket: {bucket_name}, prefix: {prefix}")
            image_objects = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    image_objects.extend([obj for obj in page['Contents']
                                       if any(obj['Key'].lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])])

            if not image_objects:
                raise Exception("No images found in the specified S3 bucket")

            def download_image(obj):
                print(f"Downloading image: {obj['Key']}")
                local_path = os.path.join(images_dir, os.path.basename(obj['Key']))
                self.s3_client.download_file(bucket_name, obj['Key'], local_path)
                self.temp_files.append(local_path)
                return local_path

            # Limitar o número de workers com base no número de imagens
            max_workers = min(MAX_WORKERS_PROCESS_IMAGES, len(image_objects))  # Máximo de 5 threads, ou menos se houver menos imagens
            print(f"Using {max_workers} workers to download {len(image_objects)} images")

            image_paths = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(download_image, obj) for obj in image_objects]
                for future in concurrent.futures.as_completed(futures):
                    if result := future.result():
                        image_paths.append(result)

            if not image_paths:
                raise Exception("Failed to download any images")

            print(f"Successfully downloaded {len(image_paths)} images")
            return sorted(image_paths)

        except Exception as e:
            print(f"Error in get_images_from_event: {str(e)}")
            raise


    def process_template_with_images(
        self, template_clip: Dict, image_paths: List[str], temp_dir: str
    ) -> str:
        try:
            template_path = os.path.join(
                temp_dir, f"template_{os.path.basename(template_clip['link'])}"
            )
            self.s3_client.download_file(
                BUCKET_NAME, template_clip["link"], template_path
            )
            self.temp_files.append(template_path)

            rotation_config = template_clip["metadata"]["rotation"]
            output_path = os.path.join(
                temp_dir, f"processed_{os.path.basename(template_clip['link'])}"
            )

            # Validar número de imagens
            num_images = len(image_paths)

            # Calcular duração por imagem
            total_duration = 6  # Duração total do bloco de mídia em segundos
            if num_images <= 2:
                duration_per_image = (
                    total_duration // num_images
                )  # 3s por imagem para 2 fotos
            else:
                duration_per_image = (
                    total_duration / num_images
                )  # 0.5s por imagem para 12 fotos

            # Criar vídeo com as imagens
            filter_complex_parts = []
            for idx, image_path in enumerate(image_paths):
                image_output_path = os.path.join(temp_dir, f"image_{idx}.png")

                # Calcular dimensões mantendo a proporção
                target_w = rotation_config['width']
                target_h = rotation_config['height']

                # Adicionar entrada ao filtro complexo com scale que preserva proporção
                filter_complex_parts.append(
                    f"[{idx}:v]scale=w={target_w}:h={target_h}:force_original_aspect_ratio=decrease,"
                    f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=#E9E7E9@1,"
                    f"setsar=1[v{idx}]"
                )

            # Construir o filtro complexo para concatenação
            concat_filter = (
                ";".join(filter_complex_parts)
                + f";{''.join(f'[v{idx}]' for idx in range(len(image_paths)))}concat=n={len(image_paths)}:v=1:a=0[outv]"
            )

            # Comando FFmpeg para criar vídeo
            ffmpeg_inputs = []
            for image_path in image_paths:
                ffmpeg_inputs.extend(
                    ["-loop", "1", "-t", str(duration_per_image), "-i", image_path]
                )

            template_id = os.path.splitext(os.path.basename(template_clip['link']))[0]
            image_video_output = os.path.join(temp_dir, f"image_video_{template_id}.mp4")
            
            ffmpeg_command = [
                self.ffmpeg_path,
                *ffmpeg_inputs,
                "-filter_complex",
                concat_filter,
                "-map",
                "[outv]",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",  # Mudado de 'medium' para 'ultrafast'
                "-tune",
                "fastdecode",  # Adicionado para otimizar decodificação
                "-threads",
                "auto",  # Permite que o FFmpeg gerencie as threads
                "-pix_fmt",
                "yuv420p",
                "-y",
                image_video_output,
            ]
            subprocess.run(ffmpeg_command, check=True)

            # Sobrepor o vídeo gerado no template
            x_position = rotation_config["x"]
            y_position = rotation_config["y"]
            ffmpeg_process = subprocess.Popen(
                [
                    self.ffmpeg_path,
                    "-i",
                    template_path,
                    "-i",
                    image_video_output,
                    "-filter_complex",
                    f"[0:v][1:v]overlay=x={x_position}:y={y_position}:format=auto[outv]",
                    "-map",
                    "[outv]",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",  # Mudado para ultrafast
                    "-tune",
                    "fastdecode",  # Otimização para decodificação
                    "-threads",
                    "auto",  # Gerenciamento automático de threads
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    output_path,
                ],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )

            stdout, stderr = ffmpeg_process.communicate()
            if ffmpeg_process.returncode != 0:
                print(f"Erro ao processar o vídeo {output_path}: {stderr.decode()}")

            return output_path
        except Exception as e:
            print(f"Error processing template video", e)
            raise


    def setup_ffmpeg(self) -> str:
        print("Setting up FFmpeg...")
        if ENVIRONMENT == 'local':
            return FFMPEG_LAYER_PATH_LOCAL

        if os.environ.get('AWS_LAMBDA_FUNCTION_NAME'):
            tmp_ffmpeg = '/tmp/ffmpeg'
            if not os.path.exists(tmp_ffmpeg):
                os.system(f'cp {FFMPEG_LAYER_PATH} {tmp_ffmpeg}')
                os.chmod(tmp_ffmpeg, stat.S_IEXEC | stat.S_IREAD)
            return tmp_ffmpeg

        os.chmod(FFMPEG_LAYER_PATH, stat.S_IEXEC | stat.S_IREAD)
        return FFMPEG_LAYER_PATH
    
    def setup_ffmpeg_with_draw_text(self) -> str:
        print("Setting up FFmpeg with draw text...")
        if ENVIRONMENT == 'local':
            return FFMPEG_LAYER_PATH_LOCAL

        if os.environ.get('AWS_LAMBDA_FUNCTION_NAME'):
            tmp_ffmpeg = '/tmp/ffmpeg2'
            if not os.path.exists(tmp_ffmpeg):
                os.system(f'cp {FFMPEG_LAYER_PATH_WITH_DRAW_TEXT} {tmp_ffmpeg}')
                os.chmod(tmp_ffmpeg, stat.S_IEXEC | stat.S_IREAD)
            return tmp_ffmpeg

        os.chmod(FFMPEG_LAYER_PATH_WITH_DRAW_TEXT, stat.S_IEXEC | stat.S_IREAD)
        return FFMPEG_LAYER_PATH_WITH_DRAW_TEXT

    def configure_s3_client(self):
        print("Configuring S3 client...")
        config = Config(
            max_pool_connections=25,
            retries={'max_attempts': 2},
            tcp_keepalive=True
        )
        return boto3.client(
            's3',
            aws_access_key_id="AKIA4RCAORASWOCO5BQU",
            aws_secret_access_key="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",
            region_name='us-east-1',
            config=config
        )

    def prepare_resources(self) -> tuple[List[Dict], Dict[str, int]]:
        print("Preparing resources...")
        video_config = self.event['videoConfig']
        static_resources = self.event['static']
        track_order = video_config['trackOrder']

        resource_clips = {}
        current_index = {'template': 0, 'card': 0}
        final_video_order = []

        for resource_type, resource_data in static_resources.items():
            resource_clips[resource_type] = resource_data['clips']

        for track in track_order:
            if track in ['initial', 'final']:
                if clips := resource_clips.get(track):
                    final_video_order.extend(clips)
            else:
                resource_type = 'templates' if track == 'template' else 'cards'
                if clips := resource_clips.get(resource_type):
                    index = current_index[track] % len(clips)
                    final_video_order.append(clips[index])
                    current_index[track] += 1

        print(f"Prepared {len(final_video_order)} clips for processing")
        return final_video_order, current_index

    def process_videos(self, final_video_order: List[Dict]) -> str:
        print("Processing videos...")
        temp_dir = "/tmp" if ENVIRONMENT != "local" else "./temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        concat_file = NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False, dir=temp_dir
        )
        self.temp_files.append(concat_file.name)

        templates = [
            clip for clip in final_video_order if clip["metadata"]["clipType"] == "template"
        ]
        num_templates = len(templates)

        # Definir quantidade mínima e máxima de imagens por template
        min_images = 2
        max_images = 12

        # Calcular número inicial de imagens por template
        images_per_template = max(len(self.images) // num_templates, min_images)
        images_per_template = min(images_per_template, max_images)

        # Ajustar imagens excedentes
        excess_images = len(self.images) - (images_per_template * num_templates)

        if excess_images < 0:
            print("Not enough images for all templates, reducing the count.")
        else:
            print(f"Distributing {excess_images} extra images.")

        processed = []
        image_index = 0

        def process_clip(args):
            nonlocal image_index, excess_images
            index, clip = args
            clip_type = clip["metadata"]["clipType"]
            print(f"Processing clip {index} of type {clip_type}")

            # For nome_pet_e_dono.mp4 processing
            if clip["link"].endswith("nome_pet_e_dono.mp4"):
                try:
                    # Video settings
                    VIDEO_WIDTH = 1080
                    VIDEO_HEIGHT = 1920
                    FONT_SIZE = 110
                    FONT_COLOR = "#158e4d"
                    VERTICAL_SPACING = 140

                    temp_path = os.path.join(
                        temp_dir, f"{index}_{clip_type}_{os.path.basename(clip['link'])}"
                    )
                    self.s3_client.download_file(BUCKET_NAME, clip["link"], temp_path)
                    
                    # Verify input files exist
                    if not os.path.exists(temp_path):
                        raise Exception(f"Input file not found at {temp_path}")
                    if not os.path.exists(self.font_path):
                        raise Exception(f"Font file not found at {self.font_path}")

                    text_options = self.event['videoConfig'].get('textOptions', {})
                    text_filters = []
                    base_y = 1050

                    # Add text filters
                    for i, key in enumerate(['firstLine', 'secondLine', 'thirdLine']):
                        if text := text_options.get(key):
                            text_filters.append(
                                f"drawtext=text='{text}'"
                                f":fontfile='{self.font_path}'"
                                f":fontsize={FONT_SIZE}"
                                f":fontcolor='{FONT_COLOR}'"
                                f":x=(w-text_w)/2:y={base_y + (VERTICAL_SPACING * i)}"
                            )

                    filter_complex = ','.join(text_filters)
                    
                    # Define all output paths
                    temp_output_path = os.path.join(temp_dir, f"temp_text_{os.path.basename(clip['link'])}")
                    final_path = os.path.join(temp_dir, f"processed_final_{index}_{os.path.basename(clip['link'])}")

                    # First command - Apply text
                    text_command = [
                        self.ffmpeg_with_draw_text_path,
                        "-i", temp_path,
                        "-vf", filter_complex,
                        "-c:a", "copy",
                        "-y",
                        temp_output_path
                    ]
                    subprocess.run(text_command, check=True, capture_output=True, text=True)

                    # Second command - Encode with specifications
                    final_command = [
                        self.ffmpeg_path,
                        "-i", temp_output_path,
                        "-c:v", "libx264",
                        "-profile:v", "high",
                        "-level:v", "4.1",
                        "-pix_fmt", "yuv420p",
                        "-r", "29.97",
                        "-b:v", "1301k",
                        "-video_track_timescale", "30k",
                        "-color_primaries", "bt709",
                        "-color_trc", "bt709",
                        "-colorspace", "bt709",
                        "-g", "60",
                        "-bf", "3",
                        "-refs", "3",
                        "-tag:v", "avc1",
                        "-y",
                        final_path
                    ]
                    subprocess.run(final_command, check=True, capture_output=True, text=True)

                    # Cleanup temp file
                    if os.path.exists(temp_output_path):
                        os.remove(temp_output_path)

                    return index, final_path

                except subprocess.CalledProcessError as e:
                    error_msg = f"FFmpeg error: {e.stderr}"
                    print(error_msg)
                    raise Exception(error_msg)
                except Exception as e:
                    error_msg = f"Error processing text overlay: {str(e)}"
                    print(error_msg)
                    raise Exception(error_msg)

            elif clip_type == "template":
                # Calcular número de imagens para este template
                num_images = images_per_template
                if excess_images > 0:
                    num_images += 1
                    excess_images -= 1

                # Pegar imagens disponíveis
                image_paths = self.images[image_index:image_index + num_images]
                image_index += len(image_paths)

                # Validar número mínimo de imagens
                if len(image_paths) < min_images:
                    print(f"Skipping template at index {index} due to insufficient images.")
                    return None

                # Processar template
                return index, self.process_template_with_images(clip, image_paths, temp_dir)
            else:
                # Processar outros tipos de clipe
                temp_path = os.path.join(
                    temp_dir, f"{index}_{clip_type}_{os.path.basename(clip['link'])}"
                )
                self.s3_client.download_file(BUCKET_NAME, clip["link"], temp_path)
                return index, temp_path
                
        max_workers = min(MAX_WORKERS_PROCESS_CLIPS, len(final_video_order))  # Maximum of 5 threads, or less if fewer clips
        print(f"Using {max_workers} workers to process {len(final_video_order)} clips")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_clip, (i, clip))
                for i, clip in enumerate(final_video_order)
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:  # Ignorar templates omitidos
                        processed.append(result)
                        print(f"Successfully processed clip {result[0]}")
                except Exception as e:
                    print(f"Error processing clip: {str(e)}")
                    raise

        # Ordenar clipes processados e escrever no arquivo de concatenação
        processed.sort(key=lambda x: x[0])
        for _, temp_path in processed:
            self.temp_files.append(temp_path)
            concat_file.write(f"file '{os.path.abspath(temp_path)}'\n")

        concat_file.close()
        print(f"Created concat file with {len(processed)} entries")
        return concat_file.name

    def merge_videos(self, concat_file_path: str) -> str:
        print("Merging videos...")
        temp_output = NamedTemporaryFile(suffix='_temp.mp4', delete=False)
        final_output = NamedTemporaryFile(suffix='_final.mp4', delete=False)
        self.temp_files.extend([temp_output.name, final_output.name])

        # Download audio file
        audio_path = '/tmp/audio.mp3' if ENVIRONMENT != 'local' else './temp/audio.mp3'
        print("Downloading audio file...")
        self.s3_client.download_file(BUCKET_NAME, 'audio/audio.mp3', audio_path)
        self.temp_files.append(audio_path)

        # First merge videos without audio
        merge_cmd = [
            self.ffmpeg_path, '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_file_path,
            '-c:v', 'libx264',
            '-crf', '28',
            '-preset', 'veryfast',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-threads', 'auto',
            '-tune', 'fastdecode',
            '-an',  # Remove any existing audio
            temp_output.name
        ]

        print("Executing merge command...")
        try:
            result = subprocess.run(merge_cmd, capture_output=True, text=True, check=True)
            print(f"Merge command output: {result.stdout}")
            print(f"Merge command error: {result.stderr}")
        
            # Add audio to the merged video with improved settings
            print("Adding audio to video...")
            audio_cmd = [
                self.ffmpeg_path,
                '-i', temp_output.name,
                '-i', audio_path,
                '-c:v', 'copy',            # Copy video without re-encoding
                '-c:a', 'aac',             # Use AAC codec for audio
                '-map', '0:v:0',           # Map video from first input
                '-map', '1:a:0',           # Map audio from second input
                '-y',                      # Overwrite output file if exists
                final_output.name
            ]
            
            result = subprocess.run(audio_cmd, capture_output=True, text=True, check=True)
            print(f"Audio addition output: {result.stdout}")
            print(f"Audio addition error: {result.stderr}")

        except subprocess.CalledProcessError as e:
            print(f"Error in video processing: {e.stderr}")
            raise Exception(f"Failed to process video: {e.stderr}")

        return final_output.name

    def cleanup(self):
        print("Cleaning up temporary files...")

    def process(self) -> Dict[str, Any]:
        try:
            print("Starting video processing...")
            final_video_order, _ = self.prepare_resources()
            concat_file_path = self.process_videos(final_video_order)
            output_path = self.merge_videos(concat_file_path)

            output_key = f'users_videos/{self.event["videoConfig"]["uploadedResources"]["folderId"]}.mp4'

            print(f"Uploading final video to S3: {output_key}")
            self.s3_client.upload_file(output_path, OUTPUT_BUCKET_NAME, output_key)

            processing_time = time.time() - self.start_time
            print(f"Processing completed in {processing_time:.2f} seconds")

            return {
                'statusCode': 200,
                'body': {
                    'url': f's3://{OUTPUT_BUCKET_NAME}/{output_key}',
                    'processingTime': f"{processing_time:.2f} segundos",
                    'clipCount': len(final_video_order)
                }
            }

        except Exception as e:
            print(f"Error in process: {str(e)}")
            return {
                'statusCode': 500,
                'body': f'Error: {str(e)}'
            }
        finally:
            self.cleanup()

def lambda_handler(event, context):
    processor = VideoProcessor(event)
    return processor.process()

mockInput = {
    "videoConfig": {
        "trackOrder": [
            "initial",
            "template",
            "card",
            "template",
            "card",
            "template",
            "card",
            "template",
            "card",
            "template",
            "final"
        ],
        "uploadedResources": {
            "bucketKey": "s3://retrospet-photos-users/users_photos/test",
            "folderId": "9d91b9ad-be6b-40fd-9a11-152cc6f55c87"
        },
        "textOptions": {
            "firstLine": "Diogo, asdasdsads",
            "secondLine": "sadasdsada, sadasdsads",
            "thirdLine": "& sadasdasda"
        }
    },
    "static": {
        "initial": {
            "refId": "initial",
            "quantity": 2,
            "clips": [
                {
                    "link": "static/initial/capa.mp4",
                    "type": "video",
                    "metadata": {
                        "pets": [
                            {
                                "name": "asdasdsads",
                                "type": "GATO"
                            },
                            {
                                "name": "sadasdsada",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdsads",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdasda",
                                "type": "CAO"
                            }
                        ],
                        "clipType": "initial"
                    }
                },
                {
                    "link": "static/initial/nome_pet_e_dono.mp4",
                    "type": "video",
                    "metadata": {
                        "pets": [
                            {
                                "name": "asdasdsads",
                                "type": "GATO"
                            },
                            {
                                "name": "sadasdsada",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdsads",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdasda",
                                "type": "CAO"
                            }
                        ],
                        "clipType": "initial"
                    }
                }
            ],
            "ordened": 1
        },
        "cards": {
            "refId": "card",
            "quantity": 5,
            "clips": [
                {
                    "link": "static/cartelas/cao/VerdeClaro-01.mp4",
                    "type": "video",
                    "metadata": {
                        "pets": [
                            {
                                "name": "asdasdsads",
                                "type": "GATO"
                            },
                            {
                                "name": "sadasdsada",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdsads",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdasda",
                                "type": "CAO"
                            }
                        ],
                        "clipType": "card"
                    }
                },
                {
                    "link": "static/cartelas/cao/VerdeEscuro-08.mp4",
                    "type": "video",
                    "metadata": {
                        "pets": [
                            {
                                "name": "asdasdsads",
                                "type": "GATO"
                            },
                            {
                                "name": "sadasdsada",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdsads",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdasda",
                                "type": "CAO"
                            }
                        ],
                        "clipType": "card"
                    }
                },
                {
                    "link": "static/cartelas/cao/VerdeClaro-06.mp4",
                    "type": "video",
                    "metadata": {
                        "pets": [
                            {
                                "name": "asdasdsads",
                                "type": "GATO"
                            },
                            {
                                "name": "sadasdsada",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdsads",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdasda",
                                "type": "CAO"
                            }
                        ],
                        "clipType": "card"
                    }
                },
                {
                    "link": "static/cartelas/cao/VerdeEscuro-02.mp4",
                    "type": "video",
                    "metadata": {
                        "pets": [
                            {
                                "name": "asdasdsads",
                                "type": "GATO"
                            },
                            {
                                "name": "sadasdsada",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdsads",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdasda",
                                "type": "CAO"
                            }
                        ],
                        "clipType": "card"
                    }
                },
                {
                    "link": "static/cartelas/cao/VerdeClaro-05.mp4",
                    "type": "video",
                    "metadata": {
                        "pets": [
                            {
                                "name": "asdasdsads",
                                "type": "GATO"
                            },
                            {
                                "name": "sadasdsada",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdsads",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdasda",
                                "type": "CAO"
                            }
                        ],
                        "clipType": "card"
                    }
                }
            ],
            "ordened": 0
        },
        "templates": {
            "refId": "template",
            "quantity": 5,
            "clips": [
                {
                    "link": "static/templates/FundoFotoVerdeEscuro-06.mp4",
                    "type": "video",
                    "metadata": {
                        "pets": [
                            {
                                "name": "asdasdsads",
                                "type": "GATO"
                            },
                            {
                                "name": "sadasdsada",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdsads",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdasda",
                                "type": "CAO"
                            }
                        ],
                        "clipType": "template",
                        "rotation": {
                            "width": 878,
                            "height": 870,
                            "rotation": 1,
                            "x": 97,
                            "y": 457
                        },
                        "initialTimestamp": 6
                    }
                },
                {
                    "link": "static/templates/FundoFotoVerdeEscuro-03.mp4",
                    "type": "video",
                    "metadata": {
                        "pets": [
                            {
                                "name": "asdasdsads",
                                "type": "GATO"
                            },
                            {
                                "name": "sadasdsada",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdsads",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdasda",
                                "type": "CAO"
                            }
                        ],
                        "clipType": "template",
                        "rotation": {
                            "width": 900,
                            "height": 720,
                            "rotation": 3,
                            "x": 98,
                            "y": 560
                        },
                        "initialTimestamp": 15
                    }
                },
                {
                    "link": "static/templates/FundoFotoVerdeEscuro-02.mp4",
                    "type": "video",
                    "metadata": {
                        "pets": [
                            {
                                "name": "asdasdsads",
                                "type": "GATO"
                            },
                            {
                                "name": "sadasdsada",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdsads",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdasda",
                                "type": "CAO"
                            }
                        ],
                        "clipType": "template",
                        "rotation": {
                            "width": 800,
                            "height": 1060,
                            "rotation": 2.2,
                            "x": 129,
                            "y": 369
                        },
                        "initialTimestamp": 24
                    }
                },
                {
                    "link": "static/templates/FundoFotoVerdeEscuro-01.mp4",
                    "type": "video",
                    "metadata": {
                        "pets": [
                            {
                                "name": "asdasdsads",
                                "type": "GATO"
                            },
                            {
                                "name": "sadasdsada",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdsads",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdasda",
                                "type": "CAO"
                            }
                        ],
                        "clipType": "template",
                        "rotation": {
                            "width": 900,
                            "height": 900,
                            "rotation": 1.7,
                            "x": 85,
                            "y": 447
                        },
                        "initialTimestamp": 30
                    }
                },
                {
                    "link": "static/templates/FundoFotoVerdeMedio-06.mp4",
                    "type": "video",
                    "metadata": {
                        "pets": [
                            {
                                "name": "asdasdsads",
                                "type": "GATO"
                            },
                            {
                                "name": "sadasdsada",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdsads",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdasda",
                                "type": "CAO"
                            }
                        ],
                        "clipType": "template",
                        "rotation": {
                            "width": 878,
                            "height": 870,
                            "rotation": 1,
                            "x": 97,
                            "y": 457
                        },
                        "initialTimestamp": 39
                    }
                }
            ],
            "ordened": 0
        },
        "final": {
            "refId": "final",
            "quantity": 3,
            "clips": [
                {
                    "link": "static/final/final_step_1.mp4",
                    "type": "video",
                    "metadata": {
                        "pets": [
                            {
                                "name": "asdasdsads",
                                "type": "GATO"
                            },
                            {
                                "name": "sadasdsada",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdsads",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdasda",
                                "type": "CAO"
                            }
                        ],
                        "clipType": "final"
                    }
                },
                {
                    "link": "static/final/final_step_2.mp4",
                    "type": "video",
                    "metadata": {
                        "pets": [
                            {
                                "name": "asdasdsads",
                                "type": "GATO"
                            },
                            {
                                "name": "sadasdsada",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdsads",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdasda",
                                "type": "CAO"
                            }
                        ],
                        "clipType": "final"
                    }
                },
                {
                    "link": "static/final/final_step_3.mp4",
                    "type": "video",
                    "metadata": {
                        "pets": [
                            {
                                "name": "asdasdsads",
                                "type": "GATO"
                            },
                            {
                                "name": "sadasdsada",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdsads",
                                "type": "CAO"
                            },
                            {
                                "name": "sadasdasda",
                                "type": "CAO"
                            }
                        ],
                        "clipType": "final"
                    }
                }
            ],
            "ordened": 1
        }
    }
}

if __name__ == "__main__":
    lambda_handler(mockInput, None)
