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
FFMPEG_LAYER_PATH_LOCAL = 'ffmpeg'
ENVIRONMENT = 'local'  # production | local
BUCKET_NAME = 'photos-processing'
OUTPUT_BUCKET_NAME = 'retrospet-photos-users'

MAX_WORKERS_PROCESS_CLIPS = 14
MAX_WORKERS_PROCESS_IMAGES = 20

class VideoProcessor:
    def __init__(self, event: Dict[str, Any]):
        print("Initializing VideoProcessor...")
        self.event = event
        self.ffmpeg_path = self.setup_ffmpeg()
        self.s3_client = self.configure_s3_client()
        self.temp_files: List[str] = []
        self.start_time = time.time()
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

                # Ajustar tamanho e aplicar rotação na imagem
                with Image.open(image_path).convert("RGBA") as img:
                #img = self.adjust_image(img, rotation_config)
                    img.save(image_output_path)

                # Adicionar entrada ao filtro complexo
                filter_complex_parts.append(
                    f"[{idx}:v]scale={rotation_config['width']}:{rotation_config['height']},setsar=1[v{idx}]"
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
                "ffmpeg",
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
                    "ffmpeg",
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

            if clip_type == "template":
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
        final_output = NamedTemporaryFile(suffix='_final.mp4', delete=False)
        self.temp_files.append(final_output.name)

        merge_cmd = [
            self.ffmpeg_path, '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_file_path,
            '-c:v', 'libx264',
            '-crf', '28',  # Aumentado de 23 para 28 (menor qualidade mas mais rápido)
            '-preset', 'veryfast',  # Mudado de 'medium' para 'veryfast'
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-threads', 'auto',
            '-tune', 'fastdecode',  # Adicionado para otimizar decodificação
            '-an',  # No audio
            final_output.name
        ]

        print("Executing merge command...")
        try:
            result = subprocess.run(merge_cmd, capture_output=True, text=True, check=True)
            print(f"Merge command output: {result.stdout}")
            print(f"Merge command error: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"Error merging videos: {e.stderr}")
            raise Exception(f"Failed to merge videos: {e.stderr}")

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

# if __name__ == "__main__":
#     lambda_handler(mockInput, None)
