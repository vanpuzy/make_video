import os
import subprocess
import random
import cv2
import numpy as np

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file tại đường dẫn: {file_path}")

def get_video_duration(input_path):
    """Lấy tổng thời lượng của video bằng OpenCV"""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return frame_count / fps
    return 0

def process_pipeline(input_file, final_output):
    try:
        check_file_exists(input_file)
        print("[*] Đang thiết lập các thông số biến đổi hình ảnh nâng cao và cắt khúc ngẫu nhiên...")
        
        duration = get_video_duration(input_file)
        if duration < 5:
            raise ValueError("Video quá ngắn để thực hiện chia đoạn.")

        # Chia video thành các chunk khoảng 4-8 giây
        chunk_size = random.uniform(4.0, 8.0)
        chunks = []
        start_t = random.uniform(1.0, 3.0) # Bỏ qua đoạn đầu một chút
        
        while start_t < duration - 3:
            end_t = min(start_t + chunk_size, duration - 1)
            chunks.append((start_t, end_t))
            start_t = end_t

        # Lặp lại ngẫu nhiên 1 đoạn nhỏ để phá cấu trúc thời gian đánh lừa AI
        if len(chunks) >= 3:
            random_idx = random.randint(0, len(chunks) - 2)
            chunks.insert(random_idx + 1, chunks[random_idx])
            print(f"[*] Đã kích hoạt tính năng: Lặp ngẫu nhiên đoạn từ {chunks[random_idx][0]:.2f}s đến {chunks[random_idx][1]:.2f}s")

        # Tạo biểu thức select cho FFmpeg
        select_expr = "+".join([f"between(t,{c[0]},{c[1]})" for c in chunks])
        
        # Biên độ ngẫu nhiên các hiệu ứng hình ảnh
        contrast_val = round(random.uniform(1.05, 1.12), 2)
        brightness_val = round(random.uniform(-0.02, 0.03), 3)
        saturation_val = round(random.uniform(1.04, 1.10), 2)
        noise_val = random.randint(12, 22)
        k1_val = round(random.uniform(0.015, 0.035), 4)
        speed_factor = round(random.uniform(1.05, 1.12), 3)
        
        zoom_val = round(random.uniform(1.05, 1.12), 3)
        crop_val = round(random.uniform(1.02, 1.05), 3)
        
        # Lật ngang ngẫu nhiên
        do_hflip = random.choice([True, False])
        hflip_filter = "hflip," if do_hflip else ""
        if do_hflip:
            print("[*] Kích hoạt chế độ: Lật ngang video (hflip).")

        vignette_filter = "vignette=angle=PI/4,"
        
        # Chuỗi filter hoàn chỉnh cho hình ảnh
        filter_complex = (
            f"select='{select_expr}',setpts=N/FRAME_RATE/TB,"
            f"{hflip_filter}"
            f"eq=contrast={contrast_val}:brightness={brightness_val}:saturation={saturation_val},"
            f"scale=iw*{zoom_val}:ih*{zoom_val},"
            f"crop=iw/{crop_val}:ih/{crop_val},"
            f"noise=alls={noise_val}:allf=t+u,"
            f"{vignette_filter}"
            f"lenscorrection=cx=0.5:cy=0.5:k1={k1_val}:k2=0.008,"
            f"setpts=1/{speed_factor}*PTS"
        )
        
        # Lệnh FFmpeg giữ lại audio gốc của video đầu vào (-map 0:a:0)
        cmd = [
            "ffmpeg", "-y",
            "-i", input_file,
            "-vf", filter_complex,
            "-af", f"atempo={speed_factor}", # Đẩy tốc độ audio khớp theo speed_factor của video
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "0:a:0",
            final_output
        ]
        
        print("[*] Đang thực thi render video với các hiệu ứng hình ảnh nâng cao (Dùng audio gốc)...")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f"[!] Lỗi FFmpeg:\n{result.stderr}")
            raise RuntimeError("FFmpeg xử lý thất bại.")
            
        print(f"\n[SUCCESS] Hoàn tất toàn bộ quy trình! Thành phẩm: {final_output}")
        
    except Exception as e:
        print(f"\n[ERROR] Lỗi khi thực thi Pipeline: {e}")

if __name__ == "__main__":
    INPUT_VIDEO = "input_videos.mp4"
    OUTPUT_VIDEO = "output_full_optimized.mp4"
    
    if os.path.exists(INPUT_VIDEO):
        process_pipeline(INPUT_VIDEO, OUTPUT_VIDEO)
    else:
        print(f"Không tìm thấy file nguồn '{INPUT_VIDEO}' để thực thi.")