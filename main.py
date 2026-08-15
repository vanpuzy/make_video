import os
os.environ["FLAGS_use_mkldnn"] = "0"  # Vô hiệu hóa oneDNN để tránh lỗi tính toán trên CPU

import subprocess
import random
import cv2
import numpy as np
from gtts import gTTS

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file tại đường dẫn: {file_path}")

def get_video_duration(input_path):
    """Lấy tổng thời lượng của video bằng ffprobe/opencv"""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return frame_count / fps
    return 0

def step_0_auto_remove_subtitles_and_tts(input_path, audio_output_path):
    print("[*] Bước 0: Đang khởi động AI quét, dịch phụ đề và tạo âm thanh tiếng Việt...")
    check_file_exists(input_path)
    
    try:
        from paddleocr import PaddleOCR
        from deep_translator import GoogleTranslator
    except ImportError:
        print("[!] Thiếu thư viện PaddleOCR/deep-translator. Dùng audio mặc định.")
        tts = gTTS(text="Video hấp dẫn.", lang='vi', slow=False)
        tts.save(audio_output_path)
        return audio_output_path

    ocr = PaddleOCR(lang='en')
    translator = GoogleTranslator(source='auto', target='vi')
    
    cap = cv2.VideoCapture(input_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    sub_y1 = int(height * 0.75)
    frame_idx = 0
    all_translated_sentences = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        sub_area = frame[sub_y1:height, 0:width]
        
        try:
            output_results = ocr.predict(sub_area)
            for res in output_results:
                if hasattr(res, 'json') and 'text_det_res' in res.json:
                    rec_texts = res.json.get('text_rec_res', [])
                    for text_item in rec_texts:
                        text = text_item.get('text', '') if isinstance(text_item, dict) else str(text_item)
                        try:
                            translated_text = translator.translate(text)
                            if translated_text and translated_text not in all_translated_sentences:
                                all_translated_sentences.append(translated_text)
                        except Exception:
                            pass
        except Exception:
            pass
            
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"[*] Quét khung hình OCR: {frame_idx}/{total_frames}", end='\r')
            
    cap.release()
    
    print("\n[*] Đang tổng hợp giọng đọc tiếng Việt (Text-to-Speech)...")
    full_text = ". ".join(all_translated_sentences) if all_translated_sentences else "Video hấp dẫn."
    tts = gTTS(text=full_text, lang='vi', slow=False)
    tts.save(audio_output_path)
    print(f"[*] Đã tạo xong file lồng tiếng: {audio_output_path}")
    
    return audio_output_path

def process_pipeline(input_file, final_output):
    temp_audio = "temp_tts_audio.mp3"
    
    try:
        # Bước 0: Tạo file Audio tiếng Việt từ OCR & Dịch
        audio_path = step_0_auto_remove_subtitles_and_tts(input_file, temp_audio)
        
        print("[*] Đang thiết lập các thông số biến đổi hình ảnh nâng cao và cắt khúc ngẫu nhiên...")
        
        duration = get_video_duration(input_file)
        if duration < 5:
            raise ValueError("Video quá ngắn để thực hiện chia đoạn.")

        # Ý TƯỞNG MỚI: Cắt video thành các đoạn nhỏ và xáo trộn/lặp lại ngẫu nhiên một đoạn
        # Ví dụ: Chia video thành các chunk khoảng 4-8 giây
        chunk_size = random.uniform(4.0, 8.0)
        chunks = []
        start_t = random.uniform(1.0, 3.0) # Bỏ qua đoạn đầu một chút
        
        while start_t < duration - 3:
            end_t = min(start_t + chunk_size, duration - 1)
            chunks.append((start_t, end_t))
            start_t = end_t

        # Nếu có từ 3 chunk trở lên, ta có thể lặp lại ngẫu nhiên 1 đoạn nhỏ để đánh lừa AI
        if len(chunks) >= 3:
            # Chọn ngẫu nhiên 1 đoạn để lặp lại 2 lần liên tiếp (tạo hiệu ứng giật tít, giữ chân người xem hoặc phá cấu trúc)
            random_idx = random.randint(0, len(chunks) - 2)
            chunks.insert(random_idx + 1, chunks[random_idx]) # Chèn lặp lại đoạn đó
            print(f"[*] Đã kích hoạt tính năng: Lặp ngẫu nhiên đoạn từ {chunks[random_idx][0]}s đến {chunks[random_idx][1]}s")

        # Tạo chuỗi filter_complex cho FFmpeg để cắt ghép các đoạn (Select/Trim qua expression hoặc dùng concat filter)
        # Cách tối ưu trong 1 câu lệnh FFmpeg duy nhất cho nhiều phân đoạn:
        # Sử dụng enable='between(t,start1,end1)+between(t,start2,end2)...' hoặc dùng lệnh cắt phức hợp.
        # Ở đây dùng phương pháp `select` cực kỳ mạnh mẽ của FFmpeg để giữ lại và xắp xếp các khoảng thời gian:
        
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
        
        # Chuỗi filter hoàn chỉnh gồm: Chọn khung hình theo phân đoạn -> Lật -> Màu -> Zoom -> Crop -> Noise -> Vignette -> Lens -> Tốc độ
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
        
        cmd = [
            "ffmpeg", "-y",
            "-i", input_file,
            "-i", audio_path,
            "-vf", filter_complex,
            "-af", "atempo=1.0", # Giữ ổn định âm thanh nền nếu cần, ở đây dùng audio lồng tiếng riêng nên dùng -map audio mới
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            final_output
        ]
        
        print("[*] Đang thực thi render video với cắt khúc ngẫu nhiên & lặp đoạn...")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f"[!] Lỗi FFmpeg:\n{result.stderr}")
            raise RuntimeError("FFmpeg xử lý thất bại.")
            
        print(f"\n[SUCCESS] Hoàn tất toàn bộ quy trình! Thành phẩm: {final_output}")
        
    except Exception as e:
        print(f"\n[ERROR] Lỗi khi thực thi Pipeline: {e}")
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

if __name__ == "__main__":
    INPUT_VIDEO = "input_videos.mp4"
    OUTPUT_VIDEO = "output_full_optimized.mp4"
    
    if os.path.exists(INPUT_VIDEO):
        process_pipeline(INPUT_VIDEO, OUTPUT_VIDEO)
    else:
        print(f"Không tìm thấy file nguồn '{INPUT_VIDEO}' để thực thi.")