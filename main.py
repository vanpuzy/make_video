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

def step_0_auto_remove_subtitles_and_tts(input_path, output_path, audio_output_path):
    """
    Bước 0: Quét phụ đề gốc, dịch sang tiếng Việt, xóa sub cũ bằng Inpainting,
    đồng thời gom nội dung dịch để tạo file thuyết minh tiếng Việt (TTS).
    """
    print("[*] Bước 0: Đang khởi động AI quét, dịch phụ đề, xóa sub gốc và tạo âm thanh tiếng Việt...")
    check_file_exists(input_path)
    
    try:
        from paddleocr import PaddleOCR
        from deep_translator import GoogleTranslator
    except ImportError:
        print("[!] Thiếu thư viện PaddleOCR/deep-translator.")
        return input_path

    ocr = PaddleOCR(lang='en')
    translator = GoogleTranslator(source='auto', target='vi')
    
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0:
        fps = 30.0
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
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
                    det_boxes = res.json['text_det_res'].get('box', [])
                    rec_texts = res.json.get('text_rec_res', [])
                    
                    if det_boxes:
                        mask = np.zeros((height, width), dtype=np.uint8)
                        for box, text_item in zip(det_boxes, rec_texts):
                            text = text_item.get('text', '') if isinstance(text_item, dict) else str(text_item)
                            try:
                                translated_text = translator.translate(text)
                                if translated_text and translated_text not in all_translated_sentences:
                                    all_translated_sentences.append(translated_text)
                            except Exception:
                                pass

                            pts = np.array(box, dtype=np.int32)
                            pts[:, 1] += sub_y1
                            cv2.fillPoly(mask, [pts], 255)
                        
                        frame = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        except Exception:
            pass
            
        out.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"[*] Xử lý khung hình xóa sub: {frame_idx}/{total_frames}", end='\r')
            
    cap.release()
    out.release()
    
    print("\n[*] Đang tổng hợp giọng đọc tiếng Việt (Text-to-Speech)...")
    full_text = ". ".join(all_translated_sentences) if all_translated_sentences else "Video hấp dẫn."
    tts = gTTS(text=full_text, lang='vi', slow=False)
    tts.save(audio_output_path)
    print(f"[*] Đã tạo xong file lồng tiếng: {audio_output_path}")
    
    return output_path

def step_1_trim_and_cut(input_path, temp_output):
    random_start = round(random.uniform(1.5, 3.5), 2)
    print(f"[*] Bước 1: Ngẫu nhiên cắt bỏ {random_start}s đầu của video...")
    check_file_exists(input_path)
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(random_start),
        "-i", input_path,
        "-c", "copy",
        temp_output
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return temp_output

def step_2_advanced_visual_mutation(input_path, temp_output):
    print("[*] Bước 2: Áp dụng biến đổi hình ảnh đa tầng chống AI quét bản quyền (Bản tối ưu an toàn)...")
    check_file_exists(input_path)
    
    contrast_val = round(random.uniform(1.04, 1.09), 2)
    brightness_val = round(random.uniform(0.015, 0.03), 3)
    saturation_val = round(random.uniform(1.03, 1.07), 2)
    noise_val = random.randint(10, 15)
    box_x = random.randint(2, 20)
    box_y = random.randint(2, 20)
    
    # Sử dụng chuỗi filter tối ưu, tránh lỗi biểu thức scale eval=frame trên một số phiên bản FFmpeg của Linux
    advanced_filters = (
        "hflip,"
        f"eq=contrast={contrast_val}:brightness={brightness_val}:saturation={saturation_val},"
        "scale=iw*1.04:ih*1.04,"
        "crop=iw/1.02:ih/1.02,"
        f"noise=alls={noise_val}:allf=t+u,"
        f"drawbox=x={box_x}:y={box_y}:w=4:h=4:color=white@0.02:t=fill"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", advanced_filters,
        "-c:v", "libx264",
        "-preset", "veryfast",  # Đẩy tốc độ render nhanh hơn trên CPU EC2
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-an",
        temp_output
    ]
    
    # Chạy lệnh và in ra lỗi chi tiết nếu có sự cố từ FFmpeg
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"[!] Lỗi FFmpeg chi tiết:\n{result.stderr}")
        raise RuntimeError("FFmpeg xử lý hình ảnh thất bại ở Bước 2.")
        
    return temp_output

def step_3_warp_and_mix_audio(input_video_path, input_audio_path, output_path):
    print("[*] Bước 3: Biến đổi méo quang học, dịch chuyển thời gian & mix âm thanh tiếng Việt...")
    check_file_exists(input_video_path)
    check_file_exists(input_audio_path)
    
    k1_val = round(random.uniform(0.012, 0.025), 4)
    speed_factor = round(random.uniform(1.03, 1.07), 3)
    
    temporal_filters = (
        f"lenscorrection=cx=0.5:cy=0.5:k1={k1_val}:k2=0.005,"
        f"setpts=1/{speed_factor}*PTS"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video_path,
        "-i", input_audio_path,
        "-vf", temporal_filters,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return output_path

def process_pipeline(input_file, final_output):
    temp_clean_sub = "temp_clean_sub.mp4"
    temp_audio = "temp_tts_audio.mp3"
    temp_step1 = "temp_step1.mp4"
    temp_step2 = "temp_step2.mp4"
    
    try:
        current_input = step_0_auto_remove_subtitles_and_tts(input_file, temp_clean_sub, temp_audio)
        
        step_1_trim_and_cut(current_input, temp_step1)
        step_2_advanced_visual_mutation(temp_step1, temp_step2)
        step_3_warp_and_mix_audio(temp_step2, temp_audio, final_output)
        
        print(f"\n[SUCCESS] Hoàn tất toàn bộ quy trình nâng cao! Thành phẩm: {final_output}")
        
    except Exception as e:
        print(f"\n[ERROR] Lỗi khi thực thi Pipeline: {e}")
    finally:
        for temp_file in [temp_clean_sub, temp_audio, temp_step1, temp_step2]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    INPUT_VIDEO = "input_videos.mp4"
    OUTPUT_VIDEO = "output_ultimate_advanced_copyright_bypass.mp4"
    
    if os.path.exists(INPUT_VIDEO):
        process_pipeline(INPUT_VIDEO, OUTPUT_VIDEO)
    else:
        print(f"Không tìm thấy file nguồn '{INPUT_VIDEO}'.")