import subprocess
import os
import random
import cv2
import numpy as np

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file tại đường dẫn: {file_path}")

def step_0_auto_remove_subtitles(input_path, output_path):
    """
    Bước 0: Dùng AI PaddleOCR quét vùng phụ đề phía dưới video, 
    dùng deep-translator dịch nội dung, và dùng OpenCV Inpainting để xóa bỏ phụ đề gốc.
    """
    print("[*] Bước 0: Đang khởi động AI quét, dịch phụ đề và xóa phụ đề gốc bên dưới video...")
    check_file_exists(input_path)
    
    try:
        from paddleocr import PaddleOCR
        from deep_translator import GoogleTranslator
    except ImportError:
        print("[!] Thiếu thư viện. Hãy chạy lệnh: pip install paddleocr opencv-python deep-translator")
        print("[!] Bỏ qua bước tự động xóa phụ đề và tiếp tục chạy các bước sau...")
        return input_path

    # Khởi động mô hình OCR nhận diện chữ (mặc định 'ch' cho tiếng Trung, bạn có thể đổi sang 'en' nếu video tiếng Anh)
    # Khởi động mô hình OCR chuẩn theo phiên bản thư viện hiện tại
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    translator = GoogleTranslator(source='auto', target='vi')
    
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Vùng chứa phụ đề nằm ở 25% phía dưới khung hình
    sub_y1 = int(height * 0.75)
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Cắt vùng phía dưới để OCR quét nhanh hơn
        sub_area = frame[sub_y1:height, 0:width]
        result = ocr.ocr(sub_area, cls=True)
        
        if result and result[0]:
            mask = np.zeros((height, width), dtype=np.uint8)
            for line in result[0]:
                box = line[0]
                text = line[1][0]
                
                # Ví dụ minh họa gọi deep-translator dịch chữ quét được sang tiếng Việt
                try:
                    translated_text = translator.translate(text)
                    # Ở đây bạn có thể lưu lại mốc thời gian (timestamp) và chữ dịch ra file .srt nếu muốn render lại sub Việt
                except Exception:
                    pass

                # Tạo mặt nạ để xóa vùng chữ gốc
                pts = np.array(box, dtype=np.int32)
                pts[:, 1] += sub_y1
                cv2.fillPoly(mask, [pts], 255)
            
            # Xóa vùng chữ bằng thuật toán Inpainting lấy mẫu nền xung quanh
            frame = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            
        out.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"[*] Đang xử lý xóa phụ đề gốc: {frame_idx}/{total_frames} frames...", end='\r')
            
    cap.release()
    out.release()
    print(f"\n[*] Đã hoàn tất xử lý phụ đề thành công!")
    return output_path

def step_1_trim_and_cut(input_path, temp_output):
    """
    Bước 1: Ngẫu nhiên hóa thời gian cắt bỏ đoạn đầu (từ 1.5 đến 3.5 giây).
    """
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

def step_2_extreme_visual_transform(input_path, temp_output):
    """
    Bước 2: Áp dụng Dynamic Zoom biến thiên theo khung hình & biến đổi màu sắc an toàn.
    """
    print("[*] Bước 2: Áp dụng Dynamic Zoom biến thiên theo khung hình & biến đổi màu sắc...")
    check_file_exists(input_path)
    
    contrast_val = round(random.uniform(1.03, 1.08), 2)
    brightness_val = round(random.uniform(0.01, 0.025), 3)
    noise_val = random.randint(10, 16)
    box_x = random.randint(1, 15)
    box_y = random.randint(1, 15)
    
    zoom_expr = "iw*(1.02 + 0.03*sin(n/25))"
    h_zoom_expr = "ih*(1.02 + 0.03*sin(n/25))"
    
    extreme_filters = (
        "hflip,"
        f"eq=contrast={contrast_val}:brightness={brightness_val}:saturation=1.04:gamma=0.97,"
        f"scale={zoom_expr}:{h_zoom_expr}:eval=frame,"
        f"crop=iw/1.03:ih/1.03,"
        f"noise=alls={noise_val}:allf=t+u,"
        f"drawbox=x={box_x}:y={box_y}:w=3:h=3:color=white@0.03:t=fill"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", extreme_filters,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "21",
        "-g", str(random.choice([24, 25, 30])),
        "-pix_fmt", "yuv420p",
        "-an",
        temp_output
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return temp_output

def step_3_warp_and_temporal_shift(input_path, output_path):
    """
    Bước 3: Ngẫu nhiên hóa méo thấu kính (Lens correction) và hệ số thời gian (setpts).
    """
    print("[*] Bước 3: Biến đổi méo hình học và dịch chuyển thời gian ngẫu nhiên...")
    check_file_exists(input_path)
    
    k1_val = round(random.uniform(0.01, 0.02), 4)
    speed_factor = round(random.uniform(1.02, 1.06), 3)
    
    advanced_filters = (
        f"lenscorrection=cx=0.5:cy=0.5:k1={k1_val}:k2=0.005,"
        f"setpts=1/{speed_factor}*PTS"
    )
    
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", advanced_filters,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "21",
        "-pix_fmt", "yuv420p",
        "-an",
        output_path
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return output_path

def process_pipeline(input_file, final_output):
    temp_clean_sub = "temp_clean_sub.mp4"
    temp_step1 = "temp_step1.mp4"
    temp_step2 = "temp_step2.mp4"
    
    try:
        # Bước 0: Phát hiện, dịch và xóa phụ đề gốc
        current_input = step_0_auto_remove_subtitles(input_file, temp_clean_sub)
        
        # Các bước lách bản quyền FFmpeg tiếp theo
        step_1_trim_and_cut(current_input, temp_step1)
        step_2_extreme_visual_transform(temp_step1, temp_step2)
        step_3_warp_and_temporal_shift(temp_step2, final_output)
        
        print(f"\n[SUCCESS] Hoàn tất toàn bộ quy trình! Thành phẩm: {final_output}")
        
    except Exception as e:
        print(f"\n[ERROR] Lỗi khi thực thi Pipeline: {e}")
    finally:
        for temp_file in [temp_clean_sub, temp_step1, temp_step2]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    INPUT_VIDEO = "input_videos.mp4"
    OUTPUT_VIDEO = "output_ultimate_translated.mp4"
    
    if os.path.exists(INPUT_VIDEO):
        process_pipeline(INPUT_VIDEO, OUTPUT_VIDEO)
    else:
        print(f"Không tìm thấy file nguồn '{INPUT_VIDEO}'.")