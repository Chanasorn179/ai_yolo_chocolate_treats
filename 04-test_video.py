from ultralytics import YOLO

# โหลดโมเดลที่ผ่านการฝึก (Trained Model)
model = YOLO("runs/detect/train-3/weights/best.pt")

# กำหนดชื่อไฟล์วิดีโอที่ต้องการนำมาทดสอบ
video_to_test = "test_chocolate_treats_video.mp4"

# นำโมเดลไปทดสอบกับวิดีโอ
results = model.predict(
    source=video_to_test,   # กำหนดไฟล์วิดีโอที่ต้องการทดสอบ
    save=True,              # บันทึกวิดีโอผลลัพธ์หลังจากตรวจจับวัตถุ
    show=True,              # แสดงผลการตรวจจับวัตถุแบบเรียลไทม์ขณะประมวลผล
    conf=0.5                # กำหนดค่า Confidence ขั้นต่ำที่ 50%
)

# แสดงข้อความเมื่อการทดสอบเสร็จสิ้น
# Ultralytics สร้างโฟลเดอร์ผลลัพธ์ใหม่ทุกครั้ง (predict, predict-2, ...) จึงอ่านชื่อจริงจากผลลัพธ์
print(f"ทดสอบเสร็จสิ้น! สามารถเข้าดูวิดีโอผลลัพธ์ได้ที่โฟลเดอร์ {results[0].save_dir}")