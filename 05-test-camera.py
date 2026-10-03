import cv2
from ultralytics import YOLO


def main():
    # โหลดโมเดลที่ผ่านการฝึก (Trained Model)
    model = YOLO("runs/detect/train-v2/weights/best.pt")

    # เปิดใช้งานกล้องเว็บแคม
    # เลข 0 หมายถึงกล้องตัวแรกที่เชื่อมต่อกับคอมพิวเตอร์
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # ตั้งค่ากล้อง
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # ตรวจสอบว่าสามารถเปิดกล้องได้หรือไม่
    if not cap.isOpened():
        print("ไม่สามารถเปิดกล้องได้")
        return

    # แสดงคำแนะนำการใช้งาน
    print("กด 'q' เพื่อออกจากโปรแกรม")

    # เริ่มการตรวจจับแบบ Real-time
    while True:
        # อ่านภาพจากกล้องทีละเฟรม
        success, frame = cap.read()

        # ตรวจสอบว่าสามารถอ่านภาพจากกล้องได้
        if success:

            # นำภาพจากกล้องเข้าสู่โมเดล YOLO
            results = model.predict(
                source=frame,
                device=0,       # ใช้ GPU ในการประมวลผล
                verbose=False   # ไม่พิมพ์ log ทุกเฟรม
            )

            # วาดกรอบ Bounding Box และชื่อ Class ลงบนภาพ (ส่งเข้าไป 1 เฟรม จึงได้ผลลัพธ์ 1 ชุด)
            annotated_frame = results[0].plot()

            # แสดงภาพที่ผ่านการตรวจจับบนหน้าต่าง
            cv2.imshow("YOLO26 Real-time Detection", annotated_frame)

            # กดปุ่ม 'q' เพื่อออกจากโปรแกรม
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # หากไม่สามารถอ่านภาพจากกล้องได้ ให้หยุดการทำงาน
        else:
            break

    # ปิดการเชื่อมต่อกับกล้อง
    cap.release()

    # ปิดหน้าต่างแสดงผลทั้งหมด
    cv2.destroyAllWindows()


# เรียกใช้งานฟังก์ชัน main()
if __name__ == '__main__':
    main()