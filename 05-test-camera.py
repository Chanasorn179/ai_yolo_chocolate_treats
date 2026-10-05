import os
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# จำนวนเฟรมย้อนหลังที่ใช้โหวต class ของวัตถุแต่ละชิ้น (มากขึ้น = นิ่งขึ้น แต่เปลี่ยนช้าลง)
VOTE_FRAMES = 15

# โฟลเดอร์เก็บภาพจากกล้องเมื่อกด 's' (นำไป label ใน Label Studio แล้วเทรนใหม่)
SAVE_DIR = "frame/images"


def main():
    # โหลดโมเดลที่ผ่านการฝึก (Trained Model)
    model = YOLO("runs/detect/train-5/weights/best.pt")
    names = model.names

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

    # ประวัติ (class, conf) ของแต่ละ track id ใช้โหวตแก้ปัญหา class กระพริบ เช่น beng-beng กลายเป็น sumo บางเฟรม
    history = defaultdict(lambda: deque(maxlen=VOTE_FRAMES))
    saved = 0

    # แสดงคำแนะนำการใช้งาน
    print("กด 'q' เพื่อออกจากโปรแกรม")
    print(f"กด 's' เพื่อบันทึกภาพจากกล้องลง {SAVE_DIR}/ (ใช้ label เพิ่มแล้วเทรนใหม่)")

    # เริ่มการตรวจจับแบบ Real-time
    while True:
        # อ่านภาพจากกล้องทีละเฟรม
        success, frame = cap.read()

        # หากไม่สามารถอ่านภาพจากกล้องได้ ให้หยุดการทำงาน
        if not success:
            break

        # ใช้ tracking แทน predict เพื่อให้วัตถุชิ้นเดิมได้ id เดิมในทุกเฟรม
        results = model.track(
            source=frame,
            persist=True,   # จำ track ข้ามเฟรม
            conf=0.3,
            device=0,       # ใช้ GPU ในการประมวลผล
            verbose=False   # ไม่พิมพ์ log ทุกเฟรม
        )
        boxes = results[0].boxes

        annotator = Annotator(frame.copy(), line_width=3)
        if boxes.id is not None:
            for xyxy, track_id, cls, conf in zip(
                boxes.xyxy.tolist(), boxes.id.int().tolist(),
                boxes.cls.int().tolist(), boxes.conf.tolist()
            ):
                history[track_id].append((cls, conf))

                # โหวตแบบถ่วงน้ำหนักด้วย confidence จากเฟรมย้อนหลัง
                score = defaultdict(float)
                for c, p in history[track_id]:
                    score[c] += p
                best = max(score, key=score.get)
                avg_conf = score[best] / len(history[track_id])

                annotator.box_label(xyxy, f"{names[best]} {avg_conf:.2f}", color=colors(best, True))

        # แสดงภาพที่ผ่านการตรวจจับบนหน้าต่าง
        cv2.imshow("YOLO26 Real-time Detection", annotator.result())

        key = cv2.waitKey(1) & 0xFF
        # กดปุ่ม 'q' เพื่อออกจากโปรแกรม
        if key == ord('q'):
            break
        # กดปุ่ม 's' เพื่อบันทึกภาพดิบ (ไม่มีกรอบ) ไว้ label เพิ่ม
        if key == ord('s'):
            os.makedirs(SAVE_DIR, exist_ok=True)
            while os.path.exists(path := os.path.join(SAVE_DIR, f"cam_{saved:04d}.jpg")):
                saved += 1
            cv2.imwrite(path, frame)
            print(f"บันทึกภาพ: {path}")

    # ปิดการเชื่อมต่อกับกล้อง
    cap.release()

    # ปิดหน้าต่างแสดงผลทั้งหมด
    cv2.destroyAllWindows()


# เรียกใช้งานฟังก์ชัน main()
if __name__ == '__main__':
    main()
