import socket
import cv2
import numpy as np
import struct
import pickle
import time

import tensorrt as trt
from cuda import cudart  # Requires: pip3 install nvidia-cuda-python

import posix_ipc
import mmap
import argparse
import torch

# 공유 메모리 설정
SHARED_MEMORY_NAME = "/object_detection"
MEMORY_SIZE = 4096  # 1KB


class YOLOv5TRTNoPyCUDA:
    def __init__(self, engine_path, input_hw=(640, 640)):
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)
        self.input_hw = input_hw

        with open(engine_path, "rb") as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if not self.engine:
            raise RuntimeError("Failed to load TensorRT engine.")

        self.context = self.engine.create_execution_context()
        self.input_idx = self.engine.get_binding_index("input")
        self.output_idx = self.engine.get_binding_index("output")

        self.input_shape = list(self.engine.get_binding_shape(self.input_idx))
        self.output_shape = list(self.engine.get_binding_shape(self.output_idx))

        self.input_dtype = trt.nptype(self.engine.get_binding_dtype(self.input_idx))
        self.output_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_idx))

        self.host_input = np.zeros(self.input_shape, dtype=self.input_dtype)
        self.host_output = np.zeros(self.output_shape, dtype=self.output_dtype)

        self.d_input = None
        self.d_output = None

        self.context.set_binding_shape(self.input_idx, self.input_shape)

    def infer(self, frame):
        H, W = self.input_hw
        img = cv2.resize(frame, (W, H))
        img = img.astype(self.input_dtype) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        self.host_input[...] = img

        if self.d_input is None:
            n_input_bytes = self.host_input.nbytes
            n_output_bytes = self.host_output.nbytes
            self.d_input = cudart.cudaMalloc(n_input_bytes)[1]
            self.d_output = cudart.cudaMalloc(n_output_bytes)[1]

        # Host->Device
        cudart.cudaMemcpy(
            self.d_input,
            self.host_input.ctypes.data,
            self.host_input.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )

        bindings = [0] * self.engine.num_bindings
        bindings[self.input_idx] = self.d_input
        bindings[self.output_idx] = self.d_output

        self.context.execute_v2(bindings)

        # Device->Host
        cudart.cudaMemcpy(
            self.host_output.ctypes.data,
            self.d_output,
            self.host_output.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )

        return self.host_output

    import cv2

    def postprocess(self, output, original_hw, conf_threshold=0.25, nms_threshold=0.4):
        original_h, original_w = original_hw
        model_h, model_w = self.input_hw
        scale_h = original_h / model_h
        scale_w = original_w / model_w

        boxes = []
        confidences = []
        class_ids = []

        # YOLO 모델의 출력에서 바운딩 박스 정보 추출
        for det in output[0]:
            conf = det[4]
            if conf > conf_threshold:
                x1, y1, x2, y2 = det[0:4]
                x1 = int(x1 * scale_w)
                x2 = int(x2 * scale_w)
                y1 = int(y1 * scale_h)
                y2 = int(y2 * scale_h)
                obj_label = int(det[5])

                boxes.append([x1, y1, x2 - x1, y2 - y1])  # OpenCV NMS는 [x, y, w, h] 형식 사용
                confidences.append(float(conf))
                class_ids.append(obj_label)

        # NMS 적용
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        result = []
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, w, h = boxes[i]
                x2 = x1 + w
                y2 = y1 + h
                conf = confidences[i]
                obj_label = class_ids[i]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                result.append((x1, y1, x2, y2, conf, center_x, center_y, obj_label))

        return result


def start_camera_stream_client(engine_path="/home/group5/AI_Cannon/milestone2/model/yolov11/yolov11s_final.pt.engine"):
    # Clean up existing shared memory (if any)
    try:
        posix_ipc.unlink_shared_memory(SHARED_MEMORY_NAME)
        posix_ipc.unlink_semaphore(SHARED_MEMORY_NAME)
    except posix_ipc.ExistentialError:
        pass
    
    # Create shared memory and semaphore
    shared_memory = posix_ipc.SharedMemory(SHARED_MEMORY_NAME, posix_ipc.O_CREX, size=MEMORY_SIZE)
    semaphore = posix_ipc.Semaphore(SHARED_MEMORY_NAME, posix_ipc.O_CREX)
    memory_map = mmap.mmap(shared_memory.fd, MEMORY_SIZE)
    shared_memory.close_fd()

    # Initialize YOLO
    yolo = YOLOv5TRTNoPyCUDA(engine_path, input_hw=(640, 640))

    # Initialize USB camera
    cap = cv2.VideoCapture(0)  # 0 or the index of your USB camera
    if not cap.isOpened():
        print("Cannot open camera.")
        return
    
    # Setting TCP Server
    HOST = '0.0.0.0'
    PORT = 5001
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    print(f"Waiting for connection on {PORT}...")
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame data from camera.")
            break

        # Optional: flip frame if needed
        frame = cv2.flip(frame, -1)

        original_hw = frame.shape[:2]
        # print(original_hw)
        raw_output = yolo.infer(frame)
        # print(f"raw output: {raw_output}")
        detections = yolo.postprocess(raw_output, original_hw, conf_threshold=0.1, nms_threshold=0.4)
        # detections = yolo.postprocess(raw_output, original_hw)
        print("detection is:", detections)

        # Pack detection results into shared memory
        binary_data = struct.pack('i' + 'f' * 8 * len(detections),
                                  len(detections),
                                  *[value for bbox in detections for value in bbox])
        
        semaphore.acquire()
        memory_map.seek(0)
        memory_map.write(binary_data)
        semaphore.release()

        # TCP Connection
        data = pickle.dumps(frame)
        message_size = struct.pack("L", len(data))
        conn.sendall(message_size + data)

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(
             frame,
             f"FPS: {fps:.2f}",
             (10, 30),
             cv2.FONT_HERSHEY_SIMPLEX,
             1,
             (0, 0, 255),
             2
         )
        classes_dict = {2:"zombie", 1:"human", 0:"cat"}
        #classes_dict = {1:"zombie", 0:"cat"}
        # Draw bounding boxes

        for det in detections:
            x1, y1, x2, y2, conf, _, _, class_id = det
            if class_id == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{classes_dict[class_id]} ({conf:.2f})",
                    # f"class id:{class_id} ({conf:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
            elif class_id == 2:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"{classes_dict[class_id]} ({conf:.2f})",
                    # f"class id:{class_id} ({conf:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"{classes_dict[class_id]} ({conf:.2f})",
                    # f"class id:{class_id} ({conf:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )

        # Show the processed frame
        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    memory_map.close()
    shared_memory.unlink()
    semaphore.unlink()
    cap.release()
    cv2.destroyAllWindows()
    conn.close()
    server_socket.close()


if __name__ == "__main__":
    start_camera_stream_client(
        # engine_path="/home/group5/AI_Cannon/milestone2/model/yolov5/yolov5s_custom.engine",
        # engine_path="/home/group5/AI_Cannon/milestone2/model/yolov11/best.pt.engine",
        # engine_path="/home/group5/AI_Cannon/src/yolov5/bestv5.pt.engine",
        # engine_path="/home/group5/AI_Cannon/milestone2/model/yolov11/bestv11.pt.engine"
        engine_path="/home/group5/AI_Cannon/milestone2/model/yolov11/yolov11s_final.pt.engine"
    )
