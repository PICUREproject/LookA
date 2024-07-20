import cv2
import mediapipe as mp
import numpy as np
import time

class FaceLook:
    def __init__(self):
        # 웹캠 열기
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open webcam.")
            return

        # MediaPipe FaceMesh 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()

        # 추가적인 초기화 작업
        self.text_box = None  # 여기에 텍스트 박스 초기화 코드 추가

    def calculate_rotation_angle(self, left_eye, right_eye, nose_tip):
        # 두 눈의 중심 계산
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)

        # 눈 중심과 코 끝점 간의 벡터 계산
        eye_center = (left_eye_center + right_eye_center) / 2
        nose_vector = nose_tip - eye_center

        # 벡터의 각도 계산
        angle = np.arctan2(nose_vector[1], nose_vector[0])
        angle = np.degrees(angle)

        return angle

    def capture(self):
        start_time = time.time()
        dangerous_count = 0
        good_count = 0

        while time.time() - start_time < 10:
            # 웹캠에서 프레임 읽기
            ret, frame = self.cap.read()
            if not ret:
                break

            # 좌우반전
            frame = cv2.flip(frame, 1)

            # 이미지 크기 얻기
            image_height, image_width, _ = frame.shape

            # 이미지를 RGB 형식으로 변환
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 얼굴 특성 추출
            results = self.face_mesh.process(image_rgb)

            # 추출된 얼굴 랜드마크를 리스트로 변환합니다.
            landmarks = []
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * image_width)
                        y = int(landmark.y * image_height)
                        landmarks.append((x, y))

                # 필요한 랜드마크 인덱스를 정의합니다.
                nose_tip = np.array(landmarks[1])  # 코 끝점
                left_eye = np.array(landmarks[133:153])  # 왼쪽 눈
                right_eye = np.array(landmarks[362:382])  # 오른쪽 눈

                # 회전 각도 계산
                angle = self.calculate_rotation_angle(left_eye, right_eye, nose_tip)

                # 각도 범위 확인하여 dangerous 또는 good 세기
                if angle < -120 or angle > -30:
                    dangerous_count += 1
                else:
                    good_count += 1

                # 얼굴 중심점 표시
                cv2.circle(frame, tuple(nose_tip), 5, (0, 0, 255), -1)  # 코 끝점을 빨간색 점으로 표시

                # 이미지 창에 랜드마크 그리기
                for landmark in landmarks:
                    cv2.circle(frame, tuple(landmark), 2, (0, 255, 0), -1)  # 랜드마크를 초록색 점으로 표시

            # 이미지 창에 출력
            cv2.imshow('Face Landmarks', frame)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 웹캠 해제 및 창 닫기
        self.cap.release()
        cv2.destroyAllWindows()

        # 결과 출력
        if dangerous_count > good_count:
            print(True)
        else:
            print(False)

    def __del__(self):
        # 웹캠 해제 및 창 닫기
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

# FaceLook 객체 생성 및 capture 메서드 호출
face_look = FaceLook()
face_look.capture()
