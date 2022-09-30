import face_recognition
import cv2
from datetime import datetime, timedelta
import numpy as np
import platform
import pickle


known_face_encodings = []
known_face_metadata = []

def save_known_faces():
    with open("known_faces.dat", "wb") as face_data_file:
        face_data = [known_face_encodings, known_face_metadata]
        pickle.dump(face_data, face_data_file)
        print("Known faces backed up to disk.")


def load_known_faces():
    global known_face_encodings, known_face_metadata

    try:
        with open("known_faces.dat", "rb") as face_data_file:
            known_face_encodings, known_face_metadata = pickle.load(face_data_file)
            print("Known faces loaded from disk.")
    except FileNotFoundError as e:
        print("No previous face data found - starting with a blank known face list.")
        pass



def register_new_face(face_encoding, face_image):

    known_face_encodings.append(face_encoding)
    known_face_metadata.append({
        "first_seen": datetime.now(),
        "first_seen_this_interaction": datetime.now(),
        "last_seen": datetime.now(),
        "seen_count": 1,
        "seen_frames": 1,
        "face_image": face_image,
    })


def lookup_known_face(face_encoding):
    metadata = None

    if len(known_face_encodings) == 0:
        return metadata

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

    # ищем наиболее похожее лицо среди незнакомцев
    best_match_index = np.argmin(face_distances)

    # distance under 0.66 => face match.
    if face_distances[best_match_index] < 0.65:
        metadata = known_face_metadata[best_match_index]

        metadata["last_seen"] = datetime.now()
        metadata["seen_frames"] += 1

    return metadata


def main_loop():
    video_capture = cv2.VideoCapture(0)
    number_of_faces_since_save = 0
    while True:
        ret, frame = video_capture.read()
        # для увеличения скорости распознавания лиц
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Конвертируем в RGB
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # проверяем каждое обнаруженное лицо и если мы видели человека, то присвоим ему метку поверх видео
        face_labels = []
        for face_location, face_encoding in zip(face_locations, face_encodings):
            # знакомые лица
            metadata = lookup_known_face(face_encoding)
            if metadata is not None:
                time_at_door = datetime.now() - metadata['first_seen_this_interaction']
                face_label = f"At door {int(time_at_door.total_seconds())}s"

            # незнакомое лицо, регистрируем его
            else:
                face_label = "New visitor!"
                top, right, bottom, left = face_location
                face_image = small_frame[top:bottom, left:right]
                face_image = cv2.resize(face_image, (150, 150))
                register_new_face(face_encoding, face_image)
            face_labels.append(face_label)

        # баундинг бокс вокруг каждого лица
        for (top, right, bottom, left), face_label in zip(face_locations, face_labels):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        number_of_recent_visitors = 0
        for metadata in known_face_metadata:
            x_position = number_of_recent_visitors * 150
            frame[30:180, x_position:x_position + 150] = metadata["face_image"]
            number_of_recent_visitors += 1
            visits = metadata['seen_count']
            visit_label = f"{visits} visits"
            if visits == 1:
                visit_label = "First visit"
            cv2.putText(frame, visit_label, (x_position + 10, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        if number_of_recent_visitors > 0:
            cv2.putText(frame, "Visitors at Door", (5, 18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)


        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            save_known_faces()
            break


        # обнуляем список лиц во избежание падений
        # if len(face_locations) > 0 and number_of_faces_since_save > 100:
        #     save_known_faces()
        #     number_of_faces_since_save = 0
        # else:
        #     number_of_faces_since_save += 1


    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    load_known_faces()
    main_loop()