import face_recognition
from PIL import Image, ImageDraw

img_path = ''

def face_rec():
    face_img = face_recognition.load_image_file('')
    face_loc = face_recognition.face_locations(face_img)
    print(face_loc)
    print('Надено {len(face_loc)} лиц')

    pil_img1 = Image.fromarray(face_img)
    draw1 = ImageDraw.Draw(pil_img1)

    for (top, right, bottom, left) in face_loc:
        draw1.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0),width=4)

    del draw1
    pil_img1.save('images/new_img1.jpg')


def extracting_faces(img_path):
    count = 0
    faces = face_recognition.load_image_file(img_path)
    faces_locations = face_recognition.face_locations(faces)
    for face_location in faces_locations:
        top, right, bottom, left = face_location
        face_img = faces[top:bottom, left:right]
        pil_img = Image.fromarray(face_img)
        pil_img.save(f'images/{count}_face_img.jpg')
        count +=1

def compare_faces(img1_path, img2_path):
    img1 = face_recognition.load_image_file(img1_path)
    img1_encodings = face_recognition.face_encodings(img1)[0]

    img2 = face_recognition.load_image_file(img2_path)
    img2_encodings = face_recognition.face_encodings(img2)[0]

    result = face_recognition.compare_faces([img1_encodings], img2_encodings)
    print(result)


def main():
    # face_rec()
    print(extracting_faces('images/ne_tolko_sveta.jpg'))

if __name__ == '__main__':
    main()