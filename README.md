### RECOGNITION
1. face_recognition: отложила запуск скриптов, так как dlib уже мало кем используется + проблемы с установкой этой библиотеки из-за вижуал студио
2. Opencv: умею находить растояние между снимками (их сходство), распознавать лица и части лица с помощью хаар каскада, работать с IP камерами и обрабатывать снимки 
3. Mediapipe: обнаруживать лицо (+написан модуль, чтобы импортировать и применять его в проектах) с точностью выше opencv, отрисовывать маску на человеке по вебке или видеозаписи
4. mtcnn: хорошо находит главные лицевые точки и координаты даже на некачественных темных снимках с вебки, на видео не проверяла
5. retinaface:
6. facenet-pytorch: 
7. Посмотреть библиотеки: deepface, Dlib MMOD, Dlib CNN, S3FD и InsightFace.
***Retinaface is better than MTCNN. SCRFD is better than Retinaface.***


### Motion Analysis
1. mediapipe: умею трекать движение руки и пальцев, отрисовывать скелет (позу) человека с вебки и на видеозаписи