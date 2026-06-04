import cv2


def extract_face(image_path):
    """
    Detect and crop face from image.
    Returns cropped face image.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read image")

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return image

    x, y, w, h = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    face = image[y:y+h, x:x+w]

    return face