import cv2
import numpy as np

# ======= 1. Wczytanie zdjęcia referencyjnego i wykrycie twarzy =======

# Ładowanie Haar Cascade do wykrywania twarzy
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Wczytanie obrazu referencyjnego w skali szarości
reference_img = cv2.imread("ref.jpg", cv2.IMREAD_GRAYSCALE)
if reference_img is None:
    print("Błąd: Nie udało się wczytać ref.jpg!")
    exit()

# Wykrycie twarzy na zdjęciu referencyjnym
faces = face_cascade.detectMultiScale(reference_img, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

if len(faces) == 0:
    print("Błąd: Nie znaleziono twarzy w ref.jpg!")
    exit()

# Pobranie pierwszej wykrytej twarzy jako wzorca
(x, y, w, h) = faces[0]
reference_face = reference_img[y:y+h, x:x+w]

# Normalizacja histogramu dla lepszego kontrastu
reference_face = cv2.equalizeHist(reference_face)

# Skalowanie do jednolitego rozmiaru
fixed_size = (150, 150)
reference_face = cv2.resize(reference_face, fixed_size)


# ======= 2. Funkcja augmentacji obrazu =======
def augment_image(image):
    flipped = cv2.flip(image, 1)  # Odbicie lustrzane
    rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
    rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)  # Rozmycie
    bright = cv2.convertScaleAbs(image, alpha=1.2, beta=30)  # Rozjaśnienie
    dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-30)  # Przyciemnienie
    high_contrast = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    low_contrast = cv2.convertScaleAbs(image, alpha=0.5, beta=0)
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    return [image, flipped, blurred, bright, dark, noisy, high_contrast, low_contrast, rotated_90, rotated_180, rotated_270]


# Tworzenie zestawu treningowego
augmented_faces = augment_image(reference_face)
labels = [0] * len(augmented_faces)

# ======= 3. Trening LBPH Face Recognizer =======
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(augmented_faces, np.array(labels))

# ======= 4. Rozpoznawanie twarzy w kamerze =======

# Uruchomienie kamerki
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Błąd: Brak obrazu z kamerki.")
        break

    # Konwersja obrazu na skalę szarości
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wykrycie twarzy w kamerze
    detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in detected_faces:
        # Pobranie obszaru twarzy
        face = gray[y:y+h, x:x+w]
        face = cv2.equalizeHist(face)  # Normalizacja kontrastu
        face = cv2.resize(face, fixed_size)  # Skalowanie do spójnego rozmiaru

        # Rozpoznanie twarzy
        label, confidence = face_recognizer.predict(face)

        print(f"Confidence: {confidence}")
        print(f"Label: {label}")

        # Jeśli confidence < 60, twarz pasuje (niższa wartość = lepsze dopasowanie)
        if confidence < 60:
            color = (0, 255, 0)
            text = f"Matched ({round(confidence, 2)})"
        else:
            color = (0, 0, 255)
            text = "Not Matched"

        # Rysowanie prostokąta i tekstu
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Wyświetlenie obrazu
    cv2.imshow("Face Recognition", frame)

    # Wyjście po wciśnięciu 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnienie zasobów
cap.release()
cv2.destroyAllWindows()
