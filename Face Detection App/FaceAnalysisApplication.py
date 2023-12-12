# pylint: disable=no-member
# Gerekli kütüphaneleri içe aktarın
# Import required libraries
import cv2
import joblib
import mediapipe as mp
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# MediaPipe yüz ağı algılama modelini yükleyin
# Load MediaPipe face mesh detection model
mp_face_mesh = mp.solutions.face_mesh
#Sadece bir yüz algılanacak
#Only one face will be detected
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.2, min_tracking_confidence=0.2)

# Yapay zeka modelini yükle

model = DecisionTreeClassifier()

# Tüm etiketleri önceden tanımlayın
# Pre-define all tags
all_labels = ['No Skin', 'Acne', 'Kuru Cilt']
le = LabelEncoder()
le.fit(all_labels)

def load_and_preprocess_images():
    images = []
    labels = []

    def process_images(label, folder, count):
        for i in range(1, count + 1):
            image_path = f"{folder}/{i}.jpg"
            #Görüntü okunur
            image = cv2.imread(image_path)
            if image is not None:
                #Görüntü boyutlandırılır ve renk normalizasyonu yapılır
                resized_image, normalized_image = preprocess_image(image)
                if resized_image is not None and normalized_image is not None:
                  #images ve label listelerine eklenir
                    images.append(normalized_image)
                    labels.append(label)

    #Bu projede sadece 1er örnek kullanılmıştır. Daha doğru tahminler için örnek sayıları arttırılabilir
    #Only 1 sample was used in this project. Sample numbers can be increased for more accurate predictions
    process_images("No Skin", "NoSkinProblem", 1)
    process_images("Acne", "Acne", 1)
    process_images("Kuru Cilt", "DrySkin", 1)

    #Pixel değerlerini içerir
    #Contains pixel values
    images = np.array(images)
    #Label değerlerini içerir
    #Contains label values
    labels = np.array(labels)
    le.fit(labels)
    return images, labels

def preprocess_image(image):
    if image is None or image.size == 0:
        print("Hata: Geçersiz veya boş bir görüntü.")
        return None, None

    try:
        resized_image = cv2.resize(image, (224, 224))
        normalized_image = resized_image / 255.0
        return resized_image, normalized_image
    except Exception as e:
        print(f"Error: An error occurred while resizing the image. Error: {e}")
        return None, None

#X_train görüntülerin pixel değerlerini içeren bir NumPy dizisi
#y_train eğitim veri setinin hedef etiketleri, sınıf bilgilerini içeren numPY dizisi
#X_train is a NumPy array containing pixel values of​​  images
#y_train numPY array containing target labels, class information of the training dataset
def train_and_save_model(X_train, y_train, model_filename='decision_tree_model.joblib'):
    #X_traini düzleştirir| Veri setini modelin kullanabileceği formata getirir
    #Flattenes the x_train| Converts the data set into a format that the model can use
    X_train_flattened = X_train.reshape(X_train.shape[0], -1)
    #Düzleştirilmiş veri setini hedef etiketleri kullanarak modeli eğitir
    #Fit fonksiyonu modelin öğrenmesini sağlar
    #Trains the model using target labels on the flattened dataset
    #Fit function allows the model to learn
    model.fit(X_train_flattened, y_train)
    #Eğitilen modeli belirtilen dosya adıyla bir dosyaya kaydeder.
    #Save the trained model to a file with the specified file name.
    joblib.dump(model, model_filename)

#Eğitilen model dosyası yüklenir
#The trained model file is loaded
def load_model(model_filename='decision_tree_model.joblib'):
    return joblib.load(model_filename)

#Bir frame üzerinden yüz analizi yapmak için kullanılır
#Used to perform facial analysis over a frame
def analyze_face(frame, model, le, X_test):
    results = face_mesh.process(frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            facial_landmarks = np.array([(landmark.x, landmark.y, landmark.z) for landmark in landmarks.landmark]).flatten()
            facial_landmarks_flat = facial_landmarks.flatten()[:X_test.shape[1]]
            #Düzenlenen yüz noktalarını içeren vektör,eğitimde kullanılan özellik sayısı kadar kırpılır
            num_features_used_in_training = X_test.shape[1]
            facial_landmarks_flat = facial_landmarks.flatten()[:num_features_used_in_training]
            #Eksik olan özellikleri tamamlar
            if len(facial_landmarks_flat) < num_features_used_in_training:
                facial_landmarks_flat = np.pad(facial_landmarks_flat, (0, num_features_used_in_training - len(facial_landmarks_flat)))
            #Özellik vektörü haline getilir
            facial_landmarks_flat = facial_landmarks_flat.reshape(1, -1)[:,:num_features_used_in_training]

            print("Number of training data features:", X_test.shape[1])
            print("Number of real-time features:", facial_landmarks_flat.shape[1])
            print("Classes used in education:", le.classes_)
            X_test_flattened = X_test.reshape(X_test.shape[0], -1)
            #Model özellik vektörü üzerinde tahmin yapar
            #The model makes predictions on the feature vector
            prediction = model.predict(X_test_flattened)
            #Sonucu bir array içerisine kayıt eder
            #Save the result into an array
            prediction_array = np.array(prediction)
            print("Prediction Array:", prediction_array)
            #Yapılan cilt durumu tahminini ekrana yazdırır
            #Prints the skin condition estimate to the screen
            skin_condition = prediction_array[0]
            cv2.putText(frame, f"Cilt Durumu: {skin_condition}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
    return frame

# Eğitim için veri setini yükle
# Load dataset for training
X, y = load_and_preprocess_images()

# Veriyi eğitim ve test setlerine bölün
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğitin ve kaydedin
# Train and save the model
train_and_save_model(X_train, y_train)

# Modeli yükleyin
# Load the model
loaded_model = load_model()

# Kamera açma ve analiz yapma
# Open camera and analyze
cap = cv2.VideoCapture(0)

while True:
    # Kameradan bir kare al
    # Take a frame from the camera
    ret, frame = cap.read()

    # Yüz ağı algılama işlemini uygula ve cilt durumunu analiz et
    # Apply facial mesh detection and analyze skin condition
    frame = analyze_face(frame, loaded_model, le, X_test)

    # Çerçeveyi göster
    # Show frame
    cv2.imshow("Face Analysis Application", frame)

    # Çıkış için 'q' tuşuna basın
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
# Release resources
cap.release()
cv2.destroyAllWindows()
