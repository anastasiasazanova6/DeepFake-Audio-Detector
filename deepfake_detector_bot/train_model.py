import os
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from audio_processor import AudioFeatureExtractor
from config import DATA_DIR, MODELS_DIR, TEST_SIZE, RANDOM_STATE, THRESHOLDS

def load_dataset():
    print("Загрузка данных...")
    
    X = []
    y = []
    
    real_dir = os.path.join(DATA_DIR, 'real')
    if os.path.exists(real_dir):
        for filename in os.listdir(real_dir):
            if filename.endswith(('.wav', '.mp3', '.ogg', '.m4a', '.flac')):
                file_path = os.path.join(real_dir, filename)
                features = AudioFeatureExtractor.extract_features_for_model(file_path)
                if features is not None:
                    X.append(features)
                    y.append(0)  
    
    fake_dir = os.path.join(DATA_DIR, 'fake')
    if os.path.exists(fake_dir):
        for filename in os.listdir(fake_dir):
            if filename.endswith(('.wav', '.mp3', '.ogg', '.m4a', '.flac')):
                file_path = os.path.join(fake_dir, filename)
                features = AudioFeatureExtractor.extract_features_for_model(file_path)
                if features is not None:
                    X.append(features)
                    y.append(1)  
    
    if len(X) == 0:
        print("Ошибка: нет данных для обучения")
        return None, None
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Загружено: {len(X)} записей")
    print(f"  Реальных: {sum(y==0)}")
    print(f"  Фейковых: {sum(y==1)}")
    
    return X, y

def train_and_evaluate(X, y):
    print("\nОбучение модели...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"  Обучающая выборка: {len(X_train)} записей")
    print(f"  Тестовая выборка: {len(X_test)} записей")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        class_weight=class_weight_dict,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nРезультаты оценки:")
    print(f"  Точность: {accuracy*100:.1f}%")
    print("\nДетальный отчет:")
    print(classification_report(y_test, y_pred, target_names=['Реальное', 'Фейк']))
    
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    print(f"\nАнализ для порогов {THRESHOLDS['REAL_MAX']*100:.0f}%/{THRESHOLDS['UNCERTAIN_MAX']*100:.0f}%:")
    
    real_probas = y_proba[y_test == 0]
    real_below_75 = np.sum(real_probas < THRESHOLDS['REAL_MAX'])
    print(f"  Реальные аудио:")
    print(f"    Среднее: {real_probas.mean()*100:.1f}%")
    print(f"    < {THRESHOLDS['REAL_MAX']*100:.0f}%: {real_below_75}/{len(real_probas)} "
          f"({real_below_75/len(real_probas)*100:.1f}%)")
    
    fake_probas = y_proba[y_test == 1]
    fake_above_85 = np.sum(fake_probas > THRESHOLDS['FAKE_MIN'])
    print(f"  Фейковые аудио:")
    print(f"    Среднее: {fake_probas.mean()*100:.1f}%")
    print(f"    > {THRESHOLDS['FAKE_MIN']*100:.0f}%: {fake_above_85}/{len(fake_probas)} "
          f"({fake_above_85/len(fake_probas)*100:.1f}%)")
    
    create_confusion_matrix(y_test, y_pred)
    
    return model, scaler, accuracy

def create_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Реальное', 'Фейк'],
                yticklabels=['Реальное', 'Фейк'])
    plt.title('Матрица ошибок')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.tight_layout()
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    plt.savefig(os.path.join(MODELS_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()

def save_model_with_thresholds(model, scaler, accuracy):
    model_data = {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'thresholds': THRESHOLDS
    }
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'deepfake_model.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nМодель сохранена: {model_path}")
    print(f"Точность модели: {accuracy*100:.1f}%")
    print(f"Установленные пороги:")
    print(f"  Реальное: < {THRESHOLDS['REAL_MAX']*100:.0f}%")
    print(f"  Неопределенно: {THRESHOLDS['UNCERTAIN_MIN']*100:.0f}-{THRESHOLDS['UNCERTAIN_MAX']*100:.0f}%")
    print(f"  Дипфейк: > {THRESHOLDS['FAKE_MIN']*100:.0f}%")

def main():
    print("=" * 50)
    print("ОБУЧЕНИЕ МОДЕЛИ ДЕТЕКТОРА ДИПФЕЙКОВ")
    print("=" * 50)
    print(f"Пороги классификации:")
    print(f"  ✅ < {THRESHOLDS['REAL_MAX']*100:.0f}% - Реальное")
    print(f"  ⚠️ {THRESHOLDS['UNCERTAIN_MIN']*100:.0f}-{THRESHOLDS['UNCERTAIN_MAX']*100:.0f}% - Неопределенно")
    print(f"  ❌ > {THRESHOLDS['FAKE_MIN']*100:.0f}% - Дипфейк")
    print("=" * 50)
    
    X, y = load_dataset()
    if X is None:
        print("Ошибка: недостаточно данных")
        print(f"Поместите файлы в папки:")
        print(f"  Реальные: {DATA_DIR}/real/")
        print(f"  Фейковые: {DATA_DIR}/fake/")
        return
    
    model, scaler, accuracy = train_and_evaluate(X, y)
    
    save_model_with_thresholds(model, scaler, accuracy)
    
    print("\nОбучение завершено")

if __name__ == '__main__':
    main()