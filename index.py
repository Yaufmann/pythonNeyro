import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random

def generate_data(num_samples=10000, range_min=-100, range_max=100):
    """Генерация пар чисел и их суммы """
    X = np.random.uniform(range_min, range_max, size=(num_samples, 2))
    y = np.sum(X, axis=1) 
    return X, y

num_samples = 20000  
range_min = -1000    
range_max = 1000     

X, y = generate_data(num_samples, range_min, range_max)

print(f"Размерность данных: X={X.shape}, y={y.shape}")
print(f"Пример данных:")
print(f"  Вход: {X[0]} -> Сумма: {y[0]:.4f}")
print(f"  Вход: {X[1]} -> Сумма: {y[1]:.4f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nОбучающая выборка: {X_train.shape[0]} примеров")
print(f"Тестовая выборка: {X_test.shape[0]} примеров")

def create_model():
    """Создание полносвязной нейронной сети"""
    model = keras.Sequential([
        
        layers.Input(shape=(2,)),
        
       
        layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        
        layers.Dense(32, activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        
        layers.Dense(16, activation='relu', kernel_initializer='he_normal'),
        
       
        layers.Dense(1, activation='linear')  
    ])
    
    return model

model = create_model()

print("\nАрхитектура нейронной сети:")
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',  # Mean Squared Error для регрессии
    metrics=['mae', 'mse']  # Mean Absolute Error и MSE
)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

print("\nНачало обучения...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)

test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)

print(f"\nРезультаты на тестовых данных:")
print(f"  Loss (MSE): {test_mse:.6f}")
print(f"  MAE: {test_mae:.6f}")
print(f"  RMSE: {np.sqrt(test_mse):.6f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(history.history['loss'], label='Обучающая')
axes[0, 0].plot(history.history['val_loss'], label='Валидационная')
axes[0, 0].set_title('Функция потерь (MSE)')
axes[0, 0].set_xlabel('Эпоха')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(history.history['mae'], label='Обучающая')
axes[0, 1].plot(history.history['val_mae'], label='Валидационная')
axes[0, 1].set_title('Mean Absolute Error (MAE)')
axes[0, 1].set_xlabel('Эпоха')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].legend()
axes[0, 1].grid(True)

y_pred = model.predict(X_test, verbose=0).flatten()

axes[1, 0].scatter(y_test, y_pred, alpha=0.5, s=10)
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Идеальная линия')
axes[1, 0].set_title('Предсказания vs Реальные значения')
axes[1, 0].set_xlabel('Реальная сумма')
axes[1, 0].set_ylabel('Предсказанная сумма')
axes[1, 0].legend()
axes[1, 0].grid(True)

errors = y_pred - y_test
axes[1, 1].hist(errors, bins=50, edgecolor='black')
axes[1, 1].set_title('Распределение ошибок предсказаний')
axes[1, 1].set_xlabel('Ошибка (предсказание - реальное)')
axes[1, 1].set_ylabel('Частота')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()


def test_addition(model, num1, num2):
    """Тестирование сложения двух чисел"""
    input_data = np.array([[num1, num2]])
    prediction = model.predict(input_data, verbose=0)[0][0]
    actual_sum = num1 + num2
    error = abs(prediction - actual_sum)
    
    print(f"\nТест: {num1} + {num2}")
    print(f"  Реальная сумма: {actual_sum:.4f}")
    print(f"  Предсказание НС: {prediction:.4f}")
    print(f"  Абсолютная ошибка: {error:.6f}")
    print(f"  Относительная ошибка: {(error/abs(actual_sum)*100 if actual_sum != 0 else float('inf')):.4f}%")
    
    return prediction, error

print("\n" + "="*50)
print("ТЕСТИРОВАНИЕ НА ПРОИЗВОЛЬНЫХ ЧИСЛАХ")
print("="*50)

test_cases = [
    (10.5, 20.3),
    (-15.7, 8.2),
    (0, 0),
    (100, -50),
    (3.14159, 2.71828),
    (-999.99, 999.99),
    (123.456, 789.123)
]

for i, (a, b) in enumerate(test_cases, 1):
    print(f"\nТест #{i}:")
    test_addition(model, a, b)


print("\n" + "="*50)
print("ТЕСТИРОВАНИЕ НА ЧИСЛАХ ВНЕ ДИАПАЗОНА ОБУЧЕНИЯ")
print("="*50)

extreme_cases = [
    (1500, 800),    
    (-1200, -300),  
    (2000, -2000)   
]

for a, b in extreme_cases:
    test_addition(model, a, b)



model.save('addition_model.h5')
print("\nМодель сохранена как 'addition_model.h5'")


def predict_sum(model, num1, num2):
    """Удобная функция для предсказания суммы"""
    result = model.predict(np.array([[num1, num2]]), verbose=0)[0][0]
    return result


print("\n" + "="*50)
print("ИНТЕРАКТИВНОЕ ТЕСТИРОВАНИЕ")
print("="*50)


while True:
    try:
        user_input = input("\nВведите два числа через пробел (или 'q' для выхода): ")
        if user_input.lower() == 'q':
            break
        
        a, b = map(float, user_input.split())
        prediction = predict_sum(model, a, b)
        actual = a + b
        error = abs(prediction - actual)
        
        print(f"{a} + {b} = {prediction:.4f}")
        print(f"Ошибка: {error:.6f} ({error/abs(actual)*100 if actual != 0 else 0:.4f}%)")
        
    except ValueError:
        print("Ошибка ввода! Введите два числа через пробел.")
    except KeyboardInterrupt:
        break

print("\nПрограмма завершена!")