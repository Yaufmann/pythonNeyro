import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0

x_val = x_train[:12000]
y_val = y_train[:12000]
x_train = x_train[12000:]
y_train = y_train[12000:]

model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),  
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

print("\n" + "="*60)
print("СТРУКТУРА НЕЙРОННОЙ СЕТИ:")
print("="*60)
model.summary()

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("\n" + "="*60)
print(f"ТОЧНОСТЬ НА ТЕСТОВОЙ ВЫБОРКЕ: {test_acc:.4f} ({test_acc*100:.2f}%)")
print("="*60)

print("\n" + "="*60)
print("ОБЪЯСНЕНИЕ ЧИСЛА ВЕСОВЫХ КОЭФФИЦИЕНТОВ:")
print("="*60)

print("\n1. СЛОЙ (Dense, 64 нейрона):")
print("   Вход: 784 параметра (28x28 пикселей)")
print("   Выход: 64 нейрона")
print("   Весовые коэффициенты: 784 * 64 = 50,176")
print("   Смещения (bias): 64")
print("   Итого: 50,176 + 64 = 50,240 параметров")
print("   Пояснение: Каждый из 784 входов соединен с каждым из 64 нейронов")

print("\n2. СЛОЙ (Dropout, 0.2):")
print("   Обучаемых параметров: 0")
print("   Пояснение: Dropout только обнуляет случайные активации")

print("\n3. СЛОЙ (Dense, 10 нейронов):")
print("   Вход: 64 параметра (с предыдущего слоя)")
print("   Выход: 10 нейронов (по одному на каждую цифру 0-9)")
print("   Весовые коэффициенты: 64 * 10 = 640")
print("   Смещения (bias): 10")
print("   Итого: 640 + 10 = 650 параметров")
print("   Пояснение: Каждый из 64 входов соединен с каждым из 10 нейронов")

print("\n" + "="*60)
print(f"ОБЩЕЕ ЧИСЛО ПАРАМЕТРОВ: {model.count_params():,}")
print("="*60)

if test_acc >= 0.97:
    print("\n✅ УСПЕХ: Точность на тестовой выборке ≥ 97%")
else:
    print(f"\n  Точность {test_acc*100:.2f}% немного ниже 97%")
    print("   Можно попробовать увеличить первый скрытый слой до 128 нейронов")