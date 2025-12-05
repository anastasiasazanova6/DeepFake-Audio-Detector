import os

folders = [
    'data/real',      
    'data/fake',      
    'models',         
    'temp',           
    'logs'            
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Создана папка: {folder}")

print("\n✅ Структура папок создана!")
print("Поместите ваши аудиофайлы:")
print("  Реальные → data/real/")
print("  Дипфейки → data/fake/")