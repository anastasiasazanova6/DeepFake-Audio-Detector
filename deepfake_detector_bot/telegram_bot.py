import os
import telebot
import tempfile
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from audio_processor import AudioFeatureExtractor
from config import BOT_TOKEN, MODELS_DIR, TEMP_DIR, THRESHOLDS

os.makedirs(TEMP_DIR, exist_ok=True)

class DeepfakeDetectorBot:
    def __init__(self):
        self.bot = telebot.TeleBot(BOT_TOKEN)
        self.model = None
        self.scaler = None
        self.model_accuracy = 0.0
        self.thresholds = THRESHOLDS
        
        if not self.load_model():
            print("Ошибка: не удалось загрузить модель")
            return
        
        self.setup_handlers()
        print("✅ Бот запущен")
    
    def load_model(self):
        """Загрузка обученной модели"""
        model_path = os.path.join(MODELS_DIR, 'deepfake_model.pkl')
        
        if not os.path.exists(model_path):
            print(f"Модель не найдена: {model_path}")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.model_accuracy = model_data.get('accuracy', 0.5) * 100
            
            print(f"Модель загружена (точность: {self.model_accuracy:.1f}%)")
            return True
            
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False
    
    def setup_handlers(self):
        
        @self.bot.message_handler(commands=['start'])
        def start_handler(message):
            welcome_text = (
                "🔍 **Deepfake Audio Detector**\n\n"
                "Определяю поддельные голосовые сообщения с помощью разработанной нейронной сети.\n\n"
                "**Критерии определения:**\n"
                f"✅ < {self.thresholds['REAL_MAX']*100:.0f}% - РЕАЛЬНЫЙ голос\n"
                f"⚠️ {self.thresholds['UNCERTAIN_MIN']*100:.0f}-{self.thresholds['UNCERTAIN_MAX']*100:.0f}% - НЕОПРЕДЕЛЕННО\n"
                f"❌ > {self.thresholds['FAKE_MIN']*100:.0f}% - ДИПФЕЙК\n\n"
                "**Рекомендации при обнаружении дипфейка:**\n"
                "• Перепроверьте источник информации\n"
                "• Свяжитесь с собеседником другим способом\n"
                "• Не передавайте конфиденциальную информацию\n\n"
                "📁 **Отправьте голосовое сообщение или аудиофайл**"
            )
            self.bot.send_message(message.chat.id, welcome_text, parse_mode='Markdown')
        
        @self.bot.message_handler(content_types=['voice'])
        def voice_handler(message):
            self.process_audio(message, is_voice=True)
        
        @self.bot.message_handler(content_types=['audio', 'document'])
        def file_handler(message):
            self.process_audio(message, is_voice=False)
        
        @self.bot.message_handler(func=lambda m: True)
        def text_handler(message):
            self.bot.send_message(message.chat.id, 
                "Отправьте голосовое сообщение или аудиофайл для проверки."
            )
    
    def process_audio(self, message, is_voice=True):
        try:
            chat_id = message.chat.id
            
            if self.model is None:
                self.bot.send_message(chat_id, "❌ Модель не загружена")
                return
            
            status_msg = self.bot.send_message(chat_id, "Проводится анализ")
            
            if is_voice:
                file_info = self.bot.get_file(message.voice.file_id)
                filename = "Голосовое сообщение"
            elif hasattr(message, 'audio'):
                file_info = self.bot.get_file(message.audio.file_id)
                filename = "Аудиофайл"
            else:
                file_info = self.bot.get_file(message.document.file_id)
                filename = message.document.file_name or "Файл"
            
            downloaded_file = self.bot.download_file(file_info.file_path)
            
            temp_path = tempfile.mktemp(suffix='.ogg' if is_voice else '.mp3')
            with open(temp_path, 'wb') as f:
                f.write(downloaded_file)
            
            features = AudioFeatureExtractor.extract_features_for_model(temp_path)
            
            if features is None:
                self.bot.edit_message_text("❌ Не удалось проанализировать аудио", 
                                          chat_id, status_msg.message_id)
                return
            
            features_scaled = self.scaler.transform([features])
            raw_probability = self.model.predict_proba(features_scaled)[0, 1]
            
            corrected_probability = self.adjust_probability_for_zones(raw_probability)
            
            result_text = self.format_result(corrected_probability)
            
            self.bot.edit_message_text(result_text, chat_id, status_msg.message_id, parse_mode='Markdown')
            
            try:
                os.unlink(temp_path)
            except:
                pass
            
        except Exception as e:
            print(f"Ошибка обработки аудио: {e}")
            try:
                self.bot.send_message(chat_id, "❌ Произошла ошибка при обработке")
            except:
                pass
    
    def adjust_probability_for_zones(self, raw_prob):
        if raw_prob < 0.5:
            corrected = 0.3 + (raw_prob - 0.4) * 4.0  
            corrected = min(corrected, 0.74)
        elif raw_prob < 0.65:
            corrected = 0.7 + (raw_prob - 0.5) * 1.0  
            corrected = max(0.75, min(corrected, 0.85))  
        else:
            corrected = 0.85 + (raw_prob - 0.65) * 0.67  
            corrected = max(corrected, 0.86)  
        
        return max(0.0, min(1.0, corrected))
    
    def format_result(self, probability):
        percent = probability * 100
        
        if percent < self.thresholds['REAL_MAX'] * 100:
            status = "✅ РЕАЛЬНЫЙ ГОЛОС"
            color = "🟢"
        elif percent < self.thresholds['UNCERTAIN_MAX'] * 100:
            status = "⚠️ НЕОПРЕДЕЛЕННО"
            color = "🟡"
        else:
            status = "❌ ВЕРОЯТНЫЙ ДИПФЕЙК"
            color = "🔴"
        
        bar_length = 30
        real_max_pos = int(self.thresholds['REAL_MAX'] * bar_length)  
        uncertain_max_pos = int(self.thresholds['UNCERTAIN_MAX'] * bar_length)  
        
        filled = int(percent / 100 * bar_length)
        
        bar = ""
        for i in range(bar_length):
            if i < filled:
                if i < real_max_pos:
                    bar += "█"  
                elif i < uncertain_max_pos:
                    bar += "█"  
                else:
                    bar += "█"  
            else:
                bar += "░"
        
        bar_with_marks = bar + "\n"
        if real_max_pos > 2:
            bar_with_marks += " " * (real_max_pos - 2) + "75%"
        if uncertain_max_pos - real_max_pos > 4:
            bar_with_marks += " " * (uncertain_max_pos - real_max_pos - 4) + "85%"
        
        result = (
            f"{color} **{status}**\n\n"
            f"Вероятность подделки:\n"
            f"```\n{bar_with_marks}\n{percent:.1f}%\n```\n"
            f"_Точность модели: {self.model_accuracy:.1f}%_"
        )
        
        return result
    
    def run(self):
        """Запуск бота"""
        print("Бот запущен и готов к работе")
        print(f"Пороги: <{self.thresholds['REAL_MAX']*100:.0f}% - реальное, "
              f"{self.thresholds['UNCERTAIN_MIN']*100:.0f}-{self.thresholds['UNCERTAIN_MAX']*100:.0f}% - неопределенно, "
              f">{self.thresholds['FAKE_MIN']*100:.0f}% - дипфейк")
        self.bot.polling(none_stop=True)

if __name__ == '__main__':
    bot = DeepfakeDetectorBot()
    bot.run()