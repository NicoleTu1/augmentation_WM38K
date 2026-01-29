import time
# from playsound import playsound
import pygame
# import os
import socket

# ⚠️ 注意: playsound 在 Windows 上對檔案格式和路徑比較敏感。
# 建議使用 .wav 檔案。


def play_notification_sound_pygame(file_path):
    try:
        # 1. 初始化混音器 (只需要執行一次)
        # 參數設置通常為 44.1 kHz, 16 位元, 單聲道
        pygame.mixer.init(frequency=44100, size=-16, channels=1) 
        
        # 2. 載入音效檔案
        # Pygame 支援 WAV, MP3, OGG 等多種格式
        pygame.mixer.music.load(file_path)
        
        # 3. 播放
        pygame.mixer.music.play()
        
        # 4. 關鍵：確保程式暫停足夠長的時間讓音效播放完畢
        # 這裡需要根據您的音效長度來設定延遲
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

    except Exception as e:
        print(f"使用 Pygame 播放音效時出錯: {e}")
    finally:
        # 確保在程式結束前停止混音器
        pygame.mixer.quit()

if __name__ == "__main__":
    # 取得主機名稱以決定播放哪個音效
    hostname = socket.gethostname()
    path_list = {'e2e24780b959': '2008cat.wav',    # '5070Ti'
                 '2e9c0926e682': '2008cat.wav',    # '2070'
                 '7d9716d33117': '2008cat.wav',    # 'I7-12700'
                 'IDS-RTX5090': '2008cat.wav',  # 'IDS-RTX5090'
                 'Nicole': r'C:\Nicole\Master_NTUB_11366001\Lab\Implementation\20250627-augmentation on Mixed WM38\2008cat.wav',  # 筆電
                 # 再補桌機 
                 }

    file_path = path_list.get(hostname, '2008cat.wav')  # 預設音效檔案
    play_notification_sound_pygame(file_path)
    print("腳本執行結束。")

# def play_notification_sound(file_path):
#     if not os.path.exists(file_path):
#         print(f"錯誤：找不到音效檔案 '{file_path}'")
#         return

#     try:
#         print(f"正在播放音效: {file_path}")
#         # 在 Windows 上，playsound 會自動調用系統的音訊功能播放檔案。
#         playsound(file_path)

#     except Exception as e:
#         # 如果音效格式不受支援 (例如 .mp3 遇到問題)，可能會拋出錯誤
#         print(f"播放音效時出錯: {e}")


# if __name__ == "__main__":
#     play_notification_sound(AUDIO_FILE)
    
#     print("腳本執行結束。")