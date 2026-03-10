import pyttsx3
import threading
import queue
import time

class TTSManager:
    def __init__(self, voice_id=None):
        self.speech_queue = queue.Queue()
        self.running = True
        self.voice_id = voice_id
        self.volume = 1.0
        
        # Start the background worker
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        
    def _worker(self):
        import pythoncom
        pythoncom.CoInitialize()
        while self.running:
            try:
                # Block for up to 0.5s waiting for text
                text = self.speech_queue.get(timeout=0.5)
                if text:
                    # Initialize engine PER utterance to avoid Windows runAndWait deadlocks
                    engine = pyttsx3.init()
                    if self.voice_id:
                        try:
                            engine.setProperty("voice", self.voice_id)
                        except Exception:
                            pass
                    
                    engine.setProperty('volume', self.volume)
                    engine.say(text)
                    engine.runAndWait()
                    
                    # Explicitly delete the engine and pump COM events so it cleans up
                    del engine
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS Error: {e}")
                time.sleep(1)

    def speak(self, text):
        """Adds text to the speech queue to be spoken in the background."""
        if text and str(text).strip():
            self.speech_queue.put(str(text).strip())

    def set_volume(self, volume):
        self.volume = max(0.0, min(1.0, float(volume)))

    def get_available_voices(self):
        """Helper to return a list of available voice dictionaries."""
        try:
            import pythoncom
            pythoncom.CoInitialize()
            temp_engine = pyttsx3.init()
            voices = temp_engine.getProperty('voices')
            return [{"id": v.id, "name": v.name} for v in voices]
        except Exception:
            return []
            
    def set_voice(self, voice_id):
        self.voice_id = voice_id
        # We can't safely change voice properties of a running engine from another thread in pyttsx3.
        # But we can set a flag and handle it in the worker, or simply restart the thread!
        # For simplicity, we just set self.voice_id and let the user restart the app or we can recreate the engine.
        self.stop()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
