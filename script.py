import os
import sys
import time
import base64
import hashlib
import threading
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageGrab
import pyperclip
import requests
from pynput import keyboard
from jinja2 import Environment, FileSystemLoader

try:
    import ollama
except ImportError:
    ollama = None

MIN_OLLAMA_VER = (0, 12, 7)


class TestAutomation:
    def __init__(self):
        load_dotenv()

        self.discord_webhook = os.getenv('DISCORD_WEBHOOK_URL', '')
        self.ollama_model = os.getenv('OLLAMA_MODEL')

        trigger_key_str = os.getenv('TRIGGER_KEY', 'print_screen').lower().replace(' ', '_')
        self.trigger_key = getattr(keyboard.Key, trigger_key_str, None)
        if self.trigger_key is None and len(trigger_key_str) == 1:
            self.trigger_key = keyboard.KeyCode.from_char(trigger_key_str)

        template_dir = Path(__file__).parent / 'templates'
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))

        self.execution_mode = os.getenv('EXECUTION_MODE', 'quick').lower()
        self.running = True
        self.processing = False
        self.last_hash = None

        if ollama is None:
            print("ERROR: ollama Python library not installed. Run: pip install ollama")
            sys.exit(1)

        self._check_ollama_version()
        self.image_max_dim = int(os.getenv('IMAGE_MAX_DIM', '1280'))

    def _check_ollama_version(self):
        try:
            resp = requests.get('http://localhost:11434/api/version', timeout=2)
            ver_str = resp.json().get('version', '')
            parts = [int(p) for p in ver_str.split('.')[:3]]
            if parts < list(MIN_OLLAMA_VER):
                print(f"[!] Ollama server v{ver_str} may be too old for Qwen3-VL")
                print(f"    Minimum required: v{'.'.join(str(v) for v in MIN_OLLAMA_VER)}")
        except:
            pass

    def _prepare_image(self, img):
        # Convert to RGB (some vision models fail with RGBA transparency)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        max_dim = self.image_max_dim
        w, h = img.size
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        buffered = BytesIO()
        # JPEG prevents alpha channel issues and reduces base64 payload size
        img.save(buffered, format='JPEG')
        img_data = buffered.getvalue()
        img_b64 = base64.b64encode(img_data).decode()
        return img_data, img_b64

    def handle_screenshot(self):
        if self.processing:
            return
        self.processing = True

        try:
            time.sleep(0.3)
            img = ImageGrab.grabclipboard()

            if img is None:
                print("[-] No image in clipboard")
                return

            if isinstance(img, list):
                return

            img_data, img_b64 = self._prepare_image(img)
            img_hash = hashlib.md5(img_data).hexdigest()
            if img_hash == self.last_hash:
                self.processing = False
                return
            self.last_hash = img_hash

            print(f"[+] Processing screenshot...")

            q_type = self.classify_question(img_b64)
            print(f"    -> Type: {q_type}")

            answer = self.answer_question(img_b64, q_type)
            print(f"    -> Answer: {answer[:120]}...")

            pyperclip.copy(answer)
            self.send_to_discord(answer, q_type)

            print(f"[✓] Done — clipboard + Discord")

        except Exception as e:
            print(f"[!] Error: {e}")
        finally:
            self.processing = False

    def classify_question(self, img_b64):
        router_prompt = self.jinja_env.get_template('router.jinja').render()

        response = ollama.chat(
            model=self.ollama_model,
            messages=[{
                'role': 'user',
                'content': router_prompt,
                'images': [img_b64]
            }]
        )

        q_type = response['message']['content'].strip().lower()
        valid = ['quickfire', 'maths', 'coding', 'data_analysis', 'diagram', 'psychometric']

        if q_type not in valid:
            q_type = 'psychometric'

        return q_type

    def answer_question(self, img_b64, q_type):
        try:
            template = self.jinja_env.get_template(f'{q_type}.jinja')
        except:
            template = self.jinja_env.get_template('psychometric.jinja')

        base_content = ''
        try:
            base_template = self.jinja_env.get_template('base.jinja')
            base_content = base_template.render()
        except:
            pass

        exec_mode = "QUICK_FIRE" if self.execution_mode == 'quick' else ""
        prompt = template.render(base=base_content, execution_mode=exec_mode)

        response = ollama.chat(
            model=self.ollama_model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [img_b64]
            }]
        )

        return response['message']['content']

    def send_to_discord(self, text, q_type):
        if not self.discord_webhook:
            return

        data = {
            "content": f"**[{q_type.upper()}]**\n{text}",
            "username": "Test Assistant"
        }

        try:
            resp = requests.post(self.discord_webhook, json=data, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            print(f"    Discord send failed: {e}")

    def on_press(self, key):
        if key == self.trigger_key:
            thread = threading.Thread(target=self.handle_screenshot, daemon=True)
            thread.start()
        elif key == keyboard.Key.esc:
            print("\nStopping...")
            self.running = False
            return False

    def run(self):
        print()
        print("=" * 50)
        print("  Test Automation Assistant")
        print("=" * 50)
        print(f"  Model:     {self.ollama_model}")
        print(f"  Trigger:   {os.getenv('TRIGGER_KEY', 'print_screen')}")
        print(f"  Discord:   {'Configured' if self.discord_webhook else 'NOT configured'}")
        print("=" * 50)
        print("  Press PrintScreen to capture & process")
        print("  Press ESC to stop")
        print("=" * 50)
        print()

        with keyboard.Listener(on_press=self.on_press) as listener:
            self.listener = listener
            listener.join()


if __name__ == '__main__':
    app = TestAutomation()
    app.run()
