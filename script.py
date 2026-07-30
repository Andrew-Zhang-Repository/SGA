import os
import sys
import time
import threading
from pathlib import Path

from dotenv import load_dotenv
from PIL import ImageGrab
import pyperclip
import requests
from pynput import keyboard
from jinja2 import Environment, FileSystemLoader
import pytesseract
import cv2

try:
    import ollama
except ImportError:
    ollama = None


CAPTURE_DIR = Path(__file__).parent / 'captures'
CAPTURE_PATH = CAPTURE_DIR / 'latest.jpg'


class TestAutomation:
    def __init__(self):
        CAPTURE_DIR.mkdir(exist_ok=True)

        load_dotenv()

        self.answer_model = os.getenv('OLLAMA_MODEL')
        self.text_model = os.getenv('TEXT_MODEL', self.answer_model)
        self.router_model = os.getenv('ROUTER_MODEL', self.text_model)
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK_URL', '')

        trigger_key_str = os.getenv('TRIGGER_KEY', 'print_screen').lower().replace(' ', '_')
        self.trigger_key = getattr(keyboard.Key, trigger_key_str, None)
        if self.trigger_key is None and len(trigger_key_str) == 1:
            self.trigger_key = keyboard.KeyCode.from_char(trigger_key_str)

        template_dir = Path(__file__).parent / 'templates'
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))

        self.execution_mode = os.getenv('EXECUTION_MODE', 'quick').lower()
        self.running = True
        self.processing = False

        self.VISUAL_TYPES = {'diagram', 'data_analysis'}

        if ollama is None:
            print("ERROR: ollama Python library not installed. Run: pip install ollama")
            sys.exit(1)

    def _save_image(self, img):
        img.save(CAPTURE_PATH, format='JPEG')
        return str(CAPTURE_PATH)

    def _extract_text(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return pytesseract.image_to_string(gray)

    def _classify_question(self, text):
        router_prompt = self.jinja_env.get_template('router.jinja').render()

        response = ollama.chat(
            model=self.router_model,
            messages=[{
                'role': 'user',
                'content': f"{router_prompt}\n\n---\n\n{text}"
            }]
        )

        q_type = response['message']['content'].strip().lower()
        valid = ['quickfire', 'maths', 'coding', 'data_analysis', 'diagram', 'psychometric']

        if q_type not in valid:
            q_type = 'psychometric'

        return q_type

    def _render_prompt(self, q_type):
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
        return template.render(base=base_content, execution_mode=exec_mode)

    def _answer_text(self, text, q_type):
        prompt = self._render_prompt(q_type)

        response = ollama.chat(
            model=self.text_model,
            messages=[{
                'role': 'user',
                'content': f"{prompt}\n\n---\n\n{text}"
            }]
        )

        return response['message']['content']

    def _answer_with_vision(self, image_path, q_type):
        prompt = self._render_prompt(q_type)

        response = ollama.chat(
            model=self.answer_model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_path]
            }]
        )

        return response['message']['content']

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

            image_path = self._save_image(img)

            print("[+] Extracting text for classification...")
            raw_text = self._extract_text(image_path)
            print(f"    OCR: {raw_text[:100]}...")

            print("[+] Classifying question type...")
            q_type = self._classify_question(raw_text)
            print(f"    -> Type: {q_type}")

            print("[+] Generating answer...")
            if q_type in self.VISUAL_TYPES:
                print("    Route: Vision (Gemma4 sees image)")
                answer = self._answer_with_vision(image_path, q_type)
            else:
                print("    Route: Text-only (OCR → Gemma3)")
                answer = self._answer_text(raw_text, q_type)

            print(f"    -> Answer: {answer[:120]}...")

            pyperclip.copy(answer)
            self.send_to_discord(answer, q_type)

            print("[✓] Done — clipboard + Discord")

        except Exception as e:
            print(f"[!] Error: {e}")
        finally:
            self.processing = False

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
        print(f"  Router model: {self.router_model}")
        print(f"  Text model:   {self.text_model}")
        print(f"  Vision model: {self.answer_model}")
        print(f"  Trigger:      {os.getenv('TRIGGER_KEY', 'print_screen')}")
        print(f"  Discord:      {'Configured' if self.discord_webhook else 'NOT configured'}")
        print("=" * 50)
        print("  Router:  OCR text  → Gemma3:12b  (classification)")
        print("  Text:    OCR text  → Gemma3:4b   (answer)")
        print("  Visual:  Image     → Gemma4:12b  (answer)")
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
