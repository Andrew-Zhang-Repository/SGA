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

try:
    from img2table.document import Image as T2Image
    from img2table.ocr import TesseractOCR
except ImportError:
    T2Image = None
    TesseractOCR = None


CAPTURE_DIR = Path(__file__).parent / 'captures'
CAPTURE_PATH = CAPTURE_DIR / 'latest.jpg'
RESUME_FILE = Path(__file__).parent / 'Updated_Resume.pdf'


class TestAutomation:
    def __init__(self):
        CAPTURE_DIR.mkdir(exist_ok=True)

        load_dotenv()

        self.answer_model = os.getenv('OLLAMA_MODEL')
        self.text_model = os.getenv('TEXT_MODEL', self.answer_model)
        self.router_model = os.getenv('ROUTER_MODEL', self.text_model)
        self.code_model = os.getenv('CODE_MODEL', os.getenv('code_model', self.text_model))
        self.system_design_model = os.getenv('SYSTEM_DESIGN_MODEL', os.getenv('system_design_model', self.text_model))
        self.resume_model = os.getenv('RESUME_MODEL', os.getenv('resume_model', self.text_model))
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

    def _cut_noise(self, text):
        noise_keywords = (
            'http', 'www.', '.com', '.org', '.net', 'sign in', 'sign up',
            'login', 'logout', 'menu', 'search', 'cookie', 'settings',
            'about us', 'privacy', 'terms of service', 'newsletter',
            'home page', 'back to', 'next question', 'previous question',
            'time left', 'timer', 'submit', 'save & exit', 'instructions',
            'notification', 'advertisement', 'advert', 'sponsored',
        )
        cleaned = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if len(line) < 2:
                continue
            lower = line.lower()
            if any(kw in lower for kw in noise_keywords):
                continue
            cleaned.append(line)
        return '\n'.join(cleaned)

    def _extract_tables(self, image_path):
        if T2Image is None or TesseractOCR is None:
            return None

        try:
            ocr = TesseractOCR(n_threads=1, lang="eng")
            doc = T2Image(src=image_path)
            tables = doc.extract_tables(ocr=ocr, implicit_rows=True, borderless_tables=True)
            if not tables:
                return None

            parts = []
            for table in tables:
                df = table.df
                if df.empty:
                    continue
                parts.append(df.to_string(index=False))

            return "\n\n".join(parts) if parts else None
        except Exception as e:
            print(f"    Table extraction failed: {e}")
            return None

    def _load_resume(self):
        resume_path = Path(os.getenv('RESUME_FILE', RESUME_FILE))
        if not resume_path.is_absolute():
            resume_path = Path(__file__).parent / resume_path

        if resume_path.suffix.lower() == '.pdf':
            try:
                import fitz
                doc = fitz.open(str(resume_path))
                text = '\n'.join(page.get_text() for page in doc)
                doc.close()
                return text.strip()
            except ImportError:
                print("    WARNING: pymupdf not installed. Run: pip install pymupdf")
                return ''
            except Exception as e:
                print(f"    Resume PDF read failed: {e}")
                return ''
        elif resume_path.exists():
            return resume_path.read_text(encoding='utf-8').strip()
        return ''

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
        valid = ['maths', 'coding', 'general_coding', 'system_design',
                 'data_analysis', 'diagram', 'psychometric', 'resume']

        if q_type not in valid:
            q_type = 'psychometric'

        return q_type

    def _render_prompt(self, q_type, resume_content=''):
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
        return template.render(base=base_content, execution_mode=exec_mode, resume_content=resume_content)

    def _answer_text(self, text, q_type, model=None, table_text=None, resume_content=''):
        prompt = self._render_prompt(q_type, resume_content)
        model = model or self.text_model

        cleaned_text = self._cut_noise(text)

        if table_text:
            content = (
                f"{prompt}\n\n---\n\n"
                f"QUESTION & OPTIONS (OCR text from the image):\n{cleaned_text}\n\n"
                f"---\n\n"
                f"TABLE DATA (extracted from the grid/table in the image):\n{table_text}\n\n"
                f"---\n\n"
                f"Instructions: Ignore any noise, headers, nav bars, timers, or unrelated UI text. "
                f"Focus only on the actual question, its answer options (if any), and the table data "
                f"provided above. Use the table to answer the question. Answer the question."
            )
        else:
            content = f"{prompt}\n\n---\n\n{cleaned_text}"

        response = ollama.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': content
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
            if q_type == 'diagram':
                print("    Route: Vision (Gemma4 sees image)")
                answer = self._answer_with_vision(image_path, q_type)
            elif q_type in ('coding', 'general_coding'):
                print("    Route: Coding (qwen2.5-coder) — no table extraction")
                answer = self._answer_text(raw_text, q_type, model=self.code_model)
            elif q_type == 'system_design':
                print("    Route: System design — no table extraction")
                answer = self._answer_text(raw_text, q_type, model=self.system_design_model)
            elif q_type == 'resume':
                print("    Route: Resume-based interview question — resume injected")
                resume_content = self._load_resume()
                answer = self._answer_text(raw_text, q_type, model=self.resume_model, resume_content=resume_content)
            else:
                table_text = self._extract_tables(image_path)
                if table_text:
                    print("    Route: Table detected → text model (img2table + OCR)")
                    answer = self._answer_text(raw_text, q_type, table_text=table_text)
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
        print(f"  Code model:   {self.code_model}")
        print(f"  System design model: {self.system_design_model}")
        print(f"  Resume model: {self.resume_model}")
        print(f"  Vision model: {self.answer_model}")
        print(f"  Trigger:      {os.getenv('TRIGGER_KEY', 'print_screen')}")
        print(f"  Discord:      {'Configured' if self.discord_webhook else 'NOT configured'}")
        print("=" * 50)
        print("  Router:  OCR text  → Gemma3:12b  (classification)")
        print("  Text:    OCR text  → Gemma3:4b   (answer)")
        print("  Code:    OCR text  → qwen2.5-coder (answer)")
        print("  System:  OCR text  → system design model (answer)")
        print("  Resume:  OCR text + resume → resume model (answer)")
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
