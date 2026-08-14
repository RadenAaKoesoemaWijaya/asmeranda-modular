import random
import string
from PIL import Image, ImageDraw, ImageFont
import io
import base64

class CaptchaGenerator:
    def __init__(self):
        self.width = 200
        self.height = 80
        self.length = 5
        self.font_size = 28
    
    def generate_text(self, mode="text"):
        """Generate random captcha text or math problem"""
        if mode == "math":
            num1 = random.randint(1, 10)
            num2 = random.randint(1, 10)
            operator = random.choice(['+', '-'])
            
            # Ensure no negative results for simplicity
            if operator == '-' and num1 < num2:
                num1, num2 = num2, num1
                
            expression = f"{num1} {operator} {num2}"
            if operator == '+':
                result = str(num1 + num2)
            else:
                result = str(num1 - num2)
            return expression + " = ?", result
        else:
            chars = string.ascii_uppercase + string.digits
            text = ''.join(random.choice(chars) for _ in range(self.length))
            return text, text
    
    def generate_captcha(self, text=None, display_text=None):
        """Generate captcha image"""
        if text is None or display_text is None:
            display_text, text = self.generate_text()
        
        # Create image with white background
        image = Image.new('RGB', (self.width, self.height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Add less noise for better readability
        for _ in range(30):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            draw.point((x, y), fill='gray')
        
        # Add few subtle lines
        for _ in range(2):
            x1 = random.randint(0, self.width)
            y1 = random.randint(0, self.height)
            x2 = random.randint(0, self.width)
            y2 = random.randint(0, self.height)
            draw.line([(x1, y1), (x2, y2)], fill='lightgray', width=1)
        
        # Try to use a font, fallback to default if not available
        try:
            font = ImageFont.truetype('arial.ttf', self.font_size)
        except:
            try:
                font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', self.font_size)
            except:
                font = ImageFont.load_default()
        
        # Draw text
        text_width = draw.textlength(display_text, font=font)
        x = (self.width - text_width) // 2
        y = (self.height - self.font_size) // 2
        
        # Darker colors for readability
        color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
        draw.text((x, y), display_text, font=font, fill=color)
        
        # Add slight distortion
        for _ in range(1):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            draw.arc([(x, y), (x+30, y+30)], 0, 360, fill='gray')
        
        return image, text
    
    def get_captcha_base64(self, text=None, mode="math"):
        """Generate captcha and return as base64 string"""
        if text is None:
            display_text, actual_text = self.generate_text(mode=mode)
        else:
            # When refreshing with an existing text (rarely happens unless forced), fallback
            display_text, actual_text = text, text
            
        image, final_actual_text = self.generate_captcha(actual_text, display_text)
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}", final_actual_text

def verify_captcha(input_text, actual_text, case_sensitive=True):
    """Verify captcha input"""
    if case_sensitive:
        return input_text.strip() == actual_text.strip()
    else:
        return input_text.strip().upper() == actual_text.strip().upper()

# Initialize global captcha generator
captcha_gen = CaptchaGenerator()