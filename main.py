import sys
import torch
import re
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QPropertyAnimation, QRect, QEasingCurve, QTimer
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

# Load the trained model and tokenizer
model_path = "D:/fake_news_detection/bert_fakenews_model"
tokenizer_path = "D:/fake_news_detection/bert_fakenews_tokenizer"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer with exception handling
try:
    model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model.eval()
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    sys.exit(1)


def clean_text(text):
    """Preprocess text: remove special characters, links, and normalize case."""
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r"http\S+", "", text)  # Remove links
    text = re.sub(r"[^a-zA-Z0-9.,!?'\s\"]", "", text)  # Keep only valid characters
    return text.lower().strip()


def classify_news(news_text):
    """Classify input news as Real, Fake, or Uncertain."""
    if not news_text.strip():
        return "⚠️ Please enter news text for classification."

    text = clean_text(news_text)
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)  # Convert logits to probabilities
    confidence_real = probs[0][1].item()
    confidence_fake = probs[0][0].item()

    if confidence_real >= 0.65:
        return f"✅ Real News (Confidence: {confidence_real:.2f})"
    elif confidence_fake >= 0.65:
        return f"❌ Fake News (Confidence: {confidence_fake:.2f})"
    else:
        return f"🤔 Uncertain (Real: {confidence_real:.2f}, Fake: {confidence_fake:.2f})"


class NewsDetectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        """Initialize the user interface with animations."""
        self.setWindowTitle("📰 Fake News Detector")
        self.setGeometry(100, 100, 600, 450)

        # Main layout
        layout = QVBoxLayout()
        self.setStyleSheet("background-color: #1E1E1E; color: white;")

        # Title label
        self.label = QLabel("Enter a news article:")
        self.label.setFont(QFont("Arial", 14, QFont.Bold))
        self.label.setStyleSheet("color: #FFD700;")  # Gold color
        layout.addWidget(self.label)

        # Text Input Box
        self.text_input = QTextEdit()
        self.text_input.setFont(QFont("Arial", 12))
        self.text_input.setStyleSheet(
            "background-color: #333; color: white; padding: 8px; border-radius: 5px;"
        )
        layout.addWidget(self.text_input)

        # Classify Button
        self.classify_button = QPushButton("🔍 Classify News")
        self.classify_button.setFont(QFont("Arial", 14))
        self.classify_button.setStyleSheet(
            "background-color: #008CBA; color: white; padding: 10px; border-radius: 5px;"
        )
        self.classify_button.clicked.connect(self.run_classification)
        self.classify_button.installEventFilter(self)  # For hover animation
        layout.addWidget(self.classify_button)

        # Output Label
        self.result_label = QLabel("")
        self.result_label.setFont(QFont("Arial", 14))
        self.result_label.setStyleSheet("color: #00FF00;")  # Green text for output
        layout.addWidget(self.result_label)

        # Clear Button
        self.clear_button = QPushButton("🗑️ Clear")
        self.clear_button.setFont(QFont("Arial", 12))
        self.clear_button.setStyleSheet(
            "background-color: #FF5733; color: white; padding: 8px; border-radius: 5px;"
        )
        self.clear_button.clicked.connect(self.clear_text)
        layout.addWidget(self.clear_button)

        # Exit Button
        self.exit_button = QPushButton("🚪 Exit")
        self.exit_button.setFont(QFont("Arial", 12))
        self.exit_button.setStyleSheet(
            "background-color: #A9A9A9; color: white; padding: 8px; border-radius: 5px;"
        )
        self.exit_button.clicked.connect(self.close)
        layout.addWidget(self.exit_button)

        # Set final layout
        self.setLayout(layout)

    def run_classification(self):
        """Run fake news detection on input text."""
        news_text = self.text_input.toPlainText().strip()
        result = classify_news(news_text)
        self.result_label.setText(result)

        # Animation: Fade-in effect for result text
        self.result_label.setWindowOpacity(0)
        fade_anim = QPropertyAnimation(self.result_label, b"windowOpacity")
        fade_anim.setDuration(600)
        fade_anim.setStartValue(0)
        fade_anim.setEndValue(1)
        fade_anim.start(QPropertyAnimation.DeleteWhenStopped)

    def clear_text(self):
        """Clear input and output fields."""
        self.text_input.clear()
        self.result_label.clear()

    def eventFilter(self, obj, event):
        """Apply hover animation on the classify button."""
        if obj == self.classify_button:
            if event.type() == event.Enter:
                bounce_anim = QPropertyAnimation(self.classify_button, b"geometry")
                bounce_anim.setDuration(300)
                bounce_anim.setStartValue(self.classify_button.geometry())
                bounce_anim.setEndValue(
                    QRect(
                        self.classify_button.x(),
                        self.classify_button.y() - 5,
                        self.classify_button.width(),
                        self.classify_button.height(),
                    )
                )
                bounce_anim.setEasingCurve(QEasingCurve.OutBounce)
                bounce_anim.start(QPropertyAnimation.DeleteWhenStopped)
        return super().eventFilter(obj, event)


def main():
    """Launch the application."""
    app = QApplication(sys.argv)
    window = NewsDetectorApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()