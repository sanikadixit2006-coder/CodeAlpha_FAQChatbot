# ============================================================
#  FAQ Chatbot — AI & Data Science Internship Project
#  Author  : (Your Name)
#  Intern  : CodeAlpha Artificial Intelligence Internship
#  Tech    : Python · NLTK · Scikit-learn · Tkinter
# ============================================================

# ── Standard-library imports ──────────────────────────────
import json          # To load FAQ data from the JSON file
import string        # To handle punctuation removal
import tkinter as tk # To build the graphical user interface
from tkinter import scrolledtext  # Scrollable chat display area

# ── Third-party imports ───────────────────────────────────
import nltk                                      # Natural Language Toolkit
from nltk.corpus   import stopwords             # Common English stop words
from nltk.tokenize import word_tokenize         # Splits text into word tokens
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text to numbers
from sklearn.metrics.pairwise import cosine_similarity       # Measures text similarity

# ── Download required NLTK data (runs only first time) ────
# 'punkt'     : pre-trained tokenizer models
# 'stopwords' : list of common English words to ignore
nltk.download("punkt",     quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)  # Needed in newer NLTK versions


# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 1 — Load FAQ data from the JSON file              ║
# ╚══════════════════════════════════════════════════════════╝

def load_faq_data(filepath: str = "faq_data.json") -> tuple[list, list]:
    """
    Reads faq_data.json and returns two parallel lists:
      questions — every FAQ question as a string
      answers   — the matching answer for each question
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)          # Parse the JSON file into a Python dict

    # Extract questions (keys) and answers (values) from the simple key-value format
    questions = list(data.keys())      # All questions
    answers   = list(data.values())    # All corresponding answers
    return questions, answers


# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 2 — Text pre-processing (NLP pipeline)            ║
# ╚══════════════════════════════════════════════════════════╝

def preprocess(text: str) -> str:
    """
    Cleans and normalises a text string:
      1. Lowercase  → makes matching case-insensitive
      2. Tokenise   → splits the sentence into individual words
      3. Remove punctuation → strips '?', '!', '.', etc.
      4. Remove stop words  → drops 'the', 'is', 'a', etc.
    Returns the cleaned tokens joined back into a single string.
    """
    # 1. Convert to lower-case so 'AI' and 'ai' are treated the same
    text = text.lower()

    # 2. Tokenise — split into a list of words/tokens
    tokens = word_tokenize(text)

    # 3. Keep only alphabetic tokens, removing punctuation and numbers
    tokens = [t for t in tokens if t not in string.punctuation]

    # 4. Remove stop words that carry little meaning
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]

    # Rejoin the cleaned tokens into a single string for TF-IDF
    return " ".join(tokens)


# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 3 — Find the best matching FAQ answer             ║
# ╚══════════════════════════════════════════════════════════╝

# Similarity threshold — if the best score is below this value
# we consider the question "unknown" and show an error message.
SIMILARITY_THRESHOLD = 0.15

def get_best_answer(user_input: str,
                    questions: list,
                    answers: list) -> str:
    """
    Finds the FAQ answer that best matches the user's question.

    Algorithm:
      1. Pre-process the user input and all FAQ questions.
      2. Convert all text to TF-IDF vectors.
      3. Calculate cosine similarity between the user vector
         and every FAQ vector.
      4. Return the answer with the highest similarity score,
         or a "don't know" message if nothing is close enough.
    """
    # Pre-process the live user question
    cleaned_input = preprocess(user_input)

    # Pre-process every stored FAQ question
    cleaned_questions = [preprocess(q) for q in questions]

    # Build the combined corpus: user question + all FAQ questions
    # TF-IDF needs to see all documents together to build the vocabulary
    corpus = [cleaned_input] + cleaned_questions

    # --- TF-IDF Vectorisation -------------------------------------------
    # TfidfVectorizer converts each text document into a numeric vector
    # where each dimension represents a word and its TF-IDF weight.
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    # tfidf_matrix[0]  → vector for the user's question
    # tfidf_matrix[1:] → vectors for each FAQ question

    # --- Cosine Similarity -----------------------------------------------
    # Measures the angle between the user vector and each FAQ vector.
    # Score = 1.0 → identical meaning;  Score = 0.0 → completely different
    user_vector = tfidf_matrix[0]        # User question vector (row 0)
    faq_vectors = tfidf_matrix[1:]       # All FAQ vectors (rows 1 onward)

    # cosine_similarity returns a 2-D array; we flatten it to a 1-D list
    scores = cosine_similarity(user_vector, faq_vectors).flatten()

    # Find the index of the FAQ with the highest similarity score
    best_index = scores.argmax()
    best_score = scores[best_index]

    # --- Unknown-question guard ------------------------------------------
    if best_score < SIMILARITY_THRESHOLD:
        return ("❓ Sorry, I don't understand your question.\n"
                "Please try rephrasing it, or ask about AI, Machine Learning,\n"
                "Data Science, Python, or NLP.")

    # Return the answer that corresponds to the best-matching question
    return answers[best_index]


# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 4 — Tkinter GUI                                   ║
# ╚══════════════════════════════════════════════════════════╝

# ── Colour palette & font constants ──────────────────────
BG_DARK      = "#0f1117"   # Main window background (near-black)
BG_CHAT      = "#1a1d27"   # Chat area background
BG_INPUT     = "#252836"   # Input box background
ACCENT       = "#4f8ef7"   # Blue accent — buttons, user bubbles
ACCENT_DARK  = "#3a6fd8"   # Darker blue for hover effect
BOT_BUBBLE   = "#252836"   # Bot message background
USER_BUBBLE  = "#1e3a6e"   # User message background
TEXT_PRIMARY = "#e8eaf0"   # Primary readable text
TEXT_MUTED   = "#6b7280"   # Muted/secondary text
FONT_MAIN    = ("Segoe UI", 11)
FONT_BOLD    = ("Segoe UI", 11, "bold")
FONT_TITLE   = ("Segoe UI", 14, "bold")
FONT_SMALL   = ("Segoe UI", 9)


class FAQChatbotApp:
    """
    Main application class.
    Builds the entire Tkinter window and handles all user interaction.
    """

    def __init__(self, root: tk.Tk):
        # Store the root window reference
        self.root = root
        self.root.title("FAQ Chatbot — AI & Data Science")
        self.root.geometry("780x620")
        self.root.minsize(600, 480)
        self.root.configure(bg=BG_DARK)

        # Load FAQ data once at startup
        self.questions, self.answers = load_faq_data()

        # Build all UI sections
        self._build_header()
        self._build_chat_area()
        self._build_input_area()

        # Greet the user immediately
        self._show_welcome_message()

        # Bind Enter key to send messages
        self.root.bind("<Return>", lambda event: self._send_message())

    # ── Header bar ────────────────────────────────────────

    def _build_header(self):
        """Creates the top header strip with title and subtitle."""
        header = tk.Frame(self.root, bg="#161b2e", pady=14)
        header.pack(fill=tk.X)

        # Bot avatar circle (drawn with a Canvas)
        avatar_canvas = tk.Canvas(header, width=40, height=40,
                                   bg="#161b2e", highlightthickness=0)
        avatar_canvas.pack(side=tk.LEFT, padx=(20, 10))
        # Draw filled circle
        avatar_canvas.create_oval(2, 2, 38, 38, fill=ACCENT, outline="")
        # Draw bot icon text inside circle
        avatar_canvas.create_text(20, 20, text="🤖", font=("Segoe UI", 16))

        # Title + subtitle stacked vertically
        text_frame = tk.Frame(header, bg="#161b2e")
        text_frame.pack(side=tk.LEFT)

        tk.Label(text_frame, text="AI & Data Science FAQ Chatbot",
                 font=FONT_TITLE, fg=TEXT_PRIMARY, bg="#161b2e").pack(anchor="w")
        tk.Label(text_frame,
                 text="Ask me anything about AI · ML · NLP · Python · Data Science",
                 font=FONT_SMALL, fg=TEXT_MUTED, bg="#161b2e").pack(anchor="w")

        # Status indicator (green dot + "Online")
        status_frame = tk.Frame(header, bg="#161b2e")
        status_frame.pack(side=tk.RIGHT, padx=20)
        tk.Canvas(status_frame, width=10, height=10,
                  bg="#161b2e", highlightthickness=0).pack(side=tk.LEFT)
        status_dot = tk.Canvas(status_frame, width=10, height=10,
                               bg="#161b2e", highlightthickness=0)
        status_dot.pack(side=tk.LEFT)
        status_dot.create_oval(1, 1, 9, 9, fill="#22c55e", outline="")
        tk.Label(status_frame, text=" Online", font=FONT_SMALL,
                 fg="#22c55e", bg="#161b2e").pack(side=tk.LEFT)

    # ── Chat display area ──────────────────────────────────

    def _build_chat_area(self):
        """Creates the scrollable read-only chat history area."""
        # Outer container with padding
        chat_container = tk.Frame(self.root, bg=BG_DARK, padx=12, pady=8)
        chat_container.pack(fill=tk.BOTH, expand=True)

        # ScrolledText widget — shows the conversation
        self.chat_display = scrolledtext.ScrolledText(
            chat_container,
            wrap=tk.WORD,          # Wrap long lines at word boundaries
            state=tk.DISABLED,     # Read-only — users cannot type here
            font=FONT_MAIN,
            bg=BG_CHAT,
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief=tk.FLAT,
            padx=16,
            pady=12,
            spacing3=6,            # Extra space after each paragraph
            cursor="arrow",        # No text-edit cursor
            bd=0,
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        # Define named text tags for coloured message bubbles
        # User messages — blue tint
        self.chat_display.tag_config(
            "user_label", foreground=ACCENT, font=FONT_BOLD)
        self.chat_display.tag_config(
            "user_msg",   foreground="#c7d9ff", font=FONT_MAIN,
            lmargin1=20, lmargin2=20)

        # Bot messages — neutral
        self.chat_display.tag_config(
            "bot_label",  foreground="#a78bfa", font=FONT_BOLD)
        self.chat_display.tag_config(
            "bot_msg",    foreground=TEXT_PRIMARY, font=FONT_MAIN,
            lmargin1=20, lmargin2=20)

        # Separator / divider lines
        self.chat_display.tag_config(
            "divider",    foreground=TEXT_MUTED, font=FONT_SMALL)

    # ── Input bar at the bottom ────────────────────────────

    def _build_input_area(self):
        """Creates the text-entry bar and Send button at the bottom."""
        input_frame = tk.Frame(self.root, bg=BG_DARK, padx=12, pady=10)
        input_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # Rounded container for the entry box
        entry_bg = tk.Frame(input_frame, bg=BG_INPUT,
                            highlightbackground=ACCENT,
                            highlightthickness=1)
        entry_bg.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        # The actual text-input field
        self.user_input = tk.Entry(
            entry_bg,
            font=FONT_MAIN,
            bg=BG_INPUT,
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,   # Cursor colour
            relief=tk.FLAT,
            bd=8,
        )
        self.user_input.pack(fill=tk.X, ipady=6)
        self.user_input.insert(0, "Type your question here…")    # Placeholder
        self.user_input.config(fg=TEXT_MUTED)

        # Clear placeholder on focus
        self.user_input.bind("<FocusIn>",  self._clear_placeholder)
        self.user_input.bind("<FocusOut>", self._restore_placeholder)

        # Send button
        self.send_btn = tk.Button(
            input_frame,
            text="  Send  ➤",
            font=FONT_BOLD,
            bg=ACCENT,
            fg="white",
            activebackground=ACCENT_DARK,
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",          # Hand cursor on hover
            padx=16,
            pady=8,
            command=self._send_message,
            bd=0,
        )
        self.send_btn.pack(side=tk.RIGHT)

        # Clear / reset button
        clear_btn = tk.Button(
            input_frame,
            text="🗑 Clear",
            font=FONT_SMALL,
            bg=BG_INPUT,
            fg=TEXT_MUTED,
            activebackground="#2d303e",
            activeforeground=TEXT_PRIMARY,
            relief=tk.FLAT,
            cursor="hand2",
            padx=10,
            pady=8,
            command=self._clear_chat,
            bd=0,
        )
        clear_btn.pack(side=tk.RIGHT, padx=(0, 6))

    # ── Placeholder helpers ────────────────────────────────

    def _clear_placeholder(self, event=None):
        """Removes the grey placeholder text when the user clicks the box."""
        if self.user_input.get() == "Type your question here…":
            self.user_input.delete(0, tk.END)
            self.user_input.config(fg=TEXT_PRIMARY)

    def _restore_placeholder(self, event=None):
        """Restores the placeholder if the user leaves the box empty."""
        if not self.user_input.get().strip():
            self.user_input.insert(0, "Type your question here…")
            self.user_input.config(fg=TEXT_MUTED)

    # ── Message handling ───────────────────────────────────

    def _append_message(self, sender: str, message: str):
        """
        Appends a formatted message to the chat display.
          sender — either "You" or "Bot"
          message — the text to display
        """
        # Unlock the read-only widget so we can write to it
        self.chat_display.config(state=tk.NORMAL)

        if sender == "You":
            self.chat_display.insert(tk.END, f"\n  👤 You\n", "user_label")
            self.chat_display.insert(tk.END, f"  {message}\n",  "user_msg")
        else:
            self.chat_display.insert(tk.END, f"\n  🤖 Bot\n", "bot_label")
            self.chat_display.insert(tk.END, f"  {message}\n",  "bot_msg")

        # Add a thin separator line after each exchange
        self.chat_display.insert(
            tk.END, "  " + "─" * 60 + "\n", "divider")

        # Lock the widget again so users cannot edit the history
        self.chat_display.config(state=tk.DISABLED)

        # Auto-scroll to the newest message
        self.chat_display.yview(tk.END)

    def _send_message(self):
        """
        Called when the user clicks Send or presses Enter.
        1. Reads the input, validates it.
        2. Displays the user's question.
        3. Calls get_best_answer() to find the best FAQ match.
        4. Displays the bot's answer.
        """
        raw_text = self.user_input.get().strip()

        # Ignore empty input or the placeholder string
        if not raw_text or raw_text == "Type your question here…":
            return

        # Show the user's question in the chat
        self._append_message("You", raw_text)

        # Clear the entry box
        self.user_input.delete(0, tk.END)

        # Get the best-matching answer using NLP + cosine similarity
        answer = get_best_answer(raw_text, self.questions, self.answers)

        # Show the bot's answer in the chat
        self._append_message("Bot", answer)

    def _clear_chat(self):
        """Wipes the chat display and shows the welcome message again."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)    # Delete all text
        self.chat_display.config(state=tk.DISABLED)
        self._show_welcome_message()

    def _show_welcome_message(self):
        """Displays a friendly greeting when the app first starts."""
        welcome = (
            "Hello! 👋 I'm your AI & Data Science FAQ Bot.\n\n"
            f"📚 I have **{len(self.questions)} comprehensive FAQs** on topics including:\n"
            "  • AI, Machine Learning, Deep Learning, Reinforcement Learning\n"
            "  • Supervised, Unsupervised & Ensemble Learning\n"
            "  • Neural Networks, Decision Trees, Random Forests\n"
            "  • Natural Language Processing (NLP) & Text Processing\n"
            "  • Classification, Regression, Clustering\n"
            "  • Feature Engineering, TF-IDF, Cosine Similarity\n"
            "  • Tokenization, Preprocessing, Data Handling\n"
            "  • Model Evaluation, Precision, Recall, Confusion Matrix\n"
            "  • Hyperparameters, Gradient Descent, Cross-Validation\n"
            "  • Python, Data Science, Normalization & More!\n\n"
            "💡 Try asking:\n"
            "  'What is machine learning?' | 'Explain neural networks'\n"
            "  'What is clustering?' | 'How to handle missing data?'\n\n"
            "Type your question below and press Send (or Enter)!"
        )
        self._append_message("Bot", welcome)


# ╔══════════════════════════════════════════════════════════╗
# ║  STEP 5 — Entry point                                   ║
# ╚══════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    root = tk.Tk()               # Create the main Tkinter window
    app  = FAQChatbotApp(root)   # Instantiate our chatbot application
    root.mainloop()              # Start the event loop (keeps window open)