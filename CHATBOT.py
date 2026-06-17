import nltk
import tkinter as tk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure required tokenizers are present
nltk.download('punkt', quiet=True)

# 1. Dataset setup
faqs = {
    "What is your name?": "I am a FAQ chatbot.",
    "What do you do?": "I answer basic questions using NLP techniques.",
    "How do I reset password?": "Go to settings and click reset password option.",
    "How to contact support?": "You can contact support via email or phone.",
    "What is AI?": "AI means Artificial Intelligence."
}

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

questions = list(faqs.keys())
clean_questions = [preprocess(q) for q in questions]

# 2. Vectorization
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(clean_questions)

# 3. NLP Processing Logic
def get_answer(user_input):
    user_input = preprocess(user_input)
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, faq_vectors)
    
    index = similarity.argmax()
    score = similarity[0][index]

    if score < 0.2:
        return "Sorry, I don't understand your question."

    return faqs[questions[index]]

# 4. GUI Event Interaction Function
def send_message():
    user_text = entry.get().strip()
    if not user_text:
        return
        
    # Append user question to the chat display
    chat_box.insert(tk.END, f"You: {user_text}\n")
    entry.delete(0, tk.END)
    
    # Process and append response
    bot_response = get_answer(user_text)
    chat_box.insert(tk.END, f"Bot: {bot_response}\n\n")
    chat_box.see(tk.END)

# 5. Tkinter Layout Window Setup
root = tk.Tk()
root.title("FAQ Chatbot")
root.geometry("400x500")

chat_box = tk.Text(root, bd=1, bg="white", height=20, width=50)
chat_box.pack(pady=10)

entry = tk.Entry(root, width=40)
entry.pack(pady=5)

# Bind the click action to the send_message function
send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(pady=5)

root.mainloop()
