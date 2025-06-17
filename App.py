import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pickle
import numpy as np
import re
import string
from scipy.sparse import hstack, csr_matrix
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import threading
import nltk
import pandas as pd
from tqdm import tqdm
from deep_translator import GoogleTranslator
import os
import tempfile
import time
from gtts import gTTS
import warnings
warnings.filterwarnings('ignore')


class MovieSummaryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Summary Analysis")
        self.root.geometry("900x600")
        self.root.configure(bg="#2c3e50")
        
        # Load models
        self.load_models()
        
        # Language codes for translation
        self.language_codes = {
            "English": "en",
            "Arabic": "ar",
            "Korean": "ko",
            "Japanese": "ja",
            "Spanish": "es",
            "Urdu": "ur"
        }
        
        # Create temporary directory for audio files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create UI
        self.create_ui()

    def load_models(self):
        # Load all pickle files
        with open('models/genre_prediction_logistic_simplified.pkl', 'rb') as f:
            self.model = pickle.load(f)
            
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
            
        with open('models/feature_selectors.pkl', 'rb') as f:
            self.feature_selectors = pickle.load(f)
            
        with open('models/thresholds.pkl', 'rb') as f:
            self.thresholds = pickle.load(f)
            
        with open('models/preprocessed_data.pkl', 'rb') as f:
            data = pickle.load(f)
            self.all_genres = data['all_genres']
            self.genre_to_idx = data['genre_to_idx']
            
        # Initialize preprocessing tools
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.custom_stopwords = {'the', 'a', 'an', 'of', 'and', 'or', 'to', 'in', 'on', 'at'}
    
    def load_available_voices(self):
        """Load supported languages (simplified as we're using gTTS)"""
        self.available_voices = {
            'English': 'en',
            'Arabic': 'ar',
            'Korean': 'ko',
            'Japanese': 'ja',
            'Spanish': 'es',
            'Urdu': 'ur'
        }
    
    def create_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, text="Movie Summary Analyzer", 
                              font=("Helvetica", 24, "bold"), bg="#2c3e50", fg="#ecf0f1")
        title_label.pack(pady=(0, 20))
        
        # Summary input
        input_frame = tk.Frame(main_frame, bg="#2c3e50")
        input_frame.pack(fill=tk.BOTH, expand=True)
        
        summary_label = tk.Label(input_frame, text="Enter Movie Summary:", 
                               font=("Helvetica", 12), bg="#2c3e50", fg="#ecf0f1")
        summary_label.pack(anchor="w")
        
        self.summary_text = scrolledtext.ScrolledText(input_frame, height=10, 
                                                    font=("Helvetica", 11), bg="#34495e", fg="#ecf0f1")
        self.summary_text.pack(fill=tk.BOTH, expand=True, pady=(5, 15))
        
        # Options frame
        options_frame = tk.Frame(main_frame, bg="#2c3e50")
        options_frame.pack(fill=tk.X, pady=10)
        
        # Audio conversion options
        audio_frame = tk.LabelFrame(options_frame, text="Convert to Audio", 
                                 font=("Helvetica", 11, "bold"), bg="#2c3e50", fg="#ecf0f1", padx=10, pady=10)
        audio_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Languages that we support
        languages = list(self.language_codes.keys())
        self.language_var = tk.StringVar(value=languages[0])
        
        language_label = tk.Label(audio_frame, text="Select Language:", bg="#2c3e50", fg="#ecf0f1")
        language_label.pack(anchor="w")
        
        language_menu = ttk.Combobox(audio_frame, textvariable=self.language_var, values=languages)
        language_menu.pack(fill=tk.X, pady=(5, 10))
        
        # Translation options
        translate_frame = tk.Frame(audio_frame, bg="#2c3e50")
        translate_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.translate_var = tk.BooleanVar(value=True)
        translate_check = tk.Checkbutton(translate_frame, text="Translate before playing", 
                                       variable=self.translate_var, bg="#2c3e50", fg="#ecf0f1",
                                       selectcolor="#34495e", activebackground="#2c3e50", 
                                       activeforeground="#ecf0f1")
        translate_check.pack(anchor="w")
        
        # Voice rate control
        rate_frame = tk.Frame(audio_frame, bg="#2c3e50")
        rate_frame.pack(fill=tk.X, pady=(0, 10))
        
        rate_label = tk.Label(rate_frame, text="Speech Rate:", bg="#2c3e50", fg="#ecf0f1")
        rate_label.pack(side=tk.LEFT)
        
        self.rate_var = tk.IntVar(value=200)
        rate_scale = ttk.Scale(rate_frame, from_=100, to=300, orient=tk.HORIZONTAL,
                             variable=self.rate_var, length=150)
        rate_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Convert button
        convert_btn = tk.Button(audio_frame, text="Convert and Play Audio", 
                              command=self.convert_to_audio, bg="#3498db", fg="white",
                              font=("Helvetica", 10, "bold"), padx=10, pady=5)
        convert_btn.pack(fill=tk.X)
        
        # Stop button for audio
        stop_btn = tk.Button(audio_frame, text="Stop Audio", 
                           command=self.stop_audio, bg="#e74c3c", fg="white",
                           font=("Helvetica", 10, "bold"), padx=10, pady=5)
        stop_btn.pack(fill=tk.X, pady=(5, 0))
        
        # Genre prediction options
        genre_frame = tk.LabelFrame(options_frame, text="Predict Genre", 
                                  font=("Helvetica", 11, "bold"), bg="#2c3e50", fg="#ecf0f1", padx=10, pady=10)
        genre_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        predict_btn = tk.Button(genre_frame, text="Predict Genres", 
                              command=self.predict_genre, bg="#2ecc71", fg="white",
                              font=("Helvetica", 10, "bold"), padx=10, pady=5)
        predict_btn.pack(fill=tk.X, pady=(31, 0))
        
        # Results area
        results_frame = tk.LabelFrame(main_frame, text="Results", 
                                    font=("Helvetica", 12, "bold"), bg="#2c3e50", fg="#ecf0f1")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(15, 0))
        
        # Add a loading indicator
        self.loading_var = tk.StringVar(value="")
        loading_label = tk.Label(results_frame, textvariable=self.loading_var, 
                               font=("Helvetica", 10, "italic"), bg="#2c3e50", fg="#e74c3c")
        loading_label.pack(anchor="w", padx=10, pady=(10, 0))
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=8, 
                                                    font=("Helvetica", 11), bg="#34495e", fg="#ecf0f1")
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))
    
    def clean_summary(self, summary):
        """Clean and preprocess the summary text"""
        if not summary.strip():
            return ""
            
        summary = summary.lower()
        summary = re.sub(r'https?://\S+|www\.\S+|<.*?>', '', summary)
        
        for c, e in {"n't": " not", "'s": " is", "'m": " am", "'re": " are", 
                   "'ll": " will", "'ve": " have", "'d": " would"}.items():
            summary = summary.replace(c, e)
            
        tokens = word_tokenize(summary)
        processed = []
        
        for token in tokens:
            if token in string.punctuation or token.isdigit() or \
               (len(token) <= 2 and token not in {'no', 'not'}) or token in self.custom_stopwords:
                continue
                
            lemma = self.lemmatizer.lemmatize(token)
            processed.append(lemma)
            
        return ' '.join(processed).strip()
    
    def extract_features(self, clean_summary):
        """Extract features from cleaned summary"""
        # TF-IDF features
        tfidf_features = self.vectorizer.transform([clean_summary])
        
        # Sentiment features
        sentiment = self.sentiment_analyzer.polarity_scores(clean_summary)
        sentiment_features = csr_matrix([[
            sentiment['neg'], sentiment['neu'], 
            sentiment['pos'], sentiment['compound']
        ]])
        
        # Text stats features
        tokens = clean_summary.split()
        stats_features = csr_matrix([[
            len(tokens),  # Word count
            0,  # Placeholder for part-of-speech stats
            0,  # Placeholder for part-of-speech stats
            0,  # Placeholder for part-of-speech stats
            sum(len(w) for w in tokens) / max(1, len(tokens))  # Avg word length
        ]])
        
        # Combine features
        combined_features = hstack([tfidf_features, sentiment_features, stats_features])
        
        # Extract per-genre selected features
        selected_features = {genre: selector.transform(tfidf_features) 
                           for genre, selector in self.feature_selectors.items()}
        
        return tfidf_features, combined_features, selected_features
    
    def predict_genre(self):
        """Predict genre based on summary"""
        summary = self.summary_text.get("1.0", tk.END).strip()
        
        if not summary:
            messagebox.showwarning("Input Required", "Please enter a movie summary.")
            return
            
        self.results_text.delete("1.0", tk.END)
        self.loading_var.set("Analyzing summary...")
        self.root.update()
        
        # Process in a separate thread to keep UI responsive
        threading.Thread(target=self._predict_genre_thread, args=(summary,), daemon=True).start()
    
    def _predict_genre_thread(self, summary):
        clean_summary = self.clean_summary(summary)
        
        if not clean_summary:
            self.update_results("Could not process summary. Please try a more detailed description.")
            return
            
        _, _, selected_features = self.extract_features(clean_summary)
        
        # Combine selected features
        X_combined_features = hstack(list(selected_features.values()))
        
        # Make predictions
        y_pred_proba = np.zeros((1, len(self.all_genres)))
        for i, est in enumerate(self.model.estimators_):
            y_pred_proba[0, i] = est.predict_proba(X_combined_features)[:, 1]
        
        # Apply thresholds
        y_pred = np.array([(y_pred_proba[0, i] >= t) for i, t in enumerate(self.thresholds)])
        
        # Get predicted genres
        predicted_genres = [self.all_genres[i] for i, pred in enumerate(y_pred) if pred]
        
        # Format results
        result = "Predicted Genres:\n"
        if predicted_genres:
            for genre in predicted_genres:
                probability = y_pred_proba[0, self.genre_to_idx[genre]] * 100
                result += f"• {''.join(genre)}: {probability:.1f}%\n"
        else:
            result += "No specific genres identified. The summary may be too short or vague."
        
        self.update_results(result)
        self.loading_var.set("")
    
    def convert_to_audio(self):
        """Convert summary to audio in the selected language with translation"""
        summary = self.summary_text.get("1.0", tk.END).strip()
        
        if not summary:
            messagebox.showwarning("Input Required", "Please enter a movie summary.")
            return
            
        language = self.language_var.get()
        rate = self.rate_var.get()
        translate = self.translate_var.get()
        
        self.results_text.delete("1.0", tk.END)
        
        if translate and language != "English":
            self.loading_var.set(f"Translating to {language} and converting to audio...")
            self.results_text.insert(tk.END, f"Translating summary to {language} and converting to audio...\n")
        else:
            self.loading_var.set(f"Converting to {language} audio...")
            self.results_text.insert(tk.END, f"Converting summary to {language} audio...\n")
            
        self.root.update()
        
        # Process in a separate thread to keep UI responsive
        threading.Thread(target=self._convert_audio_thread, 
                       args=(summary, language, rate, translate), daemon=True).start()

    def translate_text(self, text, target_language):
        """
        Translate text to target language using Google Translator
        
        Parameters:
        text (str): Text to translate
        target_language (str): Language code (e.g., 'es', 'ar', 'ko')
        
        Returns:
        str: Translated text
        """
        try:
            # Limit text length if needed (Google Translator has character limits)
            if len(text) > 5000:
                text = text[:5000]
            
            # Handle very long text by splitting it
            if len(text) > 1000:
                # Split into chunks and translate each chunk
                chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                translated_chunks = []
                
                for chunk in chunks:
                    translator = GoogleTranslator(source='auto', target=target_language)
                    translated_chunk = translator.translate(chunk)
                    translated_chunks.append(translated_chunk)
                    time.sleep(0.5)  # Small delay to avoid rate limiting
                    
                return ' '.join(translated_chunks)
            else:
                # For shorter text, translate in one go
                translator = GoogleTranslator(source='auto', target=target_language)
                translation = translator.translate(text)
                return translation
                
        except Exception as e:
            print(f"Translation error: {str(e)}")
            raise Exception(f"Failed to translate text: {str(e)}")

    def text_to_speech(self, text, language_code, filename):
        """
        Convert text to speech and save as audio file
        
        Parameters:
        text (str): Text to convert
        language_code (str): Language code for TTS
        filename (str): Output file path
        
        Returns:
        bool: Success status
        """
        try:
            if not text:
                print("Empty text, skipping TTS")
                return False
            
            # Map language codes to TTS compatible codes if needed
            tts_language_map = {
                'en': 'en',
                'ar': 'ar',
                'ko': 'ko',
                'ja': 'ja',
                'es': 'es',
                'ur': 'ur'
            }
            
            # Get the correct TTS language code or default to the provided code
            tts_lang = tts_language_map.get(language_code, language_code)
            
            # Try different speeds based on language
            slow_languages = ['ar', 'ur']  # Languages that might benefit from slower speech
            slow = tts_lang in slow_languages
            
            # Create and save the audio file
            tts = gTTS(text=text, lang=tts_lang, slow=slow)
            tts.save(filename)
            
            # Verify the file was created successfully
            if os.path.exists(filename) and os.path.getsize(filename) > 100:
                return True
            else:
                print(f"TTS file creation failed or file is too small: {filename}")
                return False
                
        except Exception as e:
            print(f"TTS error: {str(e)}")
            return False

    def process_summary(self, summary, target_language, output_file=None):
        """
        Process a single summary by translating it (if needed) and converting to speech
        
        Parameters:
        summary (str): The text to translate and convert
        target_language (str): Language code for translation and TTS
        output_file (str): Path to save audio file (optional)
        
        Returns:
        tuple: (translated_text, audio_file_path, success)
        """
        try:
            # Generate output filename if not provided
            if not output_file:
                # Create a unique filename using hash of content and timestamp
                timestamp = int(time.time())
                unique_id = abs(hash(f"{summary}{target_language}{timestamp}")) % 10000
                output_file = f"audio_files/summary_{target_language}_{unique_id}.mp3"
            
            # Translate if not English
            if target_language != 'en':
                translated_text = self.translate_text(summary, target_language)
            else:
                translated_text = summary
                
            # Convert to speech
            success = self.text_to_speech(translated_text, target_language, output_file)
            
            if success:
                return translated_text, output_file, True
            else:
                return translated_text, None, False
                
        except Exception as e:
            print(f"Error processing summary: {str(e)}")
            return summary, None, False
    
    def _convert_audio_thread(self, summary, language, rate, translate):
        """Thread for translation and audio conversion"""
        try:
            original_summary = summary
            
            # Get the target language code
            target_lang = self.language_codes.get(language, "en")
            
            # Skip translation if not needed
            should_translate = translate and language != "English"
            
            # Update UI with initial status
            if should_translate:
                self.update_results(f"Translating text to {language} and converting to audio...")
            else:
                self.update_results(f"Converting to {language} audio...")
            
            # Create a temporary audio file path
            audio_file = os.path.join(self.temp_dir, f"summary_{language}_{hash(summary)}.mp3")
            
            # Use the integrated process_summary method
            # This handles both translation and TTS in a single call
            if should_translate:
                translated_text, audio_path, success = self.process_summary(summary, target_lang, audio_file)
                translation_status = f"✓ Translated from English to {language}" if success else "✗ Translation failed"
            else:
                # For English or when translation is disabled
                translated_text = summary
                success = self.text_to_speech(summary, target_lang, audio_file)
                translation_status = "No translation needed"
            
            if not success or not os.path.exists(audio_file):
                raise Exception(f"Failed to create audio file for {language}")
            
            # Update UI with translation status
            self.update_results(
                f"Converting to audio in {language}...\n\n"
                f"Translation status: {translation_status}\n\n"
                f"Original:\n{original_summary}\n\n"
                f"{'Translated text:' if should_translate else 'Text:'}\n{translated_text}"
            )
            
            # Play the audio file
            self.loading_var.set(f"Playing audio in {language}...")
            self.play_audio_file(audio_file, original_summary)
            
            self.update_results(
                f"Audio playback complete.\n\n"
                f"Translation status: {translation_status}\n\n"
                f"Original:\n{original_summary}\n\n"
                f"{'Translated text:' if should_translate else 'Text:'}\n{translated_text}"
            )
            
        except Exception as e:
            error_message = f"Error processing audio: {str(e)}\n\n"
            error_message += "If you're having issues with voice playback, try:\n"
            error_message += "1. Check if you have an internet connection (required for gTTS)\n"
            error_message += "2. Try a different language\n"
            error_message += "3. Ensure you have the required audio player on your system\n"
            
            self.update_results(error_message)
        finally:
            self.loading_var.set("")
    
    def play_audio_file(self, audio_file, original_text=""):
        """Play an audio file using platform-specific methods and provide feedback"""
        import platform
        import subprocess
        
        try:
            system = platform.system()
            
            # Check if file exists
            if not os.path.exists(audio_file):
                raise Exception(f"Audio file not found: {audio_file}")
                
            # Get file size for debugging
            file_size = os.path.getsize(audio_file)
            if file_size < 100:  # Too small, likely an error
                raise Exception(f"Audio file appears to be empty or corrupted (size: {file_size} bytes)")
                
            # Update UI
            self.update_results(f"Playing audio file ({file_size/1024:.1f} KB)...\n"
                              f"If you don't hear anything, check your system volume.")
                
            # Use appropriate player based on platform
            if system == "Windows":
                os.startfile(audio_file)
            elif system == "Darwin":  # macOS
                result = subprocess.run(["afplay", audio_file], stderr=subprocess.PIPE)
                if result.returncode != 0:
                    raise Exception(f"Error playing audio: {result.stderr.decode()}")
            else:  # Linux and other Unix-like systems
                players = ["xdg-open", "mpg123", "mpg321", "play"]
                played = False
                for player in players:
                    try:
                        subprocess.run([player, audio_file], check=True, stderr=subprocess.PIPE)
                        played = True
                        break
                    except (subprocess.SubprocessError, FileNotFoundError):
                        continue
                
                if not played:
                    raise Exception("Could not find a suitable audio player on your system")
                
            # Wait for a duration proportional to summary length
            # This is an approximation, not perfect
            import time
            
            # Get file size instead of content length to handle binary files better
            file_size = os.path.getsize(audio_file)
            wait_time = min(90, max(10, file_size / 5000))  # Adjusted ratio
            
           
            # Skip extended waiting if this is a sample summary
            if any(s in original_text for s in sample_summaries):
                self.update_results("Sample summary detected. Audio playback will be brief.")
                wait_time = min(wait_time, 5)  # Short wait for samples
            
            # Show countdown
            start_time = time.time()
            while time.time() - start_time < wait_time:
                remaining = wait_time - (time.time() - start_time)
                self.loading_var.set(f"Playing audio... (approx. {remaining:.0f}s remaining)")
                self.root.update()
                time.sleep(0.5)
                
                # Check if the user pressed the stop button
                if self.loading_var.get() == "":
                    break
            
        except Exception as e:
            self.update_results(f"Error playing audio: {str(e)}\n\n"
                              "Try the following:\n"
                              "1. Check your system volume\n"
                              "2. Try a different language")

    def stop_audio(self):
        """Stop any currently playing audio"""
        try:
            # Windows-specific implementation for stopping audio
            import subprocess
            subprocess.call(["taskkill", "/F", "/IM", "wmplayer.exe"], stderr=subprocess.DEVNULL)
            self.update_results("Audio playback stopped.")
        except:
            pass
    
    def update_results(self, text):
        """Update results text widget from any thread"""
        self.root.after(0, self._update_results_main_thread, text)
    
    def _update_results_main_thread(self, text):
        """Update results text from main thread"""
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, text)
    
    def __del__(self):
        """Clean up temporary files on exit"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass

# Configure ttk style
root = tk.Tk()
style = ttk.Style()
style.theme_use('clam')

# Configure colors
style.configure("TCombobox", fieldbackground="#34495e", background="#2c3e50", 
                foreground="#ecf0f1", arrowcolor="#ecf0f1")
style.configure("TScale", background="#2c3e50", troughcolor="#34495e")

app = MovieSummaryApp(root)
root.mainloop()