# src/utils/text_cleaner.py
import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer to clean text data (supports English and Arabic).
    Removes unwanted characters, normalizes whitespace, and converts to lowercase.
    Handles pandas Series, lists, or single strings.
    """

    def __init__(self):
        """Initialize the cleaner and compile regex patterns for efficiency."""
        # Compile regex patterns once for slightly better performance
        # Pattern to keep only English letters, Arabic letters, and whitespace
        self.unwanted_chars_pattern = re.compile(r"[^a-zA-Z\u0621-\u064A\s]")
        # Pattern to replace multiple whitespace characters with a single space
        self.multiple_spaces_pattern = re.compile(r'\s+')
        # Optional: Confirmation message (can be removed in production)
        # print("✅ TextCleaner initialized with compiled regex patterns.")

    def fit(self, X, y=None):
        """
        Fit method (required by Scikit-learn).
        Doesn't learn anything from the data, just returns self.
        """
        return self

    def transform(self, X, y=None):
        """
        Applies cleaning to the input text data X.
        Determines the input type and applies cleaning accordingly.
        """
        # Helper function to clean a single piece of text
        def clean_single_text(text):
            """Cleans a single string."""
            try:
                # Ensure input is treated as string, handle None
                text_str = str(text) if text is not None else ""
                # Remove unwanted characters using the pre-compiled pattern
                cleaned = self.unwanted_chars_pattern.sub('', text_str)
                # Normalize multiple spaces using the pre-compiled pattern
                cleaned = self.multiple_spaces_pattern.sub(' ', cleaned)
                # Convert to lowercase and remove leading/trailing whitespace
                return cleaned.lower().strip()
            except Exception as e:
                # Log error and return empty string if cleaning fails unexpectedly
                print(f"⚠️ Warning: Error cleaning text: '{text}'. Error: {e}")
                return ""

        # Apply the cleaning function based on the input type
        if isinstance(X, pd.Series):
            # Apply element-wise if it's a pandas Series
            return X.apply(clean_single_text)
        elif isinstance(X, list):
            # Apply to each element if it's a list using list comprehension
            return [clean_single_text(text) for text in X]
        elif isinstance(X, str):
            # Handle single string input directly
            return clean_single_text(X)
        else:
            # Handle unexpected input types
            print(f"⚠️ Warning: Unexpected input type for TextCleaner: {type(X)}. Attempting conversion to string.")
            try:
                # Try cleaning its string representation as a fallback
                return clean_single_text(str(X))
            except Exception:
                 # Return empty string if conversion/cleaning fails
                print(f"❌ Error: Could not process input of type {type(X)}.")
                return ""

# Example usage (optional, runs only if the script is executed directly)
if __name__ == '__main__':
    cleaner = TextCleaner()
    test_texts = [
        "   Hello    World! 123  أهلاً بالعالم   ", # Mixed languages, numbers, extra spaces
        "It's great_day.", # Punctuation, underscore
        None, # None input
        12345, # Numeric input
        ["Another test.", "اختبار آخر مع _رموز_!@#"], # List input
        pd.Series(["  Check pandas.", "  مرحباً بك "]) # pandas Series input
    ]

    print("\n--- Testing TextCleaner ---")
    for text_input in test_texts:
        cleaned_output = cleaner.transform(text_input)
        print(f"Input : {text_input} ({type(text_input).__name__})")
        print(f"Output: '{cleaned_output}' ({type(cleaned_output).__name__})\n---")