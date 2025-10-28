import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Optional # Added Optional
import re
from difflib import get_close_matches # ✅ Import for fuzzy matching

class RecommendationAgent:
    """
    Provides comprehensive recommendations for a given disease.
    - Loads data on initialization and uses dictionaries for fast lookups.
    - Standardizes disease names for reliable matching.
    - ✅ Includes fallback for partial AND fuzzy disease name matches.
    - ✅ Uses a static method for name standardization.
    - ✅ Added __repr__ for better debugging.
    """

    def __init__(self, fuzzy_match_cutoff: float = 0.8): # Added cutoff parameter
        """
        Initializes the agent, loads data files, processes them into lookup maps.

        Args:
            fuzzy_match_cutoff: Similarity threshold (0.0 to 1.0) for fuzzy matching.
        """
        base_dir = Path(__file__).resolve().parents[2] # Assumes this file is in chatbot_system
        data_dir = base_dir / "data"

        self.precaution_map: Dict[str, List[str]] = {}
        self.description_map: Dict[str, str] = {}
        self.fuzzy_match_cutoff: float = max(0.0, min(1.0, fuzzy_match_cutoff)) # Ensure cutoff is valid

        # --- Load and Process Precautions Data ---
        try:
            precaution_file = data_dir / "symptom_precaution.csv"
            df_prec = pd.read_csv(precaution_file)
            df_prec.columns = [str(col).strip().lower() for col in df_prec.columns]

            if 'disease' in df_prec.columns:
                # ✅ Use static method for standardization
                df_prec['disease_clean'] = df_prec['disease'].apply(self._standardize_name)
                # Drop rows where disease name became empty after cleaning
                df_prec.dropna(subset=['disease_clean'], inplace=True)
                df_prec = df_prec[df_prec['disease_clean'] != '']

                precaution_cols = [col for col in df_prec.columns if 'precaution' in col]

                # Group by cleaned disease name to collect all precautions
                grouped = df_prec.groupby('disease_clean')

                for disease_key, group in grouped:
                    precautions_set: Set[str] = set()
                    for _, row in group.iterrows():
                        for col in precaution_cols:
                            precaution_text = row.get(col)
                            if pd.notna(precaution_text) and str(precaution_text).strip():
                                cleaned_prec = str(precaution_text).strip().capitalize()
                                precautions_set.add(cleaned_prec)
                    if precautions_set:
                        self.precaution_map[disease_key] = sorted(list(precautions_set))

                print(f"✅ RecommendationAgent: Precautions data loaded and mapped for {len(self.precaution_map)} diseases.")
            else:
                print(f"⚠️ RecommendationAgent: Precaution file ({precaution_file.name}) missing 'disease' column.")

        except FileNotFoundError:
            print(f"⚠️ RecommendationAgent: Precaution file not found at {precaution_file}.")
        except Exception as e:
            print(f"⚠️ RecommendationAgent: Error loading/processing precaution data: {e}")

        # --- Load and Process Descriptions Data ---
        try:
            description_file = data_dir / "symptom_Description.csv"
            df_desc = pd.read_csv(description_file)
            df_desc.columns = [str(col).strip().lower() for col in df_desc.columns]

            # Handle potential misnaming ('symptom' instead of 'disease')
            if 'disease' not in df_desc.columns and 'symptom' in df_desc.columns:
                 print(f"ℹ️ RecommendationAgent: Renaming 'symptom' to 'disease' in {description_file.name}.")
                 df_desc.rename(columns={'symptom': 'disease'}, inplace=True)

            if 'disease' in df_desc.columns and 'description' in df_desc.columns:
                # ✅ Use static method for standardization
                df_desc['disease_clean'] = df_desc['disease'].apply(self._standardize_name)
                # Drop rows if disease name or description is empty after cleaning/conversion
                df_desc.dropna(subset=['disease_clean', 'description'], inplace=True)
                df_desc = df_desc[df_desc['disease_clean'] != '']
                df_desc['description'] = df_desc['description'].astype(str).str.strip() # Ensure string and strip
                df_desc = df_desc[df_desc['description'] != '']

                # Create map, keeping only the first entry if duplicates exist for disease_clean
                df_desc_unique = df_desc.drop_duplicates(subset=['disease_clean'], keep='first')
                self.description_map = pd.Series(
                    df_desc_unique.description.values,
                    index=df_desc_unique.disease_clean
                ).to_dict()

                print(f"✅ RecommendationAgent: Descriptions data loaded and mapped for {len(self.description_map)} diseases.")
            else:
                print(f"⚠️ RecommendationAgent: Description file ({description_file.name}) missing required 'disease' or 'description' columns.")

        except FileNotFoundError:
            print(f"⚠️ RecommendationAgent: Description file not found at {description_file}.")
        except Exception as e:
            print(f"⚠️ RecommendationAgent: Error loading/processing description data: {e}")
        # --- End Load and Process Data ---


    # ==========================
    # 🔹 Static Helper for Name Standardization
    # ==========================
    @staticmethod
    def _standardize_name(name: str) -> str:
        """Standardizes a disease name (lowercase, strip, underscores to spaces, normalize spaces)."""
        if not isinstance(name, str):
            return ""
        try:
            # Lowercase, strip leading/trailing whitespace
            s = name.strip().lower()
            # Replace underscores with spaces
            s = s.replace('_', ' ')
            # Replace multiple whitespace characters with a single space
            s = re.sub(r'\s+', ' ', s)
            return s.strip() # Strip again just in case
        except Exception as e:
            print(f"⚠️ Error standardizing name '{name}': {e}")
            return ""

    # ==========================
    # 🔹 Fuzzy Match Helper
    # ==========================
    def _fuzzy_match(self, name: str, options: List[str]) -> Optional[str]:
        """Find the best fuzzy match for a name within a list of options."""
        if not name or not options:
            return None
        # Use get_close_matches to find the best match (n=1)
        matches = get_close_matches(name, options, n=1, cutoff=self.fuzzy_match_cutoff)
        if matches:
            return matches[0] # Return the best match found
        return None # Return None if no close match is found


    # ==========================
    # 🔹 Internal helper methods (Now using map + partial + fuzzy fallback)
    # ==========================

    def _get_precautions(self, disease_name_clean: str) -> List[str]:
        """Retrieve precautions using map, with partial and fuzzy match fallbacks."""
        # 1. Try direct match (fastest)
        result = self.precaution_map.get(disease_name_clean)
        if result is not None: # Check explicitly for None, as empty list [] is a valid (but empty) result
            return result

        # 2. Fallback: Try partial matching (simple substring check)
        # Check only if name is reasonably long to avoid too many false positives
        if len(disease_name_clean) >= 4:
            # Check if input is substring of any key
            partial_matches = [key for key in self.precaution_map if disease_name_clean in key]
            if partial_matches:
                 # Optional: prioritize shorter keys or exact word matches if multiple partial found
                 best_partial = min(partial_matches, key=len) # Simple heuristic: shortest key is often better
                 print(f"ℹ️ RecommendationAgent: Partial match found for precautions: '{disease_name_clean}' -> '{best_partial}'")
                 return self.precaution_map.get(best_partial, [])

            # Check if any key is substring of input (less common but possible)
            key_as_substring = [key for key in self.precaution_map if key in disease_name_clean]
            if key_as_substring:
                 best_key_substring = max(key_as_substring, key=len) # Longest key match
                 print(f"ℹ️ RecommendationAgent: Partial match (key in input) found for precautions: '{disease_name_clean}' -> '{best_key_substring}'")
                 return self.precaution_map.get(best_key_substring, [])


        # 3. Fallback: Try fuzzy matching
        fuzzy_key = self._fuzzy_match(disease_name_clean, list(self.precaution_map.keys()))
        if fuzzy_key:
             print(f"ℹ️ RecommendationAgent: Fuzzy match found for precautions: '{disease_name_clean}' -> '{fuzzy_key}' (Score > {self.fuzzy_match_cutoff})")
             return self.precaution_map.get(fuzzy_key, []) # Get precautions for the fuzzy matched key

        # 4. Not found
        # print(f"_get_precautions: No match found for '{disease_name_clean}'") # For debugging
        return []

    def _get_description(self, disease_name_clean: str) -> str:
        """Retrieve description using map, with partial and fuzzy match fallbacks."""
        # 1. Try direct match
        result = self.description_map.get(disease_name_clean)
        if result is not None: # Check for None, empty string "" is a valid description
            return result

        # 2. Fallback: Try partial matching
        if len(disease_name_clean) >= 4:
             partial_matches = [key for key in self.description_map if disease_name_clean in key]
             if partial_matches:
                  best_partial = min(partial_matches, key=len)
                  print(f"ℹ️ RecommendationAgent: Partial match found for description: '{disease_name_clean}' -> '{best_partial}'")
                  return self.description_map.get(best_partial, "")

             key_as_substring = [key for key in self.description_map if key in disease_name_clean]
             if key_as_substring:
                  best_key_substring = max(key_as_substring, key=len)
                  print(f"ℹ️ RecommendationAgent: Partial match (key in input) found for description: '{disease_name_clean}' -> '{best_key_substring}'")
                  return self.description_map.get(best_key_substring, "")

        # 3. Fallback: Try fuzzy matching
        fuzzy_key = self._fuzzy_match(disease_name_clean, list(self.description_map.keys()))
        if fuzzy_key:
            print(f"ℹ️ RecommendationAgent: Fuzzy match found for description: '{disease_name_clean}' -> '{fuzzy_key}' (Score > {self.fuzzy_match_cutoff})")
            return self.description_map.get(fuzzy_key, "") # Get description for the fuzzy matched key

        # 4. Not found
        # print(f"_get_description: No match found for '{disease_name_clean}'") # For debugging
        return ""


    # ==========================
    # 🔹 Main public methods
    # ==========================

    def get_precautions(self, disease_name: str) -> List[str]:
        """Public method to retrieve precautions for a given disease."""
        # Standardize the input disease name for lookup
        disease_key = self._standardize_name(disease_name)
        if not disease_key: return [] # Handle empty standardized name
        return self._get_precautions(disease_key)

    def get_details(self, disease_name: str) -> Dict:
        """Retrieves a dictionary with all available details using fast lookups + fallback."""
        # Standardize the input disease name for consistent lookup
        disease_key = self._standardize_name(disease_name)

        # Get data using internal methods which handle fallbacks
        precautions = self._get_precautions(disease_key) if disease_key else []
        description = self._get_description(disease_key) if disease_key else ""

        # Default messages if data not found even after fallbacks
        if not precautions:
            precautions = ["No specific precautions found. Please consult a healthcare professional."]
        if not description:
            description = "No detailed description is available for this condition."

        # Placeholders - consider loading these from data files too if needed
        tests = ["General blood test", "Physical examination"]
        advice = f"Based on the available information: {description} It is strongly recommended to consult a healthcare professional for an accurate diagnosis and personalized advice."

        return {
            "precautions": precautions,
            "description": description,
            "tests": tests, # Placeholder
            "advice": advice # Placeholder
        }

    def get_recommendations(self, disease_name: str) -> Dict:
        """Alias for backward compatibility."""
        return self.get_details(disease_name)

    # ==========================
    # 🔹 Debugging Representation
    # ==========================
    def __repr__(self):
        """Readable representation of the agent's loaded data state."""
        return (f"<RecommendationAgent: {len(self.precaution_map)} diseases with precautions, "
                f"{len(self.description_map)} with descriptions>")

# Example Usage (optional) - updated for fuzzy matching test
if __name__ == '__main__':
    agent = RecommendationAgent(fuzzy_match_cutoff=0.7) # Lower cutoff for more likely fuzzy matches
    print(agent) # Test __repr__

    test_disease_exact = "diabetes"
    test_disease_partial = "dia" # Should trigger partial match
    test_disease_typo = "diabates" # Should trigger fuzzy match
    test_disease_fuzzy = "diabetez" # Should trigger fuzzy match
    test_disease_unrelated = "common cold"
    test_disease_unknown = "UnknownConditionXYZ" # Should trigger no match

    print(f"\n--- Testing Agent Lookups ---")

    def test_lookup(name):
        print(f"\nTesting lookup for: '{name}' (Standardized: '{agent._standardize_name(name)}')")
        details = agent.get_details(name)
        desc_found = details['description'] != 'No detailed description is available for this condition.'
        prec_found = details['precautions'][0] != 'No specific precautions found. Please consult a healthcare professional.'
        print(f" Description found: {desc_found}")
        print(f" Precautions found: {prec_found}")
        # Optionally print the actual results for debugging
        # print(f" -> Desc: {details['description'][:50]}...")
        # print(f" -> Prec: {details['precautions']}")

    test_lookup(test_disease_exact)
    test_lookup(test_disease_partial)
    test_lookup(test_disease_typo)
    test_lookup(test_disease_fuzzy)
    test_lookup(test_disease_unrelated)
    test_lookup(test_disease_unknown)