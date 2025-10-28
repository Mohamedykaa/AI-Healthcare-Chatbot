# src/pharmacy_locator_osm.py

import requests
from geopy.distance import geodesic

class PharmacyLocator:
    """
    Class to fetch nearby pharmacies using OpenStreetMap Overpass API.
    ✅ MODIFIED to align with app_streamlit.py expectations.
    """

    OVERPASS_URL = "https://overpass-api.de/api/interpreter"

    # ✅ --- MODIFIED ---
    # __init__ is no longer needed as radius is passed directly to the method.
    # We can remove it for simplicity.
    # def __init__(self, radius: int = 5000):
    #     self.radius = radius

    # ✅ --- MODIFIED ---
    # Updated signature to accept 'radius' and 'medicine_name'
    def get_nearby_pharmacies(self, latitude: float, longitude: float, radius: int, medicine_name: str = None):
        """
        Fetch nearby pharmacies.
        :param latitude: User latitude
        :param longitude: User longitude
        :param radius: Search radius in meters (passed from UI)
        :param medicine_name: Optional name to filter by (passed from UI)
        :return: Dict with "results" list or "error" message
        """
        
        # ✅ --- MODIFIED ---
        # Build the query dynamically
        name_filter = ""
        if medicine_name:
            # Add a filter for the name, case-insensitive (",i")
            name_filter = f'["name"~"{medicine_name}",i]'

        query = f"""
        [out:json];
        (
          node["amenity"="pharmacy"]{name_filter}(around:{radius},{latitude},{longitude});
        );
        out center;
        """

        try:
            response = requests.get(self.OVERPASS_URL, params={'data': query}, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"Error fetching data from Overpass API: {e}")
            # ✅ --- MODIFIED ---
            # Return dict matching Streamlit app's error check
            return {"error": f"API request failed: {e}", "results": []}

        user_coords = (latitude, longitude)
        pharmacies = []

        for element in data.get("elements", []):
            tags = element.get("tags", {})
            
            # Use fallback for address
            addr = tags.get("addr:full", tags.get("addr:street", "No address provided"))
            
            name = tags.get("name", "Unknown Pharmacy")
            phone = tags.get("phone", "No phone number provided")
            website = tags.get("website", "No website provided")
            lat, lon = element["lat"], element["lon"]

            distance_km = round(geodesic(user_coords, (lat, lon)).kilometers, 2)

            pharmacies.append({
                "name": name,
                "address": addr,
                "latitude": lat,
                "longitude": lon,
                "distance_km": distance_km,
                "phone": phone,
                "website": website
            })

        # Sort results by distance
        if pharmacies:
            pharmacies.sort(key=lambda p: p["distance_km"])

        # ✅ --- MODIFIED ---
        # Return dict matching Streamlit app's success check
        return {"results": pharmacies}


# ✅ --- MODIFIED ---
# Updated example usage to match new method
if __name__ == "__main__":
    locator = PharmacyLocator()
    
    # Example coordinates (Cairo, Egypt)
    latitude = 30.0444
    longitude = 31.2357
    search_radius = 5000
    
    print(f"--- Searching for ALL pharmacies within {search_radius}m ---")
    results_all = locator.get_nearby_pharmacies(latitude, longitude, radius=search_radius)
    
    if not results_all.get("results"):
        print("No pharmacies found nearby.")
    else:
        print(f"Found {len(results_all.get('results', []))} pharmacies:")
        for place in results_all.get("results", [])[:5]: # Print top 5
            print(f"- {place['name']} ({place['distance_km']} km)")
            
    print("\n" + "="*40 + "\n")

    print(f"--- Searching for pharmacies matching 'Seif' within {search_radius}m ---")
    results_filtered = locator.get_nearby_pharmacies(latitude, longitude, radius=search_radius, medicine_name="Seif")
    
    if not results_filtered.get("results"):
        print("No pharmacies found with that name.")
    else:
        print(f"Found {len(results_filtered.get('results', []))} matching pharmacies:")
        for place in results_filtered.get("results", []):
            print(f"- {place['name']} ({place['distance_km']} km)")
            print(f"  Address: {place['address']}")
            print(f"  Phone: {place['phone']}")
            print("-" * 40)