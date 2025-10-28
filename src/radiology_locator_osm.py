# src/radiology_locator_osm.py

import requests
from geopy.distance import geodesic

class RadiologyLocatorOSM:
    """
    Class to fetch nearby radiology centers, clinics, and hospitals
    using OpenStreetMap Overpass API.
    ✅ MODIFIED to align with app_streamlit.py expectations.
    """

    OVERPASS_URL = "https://overpass-api.de/api/interpreter"

    # ✅ --- MODIFIED ---
    # Removed __init__ as radius is now passed directly to the method.
    # def __init__(self, radius: int = 5000):
    #     self.radius = radius

    # ✅ --- MODIFIED ---
    # Renamed method and updated signature to accept radius
    def get_nearby_radiology_centers(self, latitude: float, longitude: float, radius: int):
        """
        Fetch nearby healthcare centers related to radiology.
        :param latitude: User latitude
        :param longitude: User longitude
        :param radius: Search radius in meters (passed from UI)
        :return: Dict with "results" list or "error" message
        """
        
        # ✅ --- MODIFIED ---
        # Use the 'radius' variable from arguments, not 'self.radius'
        query = f"""
        [out:json];
        (
          node["healthcare"="radiology"](around:{radius},{latitude},{longitude});
          node["amenity"="clinic"]["healthcare:speciality"~"radiology|diagnostic"](around:{radius},{latitude},{longitude});
          node["healthcare"="laboratory"](around:{radius},{latitude},{longitude});
          node["amenity"="hospital"](around:{radius},{latitude},{longitude});
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
        places = []

        for element in data.get("elements", []):
            tags = element.get("tags", {})
            name = tags.get("name", "Unknown location")
            
            # Use fallback for address
            addr = tags.get("addr:full", tags.get("addr:street", "No address provided"))
            
            phone = tags.get("phone", "No phone number provided")
            website = tags.get("website", "No website provided")
            lat, lon = element["lat"], element["lon"]

            distance_km = round(geodesic(user_coords, (lat, lon)).kilometers, 2)

            places.append({
                "name": name,
                "address": addr,
                "latitude": lat,
                "longitude": lon,
                "distance_km": distance_km,
                "phone": phone,
                "website": website
            })

        # Sort results by distance
        if places:
            places.sort(key=lambda p: p["distance_km"])

        # ✅ --- MODIFIED ---
        # Return dict matching Streamlit app's success check
        return {"results": places}


# ✅ --- MODIFIED ---
# Updated example usage to match new method
if __name__ == "__main__":
    # Example usage
    locator = RadiologyLocatorOSM()
    
    # Example coordinates (Cairo, Egypt)
    latitude = 30.0444
    longitude = 31.2357
    search_radius = 5000 # Radius is now passed to the method
    
    results_data = locator.get_nearby_radiology_centers(latitude, longitude, search_radius)
    
    results_list = results_data.get("results", [])
    
    if not results_list:
        print("No radiology centers found nearby.")
    else:
        print(f"Nearby Radiology Centers (Found {len(results_list)}):")
        for place in results_list:
            print(f"- {place['name']} ({place['distance_km']} km)")
            print(f"  Address: {place['address']}")
            print(f"  Phone: {place['phone']}")
            print(f"  Website: {place['website']}")
            print("-" * 40)