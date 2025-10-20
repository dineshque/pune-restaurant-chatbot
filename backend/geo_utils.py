import requests

def geocode_location(location_text: str):
    """
    Convert a textual location (e.g., 'FC Road, Pune') into latitude & longitude using OpenStreetMap Nominatim.
    """
    if not location_text or not isinstance(location_text, str):
        return None, None

    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": location_text,
            "format": "json",
            "limit": 1
        }
        headers = {"User-Agent": "RestaurantChatbot/1.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=8)
        data = resp.json()
        if len(data) > 0:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            print(f"📍 Geocoded '{location_text}' → ({lat}, {lon})")
            return lat, lon
        else:
            print(f"⚠️  Geocoding failed: No results for '{location_text}'")
            return None, None
    except Exception as e:
        print(f"❌ Geocoding error for '{location_text}': {e}")
        return None, None
