import requests
import json

# Standard solved state using faces:
# UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB
# Standard solved state using colors (from frontend mapping):
# WWWWWWWWWRRRRRRRRRGGGGGGGGGYYYYYYYYYOOOOOOOOOBBBBBBBBB

url = "http://localhost:5000/solve"
# Try solved state with faces (should work)
payload_faces = {"cube_string": "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"}
# Try solved state with colors (should fail if not mapped)
payload_colors = {"cube_string": "WWWWWWWWWRRRRRRRRRGGGGGGGGGYYYYYYYYYOOOOOOOOOBBBBBBBBB"}

print("Testing with faces...")
try:
    r = requests.post(url, json=payload_faces)
    print(f"Status: {r.status_code}, Response: {r.json()}")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting with colors...")
try:
    r = requests.post(url, json=payload_colors)
    print(f"Status: {r.status_code}, Response: {r.json()}")
except Exception as e:
    print(f"Error: {e}")
