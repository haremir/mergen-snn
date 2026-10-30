import json
import requests
import datetime

token_path = r"C:\Users\PC\.config\colab-cli\token.json"
with open(token_path, "r") as f:
    data = json.load(f)

refresh_token = data["refresh_token"]
client_id = data["client_id"]
client_secret = data["client_secret"]

payload = {
    "client_id": client_id,
    "client_secret": client_secret,
    "refresh_token": refresh_token,
    "grant_type": "refresh_token"
}

resp = requests.post("https://oauth2.googleapis.com/token", data=payload)
print("Status code:", resp.status_code)
print("Response:", resp.text)

if resp.status_code == 200:
    res_data = resp.json()
    new_access_token = res_data["access_token"]
    expires_in = res_data.get("expires_in", 3600)
    
    data["token"] = new_access_token
    # Expiry in UTC ISO format
    new_expiry = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=expires_in)
    data["expiry"] = new_expiry.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    with open(token_path, "w") as f:
        json.dump(data, f)
    print("Token refreshed successfully! New expiry:", data["expiry"])
